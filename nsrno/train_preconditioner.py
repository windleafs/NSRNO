"""Learn GMRES correction directions toward converged physical fields."""
from argparse import ArgumentParser
from pathlib import Path
import json,random,time,math
import numpy as np
import torch
from nsrno.conditioned_wave import LearnedPreconditioner,differentiable_gmres
from nsrno.conditioned_data import ConditionedData,ROOT,receiver_response

def errors(system,rhs,u,truth,target):
    g=system.g
    field=((u-truth).abs().square().sum()/truth.abs().square().sum().clamp_min(1e-20))
    residual=((system(u)-rhs).abs().square().sum()/rhs.abs().square().sum().clamp_min(1e-20))
    channels=g.channels(u); exact=g.channels(truth)
    return field,residual,dict(field_relative_rms=float(field.detach().sqrt()),
        pde_relative_residual=float(residual.detach().sqrt()),
        teacher_channel_error_sq=float((channels-exact).abs().square().sum().detach()),
        teacher_channel_target_sq=float(exact.abs().square().sum()),
        rf_error_sq=float((channels*receiver_response(system.frequency)-target).abs().square().sum().detach()),
        rf_target_sq=float(target.abs().square().sum()))

def aggregate(rows):
    return dict(observations=len(rows),field_relative_rms=math.sqrt(np.mean([r['field_relative_rms']**2 for r in rows])),
        pde_relative_residual=math.sqrt(np.mean([r['pde_relative_residual']**2 for r in rows])),
        teacher_channel_relative_rms=math.sqrt(sum(r['teacher_channel_error_sq'] for r in rows)/sum(r['teacher_channel_target_sq'] for r in rows)),
        rf_relative_rms=math.sqrt(sum(r['rf_error_sq'] for r in rows)/sum(r['rf_target_sq'] for r in rows)),
        seconds_mean=float(np.mean([r['seconds'] for r in rows])) if rows and 'seconds' in rows[0] else None)

@torch.no_grad()
def evaluate(model,data,examples,iterations=8,learned=True):
    model.eval(); rows=[]
    for row,bin_index,angle in examples:
        truth=data.teacher(row,bin_index,angle); target=data.target(row,bin_index,angle)
        # Includes per-input physics assembly and preconditioner setup, excludes
        # file IO, metrics, and precomputed acquisition-only homogeneous fields.
        torch.cuda.synchronize(); start=time.perf_counter()
        system,rhs,_=data.assemble(row,bin_index,angle)
        precondition=model.setup(system) if learned else system.fixed_preconditioner()
        u=differentiable_gmres(system,precondition,rhs,iterations)
        torch.cuda.synchronize(); seconds=time.perf_counter()-start
        _,_,metrics=errors(system,rhs,u,truth,target)
        rows.append({'id':row['id'],'role':row['role'],'bin':bin_index,'angle':angle,'seconds':seconds,**metrics})
    return rows

def main():
    parser=ArgumentParser(); parser.add_argument('--steps',type=int,default=240); parser.add_argument('--iterations',type=int,default=8)
    args=parser.parse_args(); torch.set_num_threads(2); torch.manual_seed(20261105); random.seed(20261105)
    out=Path('out/v5/forward_round7_conditioned'); out.mkdir(parents=True,exist_ok=True)
    data=ConditionedData(); model=LearnedPreconditioner().cuda()
    train=[(r,b,a) for r in data.rows if r['role']=='train' for b in data.spec['bins'] for a in data.spec['angle_indices']]
    validation=[(r,b,a) for r in data.rows if r['role']=='validation' for b in data.spec['bins'] for a in data.spec['angle_indices']]
    # Fixed, anatomy-separated monitoring subset; final evaluation uses all144.
    val_rows=[r for r in data.rows if r['role']=='validation']
    monitor=[(val_rows[j],b,5) for j in (0,5,10,15) for b in data.spec['bins']]
    settings={'steps':args.steps,'iterations':args.iterations,'seed':20261105,'parameters':sum(p.numel() for p in model.parameters()),
              'loss':'relative field squared error + relative PDE residual squared',
              'lr':.002,'training_observations':len(train),'validation_observations':len(validation),
              'known_alpha':True,'spec':data.spec,'selection':'report final step; no RF validation checkpoint selection'}
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    baseline=evaluate(model,data,monitor,args.iterations,False)
    (out/'initial_monitor.json').write_text(json.dumps({'summary':aggregate(baseline),'rows':baseline},indent=2))
    print(json.dumps({'initial':aggregate(baseline)}),flush=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=.0002)
    random.shuffle(train); started=time.perf_counter(); history=[]
    for step in range(1,args.steps+1):
        if (step-1)%len(train)==0 and step>1: random.shuffle(train)
        row,bin_index,angle=train[(step-1)%len(train)]
        system,rhs,_=data.assemble(row,bin_index,angle)
        truth=data.teacher(row,bin_index,angle); target=data.target(row,bin_index,angle)
        model.train(); optimizer.zero_grad(set_to_none=True)
        u=differentiable_gmres(system,model.setup(system),rhs,args.iterations)
        field,residual,metrics=errors(system,rhs,u,truth,target); loss=field+residual
        if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite loss')
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not bool(torch.isfinite(grad_norm)): raise RuntimeError('Nonfinite parameter gradient')
        optimizer.step(); schedule.step()
        record={'step':step,'id':row['id'],'bin':bin_index,'angle':angle,'loss':float(loss.detach()),
                'grad_norm':float(grad_norm),'shift':float(model.log_shift.exp()),'elapsed_s':time.perf_counter()-started,**metrics}
        history.append(record)
        if step%10==0 or step==1: print(json.dumps(record),flush=True)
        if step%60==0 or step==args.steps:
            result=evaluate(model,data,monitor,args.iterations,True)
            summary=aggregate(result)
            (out/f'monitor_{step}.json').write_text(json.dumps({'summary':summary,'rows':result},indent=2))
            torch.save({'model':model.state_dict(),'spec':data.spec,'iterations':args.iterations,'step':step,
                        'optimizer':optimizer.state_dict(),'settings':settings},out/'last.pt')
            (out/'history.json').write_text(json.dumps(history,indent=2))
            print(json.dumps({'validation_step':step,**summary}),flush=True)
    final={}
    for iterations in (8,16):
        for learned in (False,True):
            name=('learned' if learned else 'fixed')+str(iterations)
            rows=evaluate(model,data,validation,iterations,learned)
            final[name]={'summary':aggregate(rows),'rows':rows}
            (out/'validation.json').write_text(json.dumps(final,indent=2))
            print(json.dumps({name:aggregate(rows)}),flush=True)
    # Both material inputs must remain differentiable at fixed known alpha.
    row,bin_index,angle=validation[0]
    system,rhs,medium=data.assemble(row,bin_index,angle,requires_grad=True)
    u=differentiable_gmres(system,model.setup(system),rhs,args.iterations)
    objective=system.g.channels(u).abs().square().sum()
    gradient=torch.autograd.grad(objective,medium)[0]
    check={'c_gradient_norm':float(gradient[0].norm()),'rho_gradient_norm':float(gradient[1].norm()),
        'all_finite':bool(torch.isfinite(gradient).all()),'note':'finite gradients only; no assertion of converged Jacobian accuracy'}
    (out/'input_gradient_check.json').write_text(json.dumps(check,indent=2))
    if not check['all_finite'] or min(check['c_gradient_norm'],check['rho_gradient_norm'])<=0: raise RuntimeError('Invalid input gradients')
    (out/'completion.json').write_text(json.dumps({'step':args.steps,'finished':True,'gradient_check':check},indent=2))

if __name__=='__main__': main()
