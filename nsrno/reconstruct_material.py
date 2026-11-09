"""First real joint c/rho inversion through the frozen round-7 surrogate.

Only RF and explicitly known alpha enter inversion. Material truth is read
by a separate postprocessing script after the inverse run has finished.
"""
from argparse import ArgumentParser
from pathlib import Path
import hashlib,json,math,time,os
import numpy as np
import torch
from torch.nn import functional as F
from nsrno.conditioned_surrogate import ConditionedSurrogate
from nsrno.conditioned_data import ConditionedData

DEFAULT_OUT=Path('out/v5/reconstruction_round1_train_200')

def write_json(path,value):
    temporary=path.with_suffix('.partial.json'); temporary.write_text(json.dumps(value,indent=2)); temporary.replace(path)

def build_medium(q,mask):
    scales=torch.tensor([.06,.08],device=q.device)[:,None,None]
    base=torch.tensor([1540.,1000.],device=q.device)[:,None,None]
    return base*torch.exp(scales*q*mask)

def regularization(q,mask):
    values=q*mask
    dx=values[:,:-1,1:]-values[:,:-1,:-1]
    dz=values[:,1:,:-1]-values[:,:-1,:-1]
    tv=(torch.sqrt(dx.square()+dz.square()+1e-6)-.001).mean()
    size=values.square().mean()
    return .002*tv+.0001*size,tv,size

def main():
    p=ArgumentParser(); p.add_argument('--id',default='train_200'); p.add_argument('--steps',type=int,default=200)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT); p.add_argument('--iterations',type=int,default=16)
    p.add_argument('--resume',action='store_true'); p.add_argument('--fixed-gel-mm',type=float,default=1.); args=p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(20261105); np.random.seed(20261105)
    args.out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path('out/v5/forward_round7_conditioned/last.pt')
    data=ConditionedData(); row=next(r for r in data.rows if r['id']==args.id)
    if row['role']!='validation' or row['case']!='Neg_35_Left': raise ValueError('Use fixed independent development case')
    model=ConditionedSurrogate(checkpoint,'out/v5/conditioned_wave_v1','cuda')
    model.requires_grad_(False); g=model.geometry
    alpha=torch.from_numpy(np.load(row['alpha'])).cuda()
    pairs=[(b,a) for b in model.spec['bins'] for a in model.spec['angle_indices']]
    targets={(b,a):data.target(row,b,a) for b,a in pairs}
    fit=torch.arange(192,device='cuda')%8!=0; heldout=~fit
    # Fixed geometric source plane/gel only, not an anatomy-dependent mask.
    z=torch.arange(g.h,device='cuda')*g.dx+g.spec['z0_m']
    free_depth=(z>1e-7) if args.fixed_gel_mm==0 else (z>=args.fixed_gel_mm*.001-1e-7)
    support=free_depth[:,None].expand(g.h,g.w).float()
    q=torch.nn.Parameter(torch.zeros(2,g.h,g.w,device='cuda'))
    optimizer=torch.optim.Adam([q],lr=.003)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=.0003)
    lower=torch.tensor([math.log(1400/1540)/.06,math.log(850/1000)/.08],device='cuda')[:,None,None]
    upper=torch.tensor([math.log(1700/1540)/.06,math.log(1200/1000)/.08],device='cuda')[:,None,None]
    settings={'id':args.id,'case':row['case'],'role':row['role'],'steps':args.steps,'neural_iterations':args.iterations,
        'seed':20261105,'initial_c':1540.,'initial_rho':1000.,'bounds_c':[1400.,1700.],'bounds_rho':[850.,1200.],
        'native_grid':data.spec,'unknowns':'native independent log c and log rho; below known coupling layer',
        'fixed_gel_mm':args.fixed_gel_mm,'gel_source':'build_acoustic_l11.py: GEL_MM=1, rough_sigma_mm=0, nonnegative outer relief',
        'known_true_alpha':True,'truth_used_in_optimization':False,'fit_receivers':torch.where(fit)[0].cpu().tolist(),
        'heldout_receivers':torch.where(heldout)[0].cpu().tolist(),'pairs':pairs,'tv_weight':.002,'l2_weight':.0001,
        'learning_rate':.003,'final_learning_rate':.0003,'selection':'minimum fitting data + regularization; no material truth',
        'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'alpha_sha256':hashlib.sha256(Path(row['alpha']).read_bytes()).hexdigest()}
    write_json(args.out/'settings.json',settings)
    np.save(args.out/'initial_medium.npy',build_medium(q,support).detach().cpu().numpy())
    np.save(args.out/'known_alpha.npy',alpha.cpu().numpy())
    np.savez(args.out/'observations.npz',**{f'b{b}_a{a}':targets[b,a].cpu().numpy() for b,a in pairs})
    start_step=0; best=float('inf'); history=[]; best_step=-1; previous_elapsed=0.
    if args.resume:
        state=torch.load(args.out/'resume.pt',map_location='cuda',weights_only=True)
        with torch.no_grad(): q.copy_(state['q'])
        optimizer.load_state_dict(state['optimizer']); schedule.load_state_dict(state['scheduler'])
        start_step=state['next_step']; best=state['best']; best_step=state['best_step']
        history=json.load(open(args.out/'history.json')); previous_elapsed=history[-1]['elapsed_s']
    elif (args.out/'history.json').exists(): raise FileExistsError('Use a fresh output or --resume')
    started=time.perf_counter()
    for step in range(start_step,args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        fit_losses=[]; rows=[]; spectra={}; pde=[]
        for b,a in pairs:
            medium=build_medium(q,support)
            if step==args.steps:
                with torch.no_grad(): result=model(medium,alpha,b,a,iterations=args.iterations)
            else: result=model(medium,alpha,b,a,iterations=args.iterations)
            pred=result['channels']; target=targets[b,a]
            loss=(pred[fit]-target[fit]).abs().square().sum()/target[fit].abs().square().sum().clamp_min(1e-20)
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite inversion loss')
            if step<args.steps: (loss/len(pairs)).backward()
            fit_losses.append(float(loss.detach())); pde.append(float(result['relative_residual'].detach()))
            pred=pred.detach(); spectra[f'b{b}_a{a}']=pred.cpu().numpy()
            rows.append({'bin':b,'angle':a,'fit_error_sq':float((pred[fit]-target[fit]).abs().square().sum()),
                'fit_target_sq':float(target[fit].abs().square().sum()),
                'heldout_error_sq':float((pred[heldout]-target[heldout]).abs().square().sum()),
                'heldout_target_sq':float(target[heldout].abs().square().sum())})
            del result,pred,loss,medium
        penalty,tv,l2=regularization(q,support)
        if step<args.steps: penalty.backward()
        objective=float(np.mean(fit_losses))+float(penalty.detach())
        record={'step':step,'objective':objective,'equal_acquisition_fit_rms':float(np.mean(fit_losses))**.5,
            'fit_rms':math.sqrt(sum(r['fit_error_sq'] for r in rows)/sum(r['fit_target_sq'] for r in rows)),
            'heldout_rms':math.sqrt(sum(r['heldout_error_sq'] for r in rows)/sum(r['heldout_target_sq'] for r in rows)),
            'tv':float(tv.detach()),'l2':float(l2.detach()),'mean_pde_residual':float(np.mean(pde)),
            'lr':optimizer.param_groups[0]['lr'],'elapsed_s':previous_elapsed+time.perf_counter()-started}
        if step<args.steps:
            if not bool(torch.isfinite(q.grad).all()): raise RuntimeError('Nonfinite material gradient')
            record['gradient_norms']=[float(q.grad[i].norm()) for i in range(2)]
        history.append(record)
        if objective<best:
            best=objective; best_step=step
            np.save(args.out/'reconstructed_medium.npy',build_medium(q,support).detach().cpu().numpy())
            np.savez(args.out/'predicted_spectra.npz',**spectra)
            write_json(args.out/'selected.json',{'step':step,'objective':objective,'metrics':record,'observations':rows})
        if step%25==0 or step==args.steps:
            np.save(args.out/f'medium_step_{step:03d}.npy',build_medium(q,support).detach().cpu().numpy())
            write_json(args.out/'history.json',history)
            print(json.dumps(record),flush=True)
        elif step%5==0: print(json.dumps(record),flush=True)
        if step<args.steps:
            optimizer.step(); schedule.step()
            with torch.no_grad(): q.copy_(torch.maximum(torch.minimum(q,upper),lower)*support)
            if step%25==0:
                torch.save({'q':q.detach(),'optimizer':optimizer.state_dict(),'scheduler':schedule.state_dict(),
                    'next_step':step+1,'best':best,'best_step':best_step},args.out/'resume.pt')
    np.save(args.out/'last_medium.npy',build_medium(q,support).detach().cpu().numpy())
    write_json(args.out/'completion.json',{'completed':True,'updates':args.steps,'selected_step':best_step,'best_objective':best,
        'seconds':previous_elapsed+time.perf_counter()-started,'truth_used_in_optimization':False})

if __name__=='__main__': main()
