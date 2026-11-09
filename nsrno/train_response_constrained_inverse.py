"""Primal/dual training of independent attenuation response and invariance."""
from pathlib import Path
import argparse,copy,hashlib,json,math,os,time,traceback
import numpy as np
import torch
from .angular_spectral_inverse_uno import AngularSpectralInverseUNO,load_batch,map_loss,pair_loss
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=Path,default=Path('out/v5/inverse_angular_spectral_v1/index.json'))
    p.add_argument('--warmstart',type=Path,default=Path('out/v5/inverse_angular_spectral_uno_v1/weights.pt'))
    p.add_argument('--after-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--device',type=int,default=0)
    args=p.parse_args()
    prior=json.loads((args.after_root/'run.json').read_text())
    if prior['status']!='fixed_training_pairs_evaluated':raise RuntimeError('Complete the fixed-model diagnosis first')
    if Path(f"/proc/{prior['pid']}").exists():raise RuntimeError('Prior GPU diagnosis process still exists')
    if args.out.exists():raise RuntimeError('Inspect existing constrained training; no implicit restart')
    index=json.loads(args.index.read_text())
    if index['completed']!=index['planned_records']:raise RuntimeError('Missing prepared features')
    train=[r for r in index['records'] if r['role']=='train']
    if {r['case'] for r in train}!={'Neg_07_Left'}:raise ValueError('Development contamination')
    paired=[r for r in train if 'lf' in r['features']]
    unpaired=[r for r in train if 'lf' not in r['features']]
    groups={}
    for row in train:
        if 'pair_group' in row:groups.setdefault(row['pair_group'],[]).append(row)
    pairs=[sorted(v,key=lambda r:r['pair_sign']) for v in groups.values() if len(v)==2]
    alpha=[v for v in pairs if v[0]['kind']=='alpha']
    trajectory=[v for v in pairs if v[0]['kind']=='trajectory']
    if not alpha or not trajectory or not paired or not unpaired:raise ValueError('Missing declared training strata')
    for minus,plus in alpha:
        a=np.load(minus['label']);b=np.load(plus['label'])
        if np.max(abs(a[:2]-b[:2]))>1e-3:raise ValueError('Alpha-only invariance labels changed c or rho')
    args.out.mkdir()
    mean=sum(np.load(r['label']).astype(np.float64) for r in train if r['kind']=='original')/sum(r['kind']=='original' for r in train)
    np.save(args.out/'training_mean_material.npy',mean.astype(np.float32))
    names=['train_response_constrained_inverse.py','angular_spectral_inverse_uno.py',
           'inverse_uno.py','uno_rf.py','angular_spectral_features.py']
    code={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in names}
    targets=np.array([.16,.04,.04]);multipliers=np.array([.2,1.,1.])
    record={'status':'starting','pid':os.getpid(),'steps':args.steps,'batch':args.batch,
        'device':args.device,'index':str(args.index.resolve()),
        'index_sha256':hashlib.sha256(args.index.read_bytes()).hexdigest(),'source_sha256':code,
        'warmstart':str(args.warmstart.resolve()),'warmstart_SHA256':hashlib.sha256(args.warmstart.read_bytes()).hexdigest(),
        'optimizer_reset':True,'training_records':len(train),'alpha_pairs':len(alpha),
        'trajectory_pairs':len(trajectory),'training_cases':['Neg_07_Left'],
        'development_not_used_for_gradient':True,'development_used_for_method_design':True,
        'checkpoint_selection':'fixed final step and EMA .995; no development checkpoint choice',
        'constraint_order':['alpha_relative_MSE','c_invariance_MSE_scaled_by_5','rho_invariance_MSE_scaled_by_5'],
        'constraint_targets':targets.tolist(),'dual_update_interval':20,'dual_update_rate':.1,
        'dual_maximum':50.,'base_loss_changed':False,'point_acquisitions_used':False,
        'patient_wave_solves':0,'allocator_cap_GiB':8,'model_per_card_limit_GiB':16,
        'joint_imaging_success_claimed':False,'created_unix_s':time.time()}
    atomic(args.out/'run.json',record);monitor=PressureResources(args.out)
    try:
        monitor.wait_budget([args.device],8,'before_response_constrained_training')
        torch.set_num_threads(1);torch.manual_seed(61105);np.random.seed(61105)
        rng=np.random.default_rng(61105)
        torch.cuda.set_device(args.device);device=f'cuda:{args.device}'
        torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(args.device).total_memory,args.device)
        torch.backends.cuda.matmul.allow_tf32=False
        initial=torch.load(args.warmstart,map_location='cpu',weights_only=False)
        model=AngularSpectralInverseUNO().to(device);model.load_state_dict(initial['ema'])
        del initial
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        record.update(status='training',model=model.config);atomic(args.out/'run.json',record)
        for name in code:(args.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
        started=time.time();average=None;accum=[];seen=set()
        for step in range(1,args.steps+1):
            selected=[]
            for _ in range(args.batch):
                pool=paired if rng.random()<.5 else unpaired
                row=pool[rng.integers(len(pool))];selected.append(row);seen.add(row['id'])
            x,y=load_batch(selected,device)
            flip=torch.rand(args.batch,device=device)<.5
            x[flip]=x[flip].flip(-1);y[flip]=y[flip].flip(-1)
            optimizer.zero_grad(set_to_none=True)
            base,parts=map_loss(model(x),y)
            rows=alpha[rng.integers(len(alpha))];ax,ay=load_batch(rows,device)
            if rng.random()<.5:ax=ax.flip(-1);ay=ay.flip(-1)
            prediction=model(ax)
            delta=prediction[1,:,5:]-prediction[0,:,5:]
            exact=ay[1,:,5:]-ay[0,:,5:]
            ca=(delta[2]-exact[2]).square().mean()/exact[2].square().mean().clamp_min(.02**2)
            cc=(delta[0]/5).square().mean();cr=(delta[1]/5).square().mean()
            constraints=torch.stack([ca,cc,cr]);seen.update(r['id'] for r in rows)
            traj=base.new_zeros(())
            if step%4==0:
                rows=trajectory[rng.integers(len(trajectory))];tx,ty=load_batch(rows,device)
                if rng.random()<.5:tx=tx.flip(-1);ty=ty.flip(-1)
                traj=pair_loss(model(tx),ty);seen.update(r['id'] for r in rows)
            loss=base+torch.dot(constraints,constraints.new_tensor(multipliers))+.2*traj
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite constrained loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            rate=3e-4*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/args.steps)))
            for group in optimizer.param_groups:group['lr']=rate
            optimizer.step()
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.005)
            values=constraints.detach().cpu().numpy()
            average=values.astype(np.float64) if average is None else .9*average+.1*values
            if step%20==0:multipliers=np.clip(multipliers+.1*(average-targets),0,50)
            accum.append([float(base.detach()),*values.tolist(),float(traj.detach()),*parts.detach().cpu().tolist()])
            if step%100==0 or step==1:
                values=np.mean(accum,axis=0);accum=[]
                item={'step':step,'base_map_loss':float(values[0]),'mean_constraints':values[1:4].tolist(),
                    'constraint_EMA':average.tolist(),'dual_multipliers':multipliers.tolist(),
                    'trajectory_loss':float(values[4]),'normalized_mse_c_rho_alpha':values[5:].tolist(),
                    'gradient_norm':float(norm),'unique_training_records':len(seen),
                    'seconds':time.time()-started,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
                with (args.out/'training.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
                atomic(args.out/'progress.json',item);monitor.sample('response_constrained_training')
            if step%1000==0 or step==args.steps:
                checkpoint={'step':step,'model':model.state_dict(),'ema':ema.state_dict(),
                    'optimizer':optimizer.state_dict(),'config':model.config,'run':record,
                    'dual_multipliers':multipliers,'constraint_average':average,
                    'sampler_state':rng.bit_generator.state,'torch_rng_state':torch.get_rng_state(),
                    'cuda_rng_state':torch.cuda.get_rng_state(args.device)}
                temporary=args.out/'weights.partial.pt';torch.save(checkpoint,temporary);temporary.replace(args.out/'weights.pt')
        record.update(status='training_finished',seconds=time.time()-started,
            dual_multipliers=multipliers.tolist(),training_constraint_EMA=average.tolist(),
            actual_final_constraint_satisfaction_not_yet_evaluated=True,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,finished_unix_s=time.time())
        atomic(args.out/'run.json',record)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',record);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
