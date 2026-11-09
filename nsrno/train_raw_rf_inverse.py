"""Fresh raw-RF inverse UNO training, sequential backward and bounded memory."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import os
import time
import traceback
import numpy as np
import torch
from .raw_rf_inverse_uno import RawRFInverseUNO,load_batch,mirror,map_loss,pair_loss
from .native_gn_json import atomic
from .pressure_resources import PressureResources

KINDS=('pure_c','pure_rho','alpha')
SCALES=(5.,5.,.05)
FLOORS=(5.,5.,.02)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update(model,optimizer,rows,pair,changed,trajectory,device,scale,multipliers,flip=False):
    optimizer.zero_grad(set_to_none=True)
    x,y=load_batch(rows,device,scale)
    if flip:x,y=mirror(x,y)
    base,parts=map_loss(model(x),y)
    if not torch.isfinite(base):raise RuntimeError('Nonfinite base map loss')
    base.backward();base_value=float(base.detach());part_values=parts.detach().cpu().tolist()
    del x,y,base,parts
    x,y=load_batch(pair,device,scale)
    if flip:x,y=mirror(x,y)
    prediction=model(x);delta=prediction[1,:,5:]-prediction[0,:,5:];exact=y[1,:,5:]-y[0,:,5:]
    other=[j for j in range(3) if j!=changed]
    constraints=torch.stack(((delta[changed]-exact[changed]).square().mean()/
                             exact[changed].square().mean().clamp_min(FLOORS[changed]**2),
                             *[(delta[j]/SCALES[j]).square().mean() for j in other]))
    loss=torch.dot(constraints,constraints.new_tensor(multipliers))
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite independent response loss')
    loss.backward();constraint_values=constraints.detach().cpu().numpy().astype(np.float64)
    del x,y,prediction,delta,exact,constraints,loss
    trajectory_value=0.
    if trajectory is not None:
        x,y=load_batch(trajectory,device,scale)
        if flip:x,y=mirror(x,y)
        loss=.2*pair_loss(model(x),y)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite trajectory loss')
        loss.backward();trajectory_value=float(loss.detach())/.2
        del x,y,loss
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return {'base':base_value,'map_mse':part_values,'constraints':constraint_values,
            'trajectory':trajectory_value,'gradient_norm':float(norm)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=Path,default=Path('out/v5/raw_rf_inverse_data_v1/index.json'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile-only',action='store_true')
    p.add_argument('--steps',type=int,default=8000)
    p.add_argument('--max-seconds',type=float,default=5400)
    p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    preparation=json.loads((a.index.parent/'run.json').read_text())
    if preparation['status']!='raw_RF_prepared' or Path(f"/proc/{preparation['pid']}").exists():
        raise RuntimeError('Complete and reap the CPU data preparation first')
    if a.out.exists():raise RuntimeError('Inspect existing run; do not overwrite or resume implicitly')
    if a.steps<1 or a.max_seconds<=0:raise ValueError('Positive fixed budget required')
    index=json.loads(a.index.read_text());train=[r for r in index['records'] if r['role']=='train']
    if len(train)!=112 or {r['case'] for r in train}!={'Neg_07_Left'}:raise ValueError('Training partition changed')
    grouped={}
    for row in train:
        if 'pair_group' in row:grouped.setdefault(row['pair_group'],[]).append(row)
    pairs=[sorted(v,key=lambda r:r['pair_sign']) for v in grouped.values() if len(v)==2]
    pure={k:[v for v in pairs if v[0]['kind']==k] for k in KINDS}
    trajectory=[v for v in pairs if v[0]['kind']=='trajectory']
    if [len(pure[k]) for k in KINDS]!=[8,8,16] or len(trajectory)!=8:raise ValueError('Wrong independent pair counts')
    originals=[r for r in train if r['kind'] in ['original','coverage_base']]
    a.out.mkdir()
    names=['train_raw_rf_inverse.py','raw_rf_inverse_uno.py','uno_rf.py','inverse_uno.py','pressure_resources.py']
    record={'status':'starting','pid':os.getpid(),'device':a.device,'profile_only':a.profile_only,
        'seed':611072,'planned_steps':a.steps,'walltime_budget_seconds':a.max_seconds,
        'index':str(a.index.resolve()),'index_sha256':digest(a.index),'training_records':len(train),
        'training_cases':['Neg_07_Left'],'sources_per_record':11,'receivers':192,
        'independent_pairs':{k:len(v) for k,v in pure.items()},'trajectory_pairs':len(trajectory),
        'base_batch':1,'independent_pair_batch':2,'trajectory_every':4,
        'fresh_initialization':True,'global_RF_scale':index['training_RF_component_RMS_scale'],
        'optimizer':'AdamW, lr 0.0003 cosine to 0.000015, weight decay 0.0001, gradient clip 1',
        'sequential_backward':'base, independent pair, optional trajectory; one optimizer step',
        'constraint_targets':[.16,.04,.04],'initial_duals':[[1.,1.,1.]]*3,
        'development_not_used_for_gradient':True,'development_used_for_method_design':True,
        'reserved_test_case_unread':'Neg_47_Left','checkpoint_selection':'final fixed budget EMA .995',
        'allocator_cap_GiB':8,'model_per_card_limit_GiB':16,'patient_wave_solves':0,
        'source_sha256':{n:digest(Path(__file__).parent/n) for n in names},
        'created_unix_s':time.time(),'joint_imaging_success_claimed':False}
    atomic(a.out/'run.json',record);monitor=PressureResources(a.out)
    try:
        monitor.wait_budget([a.device],8,'before_raw_RF_inverse')
        torch.set_num_threads(1);torch.manual_seed(611072);rng=np.random.default_rng(611072)
        torch.cuda.set_device(a.device);device=f'cuda:{a.device}'
        torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
        torch.backends.cuda.matmul.allow_tf32=False
        model=RawRFInverseUNO().to(device);ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        record.update(status='profiling' if a.profile_only else 'training',model=model.config,
            parameter_count=sum(p.numel() for p in model.parameters()))
        atomic(a.out/'run.json',record)
        for name in names:(a.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
        (a.out/'training_mean_material.npy').write_bytes(Path(index['training_mean_material']).read_bytes())
        scale=index['training_RF_component_RMS_scale'];multipliers=np.ones((3,3));averages=[None]*3
        counts=np.zeros(3,dtype=int);targets=np.array([.16,.04,.04]);seen=set();logs=[]
        started=time.time();last_resource=started
        steps=1 if a.profile_only else a.steps
        for step in range(1,steps+1):
            if a.profile_only:
                selected=[next(r for r in train if r['kind']=='original' and r['source_format']=='RF_90us')]
                changed=1;pair=pure['pure_rho'][0];tr=trajectory[0];flip=False
            else:
                pool=originals if rng.random()<.5 else train
                selected=[pool[rng.integers(len(pool))]];changed=int(rng.integers(3))
                pair=pure[KINDS[changed]][rng.integers(len(pure[KINDS[changed]]))]
                tr=trajectory[rng.integers(len(trajectory))] if step%4==0 else None;flip=bool(rng.random()<.5)
            rate=3e-4*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/a.steps)))
            for group in optimizer.param_groups:group['lr']=rate
            item=update(model,optimizer,selected,pair,changed,tr,device,scale,multipliers[changed],flip)
            with torch.no_grad():
                for target,source in zip(ema.parameters(),model.parameters()):target.lerp_(source,.005)
            values=item.pop('constraints');averages[changed]=values if averages[changed] is None else .9*averages[changed]+.1*values
            counts[changed]+=1
            if counts[changed]%20==0:multipliers[changed]=np.clip(multipliers[changed]+.1*(averages[changed]-targets),0,50)
            seen.update(r['id'] for r in selected+pair+(tr or []))
            torch.cuda.synchronize();elapsed=time.time()-started
            item.update(step=step,changed_parameter=changed,constraints=values.tolist(),
                seconds=elapsed,seconds_per_step=elapsed/step,unique_training_records=len(seen),
                constraint_EMA=[v.tolist() if v is not None else None for v in averages],
                dual_multipliers=multipliers.tolist(),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
            with (a.out/'training.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
            if step%25==0 or step==1:
                atomic(a.out/'progress.json',item);print(json.dumps(item),flush=True)
            if time.time()-last_resource>20 or step==1:
                monitor.sample('raw_RF_inverse_training');last_resource=time.time()
            final=step==steps or elapsed>=a.max_seconds
            if not a.profile_only and (step%1000==0 or final):
                state={'step':step,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                    'config':model.config,'run':record,'dual_multipliers':multipliers,'constraint_average':averages,
                    'constraint_counts':counts,'sampler_state':rng.bit_generator.state,
                    'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state(a.device)}
                temp=a.out/'weights.partial.pt';torch.save(state,temp);temp.replace(a.out/'weights.pt')
            if final:break
        record.update(status='full_RF_training_step_profiled' if a.profile_only else 'training_finished',
            completed_steps=step,seconds=time.time()-started,stop_reason='profile' if a.profile_only else
                ('fixed_step_budget' if step==a.steps else 'fixed_walltime_budget'),
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,finished_unix_s=time.time())
        monitor.sample('raw_RF_inverse_finished');atomic(a.out/'run.json',record);print(json.dumps(record),flush=True)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(a.out/'run.json',record);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
