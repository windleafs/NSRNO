"""Balanced independent material response constraints on a fixed coverage index."""
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
from .angular_spectral_inverse_uno import AngularSpectralInverseUNO,load_batch,map_loss,pair_loss
from .native_gn_json import atomic
from .pressure_resources import PressureResources


KINDS=('pure_c','pure_rho','alpha')
SCALES=(5.,5.,.05)
FLOORS=(5.,5.,.02)


def drop_high_frequency(x,rng,whole_pair=False):
    # Both members of a pair receive the same source mask.
    if whole_pair:
        if bool(torch.all(x[:,145,0,0]>.5)) and rng.random()<.5:
            x[:,:72]=0;x[:,144]=0
    else:
        selected=torch.tensor(rng.random(len(x))<.5,device=x.device)&(x[:,145,0,0]>.5)
        x[selected,:72]=0;x[selected,144]=0


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--warmstart',type=Path,default=Path('out/v5/inverse_response_constrained_uno_v1/weights.pt'))
    p.add_argument('--root',type=Path,default=Path('out/v5/independent_material_3mhz_v1'))
    p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    preparation=json.loads((a.root/'focus/run.json').read_text())
    if preparation['status']!='paired_features_prepared':raise RuntimeError('Require complete fixed paired features')
    if Path(f"/proc/{preparation['pid']}").exists():raise RuntimeError('Feature GPU process still exists')
    if a.out.exists():raise RuntimeError('Inspect the current coverage model; never restart implicitly')
    index=json.loads(a.index.read_text());train=[r for r in index['records'] if r['role']=='train']
    if len(train)!=590 or {r['case'] for r in train}!={'Neg_07_Left'}:
        raise ValueError('Training partition differs from the planned coverage extension')
    paired=[r for r in train if 'lf' in r['features']];unpaired=[r for r in train if 'lf' not in r['features']]
    grouped={}
    for r in train:
        if 'pair_group' in r:grouped.setdefault(r['pair_group'],[]).append(r)
    pairs=[sorted(v,key=lambda r:r['pair_sign']) for v in grouped.values() if len(v)==2]
    pure={k:[v for v in pairs if v[0]['kind']==k] for k in KINDS}
    trajectory=[v for v in pairs if v[0]['kind']=='trajectory']
    if [len(pure[k]) for k in KINDS]!=[8,8,16] or len(trajectory)!=8:
        raise ValueError('Independent parameter pair counts changed')
    for changed,kind in enumerate(KINDS):
        for minus,plus in pure[kind]:
            delta=np.load(plus['label'])-np.load(minus['label'])
            other=[j for j in range(3) if j!=changed]
            if np.max(abs(delta[other]))>1e-3:raise ValueError('Independent labels change another parameter')
            if np.sqrt(np.mean(delta[changed,5:]**2))<FLOORS[changed]:
                raise ValueError('Declared response too small for its normalization')
    a.out.mkdir()
    original=[r for r in train if r['kind']=='original']
    mean=sum(np.load(r['label']).astype(np.float64) for r in original)/len(original)
    np.save(a.out/'training_mean_material.npy',mean.astype(np.float32))
    names=['train_material_coverage_inverse.py','angular_spectral_inverse_uno.py','inverse_uno.py','uno_rf.py']
    hashes={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in names}
    targets=np.array([.16,.04,.04]);multipliers=np.ones((3,3));multipliers[:,0]=.2
    average=[None,None,None];counts=np.zeros(3,dtype=int)
    record={'status':'starting','pid':os.getpid(),'steps':a.steps,'batch':4,'device':a.device,
            'index':str(a.index.resolve()),'index_sha256':hashlib.sha256(a.index.read_bytes()).hexdigest(),
            'record_window_control_us':index['record_window_control_us'],'source_sha256':hashes,
            'warmstart':str(a.warmstart.resolve()),'warmstart_SHA256':hashlib.sha256(a.warmstart.read_bytes()).hexdigest(),
            'optimizer_reset':True,'seed':611071,'training_records':len(train),'training_LF_records':len(paired),
            'independent_pairs':{k:len(v) for k,v in pure.items()},'trajectory_pairs':len(trajectory),
            'training_cases':['Neg_07_Left'],'development_not_used_for_gradient':True,
            'development_used_for_method_design':True,'reserved_test_case_unread':'Neg_47_Left',
            'checkpoint_selection':'fixed final step and EMA .995; no development checkpoint choice',
            'constraint_order':'changed parameter relative MSE, other parameters scaled invariance MSE',
            'invariance_scales_c_rho_alpha':SCALES,'constraint_targets':targets.tolist(),
            'HF_dropout_on_LF_records':.5,'same_source_mask_on_both_pair_members':True,
            'base_map_loss_unchanged':True,'patient_wave_solves':0,'point_acquisitions_used':False,
            'allocator_cap_GiB':8,'model_per_card_limit_GiB':16,
            'joint_imaging_success_claimed':False,'created_unix_s':time.time()}
    atomic(a.out/'run.json',record);monitor=PressureResources(a.out)
    try:
        monitor.wait_budget([a.device],8,'before_material_coverage_training')
        torch.set_num_threads(1);torch.manual_seed(611071);np.random.seed(611071)
        rng=np.random.default_rng(611071);torch.cuda.set_device(a.device);device=f'cuda:{a.device}'
        torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
        torch.backends.cuda.matmul.allow_tf32=False
        checkpoint=torch.load(a.warmstart,map_location='cpu',weights_only=False)
        model=AngularSpectralInverseUNO().to(device);model.load_state_dict(checkpoint['ema']);del checkpoint
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        record.update(status='training',model=model.config);atomic(a.out/'run.json',record)
        for name in names:(a.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
        started=time.time();loss_log=[];constraint_log={k:[] for k in KINDS};seen=set()
        for step in range(1,a.steps+1):
            selected=[]
            for _ in range(4):
                pool=paired if rng.random()<.5 else unpaired
                selected.append(pool[rng.integers(len(pool))])
            x,y=load_batch(selected,device);drop_high_frequency(x,rng)
            flip=torch.rand(4,device=device)<.5;x[flip]=x[flip].flip(-1);y[flip]=y[flip].flip(-1)
            optimizer.zero_grad(set_to_none=True)
            base,parts=map_loss(model(x),y)
            changed=int(rng.integers(3));kind=KINDS[changed];rows=pure[kind][rng.integers(len(pure[kind]))]
            px,py=load_batch(rows,device);drop_high_frequency(px,rng,whole_pair=True)
            if rng.random()<.5:px=px.flip(-1);py=py.flip(-1)
            value=model(px);delta=value[1,:,5:]-value[0,:,5:];exact=py[1,:,5:]-py[0,:,5:]
            invariants=[j for j in range(3) if j!=changed]
            constraints=torch.stack([(delta[changed]-exact[changed]).square().mean()/
                                     exact[changed].square().mean().clamp_min(FLOORS[changed]**2),
                                     *[(delta[j]/SCALES[j]).square().mean() for j in invariants]])
            traj=base.new_zeros(())
            if step%4==0:
                tr=trajectory[rng.integers(len(trajectory))];tx,ty=load_batch(tr,device)
                drop_high_frequency(tx,rng,whole_pair=True)
                if rng.random()<.5:tx=tx.flip(-1);ty=ty.flip(-1)
                traj=pair_loss(model(tx),ty);seen.update(r['id'] for r in tr)
            loss=base+torch.dot(constraints,constraints.new_tensor(multipliers[changed]))+.2*traj
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite coverage training objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            rate=3e-4*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/a.steps)))
            for group in optimizer.param_groups:group['lr']=rate
            optimizer.step()
            with torch.no_grad():
                for target,source in zip(ema.parameters(),model.parameters()):target.lerp_(source,.005)
            values=constraints.detach().cpu().numpy().astype(np.float64)
            average[changed]=values if average[changed] is None else .9*average[changed]+.1*values
            counts[changed]+=1
            if counts[changed]%20==0:
                multipliers[changed]=np.clip(multipliers[changed]+.1*(average[changed]-targets),0,50)
            constraint_log[kind].append(values.tolist());seen.update(r['id'] for r in selected+rows)
            loss_log.append([float(base.detach()),float(traj.detach()),*parts.detach().cpu().tolist()])
            if step%100==0 or step==1:
                values=np.mean(loss_log,axis=0);loss_log=[]
                item={'step':step,'base_map_loss':float(values[0]),'trajectory_loss_including_skipped_steps':float(values[1]),
                      'normalized_mse_c_rho_alpha':values[2:].tolist(),
                      'per_kind_constraints':{k:{'observations':len(v),'mean':np.mean(v,axis=0).tolist() if v else None}
                                              for k,v in constraint_log.items()},
                      'constraint_EMA':[v.tolist() if v is not None else None for v in average],
                      'dual_multipliers':multipliers.tolist(),'gradient_norm':float(norm),
                      'unique_training_records':len(seen),'seconds':time.time()-started,
                      'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
                constraint_log={k:[] for k in KINDS}
                with (a.out/'training.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
                atomic(a.out/'progress.json',item);monitor.sample('independent_material_training')
            if step%1000==0 or step==a.steps:
                checkpoint={'step':step,'model':model.state_dict(),'ema':ema.state_dict(),
                            'optimizer':optimizer.state_dict(),'config':model.config,'run':record,
                            'dual_multipliers':multipliers,'constraint_average':average,'constraint_counts':counts,
                            'sampler_state':rng.bit_generator.state,'torch_rng_state':torch.get_rng_state(),
                            'cuda_rng_state':torch.cuda.get_rng_state(a.device)}
                temporary=a.out/'weights.partial.pt';torch.save(checkpoint,temporary);temporary.replace(a.out/'weights.pt')
        record.update(status='training_finished',seconds=time.time()-started,
                      peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,finished_unix_s=time.time())
        atomic(a.out/'run.json',record)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(a.out/'run.json',record);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
