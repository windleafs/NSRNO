"""Paired-window fine tuning with actual prefixes and unchanged RF operator."""
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
from .raw_rf_inverse_uno import RawRFInverseUNO,load_batch,mirror,map_loss
from .train_raw_rf_inverse import update,KINDS
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def view(row,shorter):return row['short_view'] if shorter and 'short_view' in row else row


def pair_view(pair,rng):
    flags=['short_view' in r for r in pair]
    if any(flags) and not all(flags):raise ValueError('Different coverage within an independent pair')
    shorter=all(flags) and bool(rng.random()<.5)
    return [view(r,shorter) for r in pair]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=Path,default=Path('out/v5/raw_rf_window_balanced_data_v1/index.json'))
    p.add_argument('--out',type=Path,default=Path('out/v5/inverse_raw_rf_window_balanced_v1'))
    p.add_argument('--warmstart',type=Path,default=Path('out/v5/inverse_raw_rf_uno_v1/weights.pt'))
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--max-seconds',type=float,default=5400)
    p.add_argument('--device',type=int,default=0)
    a=p.parse_args();preparation=json.loads((a.index.parent/'run.json').read_text())
    if preparation['status']!='paired_RF_windows_prepared' or Path(f"/proc/{preparation['pid']}").exists():
        raise RuntimeError('Finish and reap CPU preparation first')
    if digest(a.index)!=preparation['index_SHA256']:raise RuntimeError('Prepared input index changed')
    previous=json.loads((a.warmstart.parent/'run.json').read_text())
    if previous['status']!='training_finished' or Path(f"/proc/{previous['pid']}").exists():raise RuntimeError('Previous training still exists')
    if a.out.exists():raise RuntimeError('Inspect current paired-window training; no implicit overwrite')
    if not 0<a.steps or a.max_seconds<=0:raise ValueError('Positive budget required')
    index=json.loads(a.index.read_text());train=[r for r in index['records'] if r['role']=='train']
    long=[r for r in train if 'short_view' in r];originals=[r for r in train if r['kind']=='original']
    if len(train)!=112 or len(long)!=56 or {r['case'] for r in train}!={'Neg_07_Left'}:raise ValueError('Wrong training partition')
    if any(r['label']!=r['short_view']['label'] for r in long):raise ValueError('Matched views changed a target')
    grouped={}
    for row in train:
        if 'pair_group' in row:grouped.setdefault(row['pair_group'],[]).append(row)
    pairs=[sorted(v,key=lambda r:r['pair_sign']) for v in grouped.values() if len(v)==2]
    pure={k:[v for v in pairs if v[0]['kind']==k] for k in KINDS}
    trajectory=[v for v in pairs if v[0]['kind']=='trajectory']
    if [len(pure[k]) for k in KINDS]!=[8,8,16] or len(trajectory)!=8:raise ValueError('Independent pair counts changed')
    if digest(Path(__file__).with_name('raw_rf_inverse_uno.py'))!=previous['source_sha256']['raw_rf_inverse_uno.py']:
        raise RuntimeError('Architecture differs from the fixed warm start')
    a.out.mkdir();names=['train_raw_rf_window_balanced.py','train_raw_rf_inverse.py','raw_rf_inverse_uno.py',
        'uno_rf.py','inverse_uno.py','pressure_resources.py']
    record={'status':'starting','pid':os.getpid(),'device':a.device,'created_unix_s':time.time(),
        'index':str(a.index.resolve()),'index_sha256':digest(a.index),'warmstart':str(a.warmstart.resolve()),
        'warmstart_SHA256':digest(a.warmstart),'warmstart_state':'fixed final EMA','optimizer_reset':True,
        'planned_steps':a.steps,'walltime_budget_seconds':a.max_seconds,'one_optimizer_update_per_step':True,
        'seed':611074,'training_records':len(train),'paired_training_materials':len(long),'training_cases':['Neg_07_Left'],
        'odd_steps':'base map plus one independent parameter pair; trajectory every fourth independent step',
        'even_steps':'same-material actual RF60/RF90 map supervision plus soft prediction consistency',
        'window_map_weight':1.,'window_consistency_weight':1.,'window_consistency_scales_c_rho_alpha':[80,100,.2],
        'paired_members_use_same_window':True,'random_short_view_probability_when_available':.5,
        'full_record_remains_default_evaluation_input':True,'model_architecture_unchanged':True,
        'constraint_targets':[.16,.04,.04],'global_RF_scale_unchanged':index['training_RF_component_RMS_scale'],
        'allocator_cap_GiB':8,'model_per_card_limit_GiB':16,'new_wave_solves':0,
        'development_not_used_for_gradient':True,'development_used_for_method_design':True,
        'reserved_test_case_unread':'Neg_47_Left','checkpoint_selection':'fixed final EMA .995',
        'source_sha256':{n:digest(Path(__file__).parent/n) for n in names},'joint_imaging_success_claimed':False}
    atomic(a.out/'run.json',record);monitor=PressureResources(a.out)
    try:
        monitor.wait_budget([a.device],8,'before_paired_window_training')
        torch.set_num_threads(1);torch.manual_seed(611074);rng=np.random.default_rng(611074)
        torch.cuda.set_device(a.device);device=f'cuda:{a.device}'
        torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
        torch.backends.cuda.matmul.allow_tf32=False
        warm=torch.load(a.warmstart,map_location='cpu',weights_only=False)
        model=RawRFInverseUNO().to(device);model.load_state_dict(warm['ema'])
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        multipliers=np.array(warm['dual_multipliers'],np.float64,copy=True)
        averages=[None if v is None else np.array(v,np.float64,copy=True) for v in warm['constraint_average']]
        counts=np.array(warm['constraint_counts'],dtype=np.int64,copy=True);del warm
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        record.update(status='training',model=model.config,initial_dual_multipliers=multipliers.tolist(),
            optimizer='AdamW 1e-4 cosine to 5e-6, weight decay 1e-4, gradient clip 1')
        atomic(a.out/'run.json',record)
        for name in names:(a.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
        (a.out/'training_mean_material.npy').write_bytes(Path(index['training_mean_material']).read_bytes())
        scale=index['training_RF_component_RMS_scale'];targets=np.array([.16,.04,.04])
        started=time.time();last_resource=started;seen=set();last_window_gap=None;independent_steps=0;window_steps=0
        for step in range(1,a.steps+1):
            rate=1e-4*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/a.steps)))
            for group in optimizer.param_groups:group['lr']=rate
            flip=bool(rng.random()<.5)
            if step%2:
                independent_steps+=1;pool=originals if rng.random()<.5 else train
                selected=[view(pool[rng.integers(len(pool))],bool(rng.random()<.5))]
                changed=int(rng.integers(3));pair=pure[KINDS[changed]][rng.integers(len(pure[KINDS[changed]]))]
                pair=pair_view(pair,rng)
                tr=pair_view(trajectory[rng.integers(len(trajectory))],rng) if independent_steps%4==0 else None
                item=update(model,optimizer,selected,pair,changed,tr,device,scale,multipliers[changed],flip)
                values=item.pop('constraints');averages[changed]=values if averages[changed] is None else .9*averages[changed]+.1*values
                counts[changed]+=1
                if counts[changed]%20==0:multipliers[changed]=np.clip(multipliers[changed]+.1*(averages[changed]-targets),0,50)
                seen.update(r['id'] for r in selected+pair+(tr or []))
                item.update(phase='independent',changed_parameter=changed,constraints=values.tolist())
            else:
                window_steps+=1;row=long[rng.integers(len(long))]
                x,y=load_batch([row['short_view'],row],device,scale)
                if flip:x,y=mirror(x,y)
                optimizer.zero_grad(set_to_none=True);prediction=model(x)
                base,parts=map_loss(prediction,y);consistency,_=map_loss(prediction[:1],prediction[1:])
                loss=base+consistency
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite paired-window loss')
                last_window_gap=(prediction[1,:,5:]-prediction[0,:,5:]).square().mean((-2,-1)).sqrt().detach().cpu().tolist()
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                item={'phase':'window','base':float(base.detach()),'window_consistency':float(consistency.detach()),
                    'map_mse':parts.detach().cpu().tolist(),'gradient_norm':float(norm),'material_id':row['id']}
                seen.add(row['id']);del x,y,prediction,base,parts,consistency,loss
            with torch.no_grad():
                for target,source in zip(ema.parameters(),model.parameters()):target.lerp_(source,.005)
            torch.cuda.synchronize();elapsed=time.time()-started
            item.update(step=step,seconds=elapsed,seconds_per_step=elapsed/step,unique_training_records=len(seen),
                independent_steps=independent_steps,window_steps=window_steps,last_window_gap_RMS_c_rho_alpha=last_window_gap,
                constraint_EMA=[v.tolist() if v is not None else None for v in averages],dual_multipliers=multipliers.tolist(),
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
            with (a.out/'training.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
            if step%25==0 or step==1:
                atomic(a.out/'progress.json',item);print(json.dumps(item),flush=True)
            if time.time()-last_resource>20 or step==1:
                monitor.sample('paired_window_RF_training');last_resource=time.time()
            final=step==a.steps or elapsed>=a.max_seconds
            if step%500==0 or final:
                saved={'step':step,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                    'config':model.config,'run':record,'dual_multipliers':multipliers,'constraint_average':averages,
                    'constraint_counts':counts,'sampler_state':rng.bit_generator.state,'torch_rng_state':torch.get_rng_state(),
                    'cuda_rng_state':torch.cuda.get_rng_state(a.device)}
                temporary=a.out/'weights.partial.pt';torch.save(saved,temporary);temporary.replace(a.out/'weights.pt')
            if final:break
        record.update(status='training_finished',completed_steps=step,seconds=time.time()-started,
            independent_steps=independent_steps,window_steps=window_steps,
            stop_reason='fixed_step_budget' if step==a.steps else 'fixed_walltime_budget',
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,finished_unix_s=time.time())
        monitor.sample('paired_window_RF_training_finished');atomic(a.out/'run.json',record);print(json.dumps(record),flush=True)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time());atomic(a.out/'run.json',record);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
