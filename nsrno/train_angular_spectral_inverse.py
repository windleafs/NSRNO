"""Fixed-budget comparison of inverse-UNO RF statistics on the same partition."""
from argparse import ArgumentParser
from pathlib import Path
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


def main():
    p=ArgumentParser()
    p.add_argument('--index',type=Path,default=Path('out/v5/inverse_angular_spectral_v1/index.json'))
    p.add_argument('--out',type=Path,default=Path('out/v5/inverse_angular_spectral_uno_v1'))
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--cap-gib',type=float,default=8.)
    args=p.parse_args()
    if args.out.exists():raise RuntimeError('Inspect existing training; no implicit restart')
    preparation=json.loads((args.index.parent/'run.json').read_text())
    if preparation['status']!='features_prepared':raise RuntimeError('Full feature preparation is not complete')
    if Path(f"/proc/{preparation['pid']}").exists():raise RuntimeError('Feature GPU process still exists')
    index=json.loads(args.index.read_text())
    if index['completed']!=index['planned_records']:raise RuntimeError('Require complete features')
    if index['feature_shape_per_source']!=[72,216,192]:raise ValueError('Unexpected observation representation')
    train=[r for r in index['records'] if r['role']=='train']
    if {r['case'] for r in train}!={'Neg_07_Left'}:raise ValueError('Development data would enter training')
    paired=[r for r in train if 'lf' in r['features']]
    unpaired=[r for r in train if 'lf' not in r['features']]
    groups={}
    for row in train:
        if 'pair_group' in row:groups.setdefault(row['pair_group'],[]).append(row)
    pairs=[sorted(rows,key=lambda r:r['pair_sign']) for rows in groups.values() if len(rows)==2]
    if not paired or not unpaired or not pairs:raise ValueError('Preserve the old training sampling design')
    if not 0<args.cap_gib<=10:raise ValueError('Keep CUDA context plus allocation below 16 GiB')
    args.out.mkdir()
    originals=[r for r in train if r['kind']=='original']
    mean=sum(np.load(r['label']).astype(np.float64) for r in originals)/len(originals)
    np.save(args.out/'training_mean_material.npy',mean.astype(np.float32))
    code={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ['train_angular_spectral_inverse.py','angular_spectral_inverse_uno.py',
                     'inverse_uno.py','uno_rf.py','angular_spectral_features.py']}
    record={'status':'starting','pid':os.getpid(),'steps':args.steps,'batch':args.batch,
        'device':args.device,'training_cases':['Neg_07_Left'],'training_records':len(train),
        'paired_source_records':len(paired),'independent_parameter_pairs':len(pairs),
        'development_not_used_for_gradient':True,'development_used_for_method_design':True,
        'checkpoint_selection':'fixed final step; EMA .995; no development checkpoint selection',
        'index':str(args.index.resolve()),
        'index_sha256':hashlib.sha256(args.index.read_bytes()).hexdigest(),'source_sha256':code,
        'allocator_cap_GiB':args.cap_gib,'model_per_card_limit_GiB':16,
        'patient_wave_solves':0,'point_acquisitions_used':False,
        'joint_imaging_success_claimed':False,'created_unix_s':time.time()}
    atomic(args.out/'run.json',record)
    monitor=PressureResources(args.out)
    try:
        monitor.wait_budget([args.device],args.cap_gib,'before_angular_spectral_training')
        torch.set_num_threads(1);torch.manual_seed(61105);np.random.seed(61105)
        rng=np.random.default_rng(61105)
        torch.cuda.set_device(args.device);device=f'cuda:{args.device}'
        torch.cuda.set_per_process_memory_fraction(args.cap_gib*2**30/
            torch.cuda.get_device_properties(args.device).total_memory,args.device)
        torch.backends.cuda.matmul.allow_tf32=False
        model=AngularSpectralInverseUNO().to(device)
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        record.update(status='training',model=model.config)
        atomic(args.out/'run.json',record)
        for name in code:(args.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
        started=time.time();accum=[];seen=set()
        for step in range(1,args.steps+1):
            selected=[]
            for _ in range(args.batch):
                pool=paired if rng.random()<.5 else unpaired
                row=pool[rng.integers(len(pool))];selected.append(row);seen.add(row['id'])
            x,y=load_batch(selected,device)
            flip=torch.rand(args.batch,device=device)<.5
            x[flip]=x[flip].flip(-1);y[flip]=y[flip].flip(-1)
            optimizer.zero_grad(set_to_none=True)
            base,parts=map_loss(model(x),y);pair=base.new_zeros(())
            if step%2==0:
                rows=pairs[rng.integers(len(pairs))];px,py=load_batch(rows,device)
                if rng.random()<.5:px=px.flip(-1);py=py.flip(-1)
                pair=pair_loss(model(px),py);seen.update(r['id'] for r in rows)
            loss=base+.2*pair;loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            rate=3e-4*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/args.steps)))
            for group in optimizer.param_groups:group['lr']=rate
            optimizer.step()
            with torch.no_grad():
                for average,current in zip(ema.parameters(),model.parameters()):average.lerp_(current,.005)
            accum.append([float(base.detach()),float(pair.detach()),*parts.detach().cpu().tolist()])
            if step%100==0 or step==1:
                values=np.mean(accum,0);accum=[]
                item={'step':step,'map_loss':float(values[0]),'pair_loss':float(values[1]),
                    'normalized_mse_c_rho_alpha':values[2:].tolist(),'unique_training_records':len(seen),
                    'gradient_norm':float(norm),'lr':rate,'seconds':time.time()-started,
                    'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
                    'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30}
                with (args.out/'training.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
                atomic(args.out/'progress.json',item)
                monitor.sample('angular_spectral_training')
            if step%1000==0 or step==args.steps:
                checkpoint={'step':step,'model':model.state_dict(),'ema':ema.state_dict(),
                    'config':model.config,'run':record}
                temporary=args.out/'weights.partial.pt';torch.save(checkpoint,temporary)
                temporary.replace(args.out/'weights.pt')
        record.update(status='training_finished',seconds=time.time()-started,
            unique_training_records=len(seen),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            finished_unix_s=time.time())
        atomic(args.out/'run.json',record)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',record);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
