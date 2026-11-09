"""Fixed-budget training, Neg07 only; no development-truth checkpoint selection."""
from argparse import ArgumentParser
from pathlib import Path
import copy,hashlib,json,math,os,time,traceback
import numpy as np
import torch
from .inverse_uno import InverseUNO,load_batch,map_loss,pair_loss

def atomic_json(path,value):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def main():
    p=ArgumentParser();p.add_argument('--index',type=Path,default=Path('out/v5/inverse_focus_v1/index.json'));p.add_argument('--out',type=Path,default=Path('out/v5/inverse_uno_v1'))
    p.add_argument('--device',type=int,default=0);p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch',type=int,default=4)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'run.json').exists():raise FileExistsError('Preserve prior training runs')
    torch.set_num_threads(1);torch.manual_seed(61105);np.random.seed(61105);rng=np.random.default_rng(61105)
    torch.cuda.set_device(a.device);device=f'cuda:{a.device}'
    torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
    torch.backends.cuda.matmul.allow_tf32=False
    index=json.loads(a.index.read_text())
    if index['completed']!=index['planned_records']:raise RuntimeError('Full feature preparation has not finished')
    train=[r for r in index['records'] if r['role']=='train']
    if {r['case'] for r in train}!={'Neg_07_Left'}:raise ValueError('Development data would enter training')
    paired=[r for r in train if 'lf' in r['features']];unpaired=[r for r in train if 'lf' not in r['features']]
    groups={}
    for r in train:
        if 'pair_group' in r:groups.setdefault(r['pair_group'],[]).append(r)
    pairs=[sorted(rows,key=lambda r:r['pair_sign']) for rows in groups.values() if len(rows)==2]
    originals=[r for r in train if r['kind']=='original']
    mean=sum(np.load(r['label']).astype(np.float64) for r in originals)/len(originals)
    np.save(a.out/'training_mean_material.npy',mean.astype(np.float32))
    model=InverseUNO().to(device);ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    record={'status':'training','steps':a.steps,'batch':a.batch,'device':a.device,'model':model.config,'training_case':'Neg_07_Left','development_case_not_used_for_gradient_or_selection':'Neg_35_Left',
        'training_records':len(train),'paired_source_records':len(paired),'independent_parameter_pairs':len(pairs),'checkpoint_selection':'fixed final step, EMA .995; no development truth selection',
        'memory_limit_GiB':8,'index_sha256':hashlib.sha256(a.index.read_bytes()).hexdigest(),'code_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),Path(__file__).with_name('inverse_uno.py'),Path(__file__).with_name('uno_rf.py')]}}
    atomic_json(a.out/'run.json',record)
    for name in record['code_sha256']:(a.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    start=time.time();accum=[];seen=set();log=a.out/'training.jsonl'
    try:
        for step in range(1,a.steps+1):
            selected=[]
            for _ in range(a.batch):
                pool=paired if rng.random()<.5 else unpaired;row=pool[rng.integers(len(pool))];selected.append(row);seen.add(row['id'])
            x,y=load_batch(selected,device)
            flip=torch.rand(a.batch,device=device)<.5
            x[flip]=x[flip].flip(-1);y[flip]=y[flip].flip(-1)
            optimizer.zero_grad(set_to_none=True);prediction=model(x);base,parts=map_loss(prediction,y)
            pair=base.new_zeros(())
            if step%2==0:
                rows=pairs[rng.integers(len(pairs))];px,py=load_batch(rows,device)
                if rng.random()<.5:px=px.flip(-1);py=py.flip(-1)
                pair=pair_loss(model(px),py);seen.update(r['id'] for r in rows)
            loss=base+.2*pair;loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            rate=3e-4*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/a.steps)))
            for group in optimizer.param_groups:group['lr']=rate
            optimizer.step()
            with torch.no_grad():
                for average,current in zip(ema.parameters(),model.parameters()):average.lerp_(current,.005)
            accum.append([float(base.detach()),float(pair.detach()),*parts.detach().cpu().tolist()])
            if step%100==0 or step==1:
                values=np.mean(accum,0);accum=[]
                item={'step':step,'map_loss':float(values[0]),'pair_loss':float(values[1]),'normalized_mse_c_rho_alpha':values[2:].tolist(),
                    'unique_training_records':len(seen),'gradient_norm':float(norm),'lr':rate,'seconds':time.time()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30}
                with log.open('a') as f:f.write(json.dumps(item)+'\n')
                print(json.dumps(item),flush=True)
            if step%1000==0 or step==a.steps:
                checkpoint={'step':step,'model':model.state_dict(),'ema':ema.state_dict(),'config':model.config,'run':record}
                temporary=a.out/'weights.partial.pt';torch.save(checkpoint,temporary);temporary.replace(a.out/'weights.pt')
        record.update(status='training_finished',seconds=time.time()-start,unique_training_records=len(seen),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
        atomic_json(a.out/'run.json',record)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc());atomic_json(a.out/'run.json',record);raise

if __name__=='__main__':main()
