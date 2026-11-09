"""Prepare all-angle source-specific RF features and independent map targets."""
from pathlib import Path
from argparse import ArgumentParser
import hashlib,json,time,os
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from .focus_features import FocusFeatures,FOCUS_SPEEDS,HEIGHT,WIDTH

def atomic_json(path,value):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def label(material):
    result=[]
    for field in material:
        filtered=gaussian_filter(np.asarray(field,dtype=np.float32),sigma=1.)
        result.append(filtered[np.ix_(60+4*np.arange(HEIGHT),178+4*np.arange(WIDTH))])
    return np.stack(result).astype(np.float32)

def records():
    root=Path('out/v5/forward_cache_v1');base=json.loads((root/'index.json').read_text())['records']
    low=Path('out/v5/forward_cache_3mhz_v1');lo={r['id']:r for r in json.loads((low/'index.json').read_text())['records']}
    out=[]
    for r in base:
        sources={'hf':str(root/r['iq'])}
        if r['id'] in lo:sources['lf']=str(low/lo[r['id']]['iq'])
        out.append({'id':r['id'],'role':r['role'],'case':r['case'],'kind':'original','parent_id':r['id'],'sources':sources,'medium':str(root/r['medium']),'alpha':f"out/v5/uno_triparam_v1/alpha/{r['id']}.npy"})
    aug=json.loads(Path('out/v5/uno_triparam_v1/augmentation_index.json').read_text())['records'];groups={}
    for r in aug:
        key=Path(r['material3']).stem
        if key not in groups:groups[key]={'id':key,'role':r['role'],'case':r['case'],'kind':r['kind'],'parent_id':r['parent_id'],'pair_group':r['pair_group'],'pair_sign':r['pair_sign'],'material3':r['material3'],'sources':{}}
        groups[key]['sources']['lf' if r['source_mhz']==3 else 'hf']=r['iq']
    out.extend(groups.values())
    if {r['case'] for r in out if r['role']=='train'}!={'Neg_07_Left'}:raise ValueError('Training case contamination')
    if {r['case'] for r in out if r['role']!='train'}!={'Neg_35_Left'}:raise ValueError('Development case mismatch')
    return out

def main():
    p=ArgumentParser();p.add_argument('--out',type=Path,default=Path('out/v5/inverse_focus_v1'));p.add_argument('--device',type=int,default=0);p.add_argument('--id',nargs='*');p.add_argument('--limit',type=int)
    a=p.parse_args();torch.set_num_threads(1);torch.cuda.set_device(a.device)
    torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
    torch.backends.cuda.matmul.allow_tf32=False
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'features').mkdir(exist_ok=True);(a.out/'labels').mkdir(exist_ok=True)
    all_rows=records();selected=[r for r in all_rows if not a.id or r['id'] in a.id]
    if a.limit:selected=selected[:a.limit]
    model=FocusFeatures(f'cuda:{a.device}');start=time.time();completed=0
    for row in selected:
        for source,path in row['sources'].items():
            destination=a.out/'features'/f"{row['id']}_{source}.npy"
            if not destination.exists():
                feature=model(np.load(path),3e6 if source=='lf' else 7.5e6)
                temp=destination.with_suffix('.partial.npy');np.save(temp,feature);temp.replace(destination)
        target=a.out/'labels'/f"{row['id']}.npy"
        if not target.exists():
            if 'material3' in row:medium=np.load(row['material3'],mmap_mode='r')
            else:medium=np.concatenate([np.load(row['medium']),np.load(row['alpha'])[None]],0)
            temp=target.with_suffix('.partial.npy');np.save(temp,label(medium));temp.replace(target)
        completed+=1
        if completed%20==0 or len(selected)<5:print(json.dumps({'completed':completed,'selected':len(selected),'id':row['id'],'elapsed_seconds':time.time()-start,'memory_GiB':torch.cuda.max_memory_allocated()/2**30}),flush=True)
    full=[]
    for row in all_rows:
        item={**row,'features':{s:str((a.out/'features'/f"{row['id']}_{s}.npy").resolve()) for s in row['sources']},'label':str((a.out/'labels'/f"{row['id']}.npy").resolve())}
        if Path(item['label']).exists() and all(Path(v).exists() for v in item['features'].values()):full.append(item)
    meta={'records':full,'planned_records':len(all_rows),'completed':len(full),'focus_speeds_m_s':FOCUS_SPEEDS,'feature_shape_per_source':[30,HEIGHT,WIDTH],
        'x0':-.0191,'z0':0.,'dx':.0002,'label_role':'coarse initialization; native physical inversion still required','training_case':'Neg_07_Left','development_case':'Neg_35_Left',
        'code_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),Path(__file__).with_name('focus_features.py')]}}
    atomic_json(a.out/'index.json',meta);print(json.dumps({'prepared':len(full),'planned':len(all_rows),'seconds':time.time()-start,'peak_GiB':torch.cuda.max_memory_allocated()/2**30}),flush=True)

if __name__=='__main__':main()
