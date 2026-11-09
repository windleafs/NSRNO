"""Rebuild exact attenuation labels and bounded independent material states.

CPU-only worker processes: this module must never import PyTorch.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from argparse import ArgumentParser
import json,os,math,hashlib
import numpy as np
from scipy.ndimage import zoom
from nsrno.fullwave import load_generator,build_case
from nsrno.dataset import write_json

ROOT=Path('out/v5/uno_triparam_v1'); GENERATOR=None

def save_array(path,value):
    path=Path(path); tmp=path.with_suffix('.partial.npy')
    with tmp.open('wb') as f: np.save(f,value); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def init_worker():
    global GENERATOR
    GENERATOR=load_generator()

def prepare_one(task):
    cached,original=task; path=ROOT/'alpha'/(cached['id']+'.npy'); receipt=path.with_suffix('.json')
    if receipt.exists() and path.exists(): return cached['id']
    case=build_case(GENERATOR,original)
    m=np.stack((case['maps']['sound_speed'],case['maps']['density']))
    archived=np.load(Path('out/v5/forward_cache_v1')/cached['medium'],mmap_mode='r')
    if not np.array_equal(m,archived): raise ValueError('Material recipe mismatch for '+cached['id'])
    alpha=case['maps']['alpha_coeff'].astype(np.float32)
    save_array(path,alpha)
    write_json(receipt,{'id':cached['id'],'c_rho_exactly_match':True,
        'alpha_sha256':hashlib.sha256(alpha.tobytes()).hexdigest(),'generator':str(GENERATOR.__file__)})
    return cached['id']

def spread(rows,n):
    rows=sorted(rows,key=lambda r:r['id'])
    return [rows[i] for i in np.linspace(0,len(rows)-1,n,dtype=int)]

def main():
    p=ArgumentParser(); p.add_argument('--workers',type=int,default=3); args=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True); (ROOT/'alpha').mkdir(exist_ok=True); (ROOT/'materials').mkdir(exist_ok=True)
    high=json.load(open('out/v5/forward_cache_v1/index.json')); low=json.load(open('out/v5/forward_cache_3mhz_v1/index.json'))
    manifest=json.load(open('out/v5/manifest.json')); lookup={r['id']:r for r in manifest['records']}
    tasks=[(r,lookup[r['id']]) for r in high['records']]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker) as pool:
        work=[pool.submit(prepare_one,t) for t in tasks]
        for n,future in enumerate(as_completed(work),1):
            sample=future.result()
            if n%24==0 or n==len(work): print(json.dumps({'alpha_labels':n,'total':len(work),'id':sample}),flush=True)
    rows=[]; energies={3.:[],7.5:[]}
    for source,index,cache in [(7.5,high,Path('out/v5/forward_cache_v1')),(3.,low,Path('out/v5/forward_cache_3mhz_v1'))]:
        for row in index['records']:
            medium=Path(row['medium']); medium=medium if medium.is_absolute() else cache/medium
            iq=cache/row['iq']; alpha=ROOT/'alpha'/(row['id']+'.npy')
            item={**row,'id':('hf_' if source==7.5 else 'lf_')+row['id'],'parent_id':row['id'],
                'medium':str(medium.resolve()),'alpha':str(alpha.resolve()),'iq':str(iq.resolve()),
                'source_mhz':source,'kind':'original'}
            rows.append(item)
            if row['role']=='train':
                target=np.load(iq,mmap_mode='r'); energies[source].append(float(np.mean(target.astype(np.float64)**2)*2))
    scales={str(s):{'iq_rms':math.sqrt(float(np.mean(e))),'training_records':len(e)} for s,e in energies.items()}
    write_json(ROOT/'index.json',{'version':1,'records':rows,'scales':scales,'native_shape':[3,961,1121],
        'unknowns':['c','rho','tissue_alpha0'],'alpha_power':1.1,'known_edge_component':True,
        'true_alpha_input_to_forward_training':True,'true_alpha_input_to_inverse':False,
        'materials_reproduced_exactly':len(high['records'])})
    generator=load_generator(); case=generator.geometry_case(); x,z=case['x'],case['z']
    half=generator.medium_builder.APERTURE_HALF_MM
    edge=(6*np.clip((abs(x*1e3)-(half+1))/(x[-1]*1e3-half-1),0,1)**2).astype(np.float32)
    save_array(ROOT/'known_edge_alpha.npy',edge)
    high_lookup={r['id']:r for r in high['records']}; plan=[]
    for role,n in [('train',16),('validation',8)]:
        selected=spread([r for r in low['records'] if r['role']==role],n)
        for i,parent in enumerate(selected):
            base=high_lookup[parent['id']]; pair=np.load(Path('out/v5/forward_cache_v1')/base['medium'])
            alpha=np.load(ROOT/'alpha'/(parent['id']+'.npy')); tissue=np.maximum(.002,alpha-edge[None])
            rng=np.random.default_rng(2026110500+i+(1000 if role=='validation' else 0))
            field=zoom(rng.standard_normal((17,19)),(len(z)/17,len(x)/19),order=3)
            field=np.clip((field-field.mean())/field.std(),-2,2).astype(np.float32)
            kind='alpha' if i%2==0 else 'trajectory'; group=f'{role}_{i:02d}_{kind}'
            for sign in [-1,1]:
                material=np.concatenate((pair,alpha[None]),axis=0).copy()
                if kind=='alpha': a=tissue*np.exp(sign*.2*field)
                else:
                    lc=.25 if sign<0 else .75; lr=.75 if sign<0 else .25
                    material[0]=1540+lc*(pair[0]-1540); material[1]=1000+lr*(pair[1]-1000)
                    la=.35 if sign<0 else .65; a=((1-la)*.5+la*tissue)*np.exp(sign*.12*field)
                material[2]=np.clip(a,.002,2)+edge[None]
                gel=z<.001-1e-7; material[0,gel]=1540; material[1,gel]=1000; material[2,gel]=.002+edge[None]
                name=f'{group}_{"minus" if sign<0 else "plus"}'; path=ROOT/'materials'/(name+'.npy')
                save_array(path,material.astype(np.float32))
                plan.append({'id':name,'pair_group':group,'pair_sign':sign,'kind':kind,'role':role,
                    'parent_id':parent['id'],'case':parent['case'],'variant':parent['variant'],
                    'material3':str(path.resolve()),'seed':2026110500+i+(1000 if role=='validation' else 0),
                    'alpha_power':1.1,'source_mhz':[7.5,3.],'angles':11,
                    'sha256':hashlib.sha256(material.tobytes()).hexdigest()})
    write_json(ROOT/'augmentation_plan.json',{'materials':plan,'training_materials':32,'validation_materials':16,
        'source_records':96,'all_angles':11,'independent_alpha_pairs':12,'trajectory_pairs':12})
    write_json(ROOT/'preparation_complete.json',{'completed':True,'existing_records':len(rows),'new_materials':len(plan),'scales':scales})
    print(json.dumps({'prepared':len(rows),'new_materials':len(plan),'scales':scales}),flush=True)

if __name__=='__main__': main()
