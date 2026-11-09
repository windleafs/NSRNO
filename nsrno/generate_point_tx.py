"""Independent original-UltraWave 3 MHz diverging acquisitions of V5 media.

Never import torch here: the original OpenACC runtime must remain separate.
This changes acquisition, not the material truth or the prior 11 plane waves.
"""
from argparse import ArgumentParser
from pathlib import Path
import ctypes,fcntl,hashlib,json,os,time
if os.name=='posix':
    if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise OSError('Cannot disable process transparent huge pages')
import numpy as np
from scipy import fft
from .fullwave import initialize
from .v5_observations import canonical_id
from .v5_materials import evaluation_material,material_paths

def atomic_json(path,value):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def project(rf):
    frequency=fft.rfftfreq(rf.shape[-1],1/40e6);spectrum=fft.rfft(rf,axis=-1)
    # Exactly the same physical pass interval as the original 640-bin IQ cache.
    spectrum[..., (frequency<40/(2401/40e6)) | (frequency>=680/(2401/40e6))]=0
    return fft.irfft(spectrum,n=rf.shape[-1],axis=-1).astype(np.float32)

def source_layout(case):
    x,xe=case['x'],case['xe'];cols=np.flatnonzero((x>=xe[0])&(x<=xe[-1]))
    elements=np.rint(np.linspace(5,len(xe)-6,11)).astype(int);positions=xe[elements]
    offset=x[cols][None,:]-positions[:,None];width=.0002
    weights=np.where(abs(offset)<=width/2,.5*(1+np.cos(2*np.pi*offset/width)),0.)
    return cols,elements,positions,weights

def configure_case(g,maps=None):
    case=g.geometry_case(maps);case.update(f0=3e6,nt=36001,end=90e-6,tref=4/3e6)
    cols,_,_,_=source_layout(case);case['delays']=np.zeros(len(cols));return case

def acquire(g,case,fit,weights,label,cache=None):
    solver=g.solver_for(case,case['maps'],fit);result=[];times=[]
    for i,w in enumerate(weights):
        shot=cache/f'{label}.shot{i:02d}.npy' if cache else None
        if shot is not None and shot.exists():
            result.append(np.load(shot));times.append(None);continue
        case['weights']=w;_,_,wave=g.bench.source_arrays(case);solver.source.data[:]=wave.T
        raw,timing=solver.run();result.append(raw);times.append(timing['solve_readback_s'])
        if shot is not None:
            temp=shot.with_suffix('.partial.npy');np.save(temp,raw);temp.replace(shot)
        print(json.dumps({'kind':label,'point':i,'seconds':times[-1]}),flush=True)
    return np.stack(result),times

def main():
    p=ArgumentParser();p.add_argument('--out',type=Path,default=Path('out/v5/point_tx_3mhz_v1'));p.add_argument('--id',default='train_200')
    p.add_argument('--material',type=Path,help='Optional native absolute [c,rho,alpha] array');a=p.parse_args();a.id=canonical_id(a.id)
    a.out.mkdir(parents=True,exist_ok=True)
    generation_lock=(a.out/'generation.lock').open('a');fcntl.flock(generation_lock.fileno(),fcntl.LOCK_EX)
    previous=json.loads((a.out/'index.json').read_text()) if (a.out/'index.json').exists() else None
    if previous and any(r['id']==a.id for r in previous['records']):raise FileExistsError('Preserve indexed acquisitions')
    if (a.out/f'{a.id}.rf.npy').exists():raise FileExistsError('Preserve generated acquisitions')
    cache=a.out/'native_shots';cache.mkdir(exist_ok=True)
    material=evaluation_material(a.id,a.material);paths=material_paths(a.id,a.material)
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    provenance=cache/f'{a.id}.input.json'
    if provenance.exists():
        if json.loads(provenance.read_text())['material_sha256']!=hashes:raise ValueError('Cached shots belong to a different material')
    elif any(cache.glob(f'{a.id}.shot*.npy')):raise ValueError('Cannot identify material used by partial cached shots')
    atomic_json(provenance,{'material_sha256':hashes})
    g=initialize();case=configure_case(g);cols,elements,positions,weights=source_layout(case)
    low=json.loads(Path('out/v5/low_frequency_3mhz_v1/index.json').read_text());fit=low['absorption']
    if previous and previous['absorption']!=fit:raise ValueError('Preserve the existing reference relaxation model')
    layout_path=a.out/'source_layout.npz'
    if layout_path.exists():
        old=np.load(layout_path)
        for key,value in [('source_columns',cols),('transmit_x_m',positions),('source_weights',weights),('receiver_x_m',case['xe'])]:
            if not np.array_equal(old[key],value):raise ValueError('Existing source/receiver layout differs')
    else:
        np.savez(layout_path,source_columns=cols,element_indices=elements,transmit_x_m=positions,source_weights=weights,
            receiver_x_m=case['xe'],source_tref_s=np.full(len(weights),4/3e6),source_width_m=.0002)
    acquisition={'source_kind':'diverging_0p2mm_Hann_subaperture','source_f0_hz':3e6,'source_sigma_s':1/3e6,'source_peak_pressure_Pa':1e5,
        'native_nt':36001,'native_dt_s':2.5e-9,'rf_samples':3601,'fs_hz':40e6,'band_hz':[1.5e6,4.5e6],
        'retained_frequency_hz':[40/(2401/40e6),680/(2401/40e6)],'transmit_x_m':positions.tolist(),'receiver_x_m':case['xe'].tolist(),
        'source_width_m':.0002,'source_tref_s':(np.ones(len(weights))*4/3e6).tolist(),'same_FOV_depth_mm':43.}
    if previous and previous['acquisition']!=acquisition:raise ValueError('Preserve the acquisition definition of existing observations')
    reference_path=a.out/'reference_native.npy';start=time.time()
    if reference_path.exists():reference=np.load(reference_path,mmap_mode='r')
    else:
        reference,_=acquire(g,case,fit,weights,'reference',cache);temp=reference_path.with_suffix('.partial.npy');np.save(temp,reference);temp.replace(reference_path)
    maps=dict(sound_speed=material[0],density=material[1],alpha_coeff=material[2],BonA=np.zeros_like(material[2]))
    true_case=configure_case(g,maps);total,times=acquire(g,true_case,fit,weights,a.id,cache)
    values=[]
    for i in range(len(weights)):
        processed,_=g.sim.analytic_channels(total[i]-reference[i],2.5e-9,band=[1.5e6,4.5e6]);values.append(project(processed.T))
    rf=np.stack(values);temp=a.out/f'{a.id}.rf.partial.npy';np.save(temp,rf);temp.replace(a.out/f'{a.id}.rf.npy')
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record={'id':a.id,'rf':str((a.out/f'{a.id}.rf.npy').resolve()),'seconds_per_shot':times,
        'truth_used_only_by_generator':True,'material_sha256':hashes,'generator_source_sha256':source_hash}
    meta={'acquisition':acquisition,'absorption':fit,'reference_native':str(reference_path.resolve()),'source_layout':str((a.out/'source_layout.npz').resolve()),
        'records':[record],'generation_seconds':time.time()-start,'generator':'original UltraWave, no torch or Geo-Flow model','source_sha256':source_hash,
        'scope':'ADDITIONAL acquisitions; cannot establish sufficiency of original 11 plane waves','status':'complete'}
    if previous:
        meta.update(records=previous['records']+[record],source_sha256=previous['source_sha256'],
            generation_seconds_total=previous.get('generation_seconds_total',previous.get('generation_seconds',0.))+time.time()-start)
    else:meta['generation_seconds_total']=meta['generation_seconds']
    atomic_json(a.out/'index.json',meta);print(json.dumps({'completed':a.id,'rf_shape':list(rf.shape),'seconds':time.time()-start}),flush=True)

if __name__=='__main__':main()
