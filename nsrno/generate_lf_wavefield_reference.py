"""Fixed homogeneous 3-MHz reference fields; no unknown medium is read."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import numpy as np
import torch
from .time_acquisition import TimeAcquisition
from .native_capture import capture


def atomic(path,value):
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('out/v5/conditioned_lf_full_v1'))
    parser.add_argument('--device',type=int,default=2)
    parser.add_argument('--memory-limit-gib',type=float,default=1.1)
    parser.add_argument('--frequency-block',type=int,default=32)
    args=parser.parse_args()
    import subprocess
    used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(s.split(',')[0]):int(s.split(',')[1]) for s in used.splitlines()}
    if used[args.device]+args.memory_limit_gib*1024+768>16*1024:
        raise RuntimeError('Not enough space within the 16-GiB whole-card limit')
    torch.set_num_threads(1);torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib*2**30/torch.cuda.get_device_properties(args.device).total_memory,args.device)
    device=f'cuda:{args.device}';start=time.time()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'index.json').exists():raise FileExistsError('Reference already complete')
    # This class loads acquisition references and observations, not material maps.
    acquisition=TimeAcquisition('lf','train_200',device='cpu')
    c=acquisition.config;nz,nx=c.nz,c.nx
    medium=torch.empty((3,nz,nx),dtype=torch.float32,device=device)
    medium[0].fill_(1540);medium[1].fill_(1000);medium[2].fill_(.002)
    si,sw,ri,rw=[v.to(device) for v in [acquisition.si,acquisition.sw,acquisition.ri,acquisition.rw]]
    bins=np.arange(91,271);frequencies=bins*40e6/2401
    source_index=json.loads(Path('out/v5/forward_cache_3mhz_v1/index.json').read_text())
    low=json.loads(Path('out/v5/low_frequency_3mhz_v1/index.json').read_text())
    spec={**source_index['grid'],**source_index['acquisition'],**low['absorption'],
          'native_dt_s':acquisition.meta['dt'],'native_nt':c.nt,'pml':c.pml,'bins':bins.tolist(),
          'angle_indices':list(range(11)),'known_alpha':False,
          'reference_format':'reference/aXX.npy[freq,z,x]',
          'reference_material':'c1540,rho1000,alpha0.002 everywhere; known homogeneous acquisition reference'}
    directory=args.out/'reference';directory.mkdir(exist_ok=True)
    progress_path=args.out/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'completed':[],'checks':[]}
    provenance={'status':'running','pid':os.getpid(),'device':args.device,'cap_GiB':args.memory_limit_gib,
                'unknown_material_read':False,'truth_read':False,'dataset_measurements_generated':False,
                'purpose':'fixed acquisition-only wavefield references for joint first-order wavefield inversion',
                'all_angles':11,'receivers':192,'frequencies':180,
                'capture_clock_hz':40e6,'frequency_spacing_hz':40e6/2401,
                'capture_precision':'FP64 accumulation of unchanged FP32 native pressure',
                'sources_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                  for n in ['native_time.cu','native_capture.cu','native_capture.py',Path(__file__).name]}}
    atomic(args.out/'run.json',provenance)
    for angle in range(11):
        path=directory/f'a{angle:02d}.npy'
        fields=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.complex64,shape=(len(bins),nz,nx))
        wave=acquisition.waveform(angle).to(device)
        reference=np.asarray(acquisition.reference[angle])
        for begin in range(0,len(bins),args.frequency_block):
            end=min(len(bins),begin+args.frequency_block);key=f'{angle}:{begin}:{end}'
            if key in progress['completed']:continue
            torch.cuda.reset_peak_memory_stats()
            pressure,raw=capture(medium,wave,si,sw,ri,rw,c,frequencies[begin:end])
            actual_peak=torch.cuda.max_memory_allocated()/2**30
            # These comparisons inspect the same reference run. There is no
            # extra material simulation, perturbation or identifiability test.
            relative=float(np.linalg.norm(raw.astype(np.float64)-reference)/np.linalg.norm(reference))
            samples=raw[::10]
            phase=np.exp(-2j*np.pi*frequencies[begin:end,None]*np.arange(len(samples))[None,:]/40e6)/2401
            trace=phase@samples
            iz=np.rint((acquisition.geometry['receiver_coordinates'][:,1]-acquisition.geometry['origin'][1])/c.dx).astype(int)
            ix=np.rint((acquisition.geometry['receiver_coordinates'][:,0]-acquisition.geometry['origin'][0])/c.dx).astype(int)
            observed=pressure[:,iz,ix]
            capture_error=float(np.linalg.norm(observed-trace)/max(np.linalg.norm(trace),1e-30))
            event={'angle':angle,'begin':begin,'end':end,'native_reference_relative_RMS':relative,
                   'receiver_capture_relative_RMS':capture_error,'peak_GiB':actual_peak,'elapsed_seconds':time.time()-start}
            if relative>5e-4 or capture_error>5e-4:
                atomic(args.out/'failure.json',event);raise RuntimeError('Reference/capture discrepancy: '+str(event))
            fields[begin:end]=pressure.astype(np.complex64);fields.flush()
            progress['completed'].append(key);progress['checks'].append(event)
            atomic(progress_path,progress);print(json.dumps(event),flush=True)
            del pressure,raw
        del fields,wave
    records=[{k:r[k] for k in ['id','case','role'] if k in r} for r in source_index['records']]
    atomic(args.out/'index.json',{'spec':spec,'records':records,'truth_read':False,
         'observations':'resolved through nsrno.v5_observations.plane_observation, no material arrays',
         'source':'lf','reference_clock':'same 40-MHz sampling before the known polyphase transfer',
         'reference_normalization':'1/2401','native_update_sha256':provenance['sources_sha256']['native_time.cu']})
    provenance.update(status='complete',elapsed_seconds=time.time()-start,
                      peak_GiB=max(v['peak_GiB'] for v in progress['checks']))
    atomic(args.out/'run.json',provenance)


if __name__=='__main__':main()
