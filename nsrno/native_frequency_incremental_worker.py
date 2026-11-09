"""Exact native source-batch gradients and RF for independent c/rho/alpha."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import gc
import json
import time
import traceback
import numpy as np
import torch
from .frequency_multilevel_acquisition import LowHalfMHzAcquisition
from .full90_3mhz_acquisition import Full90ThreeMHzAcquisition
from .low_frequency_0p5 import atomic,digest
from .native_time import native_wave
from . import native_time


def save(path,value):
    temporary=path.with_name(path.stem+'.partial.npy')
    np.save(temporary,value)
    temporary.replace(path)


def worker(rank,device,jobs,results,plan_path,chunk,cap_gib,folder):
    token=-1
    try:
        folder=Path(folder)
        folder.mkdir(parents=True,exist_ok=True)
        plan=json.loads(Path(plan_path).read_text())
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(
            cap_gib*2**30/torch.cuda.get_device_properties(device).total_memory,device)
        device=f'cuda:{device}'
        edge_path='out/v5/uno_triparam_v1/known_edge_alpha.npy'
        if digest(edge_path)!=plan['known_edge_SHA256']:
            raise ValueError('Fixed known attenuation edge changed')
        edge=np.load(edge_path)
        edge=torch.from_numpy(np.broadcast_to(edge,(961,1121)).copy()).to(device)
        z,x=np.ogrid[:961,:1121]
        mask=torch.from_numpy((z>=80)&(z<961-24)&(x>=24)&(x<1121-24)).to(device)
        if abs(sum(s['coefficient'] for s in plan['families'].values())-.5)>1e-12:
            raise ValueError('Objective coefficients must sum to one half')
        cache={}
        for family,spec in plan['families'].items():
            if spec['centre']=='low':
                a=LowHalfMHzAcquisition(plan['data_root'],spec['source'],device=device,chunk=chunk)
            else:
                a=Full90ThreeMHzAcquisition(plan['high_index'],spec['source'],device=device,chunk=chunk)
            masks=[torch.from_numpy((a.frequencies>=lo)&(a.frequencies<=hi)).to(device)
                   for lo,hi in spec['bands_hz']]
            energies=[a.observed[:,:,b].abs().square().sum() for b in masks]
            weights=torch.zeros(len(a.frequencies),dtype=torch.float64,device=device)
            for b,e in zip(masks,energies):
                if not float(e)>0:
                    raise ValueError('Zero-energy observation subband')
                weights[b]+=1/(len(masks)*e)
            cache[family]={'a':a,'masks':masks,'energies':energies,
                           'weights':weights,'coefficient':spec['coefficient']}
        references={(r['family'],int(r['shot'])):r for r in plan['reference_RF']}
        atomic(folder/'ready.json',{'rank':rank,'device':device,'truth_read':False,
            'native_shape':[3,961,1121],'only_actual_native_wave_operator':True})
        while True:
            job=jobs.get()
            if job is None:
                break
            token=job['token']
            kind=job['kind']
            if kind not in ['gradient','forward','cached_forward']:
                raise ValueError(kind)
            started=time.time()
            gradient_needed=kind=='gradient'
            cached=kind=='cached_forward'
            m=None if cached else torch.from_numpy(job['material']).to(device)
            if m is not None:
                m.requires_grad_(gradient_needed)
            torch.cuda.reset_peak_memory_stats()
            loss_sum=0.
            groups={name:{'band_error_squared':np.zeros(len(v['masks']),np.float64),
                          'band_energy':[float(e) for e in v['energies']],
                          'completed_shots':[],'coefficient':v['coefficient']}
                    for name,v in cache.items()}
            for family,shot in job['sources']:
                shot=int(shot)
                if family not in cache or not 0<=shot<11:
                    raise ValueError('Unexpected acquired source')
                slot=cache[family]
                a=slot['a']
                atomic(folder/'progress.json',{'token':token,'kind':kind,'family':family,
                    'shot':shot,'state':'reading_cached_RF' if cached else 'propagating',
                    'unix_s':time.time()})
                with torch.set_grad_enabled(gradient_needed):
                    if cached:
                        reference=references[family,shot]
                        if digest(reference['path'])!=reference['SHA256']:
                            raise ValueError('Native reference RF changed')
                        prediction=torch.from_numpy(np.load(reference['path'])).to(device)
                    else:
                        wave=a.waveform(shot)
                        raw=native_wave(m,wave,a.si,a.sw,a.ri,a.rw,a.config)
                        prediction=a.scattered_coefficients(raw,shot)
                    if prediction.shape!=a.observed[shot].shape:
                        raise ValueError('Actual native RF layout mismatch')
                    residual=prediction-a.observed[shot]
                    squared=residual.abs().square()
                    loss=(job['sample_weight']*slot['coefficient']*
                          (squared*slot['weights'][None]).sum())
                    if gradient_needed:
                        loss.backward()
                loss_sum+=float(loss.detach())
                group=groups[family]
                for j,b in enumerate(slot['masks']):
                    group['band_error_squared'][j]+=float(squared[:,b].sum().detach())
                group['completed_shots'].append(shot)
                if job['save_RF'] and not cached:
                    archive=folder/'fine_RF'/f'token{token:04d}'
                    archive.mkdir(parents=True,exist_ok=True)
                    save(archive/f'{family}_{shot:02d}.npy',prediction.detach().cpu().numpy())
                del prediction,residual,squared,loss
                if not cached:
                    del raw,wave
                atomic(folder/'progress.json',{'token':token,'kind':kind,'family':family,
                    'shot':shot,'state':'completed','seconds':time.time()-started,
                    'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
                    'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30})
            gradient=None
            if gradient_needed:
                gradient=torch.stack([
                    .05*(m[0]*m.grad[0]-m[1]*m.grad[1]),
                    .10*m[1]*m.grad[1],.70*(m[2]-edge)*m.grad[2]])*mask
                gradient=gradient.detach().double().cpu().numpy()
            for v in groups.values():
                v['band_error_squared']=v['band_error_squared'].tolist()
            results.put({'rank':rank,'token':token,'kind':kind,'loss':loss_sum,
                'gradient':gradient,'groups':groups,'source_shots':len(job['sources']),
                'seconds':time.time()-started,
                'peak_Torch_GiB':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'error':None})
            del job,m,gradient
            if gradient_needed:
                native_time._TAPES.clear()
                torch.cuda.empty_cache()
            gc.collect()
    except BaseException:
        message=traceback.format_exc()
        atomic(Path(folder)/'error.json',{'rank':rank,'token':token,'error':message})
        results.put({'rank':rank,'token':token,'error':message})
        raise

