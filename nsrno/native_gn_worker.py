"""Stream full native time-domain gradients/HVPs, one shot at a time."""
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
from .time_acquisition import TimeAcquisition
from .rf_depth_weights import DepthWeights,WEIGHTS
from .native_time import native_wave
from .native_tangent import tangent,pullback
from .wri_worker import atomic

def physical_pullback(m,edge,g):
    return torch.stack([.05*(m[0]*g[0]-m[1]*g[1]),
        .1*m[1]*g[1],.7*(m[2]-edge)*g[2]])

def worker(rank,device,jobs,results,sample,chunk,cap_gib,folder,workers):
    token=-1
    try:
        torch.set_num_threads(1);torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(
            cap_gib*2**30/torch.cuda.get_device_properties(device).total_memory,device)
        device=f'cuda:{device}';folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        cache={};edge=torch.from_numpy(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')).to(device)
        for source in ['lf','point3']:
            acquisition=TimeAcquisition(source,sample,device=device,chunk=chunk)
            if len(acquisition.angles)!=11 or acquisition.observed.shape[1]!=192:
                raise ValueError('The complete 11-shot/192-receiver acquisition is required')
            depth=DepthWeights(acquisition)
            coefficients=torch.tensor([.5*w/d for w,d in zip(WEIGHTS,depth.den)],
                device=device,dtype=torch.float32)
            weights=coefficients[depth.bins.long()]
            cache[source]=(acquisition,depth,weights)
        atomic(folder/'ready.json',{'rank':rank,'device':device,'complete_acquisition':True,
            'objective':'mean of LF and point3 geometry-normalized waveform losses',
            'window_denominators':{s:cache[s][1].den for s in cache},
            'all_windows_positive':True,'truth_read':False})
        while True:
            job=jobs.get()
            if job is None:break
            token=job['token'];kind=job['kind'];started=time.time()
            torch.cuda.reset_peak_memory_stats()
            m=torch.from_numpy(job['material']).to(device)
            mask=(torch.arange(m.shape[1],device=device)[:,None]>=80).expand(m.shape[1:])
            direction=None
            if kind=='hvp':
                u=torch.from_numpy(job['direction']).to(device=device,dtype=torch.float32)
                direction=torch.stack([m[0]*.05*u[0],m[1]*(.1*u[1]-.05*u[0]),
                    (m[2]-edge)*.7*u[2]])*mask
                del u
            need_gradient=kind=='gradient'
            if need_gradient:m.requires_grad_(True)
            aggregate=torch.zeros_like(m,dtype=torch.float64)
            loss_total=0.;tangent_energy=0.;groups={}
            for source,(a,depth,weights) in cache.items():
                bins_error=np.zeros(5,np.float64);error=0.
                for shot in range(rank,11,workers):
                    atomic(folder/'progress.json',{'token':token,'kind':kind,'source':source,
                        'shot':shot,'state':'propagating','started_unix_s':time.time()})
                    waveform=a.waveform(shot)
                    if kind=='hvp':
                        with torch.no_grad():
                            raw,draw,cp=tangent(m,direction,waveform,a.si,a.sw,a.ri,a.rw,a.config)
                            delta=(a.response@draw).T
                            weighted=weights[shot]*delta
                            tangent_energy+=float((weighted.double()*delta.double()).sum())
                            residual=(a.response.T@weighted.T).contiguous()
                            gradient=pullback(m,waveform,a.si,a.sw,a.ri,a.rw,a.config,cp,residual)
                            aggregate+=physical_pullback(m,edge,gradient).double()*mask
                            del raw,draw,cp,delta,weighted,residual,gradient
                    else:
                        with torch.set_grad_enabled(need_gradient):
                            raw=native_wave(m,waveform,a.si,a.sw,a.ri,a.rw,a.config)
                            reference=torch.from_numpy(a.reference[shot].copy()).to(device)
                            prediction=(a.response@(raw-reference)).T
                            difference=prediction-a.observed[shot]
                            loss=.5*(weights[shot].double()*difference.double().square()).sum()
                            if need_gradient:loss.backward()
                        loss_total+=float(loss.detach())
                        diff=difference.detach().double()
                        error+=float(diff.square().sum())
                        for k in range(5):
                            bins_error[k]+=float(diff[depth.bins[shot]==k].square().sum())
                        del raw,reference,prediction,difference,loss,diff
                    del waveform
                    atomic(folder/'progress.json',{'token':token,'kind':kind,'source':source,
                        'shot':shot,'state':'completed','elapsed_seconds':time.time()-started,
                        'peak_Torch_GiB':torch.cuda.max_memory_allocated()/2**30})
                if kind!='hvp':
                    energy=float(a.observed.double().square().sum())
                    groups[source]={'error_squared':error,'energy':energy,
                        'window_error_squared':bins_error.tolist(),'window_energy':depth.den}
            if need_gradient:
                aggregate=physical_pullback(m.detach(),edge,m.grad).double()*mask
            result={'rank':rank,'token':token,'kind':kind,'loss':loss_total,
                'gradient':aggregate.detach().cpu().numpy() if kind in ['hvp','gradient'] else None,
                'groups':groups,'tangent_energy':tangent_energy,'seconds':time.time()-started,
                'peak_Torch_GiB':torch.cuda.max_memory_allocated()/2**30,'error':None}
            results.put(result)
            del m,aggregate,direction,job,result
            gc.collect()
    except BaseException:
        message=traceback.format_exc()
        atomic(Path(folder)/'error.json',{'rank':rank,'token':token,'error':message})
        results.put({'rank':rank,'token':token,'error':message})
        raise
