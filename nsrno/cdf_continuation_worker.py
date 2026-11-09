"""Original native physics for the two required inverse objectives, no JVPs."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable THP')
from pathlib import Path
import gc
import json
import os
import time
import traceback
import numpy as np
import torch
from .frequency_multilevel_acquisition import LowHalfMHzAcquisition
from .full90_3mhz_acquisition import Full90ThreeMHzAcquisition
from .native_time import native_wave
from . import native_time
from .native_full_source_dual import NativeWork
from .low_frequency_0p5 import atomic,digest


def array(path,value):
    tmp=path.with_name(path.stem+'.partial.npy')
    np.save(tmp,value)
    tmp.replace(path)


def worker(rank,device,jobs,results,plan_path,folder,deadline):
    token=-1
    try:
        folder=Path(folder)
        folder.mkdir(parents=True,exist_ok=True)
        plan=json.loads(Path(plan_path).read_text())
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(device).total_memory,device)
        gpu=f'cuda:{device}'
        work=NativeWork(folder)
        for prefix in ['issued','completed']:
            work.state.setdefault(prefix+'_tangent_direction_passes',0)
        edge_path=Path('out/v5/uno_triparam_v1/known_edge_alpha.npy')
        if digest(edge_path)!=plan['known_edge_SHA256']:
            raise RuntimeError('Known absorption changed')
        edge=torch.from_numpy(np.broadcast_to(np.load(edge_path),(961,1121)).copy()).to(gpu)
        z,x=np.ogrid[:961,:1121]
        active=torch.from_numpy((z>=80)&(z<937)&(x>=24)&(x<1097)).to(gpu)
        acquisitions={}
        for name,spec in plan['families'].items():
            acquisitions[name]=(LowHalfMHzAcquisition(plan['data_root'],spec['source'],gpu,128)
                if spec['centre']=='low' else Full90ThreeMHzAcquisition(plan['high_index'],spec['source'],gpu,128))
        objectives={}
        for name,definition in plan['objectives'].items():
            objective={}
            for family,spec in definition['families'].items():
                a=acquisitions[family]
                weights=torch.zeros(len(a.frequencies),dtype=torch.float64,device=gpu)
                for lo,hi in spec['bands_hz']:
                    mask=torch.from_numpy((a.frequencies>=lo)&(a.frequencies<=hi)).to(gpu)
                    energy=a.observed[:,:,mask].abs().square().sum()
                    if float(energy)<=0:
                        raise RuntimeError('Zero observed band energy')
                    weights[mask]+=1/(len(spec['bands_hz'])*energy)
                objective[family]=(a,weights,spec['coefficient'])
            objectives[name]=objective
        atomic(folder/'ready.json',{'pid':os.getpid(),'rank':rank,'device':gpu,
            'loaded_acquisition_families':list(acquisitions),'objectives':list(objectives),
            'patient_truth_read':False,'no_source_encoding':True})
        while True:
            job=jobs.get()
            if job is None:
                break
            token,kind,objective=job['token'],job['kind'],job['objective']
            if kind not in ['gradient','forward'] or objective not in objectives:
                raise ValueError('Only the necessary actual inverse gradients and candidate forwards')
            material=torch.from_numpy(job['material']).to(gpu)
            material.requires_grad_(kind=='gradient')
            aggregate=torch.zeros_like(material,dtype=torch.float64) if kind=='gradient' else None
            gradient_path=folder/f'aggregate_control_gradient_token{token:04d}.npy'
            archive=folder/'arrays'/f'token{token:04d}'
            archive.mkdir(parents=True)
            groups={name:{'loss':0.,'shots':[]} for name in objectives[objective]}
            total,started=0.,time.time()
            for family,shot in job['sources']:
                if time.time()>=deadline-60 or (folder.parent.parent/'stop_requested.flag').exists():
                    raise RuntimeError('Fixed deadline or cooperative stop; partial work retained')
                a,weights,coefficient=objectives[objective][family]
                wave=a.waveform(shot)
                atomic(folder/'progress.json',{'token':token,'kind':kind,'objective':objective,
                    'family':family,'shot':shot,'state':'actual_inverse_propagation'})
                if kind=='gradient':
                    with torch.enable_grad():
                        raw=work.run(f'{token}_{objective}_{family}_{shot}_gradient_background',
                            lambda:native_wave(material,wave,a.si,a.sw,a.ri,a.rw,a.config),physical=1)
                        prediction=a.scattered_coefficients(raw,shot)
                        residual=a.observed[shot]-prediction
                        scalar=coefficient*(weights[None]*residual.abs().square()).sum()
                        work.run(f'{token}_{objective}_{family}_{shot}_ordinary_adjoint',
                            lambda:scalar.backward(),adjoint=1)
                    with torch.no_grad():
                        block=torch.stack([.05*material[0]*material.grad[0],
                            .10*material[1]*material.grad[1],.70*(material[2]-edge)*material.grad[2]]).double()*active
                        aggregate+=block
                        array(gradient_path,aggregate.cpu().numpy())
                    material.grad=None
                    del raw,scalar,block
                else:
                    def forward():
                        with torch.no_grad():
                            raw=native_wave(material,wave,a.si,a.sw,a.ri,a.rw,a.config)
                            return a.scattered_coefficients(raw,shot).detach()
                    prediction=work.run(f'{token}_{objective}_{family}_{shot}_candidate_forward',forward,physical=1)
                    residual=a.observed[shot]-prediction
                # The required gradient's forward prediction is retained too;
                # no extra propagation to obtain a new-point baseline/cache.
                array(archive/f'{family}_{shot:02d}.npy',prediction.detach().cpu().numpy())
                value=float((coefficient*(weights[None]*residual.abs().square()).sum()).detach())
                total+=value
                groups[family]['loss']+=value
                groups[family]['shots'].append(shot)
                atomic(folder/'progress.json',{'token':token,'kind':kind,'objective':objective,
                    'family':family,'shot':shot,'state':'source_completed','source_RF_loss':value})
                del prediction,residual,wave
            if kind=='gradient':
                atomic(folder/f'gradient_receipt_token{token:04d}.json',{'path':str(gradient_path),
                    'SHA256':digest(gradient_path),'objective':objective,'completed_sources':job['sources']})
            results.put({'rank':rank,'token':token,'kind':kind,'objective':objective,
                'loss':total,'groups':groups,'sources':job['sources'],'seconds':time.time()-started,
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'error':None})
            del material,aggregate,job
            native_time._TAPES.clear()
            gc.collect()
            torch.cuda.empty_cache()
    except BaseException:
        message=traceback.format_exc()
        atomic(Path(folder)/'error.json',{'token':token,'error':message,'all_issued_work_retained':True})
        results.put({'rank':rank,'token':token,'error':message})
        raise
