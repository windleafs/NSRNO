"""Stream native jobs and return persistent per-shot curvature metadata."""
from pathlib import Path
import multiprocessing as mp
import os
import queue
import time
import json
import numpy as np
from .native_gn_block_worker import worker
from .native_gn_coordinates import dot
from .pressure_resources import PressureResources
from .wri_worker import atomic
from .rf_depth_weights import WEIGHTS


class Pool:
    def __init__(self,args,monitor):
        self.args=args
        self.monitor=monitor
        self.token=0
        context=mp.get_context('spawn')
        self.jobs=[context.Queue() for _ in args.devices]
        self.results=context.Queue()
        self.workers=[]
        monitor.wait_budget(args.devices,args.cap_gib,'before_source_resolved_GN_workers')
        for rank,device in enumerate(args.devices):
            proc=context.Process(target=worker,args=(rank,device,self.jobs[rank],self.results,
                args.sample,args.chunk,args.cap_gib,str(args.out/'workers'/f'rank{rank:02d}'),
                len(args.devices)))
            proc.start()
            self.workers.append(proc)
        atomic(args.out/'worker_processes.json',{'coordinator_pid':os.getpid(),
            'worker_pids':[p.pid for p in self.workers],'devices':args.devices,
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})

    def evaluate(self,material,kind,tag,direction=None):
        self.token+=1
        token=self.token
        started=time.time()
        job={'token':token,'kind':kind,'material':np.asarray(material,np.float32)}
        if direction is not None:job['direction']=np.asarray(direction,np.float32)
        atomic(self.args.out/'active_job.json',{'tag':tag,'token':token,'kind':kind,
            'started_unix_s':started,'truth_read':False})
        for jobs in self.jobs:jobs.put(job)
        rows=[]
        while len(rows)<len(self.workers):
            try:row=self.results.get(timeout=20)
            except queue.Empty:
                self.monitor.sample(tag)
                dead=[(i,p.exitcode) for i,p in enumerate(self.workers) if p.exitcode is not None]
                if dead:raise RuntimeError(f'Native GN workers exited during {tag}: {dead}')
                continue
            if row.get('error'):raise RuntimeError(f'Worker {row["rank"]}, {tag}:\n{row["error"]}')
            if row['token']!=token or any(r['rank']==row['rank'] for r in rows):
                raise RuntimeError('Unexpected or duplicate native worker result')
            rows.append(row)
            self.monitor.sample(tag)
        result={'tag':tag,'token':token,'kind':kind,'loss':sum(r['loss'] for r in rows),
            'seconds':time.time()-started,'peak_Torch_GiB':max(r['peak_Torch_GiB'] for r in rows),
            'groups':{},'truth_read':False}
        if kind in ['hvp','gradient']:
            gradient=sum((r['gradient'] for r in rows),np.zeros_like(material,dtype=np.float64))
        else:gradient=None
        if kind=='hvp':
            result['weighted_tangent_energy']=sum(r['tangent_energy'] for r in rows)
            result['adjoint_quadratic']=dot(direction,gradient)
            den=max(abs(result['weighted_tangent_energy']),abs(result['adjoint_quadratic']),1e-30)
            defect=abs(result['weighted_tangent_energy']-result['adjoint_quadratic'])/den
            result['routine_curvature_identity_relative_error']=defect
            if defect>2e-3:raise RuntimeError('Native HVP/tangent energy identity failed')
            images=sorted((p for r in rows for p in r['source_curvature']),
                key=lambda p:(p['source'],p['shot']))
            expected={(s,j) for s in ['lf','point3'] for j in range(11)}
            if len(images)!=22 or {(p['source'],p['shot']) for p in images}!=expected:
                raise RuntimeError('Per-shot export omitted or duplicated a native shot')
            for p in images:
                image=np.load(p['image'],mmap_mode='r')
                if image.dtype!=np.float32 or image.shape!=material.shape:
                    raise RuntimeError('Per-shot export changed dtype or native material grid')
            result['source_curvature']=images
            result['extra_wave_solves_for_source_export']=0
        else:
            if kind=='gradient':
                features=sorted((p for r in rows for p in r['residual_features']),key=lambda p:(p['source'],p['shot']))
                expected={(s,j) for s in ['lf','point3'] for j in range(11)}
                if len(features)!=22 or {(p['source'],p['shot']) for p in features}!=expected:
                    raise RuntimeError('Gradient export omitted or duplicated a shot')
                energy=sum(p['weighted_residual_energy'] for p in features)
                if abs(energy-2*result['loss'])>1e-8*max(energy,1e-30):
                    raise RuntimeError('Source residual energies disagree with full RF loss')
                result['residual_features']=features
                result['extra_wave_solves_for_gradient_export']=0
            for source in ['lf','point3']:
                values=[r['groups'][source] for r in rows]
                den=np.array(values[0]['window_energy'])
                numer=sum((np.array(v['window_error_squared']) for v in values),np.zeros(5))
                error=sum(v['error_squared'] for v in values)
                energy=values[0]['energy']
                result['groups'][source]={'relative_RMS':float(np.sqrt(error/energy)),
                    'geometry_window_relative_RMS':np.sqrt(numer/den).tolist(),
                    'geometry_loss':.5*float(np.dot(WEIGHTS,numer/den)),
                    'window_error_squared':numer.tolist(),'window_energy':den.tolist()}
            check=sum(v['geometry_loss'] for v in result['groups'].values())/2
            if abs(check-result['loss'])>1e-5*max(abs(check),1e-20):
                raise RuntimeError('Native objective disagrees with RF statistics')
        atomic(self.args.out/'jobs'/f'{token:04d}_{tag}.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['groups','source_curvature','residual_features']}),flush=True)
        return result,gradient

    def close(self):
        for jobs in self.jobs:jobs.put(None)
        for proc in self.workers:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
        self.monitor.sample('source_resolved_native_GN_workers_closed')
