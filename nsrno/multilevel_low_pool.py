"""One experiment: shot-parallel coarse updates and true native RF evaluations."""
from pathlib import Path
import multiprocessing as mp
import os
import queue
import time
import numpy as np
from .multilevel_low_worker import worker
from .low_frequency_0p5 import atomic


class MultilevelPool:
    def __init__(self,args,monitor,plan,co):
        self.args=args
        self.monitor=monitor
        self.plan=plan
        self.co=co
        self.token=0
        self.counts={'coarse_gradients':0,'native_gradients':0,'native_forward_evaluations':0}
        context=mp.get_context('spawn')
        self.jobs=[context.Queue() for _ in args.devices]
        self.results=context.Queue()
        self.workers=[]
        monitor.wait_budget(args.devices,args.cap_gib,'before_single_multilevel_experiment')
        for rank,device in enumerate(args.devices):
            p=context.Process(target=worker,args=(
                rank,device,self.jobs[rank],self.results,str(args.out/'anchor_plan.json'),
                args.chunk,args.cap_gib,str(args.out/'workers'/f'rank{rank:02d}'),len(args.devices)))
            p.start()
            self.workers.append(p)
        atomic(args.out/'worker_processes.json',{
            'coordinator_pid':os.getpid(),'worker_pids':[p.pid for p in self.workers],
            'devices':args.devices,'one_experiment_coarse_and_native_jobs_are_sequential':True})

    def collect(self,token,kind,tag):
        rows=[]
        while len(rows)<len(self.workers):
            try:
                row=self.results.get(timeout=20)
            except queue.Empty:
                used=self.monitor.sample(tag)
                if any(v['owned_MiB']>16384 for v in used.values()):
                    raise RuntimeError('Observed per-card GPU process tree exceeds 16 GiB')
                dead=[(p.pid,p.exitcode) for p in self.workers if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f'Worker stopped at {tag}: {dead}')
                continue
            if row.get('error'):
                raise RuntimeError(f"Worker {row['rank']}, {tag}:\n{row['error']}")
            if row['token']!=token or row['kind']!=kind or any(r['rank']==row['rank'] for r in rows):
                raise RuntimeError('Duplicate or unexpected multilevel worker result')
            rows.append(row)
            self.monitor.sample(tag)
        return rows

    def evaluate(self,control,kind,tag):
        if kind not in ['coarse_gradient','fine_gradient','fine_forward']:
            raise ValueError(kind)
        self.token+=1
        token=self.token
        started=time.time()
        if kind=='coarse_gradient':
            payload={'control':np.asarray(control,np.float64)}
        else:
            payload={'native_material':self.co.native_material(control)}
        atomic(self.args.out/'active_job.json',{
            'token':token,'kind':kind,'tag':tag,'started_unix_s':started,'truth_read':False})
        for jobs in self.jobs:
            jobs.put({'token':token,'kind':kind,**payload})
        rows=self.collect(token,kind,tag)
        result={
            'token':token,'kind':kind,'tag':tag,'loss':sum(r['loss'] for r in rows),
            'seconds':time.time()-started,'band_hz':self.plan['band_hz'],
            'peak_Torch_GiB':max(r['peak_Torch_GiB'] for r in rows),
            'peak_reserved_GiB':max(r['peak_reserved_GiB'] for r in rows),
            'truth_read':False,'source_shots':22,
            'coarse_operator_approximation':kind=='coarse_gradient','groups':{}}
        for family in ['lf','point3']:
            values=[r['groups'][family] for r in rows]
            shots=sorted(shot for v in values for shot in v['completed_shots'])
            if shots!=list(range(11)):
                raise RuntimeError('Incomplete or repeated source shot')
            error=sum(v['error_squared'] for v in values)
            energy=values[0]['energy']
            band_energy=np.asarray(values[0]['band_energy'])
            band_error=sum((np.asarray(v['band_error_squared']) for v in values),
                           np.zeros_like(band_energy))
            result['groups'][family]={
                'relative_RMS':float(np.sqrt(error/energy)),
                'error_squared':error,'energy':energy,
                'band_relative_RMS':np.sqrt(band_error/band_energy).tolist(),
                'band_error_squared':band_error.tolist(),'band_energy':band_energy.tolist()}
        check=.25*sum(float(np.mean(np.asarray(v['band_error_squared'])/
                                    np.asarray(v['band_energy'])))
                       for v in result['groups'].values())
        if not np.isfinite(result['loss']) or abs(check-result['loss'])>1e-10:
            raise RuntimeError('Inconsistent or nonfinite waveform loss')
        gradient=None
        if kind!='fine_forward':
            shape=self.co.shape if kind=='coarse_gradient' else self.co.native_anchor.shape
            gradient=sum((r['gradient'] for r in rows),np.zeros(shape,np.float64))
            if not np.isfinite(gradient).all():
                raise RuntimeError('Nonfinite material gradient')
        key={'coarse_gradient':'coarse_gradients','fine_gradient':'native_gradients',
             'fine_forward':'native_forward_evaluations'}[kind]
        self.counts[key]+=1
        result['cumulative_completed_jobs']=dict(self.counts)
        atomic(self.args.out/'jobs'/f'{token:04d}_{tag}.json',result)
        return result,gradient

    def save_selected(self,expected_token):
        self.token+=1
        token=self.token
        kind='save_selected_coarse_RF'
        for jobs in self.jobs:
            jobs.put({'kind':kind,'token':token,'expected_token':expected_token})
        self.collect(token,kind,'save_selected_coarse_RF')

    def close(self):
        for jobs in self.jobs:
            jobs.put(None)
        for p in self.workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        self.monitor.sample('multilevel_workers_closed')
