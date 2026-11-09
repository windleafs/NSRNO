"""Balanced source batches and one bounded native GPU experiment."""
from pathlib import Path
import multiprocessing as mp
import os
import queue
import time
import numpy as np
from .native_frequency_incremental_worker import worker
from .low_frequency_0p5 import atomic


def source_batches(families,seed,cycle):
    order=np.random.default_rng(seed+cycle).permutation(11)
    return [[(name,int(shot)) for name in families for shot in order[i:i+2]]
            for i in range(0,11,2)]


class NativeFrequencyIncrementalPool:
    def __init__(self,args,monitor,plan):
        self.args,self.monitor,self.plan=args,monitor,plan
        self.token=0
        self.counts={'gradient_jobs':0,'native_adjoint_source_shots':0,
                     'forward_only_jobs':0,'native_forward_only_source_shots':0,
                     'cached_RF_jobs':0}
        context=mp.get_context('spawn')
        self.jobs=[context.Queue() for _ in args.devices]
        self.results=context.Queue()
        self.workers=[]
        monitor.wait_budget(args.devices,args.cap_gib,'before_native_incremental_experiment')
        for rank,device in enumerate(args.devices):
            process=context.Process(target=worker,args=(rank,device,self.jobs[rank],
                self.results,str(args.out/'native_plan.json'),args.chunk,args.cap_gib,
                str(args.out/'workers'/f'rank{rank:02d}')))
            process.start()
            self.workers.append(process)
        atomic(args.out/'worker_processes.json',{'coordinator_pid':os.getpid(),
            'worker_pids':[p.pid for p in self.workers],'devices':args.devices,
            'one_GPU_experiment':True})

    def evaluate(self,material,kind,sources,tag,save_RF=False):
        sources=[(str(f),int(s)) for f,s in sources]
        if len(sources)!=len(set(sources)) or any(
                f not in self.plan['families'] or not 0<=s<11 for f,s in sources):
            raise ValueError('Repeated or unknown source')
        numbers=[sum(f==name for f,s in sources) for name in self.plan['families']]
        if not numbers or min(numbers)==0 or len(set(numbers))!=1:
            raise ValueError('Each acquired family must contribute equally many sources')
        sample_weight=11/numbers[0]
        if kind=='cached_forward' and numbers[0]!=11:
            raise ValueError('The cached baseline must include all 44 actual sources')
        self.token+=1
        token=self.token
        started=time.time()
        atomic(self.args.out/'active_job.json',{'token':token,'kind':kind,'tag':tag,
            'source_shots':len(sources),'sample_weight':sample_weight,
            'started_unix_s':started,'truth_read':False})
        payload=None if kind=='cached_forward' else np.ascontiguousarray(material,np.float32)
        if payload is not None and payload.shape!=(3,961,1121):
            raise ValueError('Full native material required')
        for rank,jobs in enumerate(self.jobs):
            jobs.put({'token':token,'kind':kind,'material':payload,
                'sources':sources[rank::len(self.jobs)],'sample_weight':sample_weight,
                'save_RF':save_RF})
        rows=[]
        while len(rows)<len(self.workers):
            try:
                row=self.results.get(timeout=20)
            except queue.Empty:
                used=self.monitor.sample(tag)
                if any(v['owned_MiB']>16384 for v in used.values()):
                    raise RuntimeError('Observed per-card owned GPU memory exceeds 16 GiB')
                stopped=[(p.pid,p.exitcode) for p in self.workers if p.exitcode is not None]
                if stopped:
                    raise RuntimeError(f'Worker stopped at {tag}: {stopped}')
                continue
            if row.get('error'):
                raise RuntimeError(f"Native worker{row['rank']}:\n{row['error']}")
            if row['token']!=token or row['kind']!=kind or any(r['rank']==row['rank'] for r in rows):
                raise RuntimeError('Unexpected or repeated worker result')
            rows.append(row)
            self.monitor.sample(tag)
        rows.sort(key=lambda r:r['rank'])
        result={'token':token,'kind':kind,'tag':tag,'loss':sum(r['loss'] for r in rows),
                'seconds':time.time()-started,'source_shots':len(sources),
                'sample_weight':sample_weight,'groups':{},'truth_read':False,
                'actual_native_wave_operator':kind!='cached_forward',
                'actual_native_RF_from_verified_parent_cache':kind=='cached_forward',
                'full_all_source_objective':numbers[0]==11,
                'peak_Torch_GiB':max(r['peak_Torch_GiB'] for r in rows),
                'peak_reserved_GiB':max(r['peak_reserved_GiB'] for r in rows)}
        check=0.
        for name in self.plan['families']:
            values=[r['groups'][name] for r in rows]
            completed=sorted(s for v in values for s in v['completed_shots'])
            expected=sorted(s for f,s in sources if f==name)
            if completed!=expected:
                raise RuntimeError('Incomplete source batch')
            energy=np.asarray(values[0]['band_energy'])
            error=sum((np.asarray(v['band_error_squared']) for v in values),np.zeros_like(energy))
            coefficient=self.plan['families'][name]['coefficient']
            check+=sample_weight*coefficient*float(np.mean(error/energy))
            result['groups'][name]={'band_relative_RMS':np.sqrt(sample_weight*error/energy).tolist(),
                'band_error_squared':error.tolist(),'band_energy':energy.tolist(),
                'coefficient':coefficient,'source_shots':len(completed)}
        if not np.isfinite(result['loss']) or abs(check-result['loss'])>1e-10*max(1.,abs(check)):
            raise RuntimeError('Inconsistent native batch objective')
        gradient=None
        if kind=='gradient':
            gradient=sum((r['gradient'] for r in rows),np.zeros((3,961,1121),np.float64))
            if not np.isfinite(gradient).all():
                raise RuntimeError('Nonfinite native source gradient')
            self.counts['gradient_jobs']+=1
            self.counts['native_adjoint_source_shots']+=len(sources)
        elif kind=='forward':
            self.counts['forward_only_jobs']+=1
            self.counts['native_forward_only_source_shots']+=len(sources)
        else:
            self.counts['cached_RF_jobs']+=1
        result['cumulative_completed_jobs']=dict(self.counts)
        atomic(self.args.out/'jobs'/f'{token:04d}_{tag}.json',result)
        return result,gradient

    def close(self):
        for jobs in self.jobs:
            jobs.put(None)
        for process in self.workers:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.monitor.sample('native_incremental_workers_closed')

