"""One source-streamed native experiment, with certified RF objective intervals."""
from pathlib import Path
import multiprocessing as mp
import os
import queue
import time
import json
import numpy as np
from .native_transport_worker import worker
from .rf_depth_weights import WEIGHTS
from .wri_worker import atomic


class Pool:
    def __init__(self,args,monitor):
        self.args=args;self.monitor=monitor;self.token=0
        context=mp.get_context('spawn');self.jobs=[context.Queue() for _ in args.devices]
        self.results=context.Queue();self.workers=[]
        monitor.wait_budget(args.devices,args.cap_gib,'before_native_transport_workers')
        options={'lambda':args.transport_lambda,'tolerance':args.inner_tolerance,
                 'maximum':args.inner_maximum,'smoothing_fraction':args.smoothing_fraction}
        for rank,device in enumerate(args.devices):
            proc=context.Process(target=worker,args=(rank,device,self.jobs[rank],self.results,args.sample,
                args.chunk,args.cap_gib,str(args.out/'workers'/f'rank{rank:02d}'),len(args.devices),options))
            proc.start();self.workers.append(proc)
        atomic(args.out/'worker_processes.json',{'coordinator_pid':os.getpid(),
            'worker_pids':[p.pid for p in self.workers],'devices':args.devices,
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})

    def evaluate(self,material,kind,tag):
        self.token+=1;token=self.token;started=time.time()
        job={'token':token,'kind':kind,'material':np.asarray(material,np.float32)}
        atomic(self.args.out/'active_job.json',{'tag':tag,'token':token,'kind':kind,
            'started_unix_s':started,'truth_read':False})
        for jobs in self.jobs:jobs.put(job)
        rows=[]
        while len(rows)<len(self.workers):
            try:row=self.results.get(timeout=20)
            except queue.Empty:
                self.monitor.sample(tag)
                dead=[(i,p.exitcode) for i,p in enumerate(self.workers) if p.exitcode is not None]
                if dead:raise RuntimeError(f'Native transport workers exited at {tag}: {dead}')
                continue
            if row.get('error'):raise RuntimeError(f'Worker {row["rank"]}, {tag}:\n{row["error"]}')
            if row['token']!=token or any(r['rank']==row['rank'] for r in rows):raise RuntimeError('Duplicate or unexpected worker result')
            rows.append(row);self.monitor.sample(tag)
        lower=sum(r['lower'] for r in rows);upper=sum(r['upper'] for r in rows)
        result={'tag':tag,'token':token,'kind':kind,'loss':.5*(lower+upper),'lower':lower,'upper':upper,
            'relative_gap':(upper-lower)/max(upper,1e-300),'raw_RF_loss':sum(r['raw_RF_loss'] for r in rows),
            'CPU_inner_seconds':sum(r['CPU_inner_seconds'] for r in rows),
            'maximum_channel_relative_gap':max(r['maximum_channel_relative_gap'] for r in rows),
            'all_inner_converged':all(r['all_inner_converged'] for r in rows),
            'RF_adjoint_L2_error_upper_bound':float(np.sqrt(sum(r['RF_adjoint_L2_error_upper_bound']**2 for r in rows))),
            'seconds':time.time()-started,'peak_Torch_GiB':max(r['peak_Torch_GiB'] for r in rows),
            'groups':{},'truth_read':False,'gradient_RF_inner_error_certified_not_zero':True}
        gradient=sum((r['gradient'] for r in rows),np.zeros_like(material,dtype=np.float64)) if kind=='gradient' else None
        for source in ['lf','point3']:
            values=[r['groups'][source] for r in rows];den=np.array(values[0]['window_energy'])
            numer=sum((np.array(v['window_error_squared']) for v in values),np.zeros(5))
            error=sum(v['error_squared'] for v in values)
            result['groups'][source]={'relative_RMS':float(np.sqrt(error/values[0]['energy'])),
                'geometry_window_relative_RMS':np.sqrt(numer/den).tolist(),
                'geometry_loss':.5*float(np.dot(WEIGHTS,numer/den)),
                'window_error_squared':numer.tolist(),'window_energy':den.tolist()}
        check=sum(v['geometry_loss'] for v in result['groups'].values())/2
        if abs(check-result['raw_RF_loss'])>1e-5*max(abs(check),1e-20):
            raise RuntimeError('Original raw-RF reporting disagrees with its fixed window statistics')
        if not np.isfinite(result['loss']) or gradient is not None and not np.isfinite(gradient).all():
            raise RuntimeError('Nonfinite RF transport result')
        atomic(self.args.out/'jobs'/f'{token:04d}_{tag}.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)
        return result,gradient

    def close(self):
        for jobs in self.jobs:jobs.put(None)
        for proc in self.workers:
            proc.join(timeout=30)
            if proc.is_alive():proc.terminate();proc.join(timeout=10)
        self.monitor.sample('native_transport_workers_closed')
