"""One bounded process pool, keeping counters across low/full objective stages."""
from pathlib import Path
import hashlib
import json
import multiprocessing as mp
import os
import queue
import time
import numpy as np
from .cdf_continuation_worker import worker
from .low_frequency_0p5 import atomic,digest


class CDFContinuationPool:
    def __init__(self,args,monitor,plan,deadline,freeze):
        self.args,self.monitor,self.plan,self.deadline,self.freeze=args,monitor,plan,deadline,freeze
        self.token=0
        self.gradient_keys=set()
        self.candidates={}
        self.final_scores=0
        ctx=mp.get_context('spawn')
        self.queues=[ctx.Queue() for _ in args.devices]
        self.results=ctx.Queue()
        self.workers=[]
        while True:
            freeze()
            usage=monitor.sample('CDF_continuation_GPU_admission')
            if all(usage[d]['owned_MiB']+10240+768<=16384 and
                usage[d]['whole_MiB']+10240+768<=usage[d]['total_MiB'] for d in args.devices):
                break
            time.sleep(20)
        for rank,device in enumerate(args.devices):
            process=ctx.Process(target=worker,args=(rank,device,self.queues[rank],self.results,
                str(args.out/'native_plan.json'),str(args.out/'workers'/f'rank{rank:02d}'),deadline))
            process.start()
            self.workers.append(process)
        atomic(args.out/'worker_processes.json',{'coordinator_pid':os.getpid(),
            'worker_pids':[p.pid for p in self.workers],'devices':args.devices,
            'one_GPU_experiment':True,'source_encoding':False})

    def actual_work(self):
        total,inflight={},0
        for rank in range(len(self.workers)):
            path=self.args.out/'workers'/f'rank{rank:02d}/native_work.json'
            if not path.exists():
                continue
            row=json.loads(path.read_text())
            for key,value in row.items():
                if key.startswith('issued_') or key.startswith('completed_'):
                    total[key]=total.get(key,0)+value
            inflight+=row['inflight_phase'] is not None
        total['unfinished_native_phases']=inflight
        if (total.get('issued_adjoint_direction_passes',0)>264 or
            total.get('issued_physical_forward_passes',0)>1100 or
            total.get('issued_tangent_direction_passes',0)!=0):
            raise RuntimeError('Declared264adjoint/1100forward/0JVP budget exceeded')
        atomic(self.args.out/'actual_native_work.json',total)
        return total

    def evaluate(self,material,kind,objective,point,final_score=False):
        self.freeze(final=final_score)
        material=np.ascontiguousarray(material,np.float32)
        signature=hashlib.sha256(material.tobytes()).hexdigest()
        if kind=='gradient':
            key=(objective,signature)
            if key in self.gradient_keys or not 1<=point<=8:
                raise RuntimeError('Do not repeat the same objective gradient at the same material')
            self.gradient_keys.add(key)
        elif kind=='forward':
            if final_score:
                if self.final_scores or objective!='full_joint':
                    raise RuntimeError('At most one mandatory final full score')
                self.final_scores+=1
            else:
                key=(objective,point)
                count=self.candidates.get(key,0)
                if count>=3 or not 1<=point<=8:
                    raise RuntimeError('At most3real candidates per declared actual point')
                self.candidates[key]=count+1
        else:
            raise ValueError('No additional derivative actions are implemented')
        self.token+=1
        entries=self.plan['objectives'][objective]['sources']
        expected=22 if objective=='low_background' else 44
        if len(entries)!=expected:
            raise RuntimeError('Unexpected required original-source count')
        atomic(self.args.out/'active_job.json',{'token':self.token,'kind':kind,'objective':objective,
            'point':point,'source_count':expected,'final_score':final_score,'started_unix_s':time.time()})
        for rank,channel in enumerate(self.queues):
            channel.put({'token':self.token,'kind':kind,'objective':objective,'material':material,
                'sources':entries[rank::len(self.queues)]})
        rows,started=[],time.time()
        while len(rows)<len(self.workers):
            try:
                row=self.results.get(timeout=20)
            except queue.Empty:
                self.freeze(final=final_score,inflight=True)
                usage=self.monitor.sample(f'{objective}_{kind}')
                self.actual_work()
                if any(v['owned_MiB']>16384 for v in usage.values()):
                    raise RuntimeError('Owned process tree exceeded16GiB')
                if any(p.exitcode is not None for p in self.workers):
                    raise RuntimeError('Inspect stopped worker; never replay partially issued work')
                continue
            if row.get('error'):
                raise RuntimeError(row['error'])
            if (row['token']!=self.token or row['kind']!=kind or row['objective']!=objective or
                any(r['rank']==row['rank'] for r in rows)):
                raise RuntimeError('Wrong or duplicate required inverse result')
            rows.append(row)
            self.monitor.sample(f'{objective}_{kind}')
        rows.sort(key=lambda r:r['rank'])
        if sorted(tuple(x) for r in rows for x in r['sources'])!=sorted(tuple(x) for x in entries):
            raise RuntimeError('An actual source is missing or duplicated')
        references=[]
        for index,(family,shot) in enumerate(entries):
            path=self.args.out/'workers'/f'rank{index%2:02d}'/'arrays'/f'token{self.token:04d}'/f'{family}_{shot:02d}.npy'
            references.append({'family':family,'shot':shot,'path':str(path.resolve()),'SHA256':digest(path)})
        result={'token':self.token,'kind':kind,'objective':objective,'point':point,
            'loss':sum(r['loss'] for r in rows),'seconds':time.time()-started,'source_count':expected,
            'all11_sources_per_family':True,'source_encoding':False,'patient_truth_read':False,
            'material_array_SHA256':signature,'RF_source_records':references,
            'groups':{},'actual_work':self.actual_work()}
        for family in self.plan['objectives'][objective]['families']:
            groups=[r['groups'][family] for r in rows]
            if sorted(s for g in groups for s in g['shots'])!=list(range(11)):
                raise RuntimeError('One source family is incomplete')
            result['groups'][family]={'loss':sum(g['loss'] for g in groups),'source_count':11}
        if kind=='gradient':
            gradient=np.zeros((3,961,1121),np.float64)
            for rank in range(len(self.workers)):
                receipt=json.loads((self.args.out/'workers'/f'rank{rank:02d}'/f'gradient_receipt_token{self.token:04d}.json').read_text())
                if receipt['objective']!=objective or digest(receipt['path'])!=receipt['SHA256']:
                    raise RuntimeError('Completed gradient receipt changed')
                gradient+=np.load(receipt['path'])
            path=self.args.out/'arrays'/f'data_gradient_token{self.token:04d}.npy'
            if path.exists() or not np.isfinite(gradient).all():
                raise RuntimeError('Preserve completed unique gradient or inspect nonfinite result')
            np.save(path,gradient)
            result.update(gradient_path=str(path.resolve()),gradient_SHA256=digest(path))
        atomic(self.args.out/'jobs'/f'{self.token:04d}_{objective}_{kind}.json',result)
        return result

    def close(self):
        for channel in self.queues:
            channel.put(None)
        for process in self.workers:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.actual_work()
        self.monitor.sample('CDF_continuation_workers_closed')
