"""Prepared bounded source-encoded native jobs; original full-source acceptance."""
from pathlib import Path
import multiprocessing as mp
import os
import queue
import time
import numpy as np
from .encoded_physical_native_worker import worker
from .low_frequency_0p5 import atomic


class EncodedPhysicalNativePool:
    def __init__(self, args, monitor, plan):
        self.args, self.monitor, self.plan = args, monitor, plan
        self.token = 0
        self.counts = {'encoded_gradient_jobs':0, 'encoded_adjoint_propagations':0,
            'encoded_forward_jobs':0, 'encoded_forward_propagations':0,
            'original_full_RF_checks':0, 'original_forward_propagations':0,
            'cached_full_RF_evaluations':0}
        self.attempted = {'encoded_adjoint_propagations_issued':0,
            'encoded_forward_propagations_issued':0, 'original_forward_propagations_issued':0}
        context = mp.get_context('spawn')
        self.jobs = [context.Queue() for _ in args.devices]
        self.results = context.Queue()
        self.workers = []
        monitor.wait_budget(args.devices, args.cap_gib, 'before_encoded_native_inverse')
        for rank, device in enumerate(args.devices):
            process = context.Process(target=worker, args=(rank, device, self.jobs[rank],
                self.results, str(args.out/'native_plan.json'), args.chunk, args.cap_gib,
                str(args.out/'workers'/f'rank{rank:02d}')))
            process.start()
            self.workers.append(process)
        atomic(args.out/'worker_processes.json', {'coordinator_pid':os.getpid(),
            'worker_pids':[p.pid for p in self.workers], 'devices':args.devices,
            'one_GPU_experiment':True})

    def evaluate(self, material, kind, tag, encodings=None, save_RF=False):
        encoded = kind in ['encoded_gradient', 'encoded_forward']
        if not encoded and kind not in ['forward', 'cached_forward']:
            raise ValueError('Bounded actual inverse jobs only')
        if encoded:
            if encodings is None or set(encodings)!=set(self.plan['families']):
                raise ValueError('Each of the four source families is required')
            entries = []
            for name in self.plan['families']:
                code = np.asarray(encodings[name])
                if code.shape!=(11,) or not np.all((code==-1)|(code==1)):
                    raise ValueError('All eleven unnormalized +-1 symbols are required')
                entries.append((name, [int(v) for v in code]))
            encodings = {name:code for name, code in entries}
        else:
            entries = [(family, shot) for family in self.plan['families'] for shot in range(11)]
        if encoded and save_RF:
            raise ValueError('Only original full-source RF is persisted for future acceptance')
        self.token += 1
        token, started = self.token, time.time()
        payload = None if kind=='cached_forward' else np.ascontiguousarray(material, np.float32)
        if payload is not None and payload.shape!=(3, 961, 1121):
            raise ValueError('Independent full native fields required')
        atomic(self.args.out/'active_job.json', {'token':token, 'kind':kind, 'tag':tag,
            'actual_acoustic_propagations':0 if kind=='cached_forward' else len(entries),
            'encoded':encoded, 'original_RF_components':44,
            'encoding':encodings, 'started_unix_s':started, 'truth_read':False})
        counter = ('encoded_adjoint_propagations_issued' if kind=='encoded_gradient' else
                   'encoded_forward_propagations_issued' if kind=='encoded_forward' else
                   'original_forward_propagations_issued' if kind=='forward' else None)
        if counter is not None:
            self.attempted[counter] += len(entries)
        atomic(self.args.out/'issued_propagations.json', {
            'token':token, **self.attempted, 'worker_completed_counters_retained_separately':True})
        for rank, jobs in enumerate(self.jobs):
            jobs.put({'token':token, 'kind':kind, 'material':payload,
                'encodings':entries[rank::len(self.jobs)] if encoded else [],
                'sources':[] if encoded else entries[rank::len(self.jobs)], 'save_RF':save_RF})
        rows = []
        while len(rows)<len(self.workers):
            try:
                row = self.results.get(timeout=20)
            except queue.Empty:
                values = self.monitor.sample(tag)
                if any(v['owned_MiB']>16384 for v in values.values()):
                    raise RuntimeError('Actual per-card owned memory exceeded 16 GiB')
                if any(p.exitcode is not None for p in self.workers):
                    raise RuntimeError('Same encoded native worker exited; inspect its error')
                continue
            if row.get('error'):
                raise RuntimeError(f"Native worker {row['rank']}:\n{row['error']}")
            if row['token']!=token or row['kind']!=kind or any(r['rank']==row['rank'] for r in rows):
                raise RuntimeError('Unexpected or duplicate source worker result')
            rows.append(row)
            self.monitor.sample(tag)
        rows.sort(key=lambda row:row['rank'])
        result = {'token':token, 'kind':kind, 'tag':tag, 'loss':sum(r['loss'] for r in rows),
            'seconds':time.time()-started, 'encoded':encoded, 'groups':{},
            'actual_acoustic_propagations':0 if kind=='cached_forward' else len(entries),
            'original_RF_components':44, 'original_all44_objective_measured':not encoded,
            'encoding':encodings, 'patient_truth_read':False,
            'peak_Torch_GiB':max(r['peak_Torch_GiB'] for r in rows),
            'peak_reserved_GiB':max(r['peak_reserved_GiB'] for r in rows)}
        check = 0.
        for name in self.plan['families']:
            groups = [r['groups'][name] for r in rows]
            energy = np.asarray(groups[0]['band_energy'])
            error = sum((np.asarray(g['band_error_squared']) for g in groups), np.zeros_like(energy))
            coefficient = self.plan['families'][name]['coefficient']
            check += coefficient*float(np.mean(error/energy))
            if encoded:
                completed = [code for g in groups for code in g['completed_encodings']]
                if completed!=[[int(v) for v in encodings[name]]]:
                    raise RuntimeError('Incomplete or mismatched source encoding')
            else:
                completed = sorted(shot for g in groups for shot in g['completed_shots'])
                if completed!=list(range(11)):
                    raise RuntimeError('Full original-source acceptance is incomplete')
            result['groups'][name] = {'band_error_squared':error.tolist(), 'band_energy':energy.tolist(),
                'coefficient':coefficient,
                'encoded_band_RMS_sample' if encoded else 'original_band_relative_RMS':
                    np.sqrt(error/energy).tolist()}
        if not np.isfinite(result['loss']) or abs(check-result['loss'])>1e-10*max(1.,abs(check)):
            raise RuntimeError('The full-family weighted objective is inconsistent')
        gradient = None
        if kind=='encoded_gradient':
            gradient = sum((r['gradient'] for r in rows), np.zeros((3, 961, 1121), np.float64))
            if not np.isfinite(gradient).all():
                raise RuntimeError('Nonfinite actual encoded inverse gradient')
            self.counts['encoded_gradient_jobs'] += 1
            self.counts['encoded_adjoint_propagations'] += len(entries)
        elif kind=='encoded_forward':
            self.counts['encoded_forward_jobs'] += 1
            self.counts['encoded_forward_propagations'] += len(entries)
        elif kind=='forward':
            self.counts['original_full_RF_checks'] += 1
            self.counts['original_forward_propagations'] += len(entries)
        else:
            self.counts['cached_full_RF_evaluations'] += 1
        result['cumulative_completed_jobs'] = dict(self.counts)
        atomic(self.args.out/'jobs'/f'{token:04d}_{tag}.json', result)
        return result, gradient

    def close(self):
        for jobs in self.jobs:
            jobs.put(None)
        for process in self.workers:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.monitor.sample('encoded_native_workers_closed')
