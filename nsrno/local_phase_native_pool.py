"""Uniform original-source minibatches; nonlinear phase is never source-encoded."""
from pathlib import Path
import multiprocessing as mp
import os
import queue
import time
import numpy as np
from .local_phase_native_worker import worker
from .low_frequency_0p5 import atomic


class LocalPhaseNativePool:
    def __init__(self, args, monitor, plan):
        self.args, self.monitor, self.plan = args, monitor, plan
        self.normalization = None
        self.token = 0
        self.counts = {'gradient_jobs':0, 'native_adjoint_source_shots':0,
            'sample_forward_jobs':0, 'sample_forward_source_shots':0,
            'original_full_checks':0, 'original_full_forward_source_shots':0,
            'cached_full_evaluations':0}
        self.attempted = {'native_adjoint_source_shots_issued':0,
            'sample_forward_source_shots_issued':0, 'original_full_forward_source_shots_issued':0}
        context = mp.get_context('spawn')
        self.jobs = [context.Queue() for _ in args.devices]
        self.results = context.Queue()
        self.workers = []
        monitor.wait_budget(args.devices, args.cap_gib, 'before_local_phase_actual_inverse')
        for rank, device in enumerate(args.devices):
            process = context.Process(target=worker, args=(rank, device, self.jobs[rank],
                self.results, str(args.out/'native_plan.json'), args.chunk, args.cap_gib,
                str(args.out/'workers'/f'rank{rank:02d}')))
            process.start()
            self.workers.append(process)
        atomic(args.out/'worker_processes.json', {'coordinator_pid':os.getpid(),
            'worker_pids':[p.pid for p in self.workers], 'devices':args.devices,
            'one_GPU_experiment':True})

    def evaluate(self, material, kind, tag, samples=None, save_RF=False):
        sampled = kind in ['gradient', 'sample_forward']
        if not sampled and kind not in ['forward', 'cached_forward']:
            raise ValueError('Actual inverse jobs only')
        if sampled:
            if (samples is None or set(samples)!=set(self.plan['families']) or
                any(not isinstance(s, (int, np.integer)) or not 0<=s<11 for s in samples.values())):
                raise ValueError('One uniformly sampled actual source in each of four families')
            entries = [(name, int(samples[name])) for name in self.plan['families']]
        else:
            entries = [(family, shot) for family in self.plan['families'] for shot in range(11)]
        if save_RF and kind!='forward':
            raise ValueError('Persist only complete original-source forward RF')
        self.token += 1
        token, started = self.token, time.time()
        payload = None if kind=='cached_forward' else np.ascontiguousarray(material, np.float32)
        if payload is not None and payload.shape!=(3, 961, 1121):
            raise ValueError('Full native independent fields required')
        atomic(self.args.out/'active_job.json', {'token':token, 'kind':kind, 'tag':tag,
            'actual_acoustic_propagations':0 if kind=='cached_forward' else len(entries),
            'original_RF_components':44, 'sampled_shots':samples,
            'sample_importance_factor':11 if sampled else 1,
            'started_unix_s':started, 'truth_read':False})
        counter = ('native_adjoint_source_shots_issued' if kind=='gradient' else
            'sample_forward_source_shots_issued' if kind=='sample_forward' else
            'original_full_forward_source_shots_issued' if kind=='forward' else None)
        if counter is not None:
            self.attempted[counter] += len(entries)
        atomic(self.args.out/'issued_propagations.json', {'token':token, **self.attempted,
            'worker_completed_counters_retained_separately':True})
        for rank, jobs in enumerate(self.jobs):
            jobs.put({'token':token, 'kind':kind, 'material':payload,
                'sources':entries[rank::len(self.jobs)], 'save_RF':save_RF,
                'normalization':self.normalization})
        rows = []
        while len(rows)<len(self.workers):
            try:
                row = self.results.get(timeout=20)
            except queue.Empty:
                values = self.monitor.sample(tag)
                if any(v['owned_MiB']>16384 for v in values.values()):
                    raise RuntimeError('Owned memory exceeded 16 GiB per card')
                if any(p.exitcode is not None for p in self.workers):
                    raise RuntimeError('Native worker exited; inspect the same run')
                continue
            if row.get('error'):
                raise RuntimeError(f"Worker {row['rank']}:\n{row['error']}")
            if row['token']!=token or row['kind']!=kind or any(r['rank']==row['rank'] for r in rows):
                raise RuntimeError('Unexpected or duplicate worker result')
            rows.append(row)
            self.monitor.sample(tag)
        rows.sort(key=lambda row:row['rank'])
        result = {'token':token, 'kind':kind, 'tag':tag, 'RF':sum(r['RF'] for r in rows),
            'phase':sum(r['phase'] for r in rows), 'seconds':time.time()-started,
            'original_all44_objective_measured':not sampled, 'sampled_shots':samples,
            'sample_importance_factor':11 if sampled else 1, 'groups':{},
            'actual_acoustic_propagations':0 if kind=='cached_forward' else len(entries),
            'patient_truth_read':False, 'peak_Torch_GiB':max(r['peak_Torch_GiB'] for r in rows),
            'peak_reserved_GiB':max(r['peak_reserved_GiB'] for r in rows)}
        check = 0.
        for name, spec in self.plan['families'].items():
            groups = [r['groups'][name] for r in rows]
            energy = np.asarray(groups[0]['band_energy'])
            error = sum((np.asarray(g['band_error_squared']) for g in groups), np.zeros_like(energy))
            check += spec['coefficient']*float(np.mean(error/energy))
            completed = sorted(s for g in groups for s in g['completed_shots'])
            if completed!=([int(samples[name])] if sampled else list(range(11))):
                raise RuntimeError('Missing or mismatched actual source')
            result['groups'][name] = {'band_error_squared':error.tolist(),
                'band_energy':energy.tolist(), 'coefficient':spec['coefficient'],
                'phase_contribution':sum(g['phase_contribution'] for g in groups)}
        if (not np.isfinite([result['RF'], result['phase']]).all() or
            abs(check-result['RF'])>1e-10*max(1., abs(check))):
            raise RuntimeError('Inconsistent actual RF objective')
        result['loss'] = (None if self.normalization is None else
            .2*result['RF']/self.normalization['RF']+.8*result['phase']/self.normalization['phase'])
        gradient = None
        if kind=='gradient':
            gradient = sum((r['gradient'] for r in rows), np.zeros((3, 961, 1121), np.float64))
            if not np.isfinite(gradient).all():
                raise RuntimeError('Nonfinite actual inverse gradient')
            self.counts['gradient_jobs'] += 1
            self.counts['native_adjoint_source_shots'] += len(entries)
        elif kind=='sample_forward':
            self.counts['sample_forward_jobs'] += 1
            self.counts['sample_forward_source_shots'] += len(entries)
        elif kind=='forward':
            self.counts['original_full_checks'] += 1
            self.counts['original_full_forward_source_shots'] += len(entries)
        else:
            self.counts['cached_full_evaluations'] += 1
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
        self.monitor.sample('local_phase_native_workers_closed')
