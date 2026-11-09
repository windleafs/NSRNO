"""Budgeted full-dual inverse jobs with separately measured native work."""
from pathlib import Path
import multiprocessing as mp
import json
import os
import queue
import time
import numpy as np
from .full_source_dual_worker import worker
from .low_frequency_0p5 import atomic


def actual_work(out):
    names = ['issued_adjoint_direction_passes', 'completed_adjoint_direction_passes',
        'issued_physical_forward_passes', 'completed_physical_forward_passes',
        'issued_auxiliary_forward_passes', 'completed_auxiliary_forward_passes']
    total = {k: 0 for k in names}
    total['unfinished_native_phases'] = 0
    for file in (out/'workers_full').glob('pool*/rank*/tasks/token*/*/native_work.json'):
        row = json.loads(file.read_text())
        for key in names:
            total[key] += row[key]
        total['unfinished_native_phases'] += row['inflight_phase'] is not None
    return total


class FullSourceDualPool:
    def __init__(self, args, monitor, plan, generation, ledger, calibration):
        self.args, self.monitor, self.plan = args, monitor, plan
        self.generation, self.ledger, self.calibration = generation, ledger, calibration
        context = mp.get_context('spawn')
        self.jobs = [context.Queue() for _ in args.devices]
        self.results, self.workers = context.Queue(), []
        monitor.wait_budget(args.devices, args.cap_gib, 'before_actual_full_source_dual_inverse')
        for rank, device in enumerate(args.devices):
            process = context.Process(target=worker, args=(rank, device, self.jobs[rank],
                self.results, str(args.out/'native_plan.json'), args.chunk, args.cap_gib,
                str(args.out/'workers_full'/f'pool{generation:02d}'/f'rank{rank:02d}'), len(args.devices)))
            process.start()
            self.workers.append(process)
        atomic(args.out/'worker_processes.json', {'coordinator_pid': os.getpid(), 'kind': 'full_source_dual',
            'generation': generation, 'worker_pids': [p.pid for p in self.workers],
            'devices': args.devices, 'one_GPU_experiment': True})

    def evaluate(self, material, sources, kind, tag, cached_RF=None):
        names = list(self.plan['families'])
        if set(sources) != set(names) or any(not 0 <= int(v) < 11 for v in sources.values()):
            raise ValueError('True source from each acquired eleven-source family required')
        if kind not in ['full_dual_gradient', 'full_dual_value']:
            raise ValueError('Actual full-space inverse jobs only')
        need_gradient = kind == 'full_dual_gradient'
        if not need_gradient and any(v.get('normalization') is None for v in self.calibration.values()):
            raise RuntimeError('Line value requires the same fully initialized objective')
        maximum_per_source = 2*self.args.max_actions+(4 if need_gradient else 2)
        charged = maximum_per_source*len(names)
        if self.ledger['charged_adjoint_direction_passes']+charged > self.args.new_work_units*4:
            raise RuntimeError('New full-dual budget exhausted; previous budget is not reset')
        self.ledger['token'] += 1
        self.ledger['charged_adjoint_direction_passes'] += charged
        self.ledger['issued_joint_jobs'] += 1
        token, started = self.ledger['token'], time.time()
        atomic(self.args.out/'full_dual_budget_ledger.json', self.ledger)
        atomic(self.args.out/'active_job.json', {'token': token, 'kind': kind, 'tag': tag,
            'started_unix_s': started, 'sources': sources, 'maximum_charged_adjoint_passes': charged,
            'charged_is_conservative_not_measured_actual_cost': True,
            'maximum_K_actions_per_source': self.args.max_actions,
            'inner_relative_numerical_gap': self.args.inner_gap, 'patient_truth_read': False})
        payload = np.ascontiguousarray(material, np.float32)
        if payload.shape != (3, 961, 1121) or not np.isfinite(payload).all():
            raise ValueError('Finite independent native material required')
        entries = [(name, int(sources[name])) for name in names]
        for rank, jobs in enumerate(self.jobs):
            jobs.put({'token': token, 'kind': kind, 'material': payload,
                'sources': entries[rank::len(self.jobs)], 'calibration': self.calibration,
                'cached_RF': cached_RF or {}, 'max_actions': self.args.max_actions,
                'relative_gap': self.args.inner_gap})
        rows = []
        while len(rows) < len(self.workers):
            try:
                row = self.results.get(timeout=20)
            except queue.Empty:
                values = self.monitor.sample(tag)
                atomic(self.args.out/'actual_native_work.json', actual_work(self.args.out))
                if any(v['owned_MiB'] > 16384 for v in values.values()):
                    raise RuntimeError('Owned per-card process tree exceeded 16 GiB')
                if any(p.exitcode is not None for p in self.workers):
                    raise RuntimeError('Same full-dual worker exited; inspect saved actual task')
                continue
            if row.get('error'):
                raise RuntimeError(f"Full-dual worker {row['rank']}:\n{row['error']}")
            if row['token'] != token or row['kind'] != kind or any(x['rank'] == row['rank'] for x in rows):
                raise RuntimeError('Wrong or duplicate full-dual result')
            rows.append(row)
            self.monitor.sample(tag)
        if sorted(tuple(x) for row in rows for x in row['sources']) != sorted(entries):
            raise RuntimeError('Incomplete true-source family processing')
        parameters = {name: value for row in rows for name, value in row['calibration'].items()}
        for name in names:
            if parameters[name]['lambda_squared'] != self.calibration[name]['lambda_squared']:
                raise RuntimeError('Frozen full-dual regularization changed')
            if self.calibration[name].get('normalization') is not None and parameters[name] != self.calibration[name]:
                raise RuntimeError('Full-dual objective normalization changed')
        self.calibration.update(parameters)
        atomic(self.args.out/'full_dual_calibration.json', self.calibration)
        ready = all(row['all_inner_ready'] for row in rows)
        result = {'token': token, 'kind': kind, 'tag': tag, 'all_inner_ready': ready,
            'loss': sum(row['loss'] for row in rows) if ready else None,
            'lower': sum(row['lower'] for row in rows) if ready else None,
            'upper': sum(row['upper'] for row in rows) if ready else None,
            'sources': sources, 'details': {name: info for row in rows for name, info in row['details'].items()},
            'seconds': time.time()-started, 'peak_Torch_GiB': max(row['peak_Torch_GiB'] for row in rows),
            'peak_reserved_GiB': max(row['peak_reserved_GiB'] for row in rows),
            'charged_budget': dict(self.ledger), 'actual_native_work': actual_work(self.args.out),
            'patient_truth_read': False, 'new_material_derivative_probes': 0}
        gradient = (sum((row['gradient'] for row in rows), np.zeros(payload.shape, np.float64))
                    if ready and need_gradient else None)
        if ready and (not np.isfinite(result['loss']) or
                      (need_gradient and not np.isfinite(gradient).all())):
            raise RuntimeError('Nonfinite full-dual objective or actual material gradient')
        self.ledger['completed_joint_jobs'] += 1
        self.ledger['converged_joint_jobs'] += int(ready)
        atomic(self.args.out/'full_dual_budget_ledger.json', self.ledger)
        atomic(self.args.out/'actual_native_work.json', result['actual_native_work'])
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
        self.monitor.sample('full_source_dual_workers_closed')
