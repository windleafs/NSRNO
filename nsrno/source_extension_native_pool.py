"""Bounded source-extension jobs, retaining issued and completed work separately."""
from pathlib import Path
import multiprocessing as mp
import os
import queue
import time
import numpy as np
from .source_extension_native_worker import worker
from .low_frequency_0p5 import atomic


class SourceExtensionNativePool:
    def __init__(self, args, monitor, plan, generation, ledger, calibration):
        self.args, self.monitor, self.plan = args, monitor, plan
        self.generation, self.ledger, self.calibration = generation, ledger, calibration
        context = mp.get_context('spawn')
        self.jobs = [context.Queue() for _ in args.devices]
        self.results, self.workers = context.Queue(), []
        monitor.wait_budget(args.devices, args.cap_gib, 'before_source_extension_actual_inverse')
        for rank, device in enumerate(args.devices):
            process = context.Process(target=worker, args=(rank, device, self.jobs[rank],
                self.results, str(args.out/'native_plan.json'), args.chunk, args.cap_gib,
                str(args.out/'workers_wri'/f'pool{generation:02d}'/f'rank{rank:02d}'), len(args.devices)))
            process.start()
            self.workers.append(process)
        atomic(args.out/'worker_processes.json', {'coordinator_pid': os.getpid(),
            'generation': generation, 'kind': 'source_extension',
            'worker_pids': [p.pid for p in self.workers], 'devices': args.devices,
            'one_GPU_experiment': True})

    def evaluate(self, material, sources, kind, tag):
        names = list(self.plan['families'])
        if set(sources) != set(names) or any(not 0 <= int(v) < 11 for v in sources.values()):
            raise ValueError('One true source from each complete eleven-source family is required')
        if kind not in ['ray_gradient', 'ray_value']:
            raise ValueError('Actual inverse work only')
        if kind == 'ray_value' and set(self.calibration) != set(names):
            raise RuntimeError('A line value must retain all frozen family calibrations')
        need_gradient = kind == 'ray_gradient'
        self.ledger['token'] += 1
        token, started = self.ledger['token'], time.time()
        payload = np.ascontiguousarray(material, np.float32)
        if payload.shape != (3, 961, 1121) or not np.isfinite(payload).all():
            raise ValueError('Independent full native material required')
        charge = (5 if need_gradient else 1)*len(names)
        self.ledger['issued_adjoint_direction_passes'] += charge
        self.ledger['issued_true_source_contributions'] += len(names)
        atomic(self.args.out/'issued_source_extension_work.json', self.ledger)
        atomic(self.args.out/'active_job.json', {'token': token, 'kind': kind, 'tag': tag,
            'sources': sources, 'started_unix_s': started,
            'charged_adjoint_direction_passes': charge, 'source_encoding': False,
            'checkpoint_replays_included_in_charge': True, 'patient_truth_read': False})
        entries = [(name, int(sources[name])) for name in names]
        for rank, jobs in enumerate(self.jobs):
            jobs.put({'token': token, 'kind': kind, 'sources': entries[rank::len(self.jobs)],
                'material': payload, 'calibration': self.calibration})
        rows = []
        while len(rows) < len(self.workers):
            try:
                row = self.results.get(timeout=20)
            except queue.Empty:
                values = self.monitor.sample(tag)
                if any(v['owned_MiB'] > 16384 for v in values.values()):
                    raise RuntimeError('Owned per-card memory exceeded 16 GiB')
                if any(p.exitcode is not None for p in self.workers):
                    raise RuntimeError('Same source-extension worker exited; inspect its saved source outputs')
                continue
            if row.get('error'):
                raise RuntimeError(f"Source-extension worker {row['rank']}:\n{row['error']}")
            if row['token'] != token or row['kind'] != kind or any(x['rank'] == row['rank'] for x in rows):
                raise RuntimeError('Wrong or duplicate actual inverse result')
            rows.append(row)
            self.monitor.sample(tag)
        completed = [tuple(entry) for row in rows for entry in row['sources']]
        if sorted(completed) != sorted(entries):
            raise RuntimeError('Incomplete true-source contributions')
        new_calibration = {name: value for row in rows for name, value in row['calibration'].items()}
        if set(new_calibration) != set(names) or any(
                self.calibration[name] != new_calibration[name] for name in self.calibration):
            raise RuntimeError('Actual line objective changed its frozen normalization')
        self.calibration.update(new_calibration)
        atomic(self.args.out/'ray_calibration.json', self.calibration)
        loss = sum(row['loss'] for row in rows)
        gradient = (sum((row['gradient'] for row in rows), np.zeros(payload.shape, np.float64))
                    if need_gradient else None)
        if not np.isfinite(loss) or (need_gradient and not np.isfinite(gradient).all()):
            raise RuntimeError('Nonfinite actual source-extension inverse result')
        self.ledger['completed_adjoint_direction_passes'] += charge
        self.ledger['completed_true_source_contributions'] += len(names)
        self.ledger['completed_gradient_jobs' if need_gradient else 'completed_line_value_jobs'] += 1
        atomic(self.args.out/'issued_source_extension_work.json', self.ledger)
        result = {'token': token, 'kind': kind, 'tag': tag, 'loss': loss,
            'seconds': time.time()-started, 'sources': sources,
            'details': {name: detail for row in rows for name, detail in row['details'].items()},
            'peak_Torch_GiB': max(row['peak_Torch_GiB'] for row in rows),
            'peak_reserved_GiB': max(row['peak_reserved_GiB'] for row in rows),
            'cumulative_work_ledger': dict(self.ledger),
            'patient_truth_read': False, 'source_encoding': False,
            'full_original44_RF_is_a_separate_acceptance_measurement': True}
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
        self.monitor.sample('source_extension_workers_closed')
