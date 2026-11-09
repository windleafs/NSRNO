"""A single low-frequency native experiment, shot-parallel across its GPU workers."""
from pathlib import Path
import multiprocessing as mp
import os
import queue
import time
import numpy as np
from .low_frequency_native_worker import worker
from .low_frequency_0p5 import atomic


class Pool:
    def __init__(self, args, monitor):
        self.args = args
        self.monitor = monitor
        self.token = 0
        context = mp.get_context('spawn')
        self.jobs = [context.Queue() for _ in args.devices]
        self.results = context.Queue()
        self.workers = []
        monitor.wait_budget(args.devices, args.cap_gib, 'before_single_low_frequency_experiment')
        for rank, device in enumerate(args.devices):
            p = context.Process(target=worker, args=(
                rank, device, self.jobs[rank], self.results, str(args.data),
                args.band_hz, args.chunk, args.cap_gib,
                str(args.out / 'workers' / f'rank{rank:02d}'), len(args.devices)))
            p.start()
            self.workers.append(p)
        atomic(args.out / 'worker_processes.json', {
            'coordinator_pid': os.getpid(), 'worker_pids': [p.pid for p in self.workers],
            'devices': args.devices, 'one_experiment_shot_parallelism': True})

    def evaluate(self, material, tag):
        self.token += 1
        token = self.token
        started = time.time()
        atomic(self.args.out / 'active_job.json', {
            'token': token, 'tag': tag, 'started_unix_s': started, 'truth_read': False})
        for jobs in self.jobs:
            jobs.put({'token': token, 'material': np.asarray(material, np.float32)})
        rows = []
        while len(rows) < len(self.workers):
            try:
                row = self.results.get(timeout=20)
            except queue.Empty:
                used = self.monitor.sample(tag)
                if any(v['owned_MiB'] > 16384 for v in used.values()):
                    raise RuntimeError('Observed per-card GPU process tree exceeds 16 GiB')
                dead = [(p.pid, p.exitcode) for p in self.workers if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f'Worker stopped at {tag}: {dead}')
                continue
            if row.get('error'):
                raise RuntimeError(f"Worker {row['rank']}, {tag}:\n{row['error']}")
            if row['token'] != token or any(r['rank'] == row['rank'] for r in rows):
                raise RuntimeError('Duplicate or unexpected native RF result')
            rows.append(row)
            self.monitor.sample(tag)
        gradient = sum((r['gradient'] for r in rows), np.zeros_like(material, dtype=np.float64))
        result = {
            'token': token, 'tag': tag, 'loss': sum(r['loss'] for r in rows),
            'seconds': time.time() - started, 'band_hz': list(self.args.band_hz),
            'peak_Torch_GiB': max(r['peak_Torch_GiB'] for r in rows),
            'peak_reserved_GiB': max(r['peak_reserved_GiB'] for r in rows),
            'truth_read': False, 'new_wave_solves': 22, 'new_material_gradient_shots': 22,
            'no_HVP_or_material_identifiability_probes': True, 'groups': {}}
        for family in ['lf', 'point3']:
            values = [r['groups'][family] for r in rows]
            complete = sorted(i for v in values for i in v['completed_shots'])
            if complete != list(range(11)):
                raise RuntimeError('Missing or repeated acquisition shot')
            energy = values[0]['energy']
            error = sum(v['error_squared'] for v in values)
            result['groups'][family] = {
                'relative_RMS': float(np.sqrt(error / energy)),
                'error_squared': error, 'energy': energy, 'completed_shots': complete}
        check = .25 * sum(v['error_squared'] / v['energy'] for v in result['groups'].values())
        if not np.isfinite(result['loss']) or not np.isfinite(gradient).all() or abs(check-result['loss']) > 1e-10:
            raise RuntimeError('Inconsistent or nonfinite native waveform objective')
        atomic(self.args.out / 'jobs' / f'{token:04d}_{tag}.json', result)
        return result, gradient

    def close(self):
        for jobs in self.jobs:
            jobs.put(None)
        for p in self.workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        self.monitor.sample('low_frequency_workers_closed')
