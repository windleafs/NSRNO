"""One native worker pool: RF evaluations and products used by actual GN steps."""
from pathlib import Path
import multiprocessing as mp
import os
import queue
import time
import numpy as np
from .encoded_physical_native_pool import EncodedPhysicalNativePool
from .neural_subspace_native_worker import worker
from .low_frequency_0p5 import atomic


class NeuralSubspaceNativePool(EncodedPhysicalNativePool):
    def __init__(self, args, monitor, plan):
        self.args, self.monitor, self.plan = args, monitor, plan
        self.token = 0
        self.counts = {'encoded_gradient_jobs': 0, 'encoded_adjoint_propagations': 0,
            'encoded_forward_jobs': 0, 'encoded_forward_propagations': 0,
            'original_full_RF_checks': 0, 'original_forward_propagations': 0,
            'cached_full_RF_evaluations': 0, 'encoded_hvp_jobs': 0,
            'encoded_hvp_background_forward_propagations': 0,
            'encoded_tangent_propagations': 0, 'encoded_hvp_adjoint_propagations': 0}
        self.attempted = {'encoded_adjoint_propagations_issued': 0,
            'encoded_forward_propagations_issued': 0, 'original_forward_propagations_issued': 0,
            'encoded_hvp_background_forward_propagations_issued': 0,
            'encoded_tangent_propagations_issued': 0, 'encoded_hvp_adjoint_propagations_issued': 0}
        context = mp.get_context('spawn')
        self.jobs = [context.Queue() for _ in args.devices]
        self.results = context.Queue()
        self.workers = []
        monitor.wait_budget(args.devices, args.cap_gib, 'before_neural_subspace_native_inverse')
        for rank, device in enumerate(args.devices):
            process = context.Process(target=worker, args=(rank, device, self.jobs[rank],
                self.results, str(args.out/'native_plan.json'), args.chunk, args.cap_gib,
                str(args.out/'workers'/f'rank{rank:02d}')))
            process.start()
            self.workers.append(process)
        atomic(args.out/'worker_processes.json', {'coordinator_pid': os.getpid(),
            'worker_pids': [p.pid for p in self.workers], 'devices': args.devices,
            'one_GPU_experiment': True})

    def hvp(self, material, direction, encodings, tag):
        if set(encodings) != set(self.plan['families']):
            raise ValueError('All four acquired source families required')
        entries = []
        for family in self.plan['families']:
            code = np.asarray(encodings[family])
            if code.shape != (11,) or not np.all((code == -1) | (code == 1)):
                raise ValueError('All eleven unchanged source encoding signs required')
            entries.append((family, code.astype(int).tolist()))
        material, direction = (np.ascontiguousarray(v, np.float32) for v in [material, direction])
        if (material.shape != (3, 961, 1121) or direction.shape != material.shape or
            not np.isfinite(material).all() or not np.isfinite(direction).all()):
            raise ValueError('Finite full native inverse material and control direction required')
        self.token += 1
        token, started = self.token, time.time()
        atomic(self.args.out/'active_job.json', {'token': token, 'kind': 'encoded_hvp',
            'tag': tag, 'started_unix_s': started, 'source_background_forward_propagations': 4,
            'source_tangent_propagations': 4, 'source_adjoint_propagations': 4,
            'used_to_solve_actual_neural_update': True, 'patient_truth_read': False})
        for key in ['encoded_hvp_background_forward_propagations_issued',
                    'encoded_tangent_propagations_issued', 'encoded_hvp_adjoint_propagations_issued']:
            self.attempted[key] += 4
        atomic(self.args.out/'issued_propagations.json', {'token': token, **self.attempted})
        for rank, jobs in enumerate(self.jobs):
            jobs.put({'token': token, 'kind': 'encoded_hvp', 'material': material,
                'direction': direction, 'encodings': entries[rank::len(self.jobs)],
                'sources': [], 'save_RF': False})
        rows = []
        while len(rows) < len(self.workers):
            try:
                row = self.results.get(timeout=20)
            except queue.Empty:
                values = self.monitor.sample(tag)
                if any(v['owned_MiB'] > 16384 for v in values.values()):
                    raise RuntimeError('Owned per-card process memory exceeded 16 GiB')
                if any(p.exitcode is not None for p in self.workers):
                    raise RuntimeError('Same native subspace worker exited; inspect error')
                continue
            if row.get('error'):
                raise RuntimeError(f"Native subspace worker {row['rank']}:\n{row['error']}")
            if row['token'] != token or row['kind'] != 'encoded_hvp' or any(
                    r['rank'] == row['rank'] for r in rows):
                raise RuntimeError('Wrong or duplicate actual inverse product')
            rows.append(row)
            self.monitor.sample(tag)
        rows.sort(key=lambda row: row['rank'])
        product = sum((r['gradient'] for r in rows), np.zeros(material.shape, np.float64))
        quadratic = 2*sum(r['loss'] for r in rows)
        check = 0.
        for family, code in entries:
            groups = [r['groups'][family] for r in rows]
            completed = [c for g in groups for c in g['completed_encodings']]
            if completed != [code]:
                raise RuntimeError('Incomplete native update product source family')
            error = sum((np.asarray(g['band_error_squared']) for g in groups),
                        np.zeros(4, np.float64))
            check += 2*self.plan['families'][family]['coefficient']*float(np.mean(
                error/np.asarray(groups[0]['band_energy'])))
        dot = float(np.sum(direction.astype(np.float64)*product))
        error = abs(dot-quadratic)/max(abs(dot), abs(quadratic), 1e-20)
        if (not np.isfinite(product).all() or not np.isfinite([dot, quadratic, check]).all()
                or abs(check-quadratic) > 1e-9*max(1., abs(check)) or error > 5e-3):
            raise RuntimeError(f'Inconsistent actual inverse GN product: {dot}, {quadratic}, {error}')
        self.counts['encoded_hvp_jobs'] += 1
        for key in ['encoded_hvp_background_forward_propagations',
                    'encoded_tangent_propagations', 'encoded_hvp_adjoint_propagations']:
            self.counts[key] += 4
        result = {'token': token, 'kind': 'encoded_hvp', 'tag': tag,
            'seconds': time.time()-started, 'quadratic_2JstarWJ': quadratic,
            'direction_dot_product': dot, 'same_product_relative_identity_error': error,
            'peak_Torch_GiB': max(r['peak_Torch_GiB'] for r in rows),
            'peak_reserved_GiB': max(r['peak_reserved_GiB'] for r in rows),
            'encoding': encodings, 'patient_truth_read': False,
            'used_to_solve_actual_neural_update': True,
            'not_an_RF_objective_evaluation': True, 'cumulative_completed_jobs': dict(self.counts)}
        atomic(self.args.out/'jobs'/f'{token:04d}_{tag}.json', result)
        return result, product
