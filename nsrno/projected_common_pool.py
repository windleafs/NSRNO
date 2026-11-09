"""Three true all44 gradient points and bounded nonlinear candidate forwards."""
from pathlib import Path
import hashlib
import json
import multiprocessing as mp
import os
import queue
import time
import numpy as np
from .full_field_prior_worker import worker
from .low_frequency_0p5 import atomic, digest


class ProjectedCommonPool:
    def __init__(self, args, monitor, plan, deadline):
        self.args, self.monitor, self.plan = args, monitor, plan
        context = mp.get_context('spawn')
        self.queues = [context.Queue() for _ in args.devices]
        self.results = context.Queue()
        self.token, self.gradient_step, self.accepted_step = 0, 0, 0
        self.forward_count, self.step_forward_count = 0, 0
        self.last_gradient_material = None
        while True:
            if time.time() >= deadline-60 or (args.out/'stop_requested.flag').exists():
                raise RuntimeError('Absolute deadline or stop reached during GPU memory admission')
            devices = monitor.sample('before_projected_common_descent')
            if all(devices[d]['owned_MiB']+args.cap_gib*1024+768 <= 16384 and
                   devices[d]['whole_MiB']+args.cap_gib*1024+768 <= devices[d]['total_MiB']
                   for d in args.devices):
                break
            print(json.dumps({'waiting_for_actual_GPU_memory': devices}), flush=True)
            time.sleep(20)
        self.workers = []
        for rank, device in enumerate(args.devices):
            process = context.Process(target=worker, args=(rank, device, self.queues[rank], self.results,
                str(args.out/'native_plan.json'), str(args.out/'workers'/f'rank{rank:02d}'),
                args.chunk, args.cap_gib, deadline))
            process.start()
            self.workers.append(process)
        atomic(args.out/'worker_processes.json', {'coordinator_pid': os.getpid(),
            'worker_pids': [p.pid for p in self.workers], 'devices': args.devices,
            'one_GPU_experiment': True, 'source_encoding': False})

    def actual_work(self):
        totals, inflight = {}, 0
        for rank in range(len(self.workers)):
            path = self.args.out/'workers'/f'rank{rank:02d}'/'native_work.json'
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            for key, value in row.items():
                if key.startswith('issued_') or key.startswith('completed_'):
                    totals[key] = totals.get(key, 0)+value
            inflight += row['inflight_phase'] is not None
        totals['unfinished_native_phases'] = inflight
        if (totals.get('issued_adjoint_direction_passes', 0) > 132 or
            totals.get('issued_physical_forward_passes', 0) > 660 or
            totals.get('issued_tangent_direction_passes', 0) != 0):
            raise RuntimeError('Declared projected-descent work ceiling exceeded')
        atomic(self.args.out/'actual_native_work.json', totals)
        return totals

    def mark_accepted(self, step):
        if step != self.gradient_step or self.step_forward_count < 1:
            raise RuntimeError('Only a completed actual candidate can advance the trajectory')
        self.accepted_step = step

    def evaluate(self, material, kind, step):
        material = np.ascontiguousarray(material, np.float32)
        if kind == 'original44_gradient':
            signature = hashlib.sha256(material.tobytes()).hexdigest()
            if (step != self.gradient_step+1 or step > 3 or self.accepted_step != step-1 or
                signature == self.last_gradient_material):
                raise RuntimeError('Do not repeat a completed gradient or advance without an accepted step')
            self.gradient_step, self.step_forward_count = step, 0
            self.last_gradient_material = signature
        elif kind == 'original44_forward':
            if step != self.gradient_step or self.step_forward_count >= 4 or self.forward_count >= 12:
                raise RuntimeError('Declared actual RF line budget exhausted')
            self.step_forward_count += 1
            self.forward_count += 1
        else:
            raise ValueError('This actual optimizer never requests tangent or diagnostic solves')
        self.token += 1
        entries = [(name, shot) for name in self.plan['families'] for shot in range(11)]
        atomic(self.args.out/'active_job.json', {'token': self.token, 'kind': kind, 'outer_step': step,
            'started_unix_s': time.time(), 'all44_sources': True, 'source_encoding': False})
        for rank, channel in enumerate(self.queues):
            channel.put({'token': self.token, 'kind': kind, 'material': material,
                         'sources': entries[rank::len(self.queues)], 'directions': None})
        rows, started = [], time.time()
        while len(rows) < len(self.workers):
            try:
                row = self.results.get(timeout=20)
            except queue.Empty:
                values = self.monitor.sample(kind)
                self.actual_work()
                if any(v['owned_MiB'] > 16384 for v in values.values()):
                    raise RuntimeError('Owned per-card process tree exceeded 16 GiB')
                if any(p.exitcode is not None for p in self.workers):
                    raise RuntimeError('Inspect exited worker and partial products; do not repeat work')
                continue
            if row.get('error'):
                raise RuntimeError(row['error'])
            if row['token'] != self.token or row['kind'] != kind or any(r['rank'] == row['rank'] for r in rows):
                raise RuntimeError('Wrong or duplicate actual inverse result')
            rows.append(row)
            self.monitor.sample(kind)
        rows.sort(key=lambda r: r['rank'])
        if sorted(tuple(x) for row in rows for x in row['sources']) != sorted(entries):
            raise RuntimeError('Incomplete original44 source set')
        result = {'token': self.token, 'kind': kind, 'outer_step': step,
            'loss': sum(r['loss'] for r in rows), 'seconds': time.time()-started,
            'actual_work': self.actual_work(), 'groups': {}, 'all44_sources': True,
            'source_encoding': False, 'patient_truth_read': False,
            'products_used_in_actual_inverse_not_validation': True}
        for family in self.plan['families']:
            groups = [r['groups'][family] for r in rows]
            if sorted(s for g in groups for s in g['shots']) != list(range(11)):
                raise RuntimeError('An acquisition source is missing')
            result['groups'][family] = {'loss': sum(g['loss'] for g in groups), 'source_count': 11}
        if kind == 'original44_gradient':
            aggregate = np.zeros((3, 961, 1121), np.float64)
            for rank in range(len(self.workers)):
                receipt = json.loads((self.args.out/'workers'/f'rank{rank:02d}'/
                    f'gradient_receipt_token{self.token:04d}.json').read_text())
                if digest(receipt['path']) != receipt['SHA256']:
                    raise RuntimeError('Completed gradient receipt changed')
                aggregate += np.load(receipt['path'])
            if not np.isfinite(aggregate).all():
                raise RuntimeError('Nonfinite complete physical gradient')
            path = self.args.out/'arrays'/f'data_control_gradient_token{self.token:04d}.npy'
            if path.exists():
                raise RuntimeError('Preserve completed gradient instead of overwriting it')
            np.save(path, aggregate)
            result.update(gradient_path=str(path.resolve()), gradient_SHA256=digest(path),
                          material_array_SHA256=self.last_gradient_material)
        atomic(self.args.out/'jobs'/f'{self.token:04d}_{kind}.json', result)
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
        self.monitor.sample('projected_common_workers_closed')
