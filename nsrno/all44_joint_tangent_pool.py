"""One exact 44-source tangent pack and at most one full physical candidate."""
from pathlib import Path
import json
import multiprocessing as mp
import os
import queue
import time
import numpy as np
from .all44_joint_tangent_worker import worker
from .low_frequency_0p5 import atomic


class All44JointPool:
    def __init__(self, args, monitor, plan, deadline):
        self.args, self.monitor, self.plan = args, monitor, plan
        context = mp.get_context('spawn')
        self.queues = [context.Queue() for _ in args.devices]
        self.results = context.Queue()
        self.workers = []
        self.token = 0
        monitor.wait_budget(args.devices, args.cap_gib, 'before_all44_actual_joint_update')
        for rank, device in enumerate(args.devices):
            p = context.Process(target=worker, args=(rank, device, self.queues[rank], self.results,
                str(args.out/'native_plan.json'), str(args.out/'workers'/f'rank{rank:02d}'),
                args.chunk, args.cap_gib, deadline))
            p.start()
            self.workers.append(p)
        atomic(args.out/'worker_processes.json', {'coordinator_pid': os.getpid(),
            'worker_pids': [p.pid for p in self.workers], 'devices': args.devices,
            'one_GPU_experiment': True, 'source_encoding': False})

    def actual_work(self):
        totals = {}
        inflight = 0
        for rank in range(len(self.workers)):
            path = self.args.out/'workers'/f'rank{rank:02d}'/'native_work.json'
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            for k, v in row.items():
                if k.startswith('issued_') or k.startswith('completed_'):
                    totals[k] = totals.get(k, 0)+v
            inflight += row['inflight_phase'] is not None
        totals['unfinished_native_phases'] = inflight
        atomic(self.args.out/'actual_native_work.json', totals)
        return totals

    def evaluate(self, material, kind, directions_path):
        if self.token == 0 and kind != 'joint_tangents' or self.token == 1 and kind != 'original44_forward':
            raise RuntimeError('Do not repeat the actual inverse products or physical candidate')
        if self.token >= 2:
            raise RuntimeError('Inherited one-candidate remaining RF budget exhausted')
        self.token += 1
        entries = [(name, shot) for name in self.plan['families'] for shot in range(11)]
        atomic(self.args.out/'active_job.json', {'token': self.token, 'kind': kind,
            'started_unix_s': time.time(), 'all44_sources': True, 'source_encoding': False})
        for rank, channel in enumerate(self.queues):
            channel.put({'token': self.token, 'kind': kind, 'material': np.ascontiguousarray(material, np.float32),
                'sources': entries[rank::len(self.queues)], 'directions': str(directions_path)})
        rows = []
        started = time.time()
        while len(rows) < len(self.workers):
            try:
                row = self.results.get(timeout=20)
            except queue.Empty:
                values = self.monitor.sample(kind)
                self.actual_work()
                if any(x['owned_MiB'] > 16384 for x in values.values()):
                    raise RuntimeError('Owned per-card process tree exceeded 16 GiB')
                if any(p.exitcode is not None for p in self.workers):
                    raise RuntimeError('Worker exited; inspect partial products without automatically repeating them')
                continue
            if row.get('error'):
                raise RuntimeError(row['error'])
            if row['token'] != self.token or row['kind'] != kind or any(r['rank'] == row['rank'] for r in rows):
                raise RuntimeError('Wrong or duplicate result')
            rows.append(row)
            self.monitor.sample(kind)
        rows.sort(key=lambda x: x['rank'])
        if sorted(tuple(x) for r in rows for x in r['sources']) != sorted(entries):
            raise RuntimeError('The exact 44-source inverse task is incomplete')
        result = {'token': self.token, 'kind': kind, 'loss': sum(r['loss'] for r in rows),
            'H': sum((r['H'] for r in rows)).tolist(),
            'linear': sum((r['linear'] for r in rows)).tolist(),
            'groups': {}, 'seconds': time.time()-started, 'actual_work': self.actual_work(),
            'peak_allocated_GiB': max(r['peak_allocated_GiB'] for r in rows),
            'peak_reserved_GiB': max(r['peak_reserved_GiB'] for r in rows),
            'all44_sources': True, 'source_encoding': False, 'patient_truth_read': False,
            'products_used_in_actual_joint_update_not_derivative_probes': True}
        for name in self.plan['families']:
            groups = [r['groups'][name] for r in rows]
            shots = sorted(s for g in groups for s in g['shots'])
            if shots != list(range(11)):
                raise RuntimeError('Missing acquisition angle or point source')
            result['groups'][name] = {'loss': sum(g['loss'] for g in groups),
                'H': sum((np.asarray(g['H']) for g in groups)).tolist(),
                'linear': sum((np.asarray(g['linear']) for g in groups)).tolist(), 'source_count': 11}
        atomic(self.args.out/'jobs'/f'{self.token:04d}_{kind}.json', result)
        return result

    def close(self):
        for q in self.queues:
            q.put(None)
        for p in self.workers:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        self.actual_work()
        self.monitor.sample('all44_joint_workers_closed')
