"""One owned pool with explicit state-replay, auxiliary and physical budgets."""
from pathlib import Path
import json
import multiprocessing as mp
import os
import queue
import time
import numpy as np
from .multi_equation_continuation_worker import worker
from .low_frequency_0p5 import atomic


class ContinuationPool:
    def __init__(self, args, monitor, deadline, freeze):
        self.args, self.monitor, self.deadline, self.freeze = args, monitor, deadline, freeze
        self.workers, self.dispatched = [], set()
        ctx = mp.get_context('spawn')
        self.queues = [ctx.Queue() for _ in args.devices]
        self.results = ctx.Queue()
        try:
            while True:
                freeze()
                usage = monitor.sample('four_channel_GPU_admission')
                if all(usage[d]['owned_MiB']+10240+768 <= 16384 and
                        usage[d]['whole_MiB']+10240+768 <= usage[d]['total_MiB'] for d in args.devices):
                    break
                time.sleep(20)
            for rank, device in enumerate(args.devices):
                process = ctx.Process(target=worker, args=(rank, device, self.queues[rank], self.results,
                    str(args.out/'native_plan.json'), str(args.out/'workers'/f'rank{rank:02d}'), deadline))
                process.start()
                self.workers.append(process)
            identities = []
            for process in self.workers:
                p = Path('/proc', str(process.pid), 'stat')
                identities.append({'pid': process.pid,
                    'start_ticks': int(p.read_text().rsplit(')', 1)[1].split()[19])})
            atomic(args.out/'worker_processes.json', {'coordinator_pid': os.getpid(),
                'worker_pids': [p.pid for p in self.workers], 'worker_identities': identities,
                'devices': args.devices, 'one_GPU_experiment': True, 'source_encoding': False})
            self.ready = self.collect('ready')
        except BaseException:
            self.close()
            raise

    def actual_work(self):
        total, inflight = {}, 0
        for rank in range(len(self.workers)):
            path = self.args.out/'workers'/f'rank{rank:02d}/native_work.json'
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            for key, value in row.items():
                if key.startswith(('issued_', 'completed_')):
                    total[key] = total.get(key, 0)+value
            inflight += row['inflight_phase'] is not None
        total['unfinished_native_phases'] = inflight
        if (total.get('issued_adjoint_direction_passes', 0) > 1144 or
                total.get('issued_auxiliary_forward_passes', 0) > 572 or
                total.get('issued_physical_forward_passes', 0) > 44 or
                total.get('issued_tangent_direction_passes', 0) != 0):
            raise RuntimeError('Declared1144state-adjoint/572auxiliary/44physical/0JVP budget exceeded')
        atomic(self.args.out/'actual_native_work.json', total)
        return total

    def collect(self, kind):
        rows = []
        while len(rows) < len(self.workers):
            try:
                row = self.results.get(timeout=20)
            except queue.Empty:
                self.freeze()
                usage = self.monitor.sample(kind)
                self.actual_work()
                if any(v['owned_MiB'] > 16384 for v in usage.values()):
                    raise RuntimeError('Owned process tree exceeded16GiB')
                if any(p.exitcode is not None for p in self.workers):
                    raise RuntimeError('Inspect original stopped worker;do not replay a partial solve')
                continue
            if row.get('error'):
                raise RuntimeError(row['error'])
            if row['kind'] != kind or any(r['rank'] == row['rank'] for r in rows):
                raise RuntimeError('Wrong or duplicate actual worker result')
            rows.append(row)
            self.monitor.sample(kind)
            self.actual_work()
        return sorted(rows, key=lambda r: r['rank'])

    def evaluate(self, kind, material):
        if kind not in ['conditional_statistics', 'physical_forward'] or kind in self.dispatched:
            raise RuntimeError('At most one all44 wavefield solve and one all44 physical candidate')
        self.freeze()
        self.dispatched.add(kind)
        atomic(self.args.out/'active_job.json', {'kind': kind, 'source_count': 44,
            'started_unix_s': time.time(), 'no_material_Jacobian_or_JVP': True})
        for channel in self.queues:
            channel.put({'kind': kind, 'material': np.ascontiguousarray(material, np.float32)})
        rows = self.collect(kind)
        entries = [tuple(v) for r in rows for v in r['sources']]
        if len(entries) != 44 or len(set(entries)) != 44:
            raise RuntimeError('A required original source is missing or duplicated')
        return rows

    def close(self):
        for channel in self.queues:
            channel.put(None)
        for process in self.workers:
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.actual_work()
        self.monitor.sample('four_channel_workers_closed')
