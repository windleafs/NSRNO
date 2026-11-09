"""Resource accounting separates this run's process tree from shared GPUs."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
from .wri_worker import atomic


def process_tree(root):
    parents = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            parents[int(entry.name)] = int(stat[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    owned = {root}
    while True:
        expanded = owned | {pid for pid, parent in parents.items() if parent in owned}
        if expanded == owned:
            return owned
        owned = expanded


class PressureResources:
    def __init__(self, root):
        self.out = root
        self.log = open(root/'resources.jsonl', 'a', buffering=1)
        self.maximum_owned = {}
        self.maximum_whole = {}
        self.boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.owned_above_limit = False

    def sample(self, phase):
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used',
            '--format=csv,noheader,nounits'], text=True)
        devices = {}
        by_uuid = {}
        for line in rows.splitlines():
            i, uuid, total, used = [s.strip() for s in line.split(',')]
            devices[int(i)] = {'total_MiB': int(total), 'whole_MiB': int(used),
                               'owned_MiB': 0, 'other_MiB': 0}
            by_uuid[uuid] = int(i)
        owned = process_tree(os.getpid())
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
            '--format=csv,noheader,nounits'], text=True)
        for line in apps.splitlines():
            pid, uuid, used = [s.strip() for s in line.split(',')]
            if uuid not in by_uuid or not used.isdigit():
                continue
            key = 'owned_MiB' if int(pid) in owned else 'other_MiB'
            devices[by_uuid[uuid]][key] += int(used)
        for device, values in devices.items():
            self.maximum_owned[device] = max(self.maximum_owned.get(device, 0), values['owned_MiB'])
            self.maximum_whole[device] = max(self.maximum_whole.get(device, 0), values['whole_MiB'])
        info = Path('/proc/meminfo').read_text().splitlines()
        available = int(next(s.split()[1] for s in info if s.startswith('MemAvailable:')))/2**20
        self.owned_above_limit |= any(v['owned_MiB'] > 16384 for v in devices.values())
        row = {'unix_s': time.time(), 'phase': phase, 'pid': os.getpid(), 'boot_id': self.boot,
            'devices': devices, 'host_available_GiB': available,
            'disk_free_GiB': shutil.disk_usage(self.out).free/2**30,
            'owned_above_16_GiB_observed': self.owned_above_limit}
        self.log.write(json.dumps(row)+'\n')
        atomic(self.out/'resource_summary.json', {'sampled_peak_owned_MiB': self.maximum_owned,
            'sampled_peak_whole_card_MiB': self.maximum_whole,
            'owned_above_16_GiB_observed': self.owned_above_limit, 'latest': row,
            'limit_scope': 'all GPU processes in this coordinator process tree, per card',
            'sampled_not_continuous': True, 'sampling_interval_seconds': 20})
        return devices

    def wait_budget(self, devices, cap, phase):
        while True:
            used = self.sample(phase)
            if all(used[d]['owned_MiB']+cap*1024+768 <= 16384
                   and used[d]['whole_MiB']+cap*1024+768 <= used[d]['total_MiB'] for d in devices):
                return
            print(json.dumps({'waiting': 'GPU memory admission', 'phase': phase,
                'devices': used, 'worker_allocator_cap_GiB': cap}), flush=True)
            time.sleep(20)
