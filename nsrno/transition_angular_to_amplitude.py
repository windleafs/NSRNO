"""Stop our verified angular-QR trainer after its first 600-step EMA is saved.

No GPU is used. This helper never launches another model or replays work.
The next phase must retain the parent's absolute walltime/update budget.
"""
from pathlib import Path
import argparse
import ctypes
import platform
import re
import hashlib
import json
import os
import signal
import time
import traceback


def atomic(path, value):
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def open_process_handle(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    if platform.machine() != 'x86_64':
        raise RuntimeError('No verified pidfd ABI is available')
    header = Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
    match = re.search(r'^#define\s+__NR_pidfd_open\s+(\d+)\s*$', header, re.MULTILINE)
    if match is None:
        raise RuntimeError('Cannot read the local kernel pidfd_open syscall number')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(int(match.group(1))), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or (args.parent_root / 'phase_transition.json').exists():
        raise RuntimeError('Inspect the existing transition; never repeat a signal implicitly')
    parent = json.loads((args.parent_root / 'run.json').read_text())
    if parent['status'] != 'training_conditional_secant' or parent['profile_only']:
        raise RuntimeError('Require the live original angular-QR training')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if parent['boot_id'] != boot:
        raise RuntimeError('Parent is from a different boot')
    if not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('A process handle is required to signal the exact trainer')
    deadline = parent['created_unix_s'] + 7200

    def identity():
        p = Path('/proc', str(parent['pid']))
        try:
            stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            ticks = int(stat[19])
            cmd = (p / 'cmdline').read_bytes().replace(bytes([0]), b' ').decode()
        except (FileNotFoundError, ProcessLookupError):
            return False
        if stat[0] == 'Z' and ticks == parent['start_ticks']:
            return True
        if ticks != parent['start_ticks'] or ' -m nsrno.train_angular_qr_inverse ' not in cmd:
            raise RuntimeError('Parent process identity changed; do not signal it')
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != boot:
            raise RuntimeError('Boot changed')
        return True

    if not identity():
        raise RuntimeError('Parent trainer is not live')
    args.out.mkdir()
    state = {
        'status': 'waiting_for_parent_600_step_checkpoint',
        'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': boot, 'created_unix_s': time.time(),
        'parent_root': str(args.parent_root.resolve()),
        'parent_pid': parent['pid'], 'parent_start_ticks': parent['start_ticks'],
        'source_SHA256': digest(__file__), 'GPU_work': False,
        'original_shared_deadline_unix_s': deadline,
        'maximum_total_training_updates': 2400, 'checkpoint_step': 600,
        'reason': 'continue_with_amplitude_input_after_checkpoint',
        'profile_or_training_automatically_launched': False,
    }
    atomic(args.out / 'run.json', state)
    try:
        weights = args.parent_root / 'weights.pt'
        while True:
            if not identity():
                raise RuntimeError('Parent exited before the requested transition')
            current = json.loads((args.parent_root / 'run.json').read_text())
            if current['status'] != 'training_conditional_secant':
                raise RuntimeError('Parent became terminal before the requested transition')
            if time.time() >= deadline - 30:
                raise RuntimeError('No remaining original budget; do not start a new phase')
            progress_path = args.parent_root / 'progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            step = progress.get('step', 0)
            if weights.exists() and 600 <= step < 1200:
                checkpoint_hash = digest(weights)
                fd = open_process_handle(parent['pid'])
                try:
                    if not identity():
                        raise RuntimeError('Parent exited before the signal')
                    state.update(status='requesting_parent_interrupt',
                                 checkpoint_SHA256=checkpoint_hash,
                                 signal_name='SIGINT', signal_unix_s=time.time())
                    atomic(args.out / 'run.json', state)
                    signal.pidfd_send_signal(fd, signal.SIGINT, None, 0)
                finally:
                    os.close(fd)
                break
            if step >= 1200:
                raise RuntimeError('Missed the first fixed checkpoint; inspect, do not silently choose another')
            state.update(last_parent_step=step, last_observed_unix_s=time.time())
            atomic(args.out / 'run.json', state)
            time.sleep(2)
        requested = time.time()
        while identity():
            if time.time() - requested > 60:
                raise RuntimeError('Parent is still live after SIGINT; inspect the same process')
            time.sleep(1)
        stopped = json.loads((args.parent_root / 'run.json').read_text())
        if stopped['status'] != 'failed' or 'KeyboardInterrupt' not in stopped.get('error', ''):
            raise RuntimeError('Parent did not record the explicit interrupt')
        last = None
        for line in reversed((args.parent_root / 'training.jsonl').read_text().splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate.get('step'), int):
                last = candidate
                break
        if last is None or not 600 <= last['step'] < 1200:
            raise RuntimeError('Invalid last completed parent update')
        if digest(weights) != checkpoint_hash:
            raise RuntimeError('Saved EMA changed across the verified transition')
        transition = {
            'reason': 'continue_with_amplitude_input_after_checkpoint',
            'pid': parent['pid'], 'start_ticks': parent['start_ticks'], 'boot_id': boot,
            'parent_process_verified_gone': True, 'stopped_unix_s': time.time(),
            'checkpoint_step': 600, 'checkpoint_SHA256': checkpoint_hash,
            'actual_last_completed_step': last['step'],
            'completed_updates_after_checkpoint_not_in_warm_state': last['step'] - 600,
            'original_shared_deadline_unix_s': deadline,
            'original_shared_maximum_updates': 2400,
            'parent_final_run_SHA256': digest(args.parent_root / 'run.json'),
            'optimizer_and_sampler_states_available': False,
            'not_a_completed_standalone_angular_QR_experiment': True,
            'not_an_imaging_failure_conclusion': True,
        }
        atomic(args.parent_root / 'phase_transition.json', transition)
        state.update(status='parent_stopped_for_amplitude_continuation',
                     finished_unix_s=time.time(), transition=transition)
        atomic(args.out / 'run.json', state)
        print(json.dumps(state), flush=True)
    except BaseException:
        state.update(status='transition_needs_inspection', error=traceback.format_exc(),
                     observed_unix_s=time.time())
        atomic(args.out / 'run.json', state)
        raise


if __name__ == '__main__':
    main()
