"""Preserve the original campaign's original-source check limit for one own PID."""
from pathlib import Path
import argparse
import json
import os
import signal
import time


def atomic(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--ticks', type=int, required=True)
    parser.add_argument('--boot', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    project = Path('/home_nvme/zhuangyang/nsrno')
    if root != project/'out/v5/reconstruction_neural_subspace_native_v1':
        raise ValueError('Only the current explicitly owned neural continuation')
    run = json.loads((root/'run.json').read_text())
    if (run['pid'] != args.pid or run['start_ticks'] != args.ticks or run['boot_id'] != args.boot
            or run['source_SHA256'].get('reconstruct_neural_subspace_native.py') !=
            '36163dab78ebceac91250cb9c6c67176fffb1786e54d481e8fb9f7c3f27e87d1'):
        raise RuntimeError('Expected inverse identity/source does not match')
    parent = json.loads((Path(run['parent_root'])/'run.json').read_text())
    old_checks = parent['original_forward_propagations']//44
    maximum = parent['maximum_original_full44_checks']-old_checks
    if parent['original_forward_propagations'] % 44 or maximum != 7:
        raise RuntimeError('Inspect changed original-source budget before controlling this PID')
    status = Path(run['artifact_root'])/'original44_budget_guard.json'
    if status.exists():
        raise RuntimeError('An existing guard receipt must be inspected, never duplicated')
    info = {'guard_pid': os.getpid(), 'guard_created_unix_s': time.time(),
        'target_pid': args.pid, 'target_start_ticks': args.ticks, 'target_boot_id': args.boot,
        'original_campaign_maximum_full44_checks': parent['maximum_original_full44_checks'],
        'old_completed_full44_checks': old_checks, 'maximum_new_full44_checks': maximum,
        'does_not_generate_data_or_use_GPU': True, 'signal_sent': False,
        'status': 'watching_specific_owned_inverse'}
    atomic(status, info)
    print(json.dumps(info), flush=True)
    while True:
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != args.boot:
            raise RuntimeError('Host boot identity changed; no signal')
        process = Path(f'/proc/{args.pid}/stat')
        if not process.exists():
            info.update(status='target_exited_without_guard_signal', finished_unix_s=time.time())
            atomic(status, info)
            return
        stat = process.read_text().rsplit(')', 1)[1].split()
        if int(stat[19]) != args.ticks:
            raise RuntimeError('PID reused; no signal')
        if stat[0] == 'Z':
            info.update(status='target_terminal_without_guard_signal', finished_unix_s=time.time())
            atomic(status, info)
            return
        run = json.loads((root/'run.json').read_text())
        active = json.loads((root/'active_job.json').read_text())
        issued = json.loads((root/'issued_propagations.json').read_text())
        original_issued = issued['original_forward_propagations_issued']
        if (run['status'] == 'original_all44_neural_subspace_acceptance' and
                active['kind'] == 'forward' and original_issued >= 44*maximum):
            # The coordinator handler finishes this already-running full check,
            # fixes its candidate, saves maps/checkpoint, and then leaves its loop.
            os.kill(args.pid, signal.SIGINT)
            info.update(status='requested_exit_after_current_original44_check', signal_sent=True,
                signal_unix_s=time.time(), active_token=active['token'],
                new_original_source_forwards_issued=original_issued,
                all_earlier_work_retained=True)
            atomic(status, info)
            print(json.dumps(info), flush=True)
            return
        time.sleep(2)


if __name__ == '__main__':
    main()
