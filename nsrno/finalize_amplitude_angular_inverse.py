"""Sequential final evaluation after the same amplitude-aware trainer exits.

This CPU coordinator never restarts training, changes its deadline, or starts
another model candidate. Only the final fixed weights are evaluated.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback


def atomic(path, value):
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--comparison-root', type=Path, required=True)
    parser.add_argument('--summary-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    permitted = (Path.cwd() / 'out/v5').resolve()
    for path in [args.train_root, args.comparison_root, args.summary_out, args.out]:
        path.resolve().relative_to(permitted)
    evaluation = args.train_root / 'evaluation'
    if args.out.exists() or evaluation.exists() or args.summary_out.exists():
        raise RuntimeError('Inspect existing finalization/evaluation; never repeat implicitly')
    training = json.loads((args.train_root / 'run.json').read_text())
    if training['status'] not in ['training_conditional_secant', 'conditional_secant_training_finished']:
        raise RuntimeError('Expected the current actual amplitude-aware training')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if training['boot_id'] != boot:
        raise RuntimeError('Training boot differs; inspect its terminal state')
    names = ['finalize_amplitude_angular_inverse.py',
             'evaluate_amplitude_angular_inverse.py', 'summarize_conditional_secant_results.py']
    hashes = {n: digest(Path(__file__).parent / n) for n in names}
    args.out.mkdir()
    state = {
        'status': 'waiting_for_same_amplitude_training',
        'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': boot, 'created_unix_s': time.time(),
        'training_root': str(args.train_root.resolve()),
        'training_pid': training['pid'], 'training_start_ticks': training['start_ticks'],
        'training_boot_id': training['boot_id'],
        'original_shared_deadline_unix_s': training['original_shared_deadline_unix_s'],
        'evaluation_root': str(evaluation.resolve()),
        'comparison_root': str(args.comparison_root.resolve()),
        'summary_out': str(args.summary_out.resolve()),
        'source_SHA256': hashes, 'coordinator_GPU_work': False,
        'per_card_limit_GiB': 16, 'evaluation_allocator_cap_GiB': 8,
        'maximum_evaluation_seconds': 1200, 'maximum_CPU_summary_seconds': 600,
        'new_training_launched': False, 'new_wave_solves': 0, 'new_material_derivatives': 0,
        'previous_model_predictions_reused_from_frozen_cache': True,
        'development_truth_used_for_neural_inference': False,
        'reserved_case_unread': 'Neg_47_Left', 'joint_imaging_success_claimed': False,
    }
    atomic(args.out / 'run.json', state)

    def same_training_is_live():
        proc = Path('/proc', str(training['pid']))
        try:
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            command = (proc / 'cmdline').read_bytes().replace(bytes([0]), b' ').decode()
        except (FileNotFoundError, ProcessLookupError):
            return False
        if int(stat[19]) != training['start_ticks']:
            raise RuntimeError('Training PID identity changed; inspect it')
        if stat[0] == 'Z':
            return True
        if ' -m nsrno.train_amplitude_angular_inverse ' not in command:
            raise RuntimeError('Unexpected process occupies the training PID')
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != boot:
            raise RuntimeError('Boot changed while waiting')
        return True

    def source_unchanged():
        for name, expected in hashes.items():
            if digest(Path(__file__).parent / name) != expected:
                raise RuntimeError('Finalization source changed: ' + name)

    def child(command, env, name, seconds):
        source_unchanged()
        log_path = args.out / (name + '.log')
        with log_path.open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, env=env)
        start_ticks = int(Path('/proc', str(process.pid), 'stat').read_text()
                          .rsplit(')', 1)[1].split()[19])
        state.update(status=name, child_pid=process.pid, child_start_ticks=start_ticks,
                     child_command=command, child_started_unix_s=time.time(),
                     child_log=str(log_path.resolve()))
        atomic(args.out / 'run.json', state)
        started = time.time()
        while process.poll() is None:
            state['child_observed_unix_s'] = time.time()
            atomic(args.out / 'run.json', state)
            if time.time() - started >= seconds:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    raise RuntimeError('Same child remains live after SIGINT; inspect it')
                raise RuntimeError(name + ' reached its fixed time allowance')
            time.sleep(5)
        state.update(child_returncode=process.returncode, child_finished_unix_s=time.time())
        atomic(args.out / 'run.json', state)
        if process.returncode:
            raise RuntimeError(name + ' failed; inspect its existing log and outputs')
        state.update(child_pid=None, child_start_ticks=None)

    try:
        while same_training_is_live():
            progress_path = args.train_root / 'progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            state.update(observed_unix_s=time.time(),
                         last_training_step=progress.get('step'),
                         last_total_optimization_updates=progress.get('global_optimization_updates'),
                         training_still_live=True)
            atomic(args.out / 'run.json', state)
            time.sleep(10)
        finished = json.loads((args.train_root / 'run.json').read_text())
        if finished['status'] != 'conditional_secant_training_finished':
            state['actual_training_terminal_run'] = finished
            raise RuntimeError('Training exited without final fixed-weight completion')
        if digest(args.train_root / 'weights.pt') != finished['weights_SHA256']:
            raise RuntimeError('Actual final EMA checksum does not match training')
        for name, expected in finished['source_SHA256'].items():
            if digest(Path(__file__).parent / name) != expected:
                raise RuntimeError('Actual trained source changed: ' + name)
        state.update(status='waiting_for_GPU_release', training_still_live=False,
                     training_finished_run_SHA256=digest(args.train_root / 'run.json'),
                     fixed_weights_SHA256=finished['weights_SHA256'],
                     training_completed_steps=finished['completed_steps'],
                     total_optimization_updates=finished['cumulative_optimization_updates'])
        atomic(args.out / 'run.json', state)
        waiting = time.time()
        while True:
            source_unchanged()
            apps = subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                text=True, timeout=10).strip()
            if not apps:
                break
            state.update(GPU_apps_before_evaluation=apps, observed_unix_s=time.time())
            atomic(args.out / 'run.json', state)
            if time.time() - waiting > 300:
                raise RuntimeError('GPU is still occupied; no new job launched')
            time.sleep(10)
        if shutil.disk_usage('.').free < 2 * 2**30 + 400 * 2**20 + 50 * 2**20:
            raise RuntimeError('Insufficient disk reserve for all three actual predictions/maps')
        if evaluation.exists() or args.summary_out.exists():
            raise RuntimeError('Another finalization already created outputs')
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMPY_MADVISE_HUGEPAGE='0', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH='.', PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
        env['CUDA_VISIBLE_DEVICES'] = '0'
        child([sys.executable, '-m', 'nsrno.evaluate_amplitude_angular_inverse',
               '--train-root', str(args.train_root), '--out', str(evaluation),
               '--comparison-root', str(args.comparison_root)],
              env, 'evaluating_fixed_final_model', 1200)
        evaluated = json.loads((evaluation / 'run.json').read_text())
        if evaluated['status'] != 'conditional_actual_predictions_and_training_secants_finished':
            raise RuntimeError('Actual prediction evaluation is not complete')
        if Path('/proc', str(evaluated['pid'])).exists():
            raise RuntimeError('Actual evaluator still exists')
        env['CUDA_VISIBLE_DEVICES'] = ''
        child([sys.executable, '-m', 'nsrno.summarize_conditional_secant_results',
               '--evaluation', str(evaluation), '--out', str(args.summary_out)],
              env, 'summarizing_locked_predictions_on_CPU', 600)
        comparison = json.loads((args.summary_out / 'comparison.json').read_text())
        expected_names = {'train_200_reference', 'val_059_reference', 'train_200_native_current'}
        if {row['name'] for row in comparison['predictions']} != expected_names:
            raise RuntimeError('Expected all three actual material predictions')
        for row in comparison['predictions']:
            if not (Path(row['folder']) / 'maps.png').exists():
                raise RuntimeError('An actual complete material map is missing')
        if comparison['weights_SHA256'] != finished['weights_SHA256']:
            raise RuntimeError('Final maps do not match the trained EMA')
        state.update(status='fixed_predictions_and_CPU_maps_ready',
                     finished_unix_s=time.time(), child_pid=None, child_start_ticks=None,
                     comparison_file=str((args.summary_out / 'comparison.json').resolve()),
                     comparison_SHA256=digest(args.summary_out / 'comparison.json'),
                     full_three_parameter_maps_ready=True,
                     actual_visual_review_still_required=True,
                     physical_RF_and_independent_joint_controls_not_yet_done=True,
                     joint_imaging_success_claimed=False)
        atomic(args.out / 'run.json', state)
        print(json.dumps(state), flush=True)
    except BaseException:
        state.update(status='finalization_needs_inspection', error=traceback.format_exc(),
                     observed_unix_s=time.time())
        atomic(args.out / 'run.json', state)
        raise


if __name__ == '__main__':
    main()
