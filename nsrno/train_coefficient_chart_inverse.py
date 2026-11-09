"""Bounded actual training of independent density/alpha output charts, no new RF."""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
import traceback
import numpy as np
import torch
from .coefficient_chart_angular_inverse_uno import CoefficientChartAngularInverseUNO
from .angular_qr_inverse_uno import secant_loss
from .central_secant_data import TrainingPairs, tensor_batch
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--max-seconds', type=float, default=2400.)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.steps <= 600 or not 60 <= args.max_seconds <= 2400:
        raise ValueError('Prespecified new structural phase budget exceeded')
    args.parent, args.out, args.checkpoint_root = [p.resolve() for p in
                                           [args.parent, args.out, args.checkpoint_root]]
    if not all(p.is_relative_to(Path.cwd().resolve() / 'out/v5') for p in [args.parent, args.out]):
        raise ValueError('Experiment records must remain in this independent project')
    if not args.checkpoint_root.is_relative_to(Path.home() / 'nsrno_artifacts'):
        raise ValueError('Use the same account dedicated artifact volume only')
    if args.out.exists() or args.checkpoint_root.exists():
        raise RuntimeError('Inspect same existing job; never overwrite or reset its budget')
    parent = json.loads((args.parent / 'run.json').read_text())
    if (parent['status'] != 'conditional_secant_training_finished' or
        Path('/proc', str(parent['pid'])).exists()):
        raise RuntimeError('Parent training must be finished and exited')
    parent_weights = args.parent / 'weights.pt'
    expected = 'a08bf2897384aaa30c99f2191099d2887d841d3297f46c3d4076726b164743e1'
    if parent['weights_SHA256'] != expected or digest(parent_weights) != expected:
        raise RuntimeError('Use the declared fixed closed parent EMA')
    for name, sha in parent['source_SHA256'].items():
        if digest(Path(__file__).parent / name) != sha:
            raise RuntimeError('Frozen parent source changed: ' + name)
    index_path = Path(parent['index'])
    if digest(index_path) != parent['index_SHA256']:
        raise RuntimeError('Prepared same-clock data index changed')
    index = json.loads(index_path.read_text())
    pairs = TrainingPairs(index)  # Enforces Neg_07_Left-only training labels.
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
            '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('A GPU experiment is already active; inspect it first')
    if shutil.disk_usage('.').free < 2 * 2**30 + 400 * 2**20 + 50 * 2**20:
        raise RuntimeError('Reserve native prediction outputs on the data volume')
    if shutil.disk_usage(Path.home()).free < 2 * 2**30 + 250 * 2**20:
        raise RuntimeError('Reserve atomic full checkpoints on the artifact volume')
    args.out.mkdir()
    args.checkpoint_root.mkdir(parents=True)
    checkpoint_path = args.checkpoint_root / 'checkpoint.pt'
    (args.out / 'checkpoint.pt').symlink_to(checkpoint_path)
    names = sorted(set(parent['source_SHA256']) | {
        Path(__file__).name, 'coefficient_chart_angular_inverse_uno.py',
        'gradient_density_angular_inverse_uno.py', 'anchored_gradient_integration.py',
        'depth_primitive.py'})
    hashes = {name: digest(Path(__file__).parent / name) for name in names}
    started = time.time()
    deadline = started + args.max_seconds
    parent_work = int(parent['cumulative_optimization_updates'])
    parent_ancestry = 1100
    run = {'status': 'initializing_coefficient_chart_inverse', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': deadline,
        'maximum_updates': args.steps, 'maximum_seconds': args.max_seconds,
        'index': str(index_path), 'index_SHA256': digest(index_path),
        'parent_root': str(args.parent), 'parent_weights_SHA256': expected,
        'source_SHA256': hashes, 'device': args.device,
        'checkpoint_path': str(checkpoint_path), 'checkpoint_interval_steps': 25,
        'parent_optimizer_work_already_consumed': parent_work,
        'parent_model_ancestry_updates': parent_ancestry,
        'parent_lost_updates_without_weights': parent.get('lost_updates_without_weights', 459),
        'previous_native_and_neural_budgets_not_reset': True,
        'fresh_AdamW_for_new_output_structure_not_exact_optimizer_resume': True,
        'first_resource_measurement_is_counted_actual_training_update': True,
        'allocator_cap_GiB': 8, 'per_card_process_tree_limit_GiB': 16,
        'training_case': 'Neg_07_Left', 'development_case': 'Neg_35_Left',
        'development_material_used_for_training': False,
        'reserved_case_unread': 'Neg_47_Left',
        'new_wave_solves': 0, 'new_acoustic_material_derivatives': 0,
        'independent_three_fields': True, 'forced_smooth_c': False,
        'shared_material_edges': False, 'new_parameters_beyond_parent': 48,
        'same_input_readout_sampling_and_loss_as_closed_parent': True,
        'weight_selection': 'fixed final EMA .995', 'joint_imaging_success_claimed': False}
    atomic(args.out / 'run.json', run)
    monitor = PressureResources(args.out)
    stop = [0]
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, lambda sig, frame: stop.__setitem__(0, int(sig)))
    signal.signal(signal.SIGTERM, lambda sig, frame: stop.__setitem__(0, int(sig)))
    completed = 0
    try:
        monitor.wait_budget([args.device], 8, 'coefficient_chart_initial_admission')
        torch.set_num_threads(1)
        torch.manual_seed(611085)
        torch.cuda.set_device(args.device)
        device = f'cuda:{args.device}'
        torch.cuda.set_per_process_memory_fraction(
            8 * 2**30 / torch.cuda.get_device_properties(args.device).total_memory, args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = CoefficientChartAngularInverseUNO().to(device)
        warm = torch.load(parent_weights, map_location='cpu', weights_only=False)
        model.load_parent_amplitude_state(warm['ema'])
        del warm
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        rng = np.random.default_rng(611085)
        counts, averages, seen = {}, {}, set()
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(p.numel() * p.element_size() for p in model.buffers())
        storage_estimate = 4 * parameter_bytes + 2 * buffer_bytes + 4 * 2**20
        run.update(status='training_coefficient_chart_inverse', model=model.config,
            parameter_count=sum(p.numel() for p in model.parameters()),
            parameter_bytes=parameter_bytes, checkpoint_estimate_bytes=storage_estimate)
        for name in names:
            (args.out / name).write_bytes((Path(__file__).parent / name).read_bytes())
        atomic(args.out / 'run.json', run)
        last_resource = 0.

        def save_full():
            if shutil.disk_usage(args.checkpoint_root).free < 2 * 2**30 + storage_estimate:
                raise RuntimeError('Insufficient artifact reserve for atomic full checkpoint')
            payload = {'step': completed, 'model': model.state_dict(), 'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(), 'numpy_rng_state': rng.bit_generator.state,
                'torch_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all(),
                'task_counts': counts, 'task_average': {k: v.tolist() for k, v in averages.items()},
                'seen_training_ids': sorted(seen), 'config': model.config, 'run': dict(run),
                'global_optimization_updates': parent_work + completed,
                'model_ancestry_updates': parent_ancestry + completed,
                'checkpoint_scope': 'model, EMA, Adam, sampler and random states after completed update'}
            temporary = args.checkpoint_root / 'checkpoint.partial.pt'
            torch.save(payload, temporary)
            temporary.replace(checkpoint_path)
            run.update(latest_checkpoint_step=completed,
                checkpoint_SHA256=digest(checkpoint_path), checkpoint_bytes=checkpoint_path.stat().st_size,
                cumulative_optimizer_work=parent_work+completed)
            atomic(args.out / 'run.json', run)

        for step in range(1, args.steps + 1):
            if time.time() >= deadline or stop[0]:
                break
            if step % 250 == 1:
                if any(digest(Path(__file__).parent / name) != sha for name, sha in hashes.items()):
                    raise RuntimeError('Frozen new phase source changed during training')
            # Same task proportions as parent, with an explicitly new sampler/Adam phase.
            sample = pairs.sample(rng, parent_work + step)
            rf, baseline, context, target, rms = tensor_batch(sample, device)
            rate = 3e-4 * (.05 + .95 * .5 * (1 + math.cos(math.pi * (step-1) / args.steps)))
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            result = model(rf, baseline, context, rms)
            loss, details = secant_loss(result, baseline, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite independent field objective')
            loss.backward()
            gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for dest, source in zip(ema.parameters(), model.parameters()):
                    dest.lerp_(source, .005)
            torch.cuda.synchronize()
            completed = step
            task = sample['task']
            counts[task] = counts.get(task, 0) + 1
            vector = np.asarray(details['relative_c_rho_alpha'])
            averages[task] = vector if task not in averages else .95*averages[task] + .05*vector
            seen.update((sample['baseline_id'], sample['target_id']))
            elapsed = time.time() - started
            item = {'step': step, 'cumulative_optimizer_work': parent_work+step,
                'model_ancestry_updates': parent_ancestry+step,
                'task': task, 'loss': float(loss.detach()), **details,
                'gradient_norm': float(gradnorm), 'seconds': elapsed,
                'seconds_per_step': elapsed/step, 'learning_rate': rate,
                'record_samples': sample['record_samples'], 'task_counts': counts.copy(),
                'task_relative_error_EMA': {k: v.tolist() for k, v in averages.items()},
                'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30}
            with (args.out/'training.jsonl').open('a') as log:
                log.write(json.dumps(item)+'\n')
            if step == 1 or step % 25 == 0:
                atomic(args.out/'progress.json', item)
                print(json.dumps({k: item[k] for k in ['step','loss','seconds','peak_reserved_GiB']}), flush=True)
            if step == 1 or time.time()-last_resource >= 20:
                monitor.sample('actual_coefficient_chart_training')
                last_resource = time.time()
                if monitor.owned_above_limit:
                    stop[0] = -16
            final = step == args.steps or time.time() >= deadline or bool(stop[0])
            if step == 1 or step % 25 == 0 or final:
                save_full()
            del rf, baseline, context, target, rms, result, loss
            if final:
                break
        if completed and run.get('latest_checkpoint_step') != completed:
            save_full()
        status = ('coefficient_chart_training_finished' if completed and not stop[0]
                  else 'coefficient_chart_training_interrupted_with_checkpoint')
        run.update(status=status, completed_steps=completed, seconds=time.time()-started,
            finished_unix_s=time.time(), stopping_signal=stop[0], task_counts=counts,
            stop_reason='update_budget' if completed == args.steps else 'wall_or_signal_budget',
            model_ancestry_updates=parent_ancestry+completed,
            cumulative_optimizer_work=parent_work+completed,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
            full_checkpoint_retained=True, final_inference_EMA_in_full_checkpoint=True)
        monitor.sample('coefficient_chart_training_finished')
        atomic(args.out/'run.json', run)
        print(json.dumps({k:run[k] for k in ['status','completed_steps','seconds',
            'cumulative_optimizer_work','peak_reserved_GiB','checkpoint_SHA256']}), flush=True)
    except BaseException:
        run.update(status='coefficient_chart_training_failed', error=traceback.format_exc(),
                   completed_steps=completed, failed_unix_s=time.time())
        atomic(args.out/'run.json', run)
        raise
    finally:
        monitor.log.close()


if __name__ == '__main__':
    main()
