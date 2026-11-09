"""Train a finite-secant correction operator on existing 3 MHz states only."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import os
import signal as process_signal
import shutil
import time
import traceback
import numpy as np
import torch
from .angular_qr_inverse_uno import secant_loss
from .amplitude_aware_angular_inverse_uno import AmplitudeAwareAngularInverseUNO
from .central_secant_data import TrainingPairs, tensor_batch
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--profile-only', action='store_true')
    p.add_argument('--profile', type=Path)
    p.add_argument('--steps', type=int, default=2400)
    p.add_argument('--max-seconds', type=float, default=7200)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--parent-root', type=Path, required=True)
    p.add_argument('--deadline-unix-s', type=float, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--resume-full', type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('Inspect existing run, never overwrite/restart implicitly')
    if not 1 <= a.steps <= 2400 or not 0 < a.max_seconds <= 7200:
        raise ValueError('Prespecified neural phase budget exceeded')
    preparation = json.loads((a.index.parent / 'run.json').read_text())
    if preparation['status'] != 'conditional_secant_data_prepared':
        raise RuntimeError('Data preparation is not complete')
    if Path('/proc', str(preparation['pid'])).exists():
        raise RuntimeError('CPU data preparer still exists')
    parent = json.loads((a.parent_root / 'run.json').read_text())
    transition = json.loads((a.parent_root / 'phase_transition.json').read_text())
    if digest(a.parent_root / 'run.json') != transition['parent_final_run_SHA256']:
        raise RuntimeError('Closed parent run record changed')
    if parent['status'] != 'failed' or 'KeyboardInterrupt' not in parent.get('error', ''):
        raise RuntimeError('Require the explicitly interrupted angular-QR parent')
    if Path('/proc', str(parent['pid'])).exists():
        raise RuntimeError('Parent process has not exited')
    if transition['reason'] != 'continue_with_amplitude_input_after_checkpoint':
        raise RuntimeError('Missing explicit parent-to-amplitude phase transition')
    if any(transition[k] != parent[k] for k in ['pid', 'start_ticks', 'boot_id']):
        raise RuntimeError('Parent process identity changed')
    if parent['index_SHA256'] != digest(a.index):
        raise RuntimeError('Parent and continuation data differ')
    parent_weights = a.parent_root / 'weights.pt'
    if digest(parent_weights) != transition['checkpoint_SHA256']:
        raise RuntimeError('Parent EMA checkpoint changed')
    if transition['checkpoint_step'] != 600:
        raise RuntimeError('This phase uses the prespecified 600-step EMA')
    parent_completed = transition['actual_last_completed_step']
    if not 600 <= parent_completed < 2400:
        raise RuntimeError('Invalid actual parent update count')
    if a.deadline_unix_s != transition['original_shared_deadline_unix_s']:
        raise RuntimeError('Continuation must not reset the original deadline')
    if a.deadline_unix_s > parent['created_unix_s'] + 7200:
        raise RuntimeError('Continuation exceeds the original shared time budget')
    if time.time() >= a.deadline_unix_s - 30:
        raise RuntimeError('No remaining shared budget for an actual phase')
    a.max_seconds = min(a.max_seconds, a.deadline_unix_s - time.time())
    for name, expected in parent['source_SHA256'].items():
        if digest(Path(__file__).parent / name) != expected:
            raise RuntimeError('Frozen parent source changed: ' + name)
    recovery = json.loads(a.receipt.read_text())
    dead_root = Path(recovery['training_root'])
    if not recovery['PID_verified_gone'] or Path('/proc', str(recovery['pid'])).exists():
        raise RuntimeError('Previous training has not been verified gone')
    if digest(dead_root / 'run.json') != recovery['run_SHA256']:
        raise RuntimeError('Abnormal-exit record changed')
    if recovery['latest_available_weights_SHA256'] != transition['checkpoint_SHA256']:
        raise RuntimeError('Recovery parent EMA differs')
    if recovery['original_shared_deadline_unix_s'] != a.deadline_unix_s:
        raise RuntimeError('Recovery must retain the original deadline')
    parent_completed = int(recovery['total_completed_optimizer_work'])
    if not 600 <= parent_completed < 2400:
        raise RuntimeError('Invalid consumed optimization work')
    resume_sha = digest(a.resume_full) if a.resume_full is not None else None
    a.steps = min(a.steps, 2400 - parent_completed)
    index = json.loads(a.index.read_text())
    pairs = TrainingPairs(index)
    if not a.profile_only:
        if a.profile is None:
            raise ValueError('A completed actual training-step memory measurement is required')
        profile = json.loads((a.profile / 'run.json').read_text())
        if profile['status'] != 'conditional_secant_full_step_profiled' or Path('/proc', str(profile['pid'])).exists():
            raise RuntimeError('Wait for the actual complete resource measurement to exit')
        if (profile.get('recovery_receipt_SHA256') != digest(a.receipt) or
                profile.get('resume_full_SHA256') != resume_sha):
            raise RuntimeError('Profile used different recovery state')
        if (profile.get('parent_checkpoint_SHA256') != transition['checkpoint_SHA256'] or
                profile.get('original_shared_deadline_unix_s') != a.deadline_unix_s or
                profile.get('parent_completed_updates') != parent_completed):
            raise RuntimeError('Actual resource measurement used a different warm phase')
        if profile['index_SHA256'] != digest(a.index) or profile['device'] != a.device:
            raise RuntimeError('Profile data/device changed')
    names = ['train_amplitude_recovery.py', 'amplitude_aware_angular_inverse_uno.py',
             'angular_qr_inverse_uno.py',
             'angular_receiver_basis.py', 'angular_qr_ray_rf_readout.py', 'ray_conditional_inverse_uno.py',
             'current_ray_rf_readout.py', 'ray_propagation.py', 'forward_model.py',
             'rf_representation.py', 'central_secant_data.py', 'conditional_secant_data.py',
             'uno_rf.py', 'pressure_resources.py', 'native_gn_json.py']
    hashes = {name: digest(Path(__file__).parent / name) for name in names}
    if not a.profile_only:
        same_model_names = names
        if any(hashes[name] != profile['source_SHA256'][name] for name in same_model_names):
            raise RuntimeError('Source changed after actual complete-step resource measurement')
        if profile.get('peak_reserved_GiB', 100.) >= 8:
            raise RuntimeError('The measured full training step exceeds the allocator budget')
    a.out.mkdir()
    run = {'status': 'starting_conditional_secant', 'pid': os.getpid(),
           'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
           'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
           'created_unix_s': time.time(), 'seed': 611083, 'device': a.device,
           'index': str(a.index.resolve()), 'index_SHA256': digest(a.index),
           'source_SHA256': hashes, 'planned_steps': a.steps, 'walltime_budget_seconds': a.max_seconds,
           'profile_only': a.profile_only, 'allocator_cap_GiB': 8, 'per_card_limit_GiB': 16,
           'training_case': 'Neg_07_Left', 'training_material_records': 112, 'training_views': 168,
           'development_case': 'Neg_35_Left', 'development_used_for_training': False,
           'development_truth_used_as_inference_baseline': False,
           'reserved_case_unread': 'Neg_47_Left', 'new_wave_solves': 0,
           'new_native_material_derivatives': 0, 'previous_native_attempts_remain': 22,
           'previous_native_completed_gradients_remain': 21,
           'new_neural_phase_does_not_reset_previous_native_budget': True,
           'training_pairing': 'same unperturbed baseline for both plus and minus changes',
           'same_training_data_and_task_frequencies_as_parent': True,
           'sampler_rng_reset_at_new_phase': a.resume_full is None,
           'profile_reuse': False,
           'new_structural_hypothesis': 'finite nonlinear heads can read residual amplitude; all QR, echo and loss mechanisms retained',
           'same_initial_parent_function_as_warm_angular_checkpoint': a.resume_full is None,
           'density_residual_value_path_is_direct_and_linear': True,
           'all_auxiliary_shape_and_gain_losses_removed': True,
           'same_original_three_field_shape_gain_loss_as_relative_parent': False,
           'independent_material_correction_loss_only': False,
           'alpha_local_and_cumulative_errors_share_one_third_weight': True,
           'combined_change_does_not_isolate_component_causality': True,
           'training_tasks': {'reference_to_material': .2, 'independent_parameter_pair': .4,
                              'cross_material_same_clock': .4},
           'checkpoint_selection': 'fixed final EMA .995; no development checkpoint selection',
           'optimizer': 'fresh AdamW from parent EMA; no old moments restored; global 2400-update cosine schedule; WD1e-4; clip1',
           'joint_imaging_success_claimed': False}
    run.update(parent_root=str(a.parent_root.resolve()),
               parent_run_SHA256=digest(a.parent_root / 'run.json'),
               parent_checkpoint_SHA256=transition['checkpoint_SHA256'],
               parent_checkpoint_step=600, parent_completed_updates=parent_completed,
               original_shared_deadline_unix_s=a.deadline_unix_s,
               original_shared_maximum_updates=2400,
               warm_start_scope='closed parent EMA only; model and new EMA initialized from it',
               optimizer_and_sampler_states_restored=False,
               phase_is_not_an_exact_optimizer_resume=True,
               resource_profile_consumes_remaining_shared_time=True)
    run.update(recovery_receipt_SHA256=digest(a.receipt),
               interrupted_training_root=str(dead_root.resolve()),
               optimizer_work_already_consumed=parent_completed,
               lost_updates_without_weights=recovery['lost_phase_updates_without_checkpoint'],
               available_warm_model_step=600, checkpoint_interval_steps=25,
               full_checkpoint_scope='model, EMA, Adam, NumPy, torch CPU/CUDA RNG, task statistics',
               resume_full_SHA256=resume_sha,
               optimizer_and_sampler_states_restored=a.resume_full is not None)
    atomic(a.out / 'run.json', run)
    monitor = PressureResources(a.out)
    try:
        monitor.wait_budget([a.device], 8, 'conditional_secant_before_training')
        torch.set_num_threads(1); torch.manual_seed(611083)
        torch.cuda.set_device(a.device); device = f'cuda:{a.device}'
        torch.cuda.set_per_process_memory_fraction(
            8 * 2**30 / torch.cuda.get_device_properties(a.device).total_memory, a.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = AmplitudeAwareAngularInverseUNO().to(device)
        warm = torch.load(parent_weights, map_location='cpu', weights_only=False)
        if warm['step'] != 600:
            raise RuntimeError('Parent checkpoint is not the declared 600-step EMA')
        model.load_parent_angular_state(warm['ema'])
        del warm
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        rng = np.random.default_rng(611083)
        resumed = None; trajectory_base = parent_completed; ancestry_base = 0
        if a.resume_full is not None:
            resumed = torch.load(a.resume_full, map_location='cpu', weights_only=False)
            if resumed['run']['index_SHA256'] != digest(a.index):
                raise RuntimeError('Full checkpoint data changed')
            if resumed['run']['original_shared_deadline_unix_s'] != a.deadline_unix_s:
                raise RuntimeError('Full checkpoint cannot reset deadline')
            if resumed['run']['source_SHA256'] != hashes:
                raise RuntimeError('Full resume requires exact frozen source')
            model.load_state_dict(resumed['model'], strict=True)
            ema.load_state_dict(resumed['ema'], strict=True)
            optimizer.load_state_dict(resumed['optimizer'])
            rng.bit_generator.state = resumed['numpy_rng_state']
            torch.set_rng_state(resumed['torch_rng_state'])
            torch.cuda.set_rng_state_all(resumed['cuda_rng_states'])
            trajectory_base = int(resumed['trajectory_step'])
            ancestry_base = int(resumed['model_updates_since_warm'])
        run.update(trajectory_start=trajectory_base, model_updates_since_warm_start=ancestry_base,
                   optimizer_and_sampler_states_restored=resumed is not None,
                   phase_is_not_an_exact_optimizer_resume=resumed is None)
        run.update(status='profiling_conditional_secant' if a.profile_only else 'training_conditional_secant',
                   model=model.config, parameter_count=sum(p.numel() for p in model.parameters()))
        atomic(a.out / 'run.json', run)
        for name in names:
            (a.out / name).write_bytes((Path(__file__).parent / name).read_bytes())
        started = time.time(); last_resource = started
        task_counts = {} if resumed is None else dict(resumed['task_counts'])
        task_average = {} if resumed is None else {k: np.asarray(v) for k, v in resumed['task_average'].items()}
        seen = set() if resumed is None else set(resumed['seen_training_ids'])
        del resumed
        stop_signal = 0
        def stop_after_update(signum, frame):
            nonlocal stop_signal
            stop_signal = int(signum)
        process_signal.signal(process_signal.SIGINT, stop_after_update)
        process_signal.signal(process_signal.SIGTERM, stop_after_update)
        checkpoint_path = a.out / 'checkpoint.pt'
        storage_estimate = (4 * sum(p.numel() * p.element_size() for p in model.parameters()) +
                            2 * sum(b.numel() * b.element_size() for b in model.buffers()) + 4 * 2**20)
        run['checkpoint_storage_estimate_bytes'] = storage_estimate

        def save_full(local_step):
            if shutil.disk_usage(a.out).free < 2 * 2**30 + 400 * 2**20 + storage_estimate:
                raise RuntimeError('Insufficient reserve for atomic full checkpoint')
            torch.cuda.synchronize()
            payload = {
                'step': local_step, 'global_optimization_updates': parent_completed + local_step,
                'trajectory_step': trajectory_base + local_step,
                'model_updates_since_warm': ancestry_base + local_step,
                'model': model.state_dict(), 'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(), 'numpy_rng_state': rng.bit_generator.state,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_states': torch.cuda.get_rng_state_all(),
                'task_counts': task_counts.copy(),
                'task_average': {k: v.tolist() for k, v in task_average.items()},
                'seen_training_ids': sorted(seen), 'config': model.config, 'run': dict(run),
                'checkpoint_scope': 'complete optimization state after a completed update'}
            temp = a.out / 'checkpoint.partial.pt'
            torch.save(payload, temp); temp.replace(checkpoint_path)
            del payload
            run.update(latest_checkpoint_step=local_step,
                       latest_checkpoint_global_work=parent_completed + local_step,
                       latest_checkpoint_SHA256=digest(checkpoint_path),
                       latest_checkpoint_bytes=checkpoint_path.stat().st_size,
                       latest_checkpoint_unix_s=time.time())
            atomic(a.out / 'run.json', run)

        for step in range(1, (1 if a.profile_only else a.steps) + 1):
            if step % 250 == 1:
                for name, expected in hashes.items():
                    if digest(Path(__file__).parent / name) != expected:
                        raise RuntimeError('Frozen training source changed: ' + name)
            global_step = parent_completed + step
            trajectory_step = trajectory_base + step
            if time.time() >= a.deadline_unix_s or stop_signal:
                step -= 1
                save_full(step)
                break
            sample = pairs.sample(rng, trajectory_step, profile=a.profile_only)
            signal, baseline, context, target, rms = tensor_batch(sample, device)
            rate = 3e-4 * (.05 + .95 * .5 * (1 + math.cos(math.pi * (trajectory_step - 1) / 2400)))
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            result = model(signal, baseline, context, rms)
            loss, details = secant_loss(result, baseline, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite finite-secant objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for dest, src in zip(ema.parameters(), model.parameters()):
                    dest.lerp_(src, .005)
            torch.cuda.synchronize()
            elapsed = time.time() - started
            task = sample['task']; task_counts[task] = task_counts.get(task, 0) + 1
            value = np.asarray(details['relative_c_rho_alpha'])
            task_average[task] = value if task not in task_average else .95 * task_average[task] + .05 * value
            seen.update((sample['baseline_id'], sample['target_id']))
            item = {'step': step, 'global_optimization_updates': global_step,
                    'trajectory_step': trajectory_step,
                    'model_updates_since_warm': ancestry_base + step,
                    'original_shared_deadline_unix_s': a.deadline_unix_s,
                    'task': task, 'baseline_id': sample['baseline_id'],
                    'target_id': sample['target_id'], 'valid_samples': sample['record_samples'],
                    'RF_difference_RMS_in_training_global_units': sample['rms'],
                    'loss': float(loss.detach()), **details, 'gradient_norm': float(norm),
                    'seconds': elapsed, 'seconds_per_step': elapsed / step,
                    'task_counts': task_counts.copy(),
                    'task_relative_error_EMA': {k: v.tolist() for k, v in task_average.items()},
                    'unique_training_ids': len(seen),
                    'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
                    'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30}
            with (a.out / 'training.jsonl').open('a') as log:
                log.write(json.dumps(item) + '\n')
            if step == 1 or step % 25 == 0:
                atomic(a.out / 'progress.json', item); print(json.dumps(item), flush=True)
            if step == 1 or time.time() - last_resource > 20:
                monitor.sample('conditional_secant_training'); last_resource = time.time()
            final = (a.profile_only or step == a.steps or elapsed >= a.max_seconds or
                     time.time() >= a.deadline_unix_s or bool(stop_signal))
            if step == 1 or step % 25 == 0 or final:
                save_full(step)
            final = final or bool(stop_signal) or time.time() >= a.deadline_unix_s
            if final and run.get('latest_checkpoint_step') != step:
                save_full(step)
            del signal, baseline, context, target, rms, result, loss
            if final:
                break
        if run.get('latest_checkpoint_step') != step:
            save_full(step)
        if a.profile_only:
            saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            assert saved['step'] == step and len(saved['optimizer']['state']) > 0
            model.load_state_dict(saved['model'], strict=True)
            ema.load_state_dict(saved['ema'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            rng.bit_generator.state = saved['numpy_rng_state']
            torch.set_rng_state(saved['torch_rng_state'])
            torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
            del saved
            torch.cuda.synchronize()
            run['profile_full_checkpoint_save_and_restore_passed'] = True
            run['profile_checkpoint_retained'] = False
            checkpoint_path.unlink()
        else:
            inference = {'step': step, 'ema': ema.state_dict(), 'config': model.config,
                         'run': dict(run), 'checkpoint_scope': 'final EMA; full state in checkpoint.pt'}
            temp = a.out / 'weights.partial.pt'
            torch.save(inference, temp); temp.replace(a.out / 'weights.pt')
            del inference
        terminal_status = (
            'conditional_secant_full_step_profiled' if a.profile_only and step == 1 and not stop_signal else
            'conditional_secant_training_finished' if not a.profile_only and step > 0 and not stop_signal else
            'conditional_secant_interrupted_with_full_checkpoint')
        run.update(stopping_signal=stop_signal, trajectory_completed=trajectory_base + step,
                   model_updates_since_warm=ancestry_base + step)
        run.update(status=terminal_status,
                   completed_steps=step, seconds=time.time() - started,
                   stop_reason='actual_training_step_profile' if a.profile_only else
                   ('shared_update_budget' if step == a.steps else 'shared_walltime_budget'),
                   peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
                   finished_unix_s=time.time(), task_counts=task_counts)
        run.update(cumulative_optimization_updates=parent_completed + step,
                   shared_seconds_since_parent_creation=time.time() - parent['created_unix_s'])
        if not a.profile_only:
            run['weights_SHA256'] = digest(a.out / 'weights.pt')
        monitor.sample('conditional_secant_finished'); atomic(a.out / 'run.json', run)
        print(json.dumps(run), flush=True)
    except BaseException:
        run.update(status='failed', error=traceback.format_exc(), failed_unix_s=time.time())
        atomic(a.out / 'run.json', run); raise
    finally:
        monitor.log.close()


if __name__ == '__main__':
    main()
