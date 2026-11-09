"""Train a finite-secant correction operator on existing 3 MHz states only."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import os
import time
import traceback
import numpy as np
import torch
from .balanced_odd_density_secant_uno import BalancedOddDensitySecantUNO, secant_loss
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
    p.add_argument('--max-seconds', type=float, default=3600)
    p.add_argument('--device', type=int, default=0)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('Inspect existing run, never overwrite/restart implicitly')
    if not 1 <= a.steps <= 2400 or not 0 < a.max_seconds <= 3600:
        raise ValueError('Prespecified neural phase budget exceeded')
    preparation = json.loads((a.index.parent / 'run.json').read_text())
    if preparation['status'] != 'conditional_secant_data_prepared':
        raise RuntimeError('Data preparation is not complete')
    if Path('/proc', str(preparation['pid'])).exists():
        raise RuntimeError('CPU data preparer still exists')
    index = json.loads(a.index.read_text())
    pairs = TrainingPairs(index)
    if not a.profile_only:
        if a.profile is None:
            raise ValueError('A completed actual training-step memory measurement is required')
        profile = json.loads((a.profile / 'run.json').read_text())
        if profile['status'] != 'conditional_secant_full_step_profiled' or Path('/proc', str(profile['pid'])).exists():
            raise RuntimeError('Wait for the actual complete resource measurement to exit')
        if profile['index_SHA256'] != digest(a.index) or profile['device'] != a.device:
            raise RuntimeError('Profile data/device changed')
    names = ['train_balanced_odd_density_secant.py', 'balanced_odd_density_secant_uno.py', 'relative_phase_secant_uno.py', 'conditional_secant_uno.py', 'central_secant_data.py', 'conditional_secant_data.py',
             'raw_rf_inverse_uno.py', 'uno_rf.py', 'inverse_uno.py', 'pressure_resources.py',
             'native_gn_json.py']
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
           'created_unix_s': time.time(), 'seed': 611081, 'device': a.device,
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
           'same_training_data_tasks_seed_budget_as_central_experiment': True,
           'profile_reuse': False,
           'new_structural_hypothesis': 'normalized odd density direction plus even gain; c and alpha retain finite nonlinearity',
           'same_initial_shared_weights_as_relative_model': True,
           'density_initial_function_changes_by_odd_projection': True,
           'density_auxiliary_shape_gain_targets_removed_after_projection': False,
           'same_original_three_field_shape_gain_loss_as_relative_parent': True,
           'raw_odd_projection_run_was_stopped_for_amplitude_reformulation': True,
           'training_tasks': {'reference_to_material': .2, 'independent_parameter_pair': .4,
                              'cross_material_same_clock': .4},
           'checkpoint_selection': 'fixed final EMA .995; no development checkpoint selection',
           'optimizer': 'AdamW 3e-4 cosine to1.5e-5; WD1e-4; global gradient clip1',
           'joint_imaging_success_claimed': False}
    atomic(a.out / 'run.json', run)
    monitor = PressureResources(a.out)
    try:
        monitor.wait_budget([a.device], 8, 'conditional_secant_before_training')
        torch.set_num_threads(1); torch.manual_seed(611081)
        torch.cuda.set_device(a.device); device = f'cuda:{a.device}'
        torch.cuda.set_per_process_memory_fraction(
            8 * 2**30 / torch.cuda.get_device_properties(a.device).total_memory, a.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = BalancedOddDensitySecantUNO().to(device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        rng = np.random.default_rng(611081)
        run.update(status='profiling_conditional_secant' if a.profile_only else 'training_conditional_secant',
                   model=model.config, parameter_count=sum(p.numel() for p in model.parameters()))
        atomic(a.out / 'run.json', run)
        for name in names:
            (a.out / name).write_bytes((Path(__file__).parent / name).read_bytes())
        started = time.time(); last_resource = started
        task_counts = {}; task_average = {}; seen = set()
        for step in range(1, (1 if a.profile_only else a.steps) + 1):
            if step % 250 == 1:
                for name, expected in hashes.items():
                    if digest(Path(__file__).parent / name) != expected:
                        raise RuntimeError('Frozen training source changed: ' + name)
            sample = pairs.sample(rng, step, profile=a.profile_only)
            signal, baseline, context, target, rms = tensor_batch(sample, device)
            rate = 3e-4 * (.05 + .95 * .5 * (1 + math.cos(math.pi * (step - 1) / a.steps)))
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
            item = {'step': step, 'task': task, 'baseline_id': sample['baseline_id'],
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
            final = a.profile_only or step == a.steps or elapsed >= a.max_seconds
            if not a.profile_only and (step % 600 == 0 or final):
                checkpoint = {'step': step, 'ema': ema.state_dict(),
                              'config': model.config, 'run': run,
                              'checkpoint_scope': 'fixed final inference weights; no implicit resume'}
                temp = a.out / 'weights.partial.pt'; torch.save(checkpoint, temp); temp.replace(a.out / 'weights.pt')
            del signal, baseline, context, target, rms, result, loss
            if final:
                break
        run.update(status='conditional_secant_full_step_profiled' if a.profile_only else 'conditional_secant_training_finished',
                   completed_steps=step, seconds=time.time() - started,
                   stop_reason='actual_training_step_profile' if a.profile_only else
                   ('fixed_step_budget' if step == a.steps else 'fixed_walltime_budget'),
                   peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
                   finished_unix_s=time.time(), task_counts=task_counts)
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
