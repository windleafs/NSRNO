"""Same TRAIN patches and denoising algorithm, corrected field populations.

One GPU experiment, fixed3000updates/1800s; no RF or patient derivative reads.
"""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable THP')
from pathlib import Path
import argparse
import copy
import json
import math
import os
import resource
import shutil
import signal
import time
import traceback
import numpy as np
import torch
from .train_independent_material_denoisers import digest, atomic, save, sample
from .independent_denoising_prior import IndependentMaterialDenoisers, training_loss
from .pressure_resources import PressureResources


def eligibility(project, products):
    bases = [r for r in json.loads((project/'out/v5/forward_cache_v1/index.json').read_text())['records']
        if r['role'] == 'train']
    aug = [r for r in json.loads((project/'out/v5/uno_triparam_v1/augmentation_index.json').read_text())['records']
        if r['role'] == 'train']
    base = {r['id']: r for r in bases}
    augmented = {str(Path(r['material3']).resolve()): r for r in aug}
    manifest = json.loads((products/'training_sources.json').read_text())
    rows = manifest['sources']
    if (len(base) != 510 or len(augmented) != 32 or len(rows) != 542 or
            manifest['development_arrays_read'] or manifest['reserved_arrays_read'] or manifest['RF_arrays_read']):
        raise RuntimeError('The unchanged original TRAIN patch bank is required')
    keep = np.ones((3, len(rows)), bool)
    counts = {str(f): {} for f in range(3)}
    for i, row in enumerate(rows):
        if row['case'] != 'Neg_07_Left' or row['role'] != 'train':
            raise RuntimeError('TRAIN-only check must precede any patch read')
        if row['kind'] == 'base':
            original = base[row['id']]
            label = original['variant']
            keep[0, i] = label not in ['calibhom', 'calibuni']
            keep[1:, i] = label != 'calibuni'
        elif row['kind'] == 'augmentation':
            original = augmented[str(Path(row['paths'][0]).resolve())]
            label = 'augmentation_'+original['kind']
            if original['kind'] not in ['alpha', 'trajectory']:
                raise RuntimeError('Unknown clean-population role')
            keep[:, i] = original['kind'] == 'alpha'
        else:
            raise RuntimeError('Unknown original TRAIN manifest kind')
        if original['case'] != 'Neg_07_Left' or original['role'] != 'train':
            raise RuntimeError('A TRAIN source index did not match its patch manifest')
        for field in range(3):
            group = counts[str(field)].setdefault(label, {'retained': 0, 'excluded': 0})
            group['retained' if keep[field, i] else 'excluded'] += 1
    numbers = keep.sum(1).tolist()
    if numbers != [466, 516, 516]:
        raise RuntimeError('Inspect metadata population mismatch before any GPU training: '+str(numbers))
    return np.repeat(keep, 5, axis=1), np.repeat([r['holdout'] for r in rows], 5), {
        'retained_materials_per_field': numbers, 'counts_by_field': counts,
        'same_original_patches_and_holdout': True, 'patches_reextracted': False,
        'development_arrays_read': False, 'reserved_arrays_read': False, 'RF_arrays_read': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--parent-root', type=Path, required=True)
    args = p.parse_args()
    project, source = Path.cwd().resolve(), Path(__file__).parent
    for name in ['out', 'artifact_root', 'parent_root']:
        setattr(args, name, getattr(args, name).resolve())
    if (args.parent_root != project/'out/v5/reconstruction_fixed_model_score_v1' or
            not args.out.is_relative_to(project/'out/v5') or args.out.exists() or
            not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or args.artifact_root.exists()):
        raise ValueError('Fresh samepatch anatomical-prior training after the completed score branch only')
    parent = json.loads((args.parent_root/'run.json').read_text())
    workers = json.loads((args.parent_root/'worker_processes.json').read_text())
    if (parent['status'] != 'matched_score_actual_stage_finished' or
            parent['material_SHA256'] != '4b4d6fd6665bd4b8eeaae0cdcfe1a60d04d5002028b6263145d7116689b809fb' or
            any(Path('/proc', str(pid)).exists() for pid in [parent['pid'], *workers['worker_pids']])):
        raise RuntimeError('No concurrent or restarted score experiment')
    oldroot = project/'out/v5/independent_denoising_prior_v1'
    old = json.loads((oldroot/'run.json').read_text())
    products = (oldroot/'products').resolve()
    if (old['status'] != 'independent_TRAIN_material_denoisers_training_finished' or old['completed_steps'] != 3000 or
            old['prior_SHA256'] != 'df440ecc661a005c7e1118ab82682d960d3e1149b7496538cf0231578f15ffeb' or
            Path('/proc', str(old['pid'])).exists()):
        raise RuntimeError('Only the completed original patch bank and fixed final3000step design')
    mask, manifest_holdout, population = eligibility(project, products)
    hashes = dict(old['source_SHA256'])
    for name, sha in hashes.items():
        if digest(source/name) != sha:
            raise RuntimeError('The original training algorithm changed: '+name)
    hashes[Path(__file__).name] = digest(source/Path(__file__).name)
    reserve = 2*2**30+400*2**20
    if shutil.disk_usage(project).free < reserve+3*2**20 or shutil.disk_usage(Path.home()).free < reserve+200*2**20:
        raise RuntimeError('Retain old disk reserves; do not copy the patch bank')
    args.out.mkdir()
    args.artifact_root.mkdir()
    (args.out/'products').symlink_to(args.artifact_root, target_is_directory=True)
    for name in hashes:
        shutil.copy2(source/name, args.artifact_root/name)
    started, deadline = time.time(), time.time()+1800
    run = {'status': 'checking_same_TRAIN_patch_bank_for_field_population', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': deadline, 'max_seconds': 1800,
        'maximum_steps': 3000, 'completed_steps': 0, 'source_SHA256': hashes,
        'parent_root': str(args.parent_root), 'old_prior_root': str(oldroot),
        'original_patch_bank_SHA256': old['patch_bank_SHA256'], 'population': population,
        'training_case': 'Neg_07_Left', 'development_arrays_read': False, 'reserved_arrays_read': False,
        'RF_arrays_read': False, 'new_acoustic_solves': 0, 'new_material_derivative_probes': 0,
        'device': 0, 'allocator_cap_GiB': 8, 'per_card_process_tree_limit_GiB': 16,
        'one_GPU_experiment': True, 'all_closed_budgets_retained': True,
        'joint_reconstruction_success_claimed': False, 'fixed_final_EMA_no_DEV_selection': True}
    atomic(args.out/'run.json', run)
    def stop(number, frame):
        (args.out/'stop_requested.flag').write_text(str(number)+'\n')
    for number in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(number, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    monitor = PressureResources(args.out)
    def guard():
        if (time.time() >= deadline-45 or (args.out/'stop_requested.flag').exists() or
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20 > 8):
            raise RuntimeError('Fixed1800second,CPU8GiB or cooperative limit reached; no budget reset')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Original disk reserves reached')
    try:
        guard()
        if digest(products/'patches.npy') != old['patch_bank_SHA256']:
            raise RuntimeError('Original TRAIN patch values changed')
        bank = np.load(products/'patches.npy', mmap_mode='r')
        scales, holdout = np.load(products/'scales.npy'), np.load(products/'holdout.npy')
        if (bank.shape != (3, 2710, 128, 128) or mask.shape != (3, 2710) or
                not np.array_equal(holdout, manifest_holdout) or not np.array_equal(scales, np.tile([0, 0, 1, 1, 2], 542))):
            raise RuntimeError('Unchanged patch slots and original partition required')
        for name in ['patches.npy', 'scales.npy', 'holdout.npy']:
            (args.artifact_root/name).symlink_to(products/name)
        np.save(args.artifact_root/'eligible_patches.npy', mask)
        atomic(args.out/'population.json', population)
        training = [np.flatnonzero(mask[f]&~holdout) for f in range(3)]
        run.update(training_patches_per_field=[len(v) for v in training],
            holdout_patches_per_field=[int(np.sum(mask[f]&holdout)) for f in range(3)],
            eligible_mask_SHA256=digest(args.artifact_root/'eligible_patches.npy'))
        atomic(args.out/'run.json', run)
        torch.set_num_threads(1)
        while True:
            guard()
            usage = monitor.sample('anatomical_prior_GPU_admission')
            if usage[0]['owned_MiB']+8192+768 <= 16384 and usage[0]['whole_MiB']+8192+768 <= usage[0]['total_MiB']:
                break
            time.sleep(20)
        torch.manual_seed(6110905)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory, 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model = IndependentMaterialDenoisers().to('cuda:0')
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        generators = [np.random.default_rng(6110907+991*f) for f in range(3)]
        run.update(status='training_anatomical_field_populations_same_denoisers', config=model.config,
            parameter_count_per_field=[sum(p.numel() for p in field.parameters()) for field in model.fields])
        atomic(args.out/'run.json', run)
        history, last_sample = [], time.time()
        for step in range(1, 3001):
            guard()
            if step%50 == 1 and any(digest(source/name) != sha for name, sha in hashes.items()):
                raise RuntimeError('Frozen same-algorithm source changed')
            rate = 3e-4*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/3000)))
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for f in range(3):
                rng = generators[f]
                clean, labels = sample(bank[f], scales, training[f], rng, 'cuda:0')
                sigma = torch.from_numpy(np.exp(rng.uniform(math.log(.03), math.log(1.5), len(clean))).astype(np.float32)).to('cuda:0')
                loss = training_loss(model.fields[f], clean, sigma, labels, torch.randn_like(clean))
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite actual training objective')
                loss.backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            with torch.no_grad():
                for dest, value in zip(ema.parameters(), model.parameters()):
                    dest.lerp_(value, .005)
            run.update(completed_steps=step, latest_weighted_denoising_loss=losses, seconds=time.time()-started)
            if step == 1 or time.time()-last_sample >= 20:
                usage = monitor.sample('charged_anatomical_denoising_update')
                if any(v['owned_MiB'] > 16384 for v in usage.values()):
                    raise RuntimeError('Owned tree exceeded16GiB')
                atomic(args.out/'run.json', run)
                last_sample = time.time()
            if step%100 == 0 or step == 1:
                history.append({'step': step, 'losses': losses, 'seconds': time.time()-started})
                atomic(args.out/'training_history.json', {'history': history})
                print(json.dumps(history[-1]), flush=True)
        save(args.artifact_root/'prior.pt', {'state': {k: v.detach().cpu() for k, v in ema.state_dict().items()},
            'config': model.config, 'completed_steps': 3000, 'training_case': 'Neg_07_Left',
            'source_SHA256': hashes, 'population': population, 'development_arrays_read': False,
            'reserved_arrays_read': False, 'same_original_patch_bank_SHA256': old['patch_bank_SHA256']})
        run['status'] = 'fixed_anatomical_prior_once_only_TRAIN_holdout'
        atomic(args.out/'run.json', run)
        validation = []
        with torch.no_grad():
            for f in range(3):
                rng = np.random.default_rng(6110911+991*f)
                for scale in range(3):
                    pool = np.flatnonzero(mask[f]&holdout&(scales == scale))
                    for noise_sigma in [.05, .2, .6]:
                        clean, labels = sample(bank[f], scales, pool, rng, 'cuda:0', batch=8)
                        noise = torch.randn_like(clean)*noise_sigma
                        output = ema.fields[f](clean+noise, noise_sigma, labels)
                        input_mse = float(noise.square().mean())
                        output_mse = float((output-clean).square().mean())
                        validation.append({'field': f, 'scale': scale, 'sigma': noise_sigma,
                            'input_MSE': input_mse, 'denoised_MSE': output_mse, 'MSE_ratio': output_mse/input_mse})
        atomic(args.out/'holdout_denoising.json', {'fixed_TRAIN_material_holdout_only': True,
            'not_patient_independent': True, 'not_imaging_evidence': True, 'rows': validation,
            'all_field_scale_average_MSE_ratios': {f'{f}_{s}': float(np.mean([r['MSE_ratio'] for r in validation
                if r['field'] == f and r['scale'] == s])) for f in range(3) for s in range(3)}})
        torch.cuda.synchronize()
        monitor.sample('anatomical_prior_training_complete')
        run.update(status='anatomical_TRAIN_material_denoisers_training_finished', prior_SHA256=digest(args.artifact_root/'prior.pt'),
            finished_unix_s=time.time(), seconds=time.time()-started,
            peak_host_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            peak_allocated_GiB=torch.cuda.max_memory_allocated(0)/2**30,
            peak_reserved_GiB=torch.cuda.max_memory_reserved(0)/2**30, holdout_evaluated_once=True,
            trained_prior_is_not_reconstruction=True)
        atomic(args.out/'run.json', run)
        print(json.dumps({k: run[k] for k in ['status', 'completed_steps', 'seconds', 'prior_SHA256',
            'peak_host_RSS_GiB', 'peak_allocated_GiB', 'peak_reserved_GiB', 'training_patches_per_field',
            'holdout_patches_per_field']}), flush=True)
    except BaseException:
        run.update(status='anatomical_prior_training_failed_for_inspection', error=traceback.format_exc(),
            seconds=time.time()-started, all_completed_updates_remain_charged=True)
        atomic(args.out/'run.json', run)
        raise
    finally:
        monitor.log.close()


if __name__ == '__main__':
    main()
