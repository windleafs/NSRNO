"""Train independent spatial material denoisers from existing TRAIN fields only.

Single bounded GPU experiment; no RF reader, acoustic solves or DEV material.
The first real update also supplies the resource observation and stays charged.
"""
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
import resource
import shutil
import signal
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from .independent_denoising_prior import IndependentMaterialDenoisers, training_loss
from .pressure_resources import PressureResources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix('.partial.pt')
    torch.save(value, tmp)
    tmp.replace(path)


def parent_identity(row):
    return row.get('parent_id', row['id'])


def is_holdout(row):
    # Only an in-TRAIN material partition; not independent patient validation.
    key = parent_identity(row)
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)%5 == 0


def prepare(tasks, products, known_edge, guard, progress):
    count = len(tasks)*5
    bank = np.lib.format.open_memmap(products/'patches.npy', mode='w+', dtype=np.float16,
                                     shape=(3, count, 128, 128))
    scales = np.tile(np.array([0, 0, 1, 1, 2], np.int64), len(tasks))
    holdout = np.repeat(np.array([is_holdout(t[1]) for t in tasks], bool), 5)
    randoms = [np.random.default_rng(6110903+997*f) for f in range(3)]
    manifest = []
    for index, (kind, row, paths) in enumerate(tasks):
        guard()
        if row['case'] != 'Neg_07_Left' or row['role'] != 'train':
            raise RuntimeError('TRAIN isolation failed before array read')
        medium = np.load(paths[0], mmap_mode='r')
        if kind == 'base':
            alpha = np.load(paths[1], mmap_mode='r')
            if medium.shape != (2, 961, 1121) or alpha.shape != (961, 1121):
                raise ValueError('Existing TRAIN native fields required')
            fields = [medium[0], medium[1], alpha]
        else:
            if medium.shape != (3, 961, 1121):
                raise ValueError('Existing three-field TRAIN augmentation required')
            fields = list(medium)
        for field in range(3):
            full = np.asarray(fields[field][80:921, 178:943], np.float32).copy()
            if field == 0:
                full = (full-1540.)/80.
            elif field == 1:
                full = (full-1000.)/100.
            else:
                full -= known_edge[80:921, 178:943]
                if np.any(full <= 0):
                    raise ValueError('Positive TRAIN tissue attenuation required')
                full = np.log(full/.5)/.7
            if not np.isfinite(full).all():
                raise ValueError('Finite TRAIN features required')
            rng = randoms[field]
            for slot, width in enumerate([128, 128, 512, 512]):
                z = int(rng.integers(full.shape[0]-width+1))
                x = int(rng.integers(full.shape[1]-width+1))
                tile = full[z:z+width, x:x+width]
                if width == 512:
                    tile = tile.reshape(128, 4, 128, 4).mean((1, 3))
                bank[field, 5*index+slot] = tile
            bank[field, 5*index+4] = F.interpolate(torch.from_numpy(full)[None, None],
                size=(128, 128), mode='bilinear', align_corners=False)[0, 0].numpy()
        manifest.append({'id': row['id'], 'case': row['case'], 'role': row['role'],
            'kind': kind, 'parent': parent_identity(row), 'holdout': bool(is_holdout(row)),
            'paths': [str(v) for v in paths]})
        if kind == 'base':
            del alpha
        del fields, medium
        if (index+1)%40 == 0:
            bank.flush()
            progress(index+1)
    bank.flush()
    np.save(products/'scales.npy', scales)
    np.save(products/'holdout.npy', holdout)
    atomic(products/'training_sources.json', {'sources': manifest,
        'only_case': 'Neg_07_Left', 'development_arrays_read': False,
        'reserved_arrays_read': False, 'RF_arrays_read': False,
        'material_holdout_is_not_patient_independent': True,
        'unique_anatomy_leakage_across_holdout_not_excluded': True,
        'independent_field_crop_random_seeds': True})
    return bank, scales, holdout


def sample(bank, scales, indices, rng, device, batch=4):
    selection = rng.choice(indices, batch, replace=len(indices)<batch)
    x = np.asarray(bank[selection], np.float32).copy()
    labels = scales[selection]
    for i in range(batch):
        if rng.random() < .5:
            x[i] = x[i, :, ::-1]
        if labels[i] != 2 and rng.random() < .5:
            x[i] = x[i, ::-1, :]
    clean = torch.from_numpy(x[:, None].copy()).to(device)
    labels = torch.from_numpy(labels.copy()).to(device)
    return clean, labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--parent-root', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--max-seconds', type=float, default=1800)
    a = p.parse_args()
    project, source = Path.cwd().resolve(), Path(__file__).parent
    out, products = a.out.resolve(), a.artifact_root.resolve()
    if (out.exists() or products.exists() or not out.is_relative_to(project/'out/v5') or
        not products.is_relative_to(Path.home()/'nsrno_artifacts') or
        a.device != 0 or a.steps != 3000 or a.max_seconds != 1800):
        raise ValueError('Only the declared fresh3000step/1800s TRAIN prior experiment is allowed')
    old = json.loads((a.parent_root/'run.json').read_text())
    workers = json.loads((a.parent_root/'worker_processes.json').read_text())
    if (old['status'] != 'projected_common_descent_native_stage_finished' or
        old['accepted_updates'] != 3 or
        any(Path('/proc', str(pid)).exists() for pid in [old['pid'], *workers['worker_pids']])):
        raise RuntimeError('Wait for the finished fixed three-step native trajectory to exit')
    qualification = json.loads(a.qualification.read_text())
    if (qualification['status'] != 'new_denoising_residual_energy_synthetic_CPU_derivative_passed' or
        qualification['source_SHA256'] != digest(source/'independent_denoising_prior.py')):
        raise RuntimeError('The new energy chain derivative has not been qualified')
    base_index = project/'out/v5/forward_cache_v1/index.json'
    aug_index = project/'out/v5/uno_triparam_v1/augmentation_index.json'
    bases = [r for r in json.loads(base_index.read_text())['records'] if r['role'] == 'train']
    aug = [r for r in json.loads(aug_index.read_text())['records'] if r['role'] == 'train']
    augmented = {r['material3']: r for r in aug}
    if (len(bases) != 510 or len(augmented) != 32 or
        {r['case'] for r in bases+list(augmented.values())} != {'Neg_07_Left'}):
        raise RuntimeError('Unexpected existing TRAIN scope; no patient arrays have been read')
    tasks = [('base', r, [base_index.parent/r['medium'],
        project/f'out/v5/uno_triparam_v1/alpha/{r["id"]}.npy']) for r in bases]
    tasks += [('augmentation', r, [Path(path).resolve()]) for path, r in sorted(augmented.items())]
    reserve = 2*2**30+400*2**20
    if shutil.disk_usage(project).free < reserve+2*2**20 or shutil.disk_usage(Path.home()).free < reserve+800*2**20:
        raise RuntimeError('Preserve project and HOME disk reserves')
    torch.set_num_threads(1)
    names = [Path(__file__).name, 'independent_denoising_prior.py', 'pressure_resources.py', 'wri_worker.py']
    hashes = {n:digest(source/n) for n in names}
    out.mkdir()
    products.mkdir()
    (out/'products').symlink_to(products, target_is_directory=True)
    for name in names:
        shutil.copy2(source/name, products/name)
    started = time.time()
    deadline = started+a.max_seconds
    run = {'status': 'preparing_existing_TRAIN_material_denoising_bank', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': deadline, 'max_seconds': 1800,
        'maximum_steps': 3000, 'completed_steps': 0, 'source_SHA256': hashes,
        'training_case': 'Neg_07_Left', 'development_arrays_read': False,
        'reserved_arrays_read': False, 'RF_arrays_read': False,
        'new_acoustic_solves': 0, 'new_material_derivative_probes': 0,
        'base_material_records': 510, 'augmented_unique_materials': 32,
        'device': 0, 'allocator_cap_GiB': 8, 'per_card_process_tree_limit_GiB': 16,
        'one_GPU_experiment': True, 'parent_root': str(a.parent_root.resolve()),
        'all_previous_closed_budgets_retained': True,
        'joint_reconstruction_success_claimed': False,
        'checkpoint_selection': 'fixed final EMA0.995,no DEV selection',
        'material_holdout_is_not_patient_independent': True}
    atomic(out/'run.json', run)
    monitor = PressureResources(out)
    def stop(number, frame):
        (out/'stop_requested.flag').write_text(str(number)+'\n')
    for number in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(number, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def guard():
        run['peak_host_RSS_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        if time.time() >= deadline-45 or (out/'stop_requested.flag').exists():
            raise InterruptedError('Fixed prior preparation deadline or requested stop')
        if run['peak_host_RSS_GiB'] > 8:
            raise RuntimeError('Declared hostRSS8GiB guard')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Existing disk reserve reached')
    def progress(number):
        run['completed_training_materials'] = number
        atomic(out/'run.json', run)
    model = ema = optimizer = None
    try:
        edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
        known_edge = np.broadcast_to(np.load(edge_path, mmap_mode='r'), (961, 1121))
        bank, scales, holdout = prepare(tasks, products, known_edge, guard, progress)
        run.update(completed_training_materials=len(tasks), patch_bank_shape=list(bank.shape),
            training_patches_per_field=int((~holdout).sum()), holdout_patches_per_field=int(holdout.sum()),
            index_SHA256={str(i):digest(i) for i in [base_index, aug_index]},
            known_edge_SHA256=digest(edge_path), patch_bank_SHA256=digest(products/'patches.npy'))
        atomic(out/'run.json', run)
        # Bounded admission, respecting foreign GPU jobs without stopping them.
        while True:
            guard()
            devices = monitor.sample('denoising_prior_GPU_admission')
            if (devices[0]['owned_MiB']+8*1024+768 <= 16384 and
                devices[0]['whole_MiB']+8*1024+768 <= devices[0]['total_MiB']):
                break
            time.sleep(min(20, max(1, deadline-time.time()-45)))
        torch.manual_seed(6110905)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory, 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = 'cuda:0'
        model = IndependentMaterialDenoisers().to(device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        run.update(status='training_independent_TRAIN_material_denoisers', config=model.config,
            parameter_count_per_field=[sum(p.numel() for p in f.parameters()) for f in model.fields])
        atomic(out/'run.json', run)
        generators = [np.random.default_rng(6110907+991*f) for f in range(3)]
        training = np.flatnonzero(~holdout)
        history = []
        stop_reason = 'declared3000steps_completed'
        last_sample = time.time()
        for step in range(1, 3001):
            if time.time() >= deadline-60 or (out/'stop_requested.flag').exists():
                stop_reason = 'declared1800seconds_or_cooperative_stop'
                break
            if step%50 == 1:
                guard()
                if any(digest(source/n) != sha for n, sha in hashes.items()):
                    raise RuntimeError('Frozen training source changed')
            rate = 3e-4*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/3000)))
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for f in range(3):
                rng = generators[f]
                clean, labels = sample(bank[f], scales, training, rng, device)
                sigma = torch.from_numpy(np.exp(rng.uniform(math.log(.03), math.log(1.5), len(clean))).astype(np.float32)).to(device)
                noise = torch.randn_like(clean)
                loss = training_loss(model.fields[f], clean, sigma, labels, noise)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite actual denoising training objective')
                loss.backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            with torch.no_grad():
                for dest, value in zip(ema.parameters(), model.parameters()):
                    dest.lerp_(value, .005)
            run.update(completed_steps=step, latest_weighted_denoising_loss=losses,
                seconds=time.time()-started,
                peak_allocated_GiB=torch.cuda.max_memory_allocated(0)/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved(0)/2**30)
            if step == 1 or time.time()-last_sample >= 20:
                usage = monitor.sample('first_charged_denoising_update' if step == 1 else 'denoising_prior_training')
                if any(v['owned_MiB'] > 16384 for v in usage.values()):
                    raise RuntimeError('Owned process tree exceeded16GiB')
                run['peak_host_RSS_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
                atomic(out/'run.json', run)
                last_sample = time.time()
            if step%100 == 0 or step == 1:
                history.append({'step':step,'losses':losses,'seconds':time.time()-started})
                atomic(out/'training_history.json', {'history':history})
                print(json.dumps(history[-1]), flush=True)
            if step%500 == 0:
                save(products/'latest.pt', {'state':ema.state_dict(), 'config':model.config,
                    'completed_steps':step, 'source_SHA256':hashes,
                    'development_arrays_read':False, 'reserved_arrays_read':False})
        if run['completed_steps'] < 100:
            raise RuntimeError('Less than100actual updates; do not deploy an untrained prior')
        run.update(status='fixed_final_EMA_prior_once_only_TRAIN_holdout_evaluation', stop_reason=stop_reason)
        atomic(out/'run.json', run)
        save(products/'prior.pt', {'state':{k:v.detach().cpu() for k,v in ema.state_dict().items()},
            'config':model.config, 'completed_steps':run['completed_steps'],
            'source_SHA256':hashes, 'training_case':'Neg_07_Left',
            'development_arrays_read':False, 'reserved_arrays_read':False})
        # One fixed evaluation only, across three fields/scales/noise levels.
        validation = []
        with torch.no_grad():
            for f in range(3):
                rng = np.random.default_rng(6110911+991*f)
                for scale in range(3):
                    pool = np.flatnonzero(holdout&(scales == scale))
                    for noise_sigma in [.05, .2, .6]:
                        clean, labels = sample(bank[f], scales, pool, rng, device, batch=8)
                        noise = torch.randn_like(clean)*noise_sigma
                        output = ema.fields[f](clean+noise, noise_sigma, labels)
                        input_mse = float(noise.square().mean())
                        output_mse = float((output-clean).square().mean())
                        validation.append({'field':f,'scale':scale,'sigma':noise_sigma,
                            'patches':len(clean),'input_MSE':input_mse,'denoised_MSE':output_mse,
                            'MSE_ratio':output_mse/input_mse})
        atomic(out/'holdout_denoising.json', {'fixed_EMA_TRAIN_material_holdout_only':True,
            'not_patient_independent':True,'new_acoustic_solves':0,'rows':validation,
            'all_field_scale_average_MSE_ratios':{
                f'{f}_{s}':float(np.mean([r['MSE_ratio'] for r in validation if r['field']==f and r['scale']==s]))
                for f in range(3) for s in range(3)}})
        torch.cuda.synchronize()
        monitor.sample('material_prior_training_complete')
        run.update(status='independent_TRAIN_material_denoisers_training_finished',
            finished_unix_s=time.time(), seconds=time.time()-started,
            prior_SHA256=digest(products/'prior.pt'),
            peak_host_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            peak_allocated_GiB=torch.cuda.max_memory_allocated(0)/2**30,
            peak_reserved_GiB=torch.cuda.max_memory_reserved(0)/2**30,
            holdout_evaluated_once=True, trained_prior_is_not_reconstruction=True)
        atomic(out/'run.json', run)
        print(json.dumps({k:run[k] for k in ['status','completed_steps','seconds','prior_SHA256',
            'peak_host_RSS_GiB','peak_allocated_GiB','peak_reserved_GiB']}), flush=True)
    except BaseException:
        run.update(status='independent_denoising_prior_training_stopped_for_inspection',
            error=traceback.format_exc(), seconds=time.time()-started,
            partial_products_and_completed_steps_retained=True)
        atomic(out/'run.json', run)
        raise


if __name__ == '__main__':
    main()
