"""Bounded CPU fit of independent conditional spatial priors using TRAIN only."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import resource
import shutil
import signal
import time
import traceback
import numpy as np
from scipy.special import logsumexp
from .conditional_patch_prior import DILATIONS, OFFSETS, LOWER, UPPER


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def save_model(path, state):
    temporary = path.with_name(path.stem+'.partial.npz')
    np.savez(temporary, **state)
    temporary.replace(path)


def fit_model(samples, seed, guard):
    started = time.time()
    center = samples.mean(0)
    centered = samples-center
    covariance = centered.T@centered/max(len(samples)-1, 1)
    eigen, basis = np.linalg.eigh(covariance)
    order = np.argsort(eigen)[::-1]
    eigen, basis = eigen[order], basis[:, order]
    x = np.ascontiguousarray(centered@basis)
    variance_floor = np.maximum(.01*np.maximum(eigen, 0.), 1e-4)
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(len(x))].copy()]
    distance = ((x-means[0])**2).sum(1)
    for _ in range(11):
        guard()
        if distance.sum() <= 1e-30:
            choice = rng.integers(len(x))
        else:
            choice = rng.choice(len(x), p=distance/distance.sum())
        means.append(x[choice].copy())
        distance = np.minimum(distance, ((x-means[-1])**2).sum(1))
    mean = np.asarray(means)
    x_squared = x*x
    for _ in range(10):
        guard()
        distances = x_squared.sum(1)[:, None]-2*x@mean.T+(mean*mean).sum(1)
        label = distances.argmin(1)
        for component in range(12):
            if np.any(label == component):
                mean[component] = x[label == component].mean(0)
    variance = np.broadcast_to(np.maximum(eigen, variance_floor), (12, 25)).copy()
    weight = np.full(12, 1/12.)
    history, previous = [], None
    for iteration in range(80):
        guard()
        precision = 1/variance
        quadratic = (x_squared@precision.T-2*x@(mean*precision).T
                     +(mean*mean*precision).sum(1))
        logp = np.log(weight)-.5*(25*np.log(2*np.pi)+np.log(variance).sum(1)+quadratic)
        normalizer = logsumexp(logp, axis=1)
        score = float(normalizer.mean())
        history.append(score)
        response = np.exp(logp-normalizer[:, None])
        count = np.maximum(response.sum(0), 1e-12)
        mean = response.T@x/count[:, None]
        variance = np.maximum(response.T@x_squared/count[:, None]-mean*mean, variance_floor)
        weight = np.maximum(count, 1e-8*len(x))
        weight /= weight.sum()
        if previous is not None and iteration >= 10 and abs(score-previous) < 1e-6:
            break
        previous = score
    return {'weight': weight, 'mean': mean, 'variance': variance,
            'pca_center': center, 'basis': basis}, {
        'seconds': time.time()-started, 'iterations': len(history),
        'mean_training_log_density_history': history,
        'minimum_variance': float(variance.min()), 'samples': len(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=900.)
    args = parser.parse_args()
    project = Path.cwd().resolve()
    out, artifact = args.out.resolve(), args.artifact_root.resolve()
    if (not out.is_relative_to(project/'out/v5') or
        not artifact.is_relative_to(Path.home()/'nsrno_artifacts') or
        out.exists() or artifact.exists() or args.max_seconds != 900):
        raise ValueError('New bounded TRAIN-only CPU fit output required')
    reserve = 2*2**30+400*2**20
    if shutil.disk_usage(project).free < reserve+2*2**20 or shutil.disk_usage(Path.home()).free < reserve+200*2**20:
        raise RuntimeError('Keep existing disk reserves')
    resource.setrlimit(resource.RLIMIT_AS, (4*2**30, 4*2**30))
    source = Path(__file__).parent
    source_hashes = {name: digest(source/name) for name in
        [Path(__file__).name, 'conditional_patch_prior.py', 'factorized_material_prior.py']}
    base_index = project/'out/v5/forward_cache_v1/index.json'
    augmentation_index = project/'out/v5/uno_triparam_v1/augmentation_index.json'
    base_rows = [r for r in json.loads(base_index.read_text())['records'] if r['role'] == 'train']
    aug_rows = [r for r in json.loads(augmentation_index.read_text())['records'] if r['role'] == 'train']
    if ({r['case'] for r in base_rows} != {'Neg_07_Left'} or
        {r['case'] for r in aug_rows} != {'Neg_07_Left'}):
        raise RuntimeError('Case-level TRAIN isolation failed before any array read')
    augmented = {r['material3']: r for r in aug_rows}
    per_base = 64
    per_aug = round(len(base_rows)*per_base/3/len(augmented))
    count = len(base_rows)*per_base+len(augmented)*per_aug
    out.mkdir()
    artifact.mkdir()
    (out/'products').symlink_to(artifact, target_is_directory=True)
    products = out/'products'
    started = time.time()
    deadline = started+args.max_seconds
    run = {'status': 'collecting_existing_TRAIN_patch_samples', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': deadline,
        'CPU_address_space_limit_GiB': 4, 'max_seconds': args.max_seconds, 'CUDA_used': False,
        'new_acoustic_solves': 0, 'source_case': 'Neg_07_Left', 'development_arrays_read': False,
        'reserved_arrays_read': False, 'RF_arrays_read': False, 'base_records': len(base_rows),
        'augmented_unique_materials': len(augmented), 'patches_per_field_scale': count,
        'source_SHA256': source_hashes, 'completed_models': [], 'joint_imaging_success': False}
    atomic(out/'run.json', run)
    for name in source_hashes:
        shutil.copy2(source/name, products/name)
    def request_stop(number, frame):
        (out/'stop_requested.flag').write_text(str(number)+'\n')
    for number in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(number, request_stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def guard():
        run['peak_CPU_RSS_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        if time.time() >= deadline or run['peak_CPU_RSS_GiB'] > 4:
            raise RuntimeError('Declared CPU prior fit time or memory budget reached')
        if (out/'stop_requested.flag').exists():
            raise InterruptedError('CPU prior fit stop requested; preserve all products')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Existing disk reserve reached')
        if any(digest(source/name) != sha for name, sha in source_hashes.items()):
            raise RuntimeError('Active fitted-prior source changed')
    try:
        samples = np.lib.format.open_memmap(products/'training_patches.npy', mode='w+',
            dtype=np.float32, shape=(3, 3, count, 25))
        randoms = {(f, d): np.random.default_rng(20261108+1000*f+17*d)
                   for f in range(3) for d in DILATIONS}
        known_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
        edge = np.broadcast_to(np.load(known_path, mmap_mode='r'), (961, 1121))
        position = 0
        manifest = []
        tasks = [('base', row, per_base) for row in base_rows]+[
            ('augmentation', row, per_aug) for path, row in sorted(augmented.items())]
        for index, (kind, row, number) in enumerate(tasks):
            guard()
            if kind == 'base':
                medium_path = base_index.parent/row['medium']
                alpha_path = project/f'out/v5/uno_triparam_v1/alpha/{row["id"]}.npy'
                medium = np.load(medium_path, mmap_mode='r')
                alpha = np.load(alpha_path, mmap_mode='r')
                if medium.shape != (2, 961, 1121) or alpha.shape != (961, 1121):
                    raise ValueError('Expected existing TRAIN native c/rho/alpha')
                fields = [medium[0], medium[1], alpha]
                paths = [str(medium_path), str(alpha_path)]
            else:
                material_path = Path(row['material3']).resolve()
                medium = np.load(material_path, mmap_mode='r')
                if medium.shape != (3, 961, 1121):
                    raise ValueError('Expected existing TRAIN native three-field augmentation')
                fields = [medium[0], medium[1], medium[2]]
                paths = [str(material_path)]
            for field in range(3):
                for scale, dilation in enumerate(DILATIONS):
                    rng = randoms[field, dilation]
                    z = rng.integers(80+2*dilation, 921-2*dilation, number)
                    x = rng.integers(178+2*dilation, 943-2*dilation, number)
                    zz = z[:, None]+OFFSETS[None, :, 0]*dilation
                    xx = x[:, None]+OFFSETS[None, :, 1]*dilation
                    values = fields[field][zz, xx].astype(np.float64)
                    if field == 0:
                        values = (values-1540.)/80.
                    elif field == 1:
                        values = (values-1000.)/100.
                    else:
                        values -= edge[zz, xx]
                        if np.any(values <= 0):
                            raise ValueError('Positive TRAIN tissue absorption required')
                        values = np.log(values/.5)/.7
                    if (not np.isfinite(values).all() or np.any(values < LOWER[field]-1e-5) or
                        np.any(values > UPPER[field]+1e-5)):
                        raise ValueError('TRAIN patch exceeds the declared original physical feature domain')
                    samples[field, scale, position:position+number] = values
            manifest.append({'kind': kind, 'id': row['id'], 'case': row['case'],
                             'role': row['role'], 'paths': paths, 'patches_per_model': number})
            position += number
            del fields, medium
            if kind == 'base':
                del alpha
            if (index+1) % 40 == 0:
                samples.flush()
                run.update(completed_training_materials=index+1, collected_patches_per_model=position)
                atomic(out/'run.json', run)
        samples.flush()
        if position != count:
            raise RuntimeError('Incomplete declared TRAIN sample collection')
        atomic(products/'training_sources.json', {'sources': manifest, 'source_case': 'Neg_07_Left',
            'index_SHA256': {str(p): digest(p) for p in [base_index, augmentation_index]},
            'known_edge_SHA256': digest(known_path), 'independent_field_scale_random_seeds': True})
        training_sha = digest(products/'training_patches.npy')
        run.update(status='fitting_independent_TRAIN_conditional_patch_models',
                   training_patches_SHA256=training_sha)
        atomic(out/'run.json', run)
        combined, fit_details = {}, {}
        for field in range(3):
            for scale, dilation in enumerate(DILATIONS):
                guard()
                key = f'f{field}_d{dilation}'
                run['active_model'] = key
                atomic(out/'run.json', run)
                data = np.asarray(samples[field, scale], np.float64)
                state, details = fit_model(data, 20261108+2000*field+19*dilation, guard)
                save_model(products/f'{key}.npz', state)
                combined.update({key+'_'+name: value for name, value in state.items()})
                fit_details[key] = details
                run['completed_models'].append(key)
                run['model_details'] = fit_details
                atomic(out/'run.json', run)
                print(json.dumps({'completed_model': key, 'seconds': details['seconds'],
                    'iterations': details['iterations'], 'peak_CPU_RSS_GiB': run['peak_CPU_RSS_GiB']}), flush=True)
                del data
                gc.collect()
        guard()
        save_model(products/'prior.npz', combined)
        run.update(status='TRAIN_conditional_patch_prior_CPU_fit_finished',
                   prior_SHA256=digest(products/'prior.npz'), seconds=time.time()-started,
                   finished_unix_s=time.time(), prior_fitted_not_reconstruction=True)
        atomic(out/'run.json', run)
        atomic(products/'metadata.json', {**run, 'tail': .1, 'dilations': list(DILATIONS),
            'patch_shape': [5, 5], 'components_per_independent_field_scale': 12,
            'energy': 'sum_fields mean_scales mean_complete_native_patches conditional_NLL_per_24_neighbors',
            'center_marginal_from_same_joint_model': True, 'shared_labels_or_anatomy': False,
            'new_patient_derivative_probes': 0, 'physical_inverse_not_yet_run': True})
        print(json.dumps({k: run[k] for k in ['status', 'seconds', 'prior_SHA256',
            'peak_CPU_RSS_GiB', 'patches_per_field_scale']}), flush=True)
    except BaseException:
        run.update(status='conditional_patch_prior_CPU_fit_stopped_for_inspection',
                   error=traceback.format_exc(), seconds=time.time()-started,
                   completed_models_and_samples_retained=True)
        atomic(out/'run.json', run)
        raise


if __name__ == '__main__':
    main()
