"""Full-space auxiliary solve used by actual three-material inverse updates."""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import gc
import hashlib
import json
import os
import shutil
import time
import traceback
import numpy as np
import torch
from .frequency_multilevel_acquisition import LowHalfMHzAcquisition
from .full90_3mhz_acquisition import Full90ThreeMHzAcquisition
from .native_full_source_dual_minres import evaluate_full_shot
from .low_frequency_0p5 import atomic, digest
from . import native_time


def worker(rank, device, jobs, results, plan_path, chunk, cap_gib, folder, nworkers):
    token = -1
    try:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        plan = json.loads(Path(plan_path).read_text())
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(
            cap_gib*2**30/torch.cuda.get_device_properties(device).total_memory, device)
        device = f'cuda:{device}'
        edge_path = Path('out/v5/uno_triparam_v1/known_edge_alpha.npy')
        if digest(edge_path) != plan['known_edge_SHA256']:
            raise RuntimeError('Known edge absorption changed')
        edge = torch.from_numpy(np.broadcast_to(np.load(edge_path), (961, 1121)).copy()).to(device)
        z, x = np.ogrid[:961, :1121]
        mask = torch.from_numpy((z >= 80)&(z < 937)&(x >= 24)&(x < 1097)).to(device)
        cache = {}
        for family in list(plan['families'])[rank::nworkers]:
            spec = plan['families'][family]
            a = (LowHalfMHzAcquisition(plan['data_root'], spec['source'], device, chunk)
                 if spec['centre'] == 'low' else
                 Full90ThreeMHzAcquisition(plan['high_index'], spec['source'], device, chunk))
            if a.observed.shape[:2] != (11, 192):
                raise ValueError('Complete eleven-source acquisition required')
            weights = torch.zeros(len(a.frequencies), dtype=torch.float64, device=device)
            for lo, hi in spec['bands_hz']:
                band = torch.from_numpy((a.frequencies >= lo)&(a.frequencies <= hi)).to(device)
                energy = a.observed[:, :, band].abs().square().sum()
                if float(energy) <= 0:
                    raise ValueError('Zero-energy acquired band')
                weights[band] += 1/(len(spec['bands_hz'])*energy)
            cache[family] = (a, weights, spec['coefficient'], .5e6 if spec['centre'] == 'low' else 3e6)
        atomic(folder/'ready.json', {'pid': os.getpid(), 'rank': rank, 'device': device,
            'families': list(cache), 'patient_truth_read': False, 'native_shape': [3, 961, 1121]})
        while True:
            job = jobs.get()
            if job is None:
                break
            token, kind = job['token'], job['kind']
            if kind not in ['full_dual_gradient', 'full_dual_value']:
                raise ValueError('Only actual full-dual inverse jobs allowed')
            need_gradient = kind == 'full_dual_gradient'
            started = time.time()
            material = torch.from_numpy(job['material']).to(device)
            aggregate = torch.zeros_like(material, dtype=torch.float64) if need_gradient else None
            total = lower = upper = 0.
            details, updated = {}, {}
            all_ready = True
            torch.cuda.reset_peak_memory_stats()
            for family, shot in job['sources']:
                a, weights, coefficient, centre = cache[family]
                task = folder/'tasks'/f'token{token:04d}'/f'{family}_{shot:02d}'
                atomic(folder/'progress.json', {'token': token, 'kind': kind, 'family': family,
                    'shot': shot, 'task_root': str(task), 'state': 'solving_actual_full_source_dual',
                    'started_unix_s': time.time(), 'patient_truth_read': False})
                parameters = job['calibration'][family]
                row = job.get('cached_RF', {}).get(family)
                prediction = None
                if row is not None:
                    if digest(Path(row['path'])) != row['SHA256']:
                        raise RuntimeError('Same-material original RF cache changed')
                    prediction = np.load(row['path'])
                resume = job.get('resume_tasks', {}).get(family)
                reused = False
                if resume:
                    old_task = Path(resume).resolve()
                    old_info = json.loads((old_task/'result.json').read_text())
                    old_work = json.loads((old_task/'native_work.json').read_text())
                    old_manifest = json.loads((old_task/'operator_manifest.json').read_text())
                    if (old_manifest['material_array_SHA256'] != hashlib.sha256(material.cpu().numpy().tobytes()).hexdigest()
                        or old_manifest['source_shot'] != shot or old_manifest['centre_hz'] != centre
                        or old_manifest['lambda_squared'] != parameters['lambda_squared']):
                        raise RuntimeError('Saved source task is not the identical physical inverse system')
                    if old_work['inflight_phase'] is not None or task.exists():
                        raise RuntimeError('Inspect partial continuation instead of repeating it')
                    task.mkdir(parents=True)
                    atomic(task/'inherited_native_work.json', old_work)
                    atomic(task/'continuation_provenance.json', {'source_task': str(old_task),
                        'source_manifest_SHA256': digest(old_task/'operator_manifest.json'),
                        'source_state_SHA256': digest(old_task/'dual_state.pt'),
                        'same_material_and_source': True, 'inherited_work_is_not_counted_twice': True})
                    if old_info['gradient_ready'] and old_info['actual_native_inner_converged']:
                        if not need_gradient or old_info['calibration'] != parameters:
                            raise RuntimeError('Wrong ready-gradient reuse request')
                        reused_gradient = torch.from_numpy(np.load(old_task/'actual_control_gradient.npy')).to(device)
                        info, value = old_info.copy(), old_info['objective']
                        info['gradient_reused_from_same_material_task'] = str(old_task)
                        info['new_native_adjoint_passes'] = 0
                        atomic(task/'result.json', info)
                        atomic(task/'gradient_reference.json', {'path': str(old_task/'actual_control_gradient.npy'),
                            'SHA256': digest(old_task/'actual_control_gradient.npy')})
                        reused = True
                    else:
                        for name in ['dual_state.pt', 'operator_manifest.json']:
                            shutil.copy2(old_task/name, task/name)
                        shutil.copytree(old_task/'operator_actions', task/'operator_actions')
                if not reused:
                    value, g, parameters, info = evaluate_full_shot(material, a, shot, weights,
                        coefficient, centre, parameters['lambda_squared'], task, parameters,
                        need_gradient=need_gradient, cached_prediction=prediction,
                        max_actions=job['max_actions'], relative_gap=job['relative_gap'],
                        warm_start_path=job.get('warm_tasks', {}).get(family))
                details[family] = info
                updated[family] = parameters
                ready = bool(info['actual_native_inner_converged'] and
                             (not need_gradient or info['gradient_ready']))
                all_ready = all_ready and ready
                if ready:
                    total += 11*value
                    lower += 11*info['objective_numerical_lower']
                    upper += 11*info['objective_numerical_upper']
                    if need_gradient:
                        control_gradient = (reused_gradient if reused else torch.stack([
                            .05*material[0]*g[0], .10*material[1]*g[1],
                            .70*(material[2]-edge)*g[2]]).double()*mask)
                        aggregate += 11*control_gradient
                        np.save(task/'actual_control_gradient.npy', control_gradient.cpu().numpy())
                        del control_gradient
                        if reused:
                            del reused_gradient
                        else:
                            del g
                atomic(folder/'progress.json', {'token': token, 'kind': kind, 'family': family,
                    'shot': shot, 'task_root': str(task),
                    'state': 'completed' if ready else 'inner_limit_saved_for_continuation',
                    'completed_unix_s': time.time(), 'actual_native_inner_converged': ready,
                    'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                    'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30})
                native_time._TAPES.clear()
                gc.collect()
                torch.cuda.empty_cache()
            results.put({'rank': rank, 'token': token, 'kind': kind, 'all_inner_ready': all_ready,
                'loss': total if all_ready else None, 'lower': lower if all_ready else None,
                'upper': upper if all_ready else None,
                'gradient': aggregate.cpu().numpy() if need_gradient and all_ready else None,
                'calibration': updated, 'sources': job['sources'], 'details': details,
                'seconds': time.time()-started,
                'peak_Torch_GiB': torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30, 'error': None})
            del aggregate, material, job
            gc.collect()
    except BaseException:
        message = traceback.format_exc()
        atomic(Path(folder)/'error.json', {'token': token, 'rank': rank, 'error': message,
            'all_completed_K_actions_and_actual_work_retained': True})
        results.put({'rank': rank, 'token': token, 'error': message})
        raise
