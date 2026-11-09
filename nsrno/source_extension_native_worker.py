"""Actual true-shot corrected source-extension updates, one source at a time."""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import gc
import json
import time
import traceback
import numpy as np
import torch
from .frequency_multilevel_acquisition import LowHalfMHzAcquisition
from .full90_3mhz_acquisition import Full90ThreeMHzAcquisition
from .low_frequency_0p5 import atomic, digest
from .source_extension_ray import evaluate_shot
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
            raise RuntimeError('Fixed known edge absorption changed')
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
                raise ValueError('All eleven true sources and 192 receivers are required')
            weights = torch.zeros(len(a.frequencies), dtype=torch.float64, device=device)
            for lo, hi in spec['bands_hz']:
                band = torch.from_numpy((a.frequencies >= lo)&(a.frequencies <= hi)).to(device)
                energy = a.observed[:, :, band].abs().square().sum()
                if float(energy) <= 0:
                    raise ValueError('Empty original dense acquisition band')
                weights[band] += 1/(len(spec['bands_hz'])*energy)
            cache[family] = (a, weights, spec['coefficient'], .5e6 if spec['centre'] == 'low' else 3e6)
        counters = {'completed_true_source_contributions': 0,
            'adjoint_direction_passes_including_checkpoint_replays': 0,
            'physical_forward_passes': 0, 'augmented_forward_passes': 0,
            'ordinary_physical_checkpoint_replay_passes': 0,
            'augmented_checkpoint_replay_passes': 0}
        atomic(folder/'ready.json', {'rank': rank, 'pid': __import__('os').getpid(),
            'device': device, 'families': list(cache), 'native_shape': [3, 961, 1121],
            'only_true_sources': True, 'patient_truth_read': False,
            'no_source_extension_propagation_before_actual_inverse_job': True})
        while True:
            job = jobs.get()
            if job is None:
                break
            token, kind = job['token'], job['kind']
            if kind not in ['ray_gradient', 'ray_value']:
                raise ValueError('Only actual inverse gradient and line values are allowed')
            need_gradient = kind == 'ray_gradient'
            started = time.time()
            material = torch.from_numpy(job['material']).to(device)
            total = 0.
            aggregate = torch.zeros_like(material, dtype=torch.float64) if need_gradient else None
            details, calibration = {}, {}
            torch.cuda.reset_peak_memory_stats()
            for family, shot in job['sources']:
                a, weights, coefficient, centre = cache[family]
                atomic(folder/'progress.json', {'token': token, 'kind': kind, 'family': family,
                    'shot': shot, 'state': 'actual_corrected_ray_gradient' if need_gradient else 'actual_ray_line_value',
                    'unix_s': time.time(), 'reserved_adjoint_direction_passes': 5 if need_gradient else 1,
                    'partial_failure_work_is_charged_by_issued_ledger': True})
                value, g, parameters, row = evaluate_shot(material, a, shot, weights,
                    coefficient, centre, job['calibration'].get(family), need_gradient)
                total += 11*value
                calibration[family] = parameters
                row['unbiased_uniform_true_source_estimator_factor'] = 11
                row['family'] = family
                details[family] = row
                if need_gradient:
                    control_gradient = torch.stack([.05*material[0]*g[0],
                        .10*material[1]*g[1], .70*(material[2]-edge)*g[2]]).double()*mask
                    aggregate += 11*control_gradient
                    path = folder/f'token{token:04d}_{family}_{shot:02d}_control_gradient.npy'
                    np.save(path, control_gradient.detach().cpu().numpy())
                    del control_gradient, g
                for name in counters:
                    counters[name] += 1 if name == 'completed_true_source_contributions' else row[name]
                atomic(folder/'completed_propagations.json', {'token': token, **counters})
                atomic(folder/f'token{token:04d}_{family}_{shot:02d}.json', row)
                atomic(folder/'progress.json', {'token': token, 'kind': kind, 'family': family,
                    'shot': shot, 'state': 'completed', 'unix_s': time.time(),
                    'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                    'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30})
                native_time._TAPES.clear()
                gc.collect()
                torch.cuda.empty_cache()
            results.put({'rank': rank, 'token': token, 'kind': kind, 'loss': total,
                'gradient': aggregate.cpu().numpy() if need_gradient else None,
                'calibration': calibration, 'sources': job['sources'], 'details': details,
                'completed_counters': dict(counters), 'seconds': time.time()-started,
                'peak_Torch_GiB': torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30, 'error': None})
            del aggregate, material, job
            gc.collect()
    except BaseException:
        message = traceback.format_exc()
        atomic(Path(folder)/'error.json', {'rank': rank, 'token': token, 'error': message,
            'issued_work_remains_charged_even_if_this_source_did_not_finish': True})
        results.put({'rank': rank, 'token': token, 'error': message})
        raise
