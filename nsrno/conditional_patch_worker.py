"""Three new spatial-prior directions and one exactly reused all44 data direction."""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import gc
import json
import os
import time
import traceback
import numpy as np
import torch
from .frequency_multilevel_acquisition import LowHalfMHzAcquisition
from .full90_3mhz_acquisition import Full90ThreeMHzAcquisition
from .native_tangent import tangent
from .native_time import native_wave
from .native_full_source_dual import NativeWork
from .low_frequency_0p5 import atomic, digest


def save_array(path, value):
    tmp = path.with_name(path.stem+'.partial.npy')
    np.save(tmp, value)
    tmp.replace(path)


def worker(rank, device, jobs, results, plan_path, folder, chunk, cap_gib, deadline):
    token = -1
    try:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        plan = json.loads(Path(plan_path).read_text())
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(cap_gib*2**30/
            torch.cuda.get_device_properties(device).total_memory, device)
        device = f'cuda:{device}'
        work = NativeWork(folder)
        for prefix in ['issued', 'completed']:
            work.state.setdefault(prefix+'_tangent_direction_passes', 0)
        cache = {}
        for name, spec in plan['families'].items():
            a = (LowHalfMHzAcquisition(plan['data_root'], spec['source'], device, chunk)
                 if spec['centre'] == 'low' else
                 Full90ThreeMHzAcquisition(plan['high_index'], spec['source'], device, chunk))
            weights = torch.zeros(len(a.frequencies), dtype=torch.float64, device=device)
            for lo, hi in spec['bands_hz']:
                mask = torch.from_numpy((a.frequencies >= lo)&(a.frequencies <= hi)).to(device)
                energy = a.observed[:, :, mask].abs().square().sum()
                if float(energy) <= 0:
                    raise ValueError('Zero acquired band energy')
                weights[mask] += 1/(len(spec['bands_hz'])*energy)
            cache[name] = a, weights, spec['coefficient']
        refs = {(r['family'], r['shot']): r for r in plan['reference_RF']}
        reused = {(r['family'], r['shot']): r for r in plan['reused_data_JVPs']}
        if digest(plan['old_physical_directions']) != plan['old_physical_directions_SHA256']:
            raise RuntimeError('Old data-direction provenance changed')
        atomic(folder/'ready.json', {'pid': os.getpid(), 'rank': rank, 'device': device,
            'all_four_families_loaded': True, 'no_source_encoding': True, 'patient_truth_read': False})
        while True:
            job = jobs.get()
            if job is None:
                break
            token, kind = job['token'], job['kind']
            if kind not in ['joint_tangents', 'original44_forward']:
                raise ValueError('Only actual joint inverse products or its physical candidate')
            material = torch.from_numpy(job['material']).to(device)
            directions = np.load(job['directions']) if kind == 'joint_tangents' else None
            n = len(directions) if directions is not None else 0
            reused_sources = 0
            if directions is not None:
                if n != 4 or digest(job['directions']) != plan['new_physical_directions_SHA256']:
                    raise RuntimeError('New direction pack changed')
                if not np.array_equal(directions[3], np.load(plan['old_physical_directions'], mmap_mode='r')[3]):
                    raise RuntimeError('Cannot reuse a JVP after changing its physical direction')
            H, linear = np.zeros((n, n)), np.zeros(n)
            groups = {name: {'loss': 0., 'H': np.zeros((n, n)), 'linear': np.zeros(n), 'shots': []}
                      for name in cache}
            total, started = 0., time.time()
            archive = folder/'arrays'/f'token{token:04d}'
            archive.mkdir(parents=True)
            torch.cuda.reset_peak_memory_stats()
            for name, shot in job['sources']:
                if time.time() >= deadline-60 or (folder.parent.parent/'stop_requested.flag').exists():
                    raise RuntimeError('Inherited absolute deadline reached; partial products retained')
                a, weights, coefficient = cache[name]
                wave = a.waveform(shot)
                atomic(folder/'progress.json', {'token': token, 'kind': kind, 'family': name,
                    'shot': shot, 'state': 'actual_inverse_propagation', 'started_unix_s': time.time()})
                if kind == 'joint_tangents':
                    row = refs[name, shot]
                    if digest(row['path']) != row['SHA256']:
                        raise RuntimeError('Accepted same-material RF cache changed')
                    prediction = torch.from_numpy(np.load(row['path'])).to(device)
                    residual = a.observed[shot]-prediction
                    columns = []
                    for j, direction in enumerate(directions[:3]):
                        if time.time() >= deadline-60 or (folder.parent.parent/'stop_requested.flag').exists():
                            raise RuntimeError('Inherited deadline reached before another required JVP')
                        delta = torch.from_numpy(direction).to(device)
                        path = archive/f'{name}_{shot:02d}_J{j:02d}.npy'
                        if path.exists():
                            raise RuntimeError('Inspect saved products instead of automatically repeating a JVP')
                        work.state['issued_tangent_direction_passes'] += 1
                        def action():
                            raw, draw, cp = tangent(material, delta, wave, a.si, a.sw,
                                a.ri, a.rw, a.config, save_checkpoints=False)
                            column = a.coefficients(draw).detach()
                            save_array(path, column.cpu().numpy())
                            work.state['completed_tangent_direction_passes'] += 1
                            del raw, draw, cp
                            return column
                        column = work.run(f'token{token}_{name}_{shot:02d}_J{j:02d}', action, physical=1)
                        columns.append(column)
                        del delta
                    reuse = reused[name, shot]
                    if digest(reuse['path']) != reuse['SHA256']:
                        raise RuntimeError('Same-material old data JVP changed')
                    old_column = torch.from_numpy(np.load(reuse['path'])).to(device)
                    if old_column.shape != prediction.shape:
                        raise RuntimeError('Cached data JVP has the wrong acquisition shape')
                    columns.append(old_column)
                    reused_sources += 1
                    J = torch.stack(columns)
                    weighted = J*weights[None, None]
                    gram = 2*coefficient*torch.einsum('arf,brf->ab', J.conj(), weighted).real
                    slope = -2*coefficient*torch.einsum('rf,arf->a', residual.conj(), weighted).real
                    Hg, lg = gram.cpu().numpy(), slope.cpu().numpy()
                    H += Hg
                    linear += lg
                    groups[name]['H'] += Hg
                    groups[name]['linear'] += lg
                    del J, columns, weighted, old_column
                else:
                    path = archive/f'{name}_{shot:02d}.npy'
                    if path.exists():
                        raise RuntimeError('Do not repeat an already stored physical candidate')
                    def action():
                        with torch.no_grad():
                            raw = native_wave(material, wave, a.si, a.sw, a.ri, a.rw, a.config)
                            result = a.scattered_coefficients(raw, shot).detach()
                        save_array(path, result.cpu().numpy())
                        return result
                    prediction = work.run(f'token{token}_{name}_{shot:02d}_original_forward', action, physical=1)
                    residual = a.observed[shot]-prediction
                loss = float(coefficient*(weights[None]*residual.abs().square()).sum())
                total += loss
                groups[name]['loss'] += loss
                groups[name]['shots'].append(shot)
                atomic(folder/'progress.json', {'token': token, 'kind': kind, 'family': name,
                    'shot': shot, 'state': 'source_completed', 'completed_unix_s': time.time(),
                    'source_RF_loss': loss, 'actual_work': work.state})
                del residual, prediction, wave
            for group in groups.values():
                group['H'] = group['H'].tolist()
                group['linear'] = group['linear'].tolist()
            results.put({'rank': rank, 'token': token, 'kind': kind, 'loss': total,
                'H': H, 'linear': linear, 'groups': groups, 'actual_work': dict(work.state),
                'seconds': time.time()-started, 'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30,
                'sources': job['sources'], 'reused_tangent_sources': reused_sources, 'error': None})
            del material, directions, job
            gc.collect()
            torch.cuda.empty_cache()
    except BaseException:
        message = traceback.format_exc()
        atomic(Path(folder)/'error.json', {'rank': rank, 'token': token, 'error': message,
                                         'partial_products_and_all_work_retained': True})
        results.put({'rank': rank, 'token': token, 'error': message})
        raise
