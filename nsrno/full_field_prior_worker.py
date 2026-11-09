"""True all44 RF gradients and JVPs for the independent-prior material update."""
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
from . import native_time
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
        edge_path = Path('out/v5/uno_triparam_v1/known_edge_alpha.npy')
        if digest(edge_path) != plan['known_edge_SHA256']:
            raise RuntimeError('Known absorption changed')
        edge = torch.from_numpy(np.broadcast_to(np.load(edge_path), (961, 1121)).copy()).to(device)
        z, x = np.ogrid[:961, :1121]
        active = torch.from_numpy((z >= 80)&(z < 937)&(x >= 24)&(x < 1097)).to(device)
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
        atomic(folder/'ready.json', {'pid': os.getpid(), 'rank': rank, 'device': device,
            'all_four_families_loaded': True, 'no_source_encoding': True, 'patient_truth_read': False})
        while True:
            job = jobs.get()
            if job is None:
                break
            token, kind = job['token'], job['kind']
            if kind not in ['original44_gradient', 'joint_tangents', 'original44_forward']:
                raise ValueError('Only actual joint inverse products or its physical candidate')
            material = torch.from_numpy(job['material']).to(device)
            material.requires_grad_(kind == 'original44_gradient')
            directions = np.load(job['directions']) if kind == 'joint_tangents' else None
            n = len(directions) if directions is not None else 0
            aggregate = (torch.zeros_like(material, dtype=torch.float64)
                         if kind == 'original44_gradient' else None)
            gradient_path = folder/f'aggregate_control_gradient_token{token:04d}.npy'
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
                    for j, direction in enumerate(directions):
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
                    J = torch.stack(columns)
                    weighted = J*weights[None, None]
                    gram = 2*coefficient*torch.einsum('arf,brf->ab', J.conj(), weighted).real
                    slope = -2*coefficient*torch.einsum('rf,arf->a', residual.conj(), weighted).real
                    Hg, lg = gram.cpu().numpy(), slope.cpu().numpy()
                    H += Hg
                    linear += lg
                    groups[name]['H'] += Hg
                    groups[name]['linear'] += lg
                    del J, columns, weighted
                elif kind == 'original44_gradient':
                    with torch.enable_grad():
                        raw = work.run(f'token{token}_{name}_{shot:02d}_gradient_background',
                            lambda: native_wave(material, wave, a.si, a.sw, a.ri, a.rw, a.config),
                            physical=1)
                        prediction = a.scattered_coefficients(raw, shot)
                        residual = a.observed[shot]-prediction
                        loss_tensor = coefficient*(weights[None]*residual.abs().square()).sum()
                        work.run(f'token{token}_{name}_{shot:02d}_physical_material_adjoint',
                            lambda: loss_tensor.backward(), adjoint=1)
                    with torch.no_grad():
                        block = torch.stack([.05*material[0]*material.grad[0],
                            .10*material[1]*material.grad[1],
                            .70*(material[2]-edge)*material.grad[2]]).double()*active
                        aggregate += block
                        save_array(gradient_path, aggregate.cpu().numpy())
                    material.grad = None
                    del raw, loss_tensor, block
                else:
                    path = archive/f'{name}_{shot:02d}.npy'
                    def action():
                        with torch.no_grad():
                            raw = native_wave(material, wave, a.si, a.sw, a.ri, a.rw, a.config)
                            result = a.scattered_coefficients(raw, shot).detach()
                        save_array(path, result.cpu().numpy())
                        return result
                    prediction = work.run(f'token{token}_{name}_{shot:02d}_original_forward', action, physical=1)
                    residual = a.observed[shot]-prediction
                loss = float((coefficient*(weights[None]*residual.abs().square()).sum()).detach())
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
            if kind == 'original44_gradient':
                atomic(folder/f'gradient_receipt_token{token:04d}.json',
                    {'path': str(gradient_path), 'SHA256': digest(gradient_path),
                     'completed_sources': job['sources'], 'independent_physical_coordinates': True})
            results.put({'rank': rank, 'token': token, 'kind': kind, 'loss': total,
                'H': H, 'linear': linear, 'groups': groups, 'actual_work': dict(work.state),
                'seconds': time.time()-started, 'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30,
                'sources': job['sources'], 'error': None})
            del material, directions, aggregate, job
            native_time._TAPES.clear()
            gc.collect()
            torch.cuda.empty_cache()
    except BaseException:
        message = traceback.format_exc()
        atomic(Path(folder)/'error.json', {'rank': rank, 'token': token, 'error': message,
                                         'partial_products_and_all_work_retained': True})
        results.put({'rank': rank, 'token': token, 'error': message})
        raise
