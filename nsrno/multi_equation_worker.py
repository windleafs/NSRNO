"""Bounded original44-source wavefield worker;no material acoustic derivatives."""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable THP')
from pathlib import Path
import gc
import json
import os
import time
import traceback
import numpy as np
import torch
from .native_full_source_dual import NativeWork
from .multi_equation_source_solve import evaluate_source
from .conditional_material_quadratic import native_body
from .frequency_multilevel_acquisition import LowHalfMHzAcquisition
from .full90_3mhz_acquisition import Full90ThreeMHzAcquisition
from .native_time import native_wave
from .low_frequency_0p5 import atomic, digest


def array(path, value):
    temporary = path.with_name(path.stem+'.partial.npy')
    np.save(temporary, value)
    temporary.replace(path)


def worker(rank, device, jobs, results, plan_path, folder, deadline):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        plan = json.loads(Path(plan_path).read_text())
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(device).total_memory, device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        gpu = f'cuda:{device}'
        work = NativeWork(folder)
        for prefix in ['issued', 'completed']:
            work.state[prefix+'_tangent_direction_passes'] = 0
        atomic(work.path, work.state)
        node = native_body()
        acquisitions, square_roots = {}, {}
        for name, spec in plan['families'].items():
            a = (LowHalfMHzAcquisition(plan['data_root'], spec['source'], gpu, 128)
                if spec['centre'] == 'low' else Full90ThreeMHzAcquisition(plan['high_index'], spec['source'], gpu, 128))
            weights = torch.zeros(len(a.frequencies), device=gpu, dtype=torch.float64)
            for lo, hi in spec['bands_hz']:
                mask = torch.from_numpy((a.frequencies >= lo)&(a.frequencies <= hi)).to(gpu)
                energy = a.observed[:, :, mask].abs().square().sum()
                if not float(energy) > 0:
                    raise RuntimeError('Original full observed band has zero energy')
                weights[mask] += 1/(len(spec['bands_hz'])*energy)
            acquisitions[name], square_roots[name] = a, (spec['coefficient']*weights).sqrt()
        entries = [tuple(v) for v in plan['entries']][rank::2]
        reference = {(v['family'], v['shot']): v for v in plan['initial_RF_source_records']}
        predictions, initial_loss = {}, 0.
        for name, shot in entries:
            record = reference[(name, shot)]
            if digest(record['path']) != record['SHA256']:
                raise RuntimeError('Immutable same-point baseline RF changed')
            predictions[(name, shot)] = torch.from_numpy(np.load(record['path'])).to(gpu)
            residual = (acquisitions[name].observed[shot]-predictions[(name, shot)])*square_roots[name][None]
            source_loss = float(residual.abs().square().sum())
            if source_loss <= 1e-24:
                raise RuntimeError('Inspect a zero saved source residual before any wavefield propagation')
            initial_loss += source_loss
        ready = {'kind': 'ready', 'rank': rank, 'pid': os.getpid(), 'initial_RF': initial_loss,
            'sources': entries, 'all_families_loaded': list(acquisitions), 'patient_truth_read': False}
        atomic(folder/'ready.json', ready)
        results.put(ready)
        while True:
            job = jobs.get()
            if job is None:
                return
            kind = job['kind']
            material = torch.from_numpy(job['material']).to(gpu)
            if material.dtype != torch.float32 or material.shape != (3, 961, 1121):
                raise RuntimeError('Independent full native physical material required')
            def guard():
                cutoff = deadline-1200 if kind == 'conditional_statistics' else deadline-60
                if time.time() >= cutoff or (folder.parent.parent/'stop_requested.flag').exists():
                    raise RuntimeError('Original absolute stage limit/cooperative stop;partial work retained')
            rows = []
            if kind == 'conditional_statistics':
                aggregate = np.zeros((8, 961, 1121), np.float64)
                for name, shot in entries:
                    guard()
                    def progress(**values):
                        atomic(folder/'progress.json', {'kind': kind, 'family': name, 'shot': shot,
                            'completed_sources': len(rows), 'unix_s': time.time(), **values})
                    stats, details = evaluate_source(material, acquisitions[name], shot, square_roots[name],
                        predictions[(name, shot)], node, folder/'source_fields'/f'{name}_{shot:02d}',
                        work, guard, progress, max_actions=12, gamma=.1, tolerance=.01)
                    aggregate += stats
                    array(folder/'conditional_statistics.npy', aggregate)
                    rows.append({'family': name, 'shot': shot, **details})
                    atomic(folder/'completed_source_summary.json', {'rows': rows, 'rank': rank,
                        'source_count': len(rows), 'patient_truth_read': False})
                    print(json.dumps({'rank': rank, 'family': name, 'shot': shot,
                        'inner_actions': details['completed_inner_actions'],
                        'inner_actual_relative_residual': details['actual_relative_normal_residual'],
                        'inner_converged': details['actual_inner_converged']}), flush=True)
                    del stats
                    gc.collect()
                    torch.cuda.empty_cache()
                result = {'kind': kind, 'rank': rank, 'rows': rows, 'sources': entries,
                    'statistics_path': str((folder/'conditional_statistics.npy').resolve()),
                    'statistics_SHA256': digest(folder/'conditional_statistics.npy')}
                del aggregate
            elif kind == 'physical_forward':
                archive = folder/'physical_candidate_RF'
                archive.mkdir(exist_ok=False)
                loss, references = 0., []
                for name, shot in entries:
                    guard()
                    a = acquisitions[name]
                    atomic(folder/'progress.json', {'kind': kind, 'family': name, 'shot': shot,
                        'completed_sources': len(references), 'unix_s': time.time()})
                    wave = a.waveform(shot)
                    def forward():
                        with torch.no_grad():
                            raw = native_wave(material, wave, a.si, a.sw, a.ri, a.rw, a.config)
                            return a.scattered_coefficients(raw, shot).detach()
                    prediction = work.run(f'{name}_{shot:02d}_original_physical_candidate', forward, physical=1)
                    residual = (a.observed[shot]-prediction)*square_roots[name][None]
                    loss += float(residual.abs().square().sum())
                    path = archive/f'{name}_{shot:02d}.npy'
                    array(path, prediction.cpu().numpy())
                    references.append({'family': name, 'shot': shot, 'path': str(path.resolve()), 'SHA256': digest(path)})
                    del prediction, residual, wave
                result = {'kind': kind, 'rank': rank, 'sources': entries, 'loss': loss, 'RF_source_records': references}
            else:
                raise RuntimeError('No additional native actions implemented')
            result.update(error=None, peak_allocated_GiB=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved(device)/2**30)
            results.put(result)
            del material, result, job
            gc.collect()
            torch.cuda.empty_cache()
    except BaseException:
        error = traceback.format_exc()
        atomic(folder/'error.json', {'error': error, 'all_issued_work_retained': True})
        results.put({'rank': rank, 'error': error})
        raise
