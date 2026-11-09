"""One source-streamed worker for actual low-frequency native RF inversion."""
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
from .acquisition_single_case_0p5 import SingleCaseLowAcquisition
from .low_frequency_0p5 import atomic
from .native_time import native_wave


def worker(rank, device, jobs, results, root, band, chunk, cap_gib, folder, workers):
    token = -1
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(
            cap_gib * 2**30 / torch.cuda.get_device_properties(device).total_memory, device)
        device = f'cuda:{device}'
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        edge = torch.from_numpy(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')).to(device)
        cache = {}
        for family in ['lf', 'point3']:
            a = SingleCaseLowAcquisition(root, family, device=device, chunk=chunk)
            keep = (a.frequencies >= band[0]) & (a.frequencies <= band[1])
            keep = torch.from_numpy(keep).to(device)
            denominator = a.observed[:, :, keep].abs().square().sum()
            if not float(denominator) > 0:
                raise ValueError('The actual target has zero selected-band RF energy')
            cache[family] = a, keep, denominator
        atomic(folder / 'ready.json', {
            'rank': rank, 'device': device, 'band_hz': list(band),
            'shots_per_family': 11, 'receivers': 192, 'truth_read': False,
            'source_fixed_and_checked_against_actual_generation': True,
            'normalization_energy': {k: float(v[2]) for k, v in cache.items()},
            'receiver_implementation': 'matrix-free double FFT, exact declared operator'})
        while True:
            job = jobs.get()
            if job is None:
                break
            token = job['token']
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            m = torch.from_numpy(job['material']).to(device)
            m.requires_grad_(True)
            groups = {}
            total_loss = 0.
            for family, (a, keep, denominator) in cache.items():
                error = 0.
                shots = []
                for shot in range(rank, 11, workers):
                    atomic(folder / 'progress.json', {
                        'token': token, 'family': family, 'shot': shot,
                        'state': 'actual_forward_and_material_gradient', 'unix_s': time.time()})
                    wave = a.waveform(shot)
                    raw = native_wave(m, wave, a.si, a.sw, a.ri, a.rw, a.config)
                    prediction = a.coefficients(raw) - a.reference[shot]
                    difference = prediction[:, keep] - a.observed[shot][:, keep]
                    # Advanced indexing of the receiver/frequency view must
                    # retain the receiver-first layout.
                    if difference.shape != prediction[:, keep].shape:
                        raise ValueError('Selected frequency layout changed')
                    e2 = difference.abs().square().sum()
                    loss = .25 * e2 / denominator
                    archive = folder / 'RF' / f'token{token:04d}'
                    archive.mkdir(parents=True, exist_ok=True)
                    temporary = archive / f'{family}_{shot:02d}.partial.npy'
                    np.save(temporary, prediction.detach().cpu().numpy())
                    temporary.replace(archive / f'{family}_{shot:02d}.npy')
                    loss.backward()
                    total_loss += float(loss.detach())
                    error += float(e2.detach())
                    shots.append(shot)
                    atomic(folder / 'progress.json', {
                        'token': token, 'family': family, 'shot': shot,
                        'state': 'completed', 'seconds': time.time() - started,
                        'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
                        'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30})
                    del wave, raw, prediction, difference, e2, loss
                groups[family] = {'error_squared': error, 'energy': float(denominator),
                                  'completed_shots': shots}
            mask = (torch.arange(m.shape[1], device=device)[:, None] >= 80)
            gradient = torch.stack([
                .05 * (m[0] * m.grad[0] - m[1] * m.grad[1]),
                .10 * m[1] * m.grad[1],
                .70 * (m[2] - edge) * m.grad[2]]) * mask
            result = {
                'rank': rank, 'token': token, 'loss': total_loss, 'groups': groups,
                'gradient': gradient.detach().double().cpu().numpy(),
                'seconds': time.time() - started,
                'peak_Torch_GiB': torch.cuda.max_memory_allocated() / 2**30,
                'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30,
                'error': None}
            results.put(result)
            del result, gradient, m, job
            gc.collect()
    except BaseException:
        message = traceback.format_exc()
        atomic(Path(folder) / 'error.json', {'rank': rank, 'token': token, 'error': message})
        results.put({'rank': rank, 'token': token, 'error': message})
        raise
