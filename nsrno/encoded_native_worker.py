"""Prepared native encoded gradients and unchanged original full-source RF checks."""
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
from .encoded_native_acquisition import EncodedNativeAcquisition
from .frequency_multilevel_acquisition import LowHalfMHzAcquisition
from .full90_3mhz_acquisition import Full90ThreeMHzAcquisition
from .low_frequency_0p5 import atomic, digest
from .native_time import native_wave
from . import native_time


def save(path, value):
    temporary = path.with_name(path.stem+'.partial.npy')
    np.save(temporary, value)
    temporary.replace(path)


def worker(rank, device, jobs, results, plan_path, chunk, cap_gib, folder):
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
        edge_path = 'out/v5/uno_triparam_v1/known_edge_alpha.npy'
        if digest(edge_path) != plan['known_edge_SHA256']:
            raise ValueError('Known edge absorption changed')
        edge = torch.from_numpy(np.broadcast_to(np.load(edge_path), (961, 1121)).copy()).to(device)
        z, x = np.ogrid[:961, :1121]
        mask = torch.from_numpy((z>=80)&(z<937)&(x>=24)&(x<1097)).to(device)
        cache = {}
        for family, spec in plan['families'].items():
            acquisition = (LowHalfMHzAcquisition(plan['data_root'], spec['source'], device, chunk)
                if spec['centre']=='low' else
                Full90ThreeMHzAcquisition(plan['high_index'], spec['source'], device, chunk))
            masks = [torch.from_numpy((acquisition.frequencies>=lo)&(acquisition.frequencies<=hi)).to(device)
                     for lo, hi in spec['bands_hz']]
            energies = [acquisition.observed[:, :, band].abs().square().sum() for band in masks]
            weights = torch.zeros(len(acquisition.frequencies), dtype=torch.float64, device=device)
            for band, energy in zip(masks, energies):
                if float(energy)<=0:
                    raise ValueError('Zero energy in original full-family subband')
                weights[band] += 1/(len(masks)*energy)
            cache[family] = {'a':acquisition, 'masks':masks, 'energies':energies,
                'weights':weights, 'coefficient':spec['coefficient'], 'encoder':None}
        references = {(row['family'], int(row['shot'])):row for row in plan['reference_RF']}
        completed_counts = {'encoded_adjoint_propagations':0,
            'encoded_forward_propagations':0, 'original_forward_propagations':0}
        atomic(folder/'ready.json', {'rank':rank, 'device':device,
            'native_shape':[3, 961, 1121], 'patient_truth_read':False,
            'encoded_sources_not_prepared_until_the_first_actual_inverse_job':True})
        while True:
            job = jobs.get()
            if job is None:
                break
            token, kind = job['token'], job['kind']
            if kind not in ['encoded_gradient', 'encoded_forward', 'forward', 'cached_forward']:
                raise ValueError('Unexpected actual inverse job')
            started = time.time()
            encoded = kind.startswith('encoded_')
            need_gradient = kind=='encoded_gradient'
            cached = kind=='cached_forward'
            material = None if cached else torch.from_numpy(job['material']).to(device)
            if material is not None:
                material.requires_grad_(need_gradient)
            torch.cuda.reset_peak_memory_stats()
            total = 0.
            groups = {name:{'band_error_squared':np.zeros(len(slot['masks']), np.float64),
                'band_energy':[float(e) for e in slot['energies']],
                'completed_shots':[], 'completed_encodings':[],
                'coefficient':slot['coefficient']} for name, slot in cache.items()}
            entries = job['encodings'] if encoded else job['sources']
            for family, code in entries:
                slot = cache[family]
                acquisition = slot['a']
                atomic(folder/'progress.json', {'token':token, 'kind':kind, 'family':family,
                    'state':'preparing_or_propagating_actual_encoding' if encoded else 'original_RF',
                    'unix_s':time.time()})
                if encoded and slot['encoder'] is None:
                    slot['encoder'] = EncodedNativeAcquisition(acquisition)
                    atomic(folder/('encoding_setup_'+family+'.json'), slot['encoder'].config)
                with torch.set_grad_enabled(need_gradient):
                    if encoded:
                        encoder = slot['encoder']
                        encoder.set_encoding(code)
                        prediction = encoder.prediction(material)
                        observed = encoder.observed
                    elif cached:
                        shot = int(code)
                        row = references[family, shot]
                        if digest(row['path']) != row['SHA256']:
                            raise ValueError('Original cached parent RF changed')
                        prediction = torch.from_numpy(np.load(row['path'])).to(device)
                        observed = acquisition.observed[shot]
                    else:
                        shot = int(code)
                        wave = acquisition.waveform(shot)
                        raw = native_wave(material, wave, acquisition.si, acquisition.sw,
                                          acquisition.ri, acquisition.rw, acquisition.config)
                        prediction = acquisition.scattered_coefficients(raw, shot)
                        observed = acquisition.observed[shot]
                    residual = prediction-observed
                    squared = residual.abs().square()
                    loss = slot['coefficient']*(squared*slot['weights'][None]).sum()
                    if need_gradient:
                        loss.backward()
                total += float(loss.detach())
                if not cached:
                    counter = ('encoded_adjoint_propagations' if need_gradient else
                               'encoded_forward_propagations' if encoded else
                               'original_forward_propagations')
                    completed_counts[counter] += 1
                    atomic(folder/'completed_propagations.json', {
                        'token':token, 'kind':kind, 'rank':rank, **completed_counts})
                group = groups[family]
                for j, band in enumerate(slot['masks']):
                    group['band_error_squared'][j] += float(squared[:, band].sum().detach())
                if encoded:
                    group['completed_encodings'].append([int(v) for v in code])
                else:
                    group['completed_shots'].append(shot)
                if job['save_RF'] and not encoded and not cached:
                    archive = folder/'fine_RF'/f'token{token:04d}'
                    archive.mkdir(parents=True, exist_ok=True)
                    save(archive/f'{family}_{shot:02d}.npy', prediction.detach().cpu().numpy())
                del prediction, observed, residual, squared, loss
                if not encoded and not cached:
                    del raw, wave
                atomic(folder/'progress.json', {'token':token, 'kind':kind, 'family':family,
                    'state':'completed', 'seconds':time.time()-started,
                    'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
                    'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30})
            gradient = None
            if need_gradient:
                gradient = torch.stack([
                    .05*(material[0]*material.grad[0]-material[1]*material.grad[1]),
                    .10*material[1]*material.grad[1],
                    .70*(material[2]-edge)*material.grad[2]])*mask
                gradient = gradient.detach().double().cpu().numpy()
            for group in groups.values():
                group['band_error_squared'] = group['band_error_squared'].tolist()
            results.put({'rank':rank, 'token':token, 'kind':kind, 'loss':total,
                'gradient':gradient, 'groups':groups, 'seconds':time.time()-started,
                'peak_Torch_GiB':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30, 'error':None})
            del material, gradient, job
            if need_gradient:
                native_time._TAPES.clear()
                torch.cuda.empty_cache()
            gc.collect()
    except BaseException:
        message = traceback.format_exc()
        atomic(Path(folder)/'error.json', {'rank':rank, 'token':token, 'error':message})
        results.put({'rank':rank, 'token':token, 'error':message})
        raise
