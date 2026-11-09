"""Multiple-centre corrected-gradient and true fine-RF jobs in one source worker.

All target data remain the originally acquired fine-grid RF. A coarse model
is locked to an actual fine material/RF anchor; its per-material bias is never
fitted against the observation.
"""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0) != 0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import gc
import json
import time
import traceback
import numpy as np
import torch
from .frequency_multilevel_coordinates import FrequencyMultilevelCoordinates
from .frequency_multilevel_acquisition import (LowHalfMHzAcquisition, coarse_acquisition)
from .full90_3mhz_acquisition import Full90ThreeMHzAcquisition
from .low_frequency_0p5 import atomic,digest
from .native_time import native_wave
from . import native_time


def save(path, value):
    temporary = path.with_name(path.stem+'.partial.npy')
    np.save(temporary, value)
    temporary.replace(path)


def worker(rank, device, jobs, results, plan_path, chunk, cap_gib, folder, workers):
    token = -1
    try:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        plan = json.loads(Path(plan_path).read_text())
        if digest(plan['native_anchor_material']) != plan['native_anchor_material_SHA256']:
            raise ValueError('Locked actual native anchor changed')
        edge = np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')
        co = FrequencyMultilevelCoordinates(np.load(plan['native_anchor_material']), edge,
                                           factor=plan['coarse_factor'])
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(
            cap_gib*2**30/torch.cuda.get_device_properties(device).total_memory, device)
        device = f'cuda:{device}'
        coarse_edge = torch.from_numpy(co.edge.copy()).to(device)
        fine_edge = torch.from_numpy(np.broadcast_to(edge, (961,1121)).copy()).to(device)
        coarse_mask = torch.from_numpy(co.mask.copy()).to(device)
        fine_mask = torch.from_numpy(co.fine.mask.copy()).to(device)
        cache = {}
        if abs(sum(s['coefficient'] for s in plan['families'].values())-.5)>1e-12:
            raise ValueError('Source-family objective coefficients must sum to one half')
        for family,spec in plan['families'].items():
            if spec['centre']=='low':
                fa=LowHalfMHzAcquisition(plan['data_root'],spec['source'],device=device,chunk=chunk)
            else:
                fa=Full90ThreeMHzAcquisition(plan['high_index'],spec['source'],device=device,chunk=chunk)
            ca=coarse_acquisition(fa,co.shape[1:],plan['coarse_factor'])
            bands=spec['bands_hz']
            band_masks=[torch.from_numpy((ca.frequencies>=lo)&(ca.frequencies<=hi)).to(device)
                        for lo,hi in bands]
            band_energy=[ca.observed[:,:,mask].abs().square().sum() for mask in band_masks]
            weights=torch.zeros(len(ca.frequencies),device=device,dtype=torch.float64)
            for mask,denominator in zip(band_masks,band_energy):
                if not float(denominator)>0:
                    raise ValueError('An acquired objective band has zero target energy')
                weights[mask]+=1/(len(bands)*denominator)
            keep=weights>0
            energy=ca.observed[:,:,keep].abs().square().sum()
            fine_anchor = {}
            for shot in range(rank,11,workers):
                rows=[r for r in plan.get('native_anchor_RF',[]) if r['family']==family and r['shot']==shot]
                if rows:
                    if len(rows)!=1 or digest(rows[0]['path'])!=rows[0]['SHA256']:
                        raise ValueError('Locked actual anchor RF changed')
                    fine_anchor[shot]=torch.from_numpy(np.load(rows[0]['path'])).to(device)
            cache[family] = {'coarse':ca,'fine':fa,'keep':keep,'energy':energy,
                             'weights':weights,'band_masks':band_masks,'band_energy':band_energy,
                             'coefficient':spec['coefficient'],
                             'fine_anchor':fine_anchor,'coarse_anchor':{}}
        atomic(folder/'ready.json',{
            'rank':rank,'device':device,'native_anchor_SHA256':plan['native_anchor_material_SHA256'],
            'family_specifications':plan['families'],'source_fixed_and_known_original_formula_retained':True,
            'truth_read':False,'coarse':co.info(),
            'known_coarse_waveform_correction_is_fixed_at_anchor':True})
        last = None
        while True:
            job = jobs.get()
            if job is None:
                break
            kind = job['kind']
            if kind == 'save_selected_coarse_RF':
                if last is None or last['token'] != job['expected_token']:
                    raise ValueError('Selected RF is not the last completed accepted coarse job')
                for family, shot, value in last['RF']:
                    save(folder/f'selected_coarse_{family}_{shot:02d}.npy',value)
                results.put({'rank':rank,'kind':kind,'token':job['token'],'error':None})
                continue
            if kind not in ['coarse_gradient','fine_forward','fine_gradient']:
                raise ValueError(kind)
            token = job['token']
            started = time.time()
            need_gradient = kind != 'fine_forward'
            is_coarse = kind == 'coarse_gradient'
            if is_coarse:
                control = np.asarray(job['control'],np.float64)
                material = co.coarse_material(control)
                is_origin = not np.any(control)
            else:
                material = np.asarray(job['native_material'],np.float32)
                is_origin = np.array_equal(material,co.native_anchor)
            m = torch.from_numpy(material).to(device)
            m.requires_grad_(need_gradient)
            torch.cuda.reset_peak_memory_stats()
            total_loss=0.
            groups={}
            saved_RF=[]
            for family, slot in cache.items():
                a=slot['coarse'] if is_coarse else slot['fine']
                keep, energy=slot['keep'],slot['energy']
                error=0.
                band_errors=np.zeros(len(slot['band_masks']),np.float64)
                shots=[]
                for shot in range(rank,11,workers):
                    atomic(folder/'progress.json',{
                        'token':token,'kind':kind,'family':family,'shot':shot,
                        'state':'propagating','unix_s':time.time()})
                    wave=a.waveform(shot)
                    with torch.set_grad_enabled(need_gradient):
                        raw=native_wave(m,wave,a.si,a.sw,a.ri,a.rw,a.config)
                        if is_coarse:
                            coefficients=a.coefficients(raw)
                            if shot not in slot['coarse_anchor']:
                                if not is_origin:
                                    raise ValueError('The coarse model must first evaluate its actual control origin')
                                slot['coarse_anchor'][shot]=coefficients.detach().clone()
                                save(folder/f'coarse_anchor_{family}_{shot:02d}.npy',
                                     coefficients.detach().cpu().numpy())
                            if shot not in slot['fine_anchor']:
                                raise ValueError('Compute the necessary actual native gradient before a new source family')
                            prediction=(coefficients-slot['coarse_anchor'][shot]+
                                        slot['fine_anchor'][shot])
                        else:
                            coefficients=None
                            prediction=a.scattered_coefficients(raw,shot)
                            if kind=='fine_gradient' and is_origin:
                                slot['fine_anchor'][shot]=prediction.detach().clone()
                                save(folder/f'fine_anchor_{family}_{shot:02d}.npy',
                                     prediction.detach().cpu().numpy())
                        difference=prediction-a.observed[shot]
                        squared=difference.abs().square()
                        e2=squared[:,keep].sum()
                        loss=slot['coefficient']*(squared*slot['weights'][None]).sum()
                        if need_gradient:
                            loss.backward()
                    value=prediction.detach().cpu().numpy()
                    if is_coarse:
                        saved_RF.append((family,shot,value))
                    else:
                        archive=folder/'fine_RF'/f'token{token:04d}'
                        archive.mkdir(parents=True,exist_ok=True)
                        save(archive/f'{family}_{shot:02d}.npy',value)
                    total_loss+=float(loss.detach())
                    error+=float(e2.detach())
                    for j,mask in enumerate(slot['band_masks']):
                        band_errors[j]+=float(squared[:,mask].sum().detach())
                    shots.append(shot)
                    atomic(folder/'progress.json',{
                        'token':token,'kind':kind,'family':family,'shot':shot,
                        'state':'completed','seconds':time.time()-started,
                        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
                        'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30})
                    del wave,raw,coefficients,prediction,difference,squared,e2,loss,value
                groups[family]={'error_squared':error,'energy':float(energy),
                                'band_error_squared':band_errors.tolist(),
                                'band_energy':[float(v) for v in slot['band_energy']],
                                'coefficient':slot['coefficient'],
                                'completed_shots':shots}
            gradient=None
            if need_gradient:
                edge_value=coarse_edge if is_coarse else fine_edge
                mask=coarse_mask if is_coarse else fine_mask
                gradient=torch.stack([
                    .05*(m[0]*m.grad[0]-m[1]*m.grad[1]),
                    .10*m[1]*m.grad[1],
                    .70*(m[2]-edge_value)*m.grad[2]])*mask
                gradient=gradient.detach().double().cpu().numpy()
            last={'token':token,'RF':saved_RF} if is_coarse else None
            results.put({
                'rank':rank,'token':token,'kind':kind,'loss':total_loss,
                'gradient':gradient,'groups':groups,'seconds':time.time()-started,
                'peak_Torch_GiB':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
                'error':None})
            del job,m,material,gradient,saved_RF
            if kind=='fine_gradient':
                # Only this worker uses its private native tape dictionary.
                # The completed fine gradient must not keep its large tape
                # resident during subsequent coarse jobs.
                native_time._TAPES.clear()
                torch.cuda.empty_cache()
            gc.collect()
    except BaseException:
        message=traceback.format_exc()
        atomic(Path(folder)/'error.json',{'rank':rank,'token':token,'error':message})
        results.put({'rank':rank,'token':token,'error':message})
        raise
