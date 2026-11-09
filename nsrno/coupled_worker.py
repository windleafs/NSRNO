"""Native full-data gradient and source-encoded positive curvature service."""
from pathlib import Path
import ctypes, json, os, time, traceback
import numpy as np
import torch
from .time_acquisition import TimeAcquisition
from .encoded_time import EncodedAcquisition
from .native_time import native_wave, recompute_tape
from .native_tangent import native_jvp
from .rf_depth_weights import DepthWeights, WEIGHTS as DEPTH_WEIGHTS
from .reconstruct_time import WINDOW_WEIGHTS, atomic_json


def worker(rank, device, assigned, all_sources, sample, banks, jobs, results, out, cap, chunk=128):
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(cap*2**30/
            torch.cuda.get_device_properties(device).total_memory, device)
        device_name = f'cuda:{device}'
        providers, cache = {}, []
        cached_material, cached_state = None, None

        def acquire(source):
            if source not in providers:
                a = TimeAcquisition(source, sample, device=device_name, chunk=chunk)
                encoded = EncodedAcquisition(a, banks[source])
                windows, denominators = encoded.denominators()
                providers[source] = (a, DepthWeights(a), windows, denominators, encoded.bank)
                # Allocate before temporary Jv storage can split this block.
                recompute_tape(a.config, torch.device(device_name))
            return providers[source]

        def full(material, gradient, token):
            m = torch.from_numpy(material).to(device_name).requires_grad_(gradient)
            value, windows_out = 0., {}
            for source in assigned:
                a, depth, windows, den, _ = acquire(source)
                sums, legacy = np.zeros(5), np.zeros(4)
                for shot in range(len(a.angles)):
                    atomic_json(out/f'worker{rank}_progress.json',
                        {'token':token,'mode':'full_gradient' if gradient else 'full_score',
                         'source':source,'shot':shot,'count':len(a.angles),'unix_s':time.time()})
                    with torch.set_grad_enabled(gradient):
                        prediction = a.predict(m, shot)
                        diff = prediction-a.observed[shot]
                        terms = depth.terms(diff, shot)
                        loss = .5*sum(w*t for w,t in zip(DEPTH_WEIGHTS, terms))/len(all_sources)
                        value += float(loss.detach())
                        sums += np.asarray([float(t.detach()) for t in terms])
                        with torch.no_grad():
                            legacy += np.asarray([float(diff[:,round(b*40):round(e*40)].square().sum()/den[j])
                                                  for j,(b,e) in enumerate(windows)])
                        if gradient:
                            loss.backward()
                        del prediction, diff, terms, loss
                windows_out[source] = sums.tolist()
                windows_out[source+':original_time_windows'] = legacy.tolist()
            g = m.grad.cpu().numpy() if gradient else None
            return value, g, windows_out

        def prepare(material, codes, state_id, token):
            m = torch.from_numpy(material).to(device_name).requires_grad_(True)
            prepared = []
            for source in assigned:
                a, _, windows, den, bank = acquire(source)
                code = np.asarray(codes[source], np.float32)
                if code.shape != (11,) or not np.all(np.abs(code)==1):
                    raise ValueError('All eleven original shots must participate in each code')
                atomic_json(out/f'worker{rank}_progress.json',
                    {'token':token,'mode':'prepare_encoded_curvature','source':source,'unix_s':time.time()})
                wave = torch.from_numpy(np.tensordot(code, bank, axes=(0,0))).to(device_name)
                reference = torch.from_numpy(np.tensordot(code, a.reference, axes=(0,0))).to(device_name)
                raw = native_wave(m, wave, a.si, a.sw, a.ri, a.rw, a.config)
                prediction = (a.response@(raw-reference)).T
                prepared.append((a, wave, prediction, windows, den))
            return m, state_id, prepared

        def metric(direction, state_id, token):
            if cached_material is None or state_id != cached_state or not cache:
                raise RuntimeError('Curvature requested for an uncached material state')
            dm = torch.from_numpy(direction).to(device_name)
            g = torch.zeros_like(cached_material)
            quadratic = 0.
            for a, wave, prediction, windows, den in cache:
                atomic_json(out/f'worker{rank}_progress.json',
                    {'token':token,'mode':'encoded_JT_W_Jv','source':a.source,'unix_s':time.time()})
                raw_jv = native_jvp(cached_material, dm, wave, a.si, a.sw, a.ri, a.rw, a.config)
                jv = (a.response@raw_jv).T
                weighted = torch.zeros_like(jv)
                for j,(b,e) in enumerate(windows):
                    s = slice(round(b*40), round(e*40))
                    weighted[:,s] = jv[:,s]*(WINDOW_WEIGHTS[j]/den[j]/len(all_sources))
                quadratic += float((jv.double()*weighted.double()).sum())
                g += torch.autograd.grad(prediction, cached_material, grad_outputs=weighted, retain_graph=True)[0]
                del raw_jv, jv, weighted
            return g.cpu().numpy(), quadratic

        while True:
            job = jobs.get()
            if job is None:
                break
            token, mode, payload, state_id, codes = job
            start = time.time()
            torch.cuda.reset_peak_memory_stats()
            value, gradient, windows, quadratic = None, None, {}, None
            if mode in ['gradient', 'score', 'prepare']:
                cache.clear()
                cached_material, cached_state = None, None
            if mode in ['gradient', 'score']:
                value, gradient, windows = full(payload, mode=='gradient', token)
            elif mode == 'prepare':
                cached_material, cached_state, cache = prepare(payload, codes, state_id, token)
            elif mode == 'hvp':
                gradient, quadratic = metric(payload, state_id, token)
            elif mode == 'clear':
                cache.clear()
                cached_material, cached_state = None, None
            else:
                raise ValueError(mode)
            peak = torch.cuda.max_memory_allocated()/2**30
            seconds = time.time()-start
            atomic_json(out/f'worker{rank}_progress.json',
                {'token':token,'mode':mode,'state':'returned','peak_GiB':peak,
                 'seconds':seconds,'unix_s':time.time()})
            results.put({'rank':rank,'token':token,'value':value,'gradient':gradient,
                         'windows':windows,'quadratic':quadratic,'peak_GiB':peak,'seconds':seconds,'error':None})
    except BaseException:
        results.put({'rank':rank,'token':-1,'error':traceback.format_exc()})
