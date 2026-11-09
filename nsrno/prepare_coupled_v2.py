"""Prepare a separate, reviewable next version. Does not start an inversion."""
from pathlib import Path
import ast
import hashlib
import json

root=Path('nsrno')
# The worker digest is checked against the frozen source in the running run,
# rather than accepting a same-name file silently changed after launch.
frozen=Path('out/v5/reconstruction_coupled_v1_train_200')
expected=json.loads((frozen/'run.json').read_text())['sources_sha256']
sources={}
for name in ['reconstruct_coupled.py','coupled_worker.py']:
    current=(root/name).read_bytes()
    archived=(frozen/name).read_bytes()
    if current!=archived:
        raise RuntimeError('Live and archived source disagree: '+name)
    if hashlib.sha256(current).hexdigest()!=expected[name]:
        raise RuntimeError('Source differs from frozen manifest: '+name)
    sources[name]=current.decode()


def replace_once(text,old,new):
    if text.count(old)!=1:
        raise RuntimeError('Unexpected source context: '+old[:100])
    return text.replace(old,new,1)


driver=sources['reconstruct_coupled.py']
driver=replace_once(driver,'from .coupled_coordinates import Coordinates, pcg, dot',
                    'from .coupled_coordinates_v2 import Coordinates, pcg, dot')
driver=replace_once(driver,'from .coupled_worker import worker','from .coupled_worker_v2 import worker')
driver=replace_once(driver,"p.add_argument('--memory-limit-gib', type=float, default=5.75)",
                    "p.add_argument('--memory-limit-gib', type=float, default=4.75)\n    p.add_argument('--minimum-host-available-gib', type=float, default=100.)")
driver=replace_once(driver,"p.add_argument('--chunk', type=int, default=128)","p.add_argument('--chunk', type=int, default=96)")
driver=replace_once(driver,'    a = p.parse_args()',
                    "    a = p.parse_args()\n    mem = dict(row.split(':',1) for row in Path('/proc/meminfo').read_text().splitlines())\n    available = float(mem['MemAvailable'].split()[0])/2**20\n    if available < a.minimum_host_available_gib:\n        raise RuntimeError(f'Two cached codes require host memory guard: {available:.2f} GiB available')")
driver=replace_once(driver,"files = ['reconstruct_coupled.py','coupled_coordinates.py','coupled_worker.py',",
                    "files = ['reconstruct_coupled_v2.py','coupled_coordinates_v2.py','coupled_worker_v2.py','coupled_coordinates.py',")
driver=replace_once(driver,"'curvature':'fixed all-11-shot Rademacher code per group and outer iteration; time-window positive metric; not exact geometry Hessian',",
                    "'curvature':'mean of two fixed all-11-shot codes with dot=1; shot-mean geometric RF weights; positive approximation, not exact source-dependent Hessian',\n              'preconditioner':'positive spatial Sobolev filter times fixed per-outer native-coordinate block scaling from current full gradient; no extra derivative probes',\n              'minimum_host_available_GiB':a.minimum_host_available_gib,")
driver=replace_once(driver,'            direction,cg_history = pcg(metric,-gp,coordinates.precondition,',
                    "            factors = coordinates.set_block_precondition(gp)\n            emit({'kind':'block_preconditioner','outer_iteration':iteration,'positive_factors':factors})\n            direction,cg_history = pcg(metric,-gp,coordinates.precondition,")
driver=replace_once(driver,'time-window curvature is a positive approximation, not the exact Hessian of',
                    'shot-mean geometric curvature is a positive approximation, not the exact Hessian of')

worker=sources['coupled_worker.py']
worker=replace_once(worker,'import ctypes, json, os, time, traceback','import ctypes, hashlib, json, os, time, traceback')
worker=replace_once(worker,'                providers[source] = (a, DepthWeights(a), windows, denominators, encoded.bank)',
                    "                depth = DepthWeights(a)\n                mean_weights = torch.zeros_like(a.observed[0])\n                for j in range(5):\n                    mean_weights += (depth.bins == j).float().mean(0)*(DEPTH_WEIGHTS[j]/depth.den[j])\n                providers[source] = (a, depth, windows, denominators, encoded.bank, mean_weights)")
worker=replace_once(worker,'a, depth, windows, den, _ = acquire(source)','a, depth, windows, den, _, _ = acquire(source)')
worker=replace_once(worker,'a, _, windows, den, bank = acquire(source)','a, _, windows, den, bank, mean_weights = acquire(source)')
old="""                wave = torch.from_numpy(np.tensordot(code, bank, axes=(0,0))).to(device_name)
                reference = torch.from_numpy(np.tensordot(code, a.reference, axes=(0,0))).to(device_name)
                raw = native_wave(m, wave, a.si, a.sw, a.ri, a.rw, a.config)
                prediction = (a.response@(raw-reference)).T
                prepared.append((a, wave, prediction, windows, den))"""
new="""                second = code.copy()
                seed = int.from_bytes(hashlib.sha256(code.tobytes()).digest()[:8], 'little')
                second[np.random.default_rng(seed).choice(11, 5, replace=False)] *= -1
                if float(np.dot(code, second)) != 1.:
                    raise RuntimeError('The two fixed codes must be almost orthogonal')
                code_list = [code, second]
                atomic_json(out/f'curvature_codes_{state_id}_{source}.json',
                    {'source':source, 'state':state_id, 'codes':[v.tolist() for v in code_list],
                     'weights':'shot-mean geometric target weights', 'normalization':'average of two code normal products'})
                for encoded_code in code_list:
                    wave = torch.from_numpy(np.tensordot(encoded_code, bank, axes=(0,0))).to(device_name)
                    reference = torch.from_numpy(np.tensordot(encoded_code, a.reference, axes=(0,0))).to(device_name)
                    raw = native_wave(m, wave, a.si, a.sw, a.ri, a.rw, a.config)
                    prediction = (a.response@(raw-reference)).T
                    prepared.append((a, wave, prediction, mean_weights, len(code_list)))"""
worker=replace_once(worker,old,new)
worker=replace_once(worker,'for a, wave, prediction, windows, den in cache:',
                    'for a, wave, prediction, mean_weights, code_count in cache:')
old="""                weighted = torch.zeros_like(jv)
                for j,(b,e) in enumerate(windows):
                    s = slice(round(b*40), round(e*40))
                    weighted[:,s] = jv[:,s]*(WINDOW_WEIGHTS[j]/den[j]/len(all_sources))"""
new="""                weighted = jv*mean_weights/(len(all_sources)*code_count)"""
worker=replace_once(worker,old,new)

coordinates='''"""Positive block balancing for truncated CPU PCG; no new forward calls."""
import numpy as np
from .coupled_coordinates import Coordinates as BaseCoordinates, pcg, dot


class Coordinates(BaseCoordinates):
    def set_block_precondition(self, gradient):
        norms=np.sqrt(np.mean(gradient*gradient,axis=(-2,-1)))
        reference=max(float(norms.max()),1e-30)
        safe=np.maximum(norms,reference*1e-4)
        geometric=float(np.exp(np.log(safe).mean()))
        factors=np.clip(geometric/safe,1/30,30)
        self.block_factors=factors[:,None,None]
        return factors.tolist()

    def precondition(self, residual):
        # Block constants commute with the positive spatial operator; therefore
        # the composition stays symmetric positive definite for ordinary PCG.
        return super().precondition(residual)*self.block_factors
'''
products={'reconstruct_coupled_v2.py':driver,'coupled_worker_v2.py':worker,'coupled_coordinates_v2.py':coordinates}
for name,value in products.items():
    ast.parse(value)
    path=root/name
    if path.exists():
        raise FileExistsError('Preserve prepared variant: '+str(path))
for name,value in products.items():
    (root/name).write_text(value)
    print(name,hashlib.sha256(value.encode()).hexdigest())
print('PREPARED ONLY: no optimizer, worker, or resource waiter was launched.')
