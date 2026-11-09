"""Measured common-midpoint phase initialization, CPU only; not a 3-field result.

Uses the existing eleven narrow point transmissions and all 192 receivers.
No propagation simulation, predicted reflector, or material truth is read.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

import numpy as np
from scipy import ndimage, signal, sparse
from scipy.optimize import minimize
import torch


def save_json(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rays(foci, apertures, seed_c, xnodes, znodes, quadrature=128):
    """Straight-ray integrals: fixed seed time + sparse nodal slowness correction."""
    ns, nq = len(apertures), len(xnodes) * len(znodes)
    blocks, times = [], []
    x0, z0, dx = -.028, -.003, .00005
    step_x, step_z = xnodes[1] - xnodes[0], znodes[1] - znodes[0]
    mid = (np.arange(quadrature) + .5) / quadrature
    for start in range(0, len(foci), 16):
        points = foci[start:start + 16]
        xf, zf = points.T
        length = np.sqrt((xf[:, None] - apertures[None, :]) ** 2 + zf[:, None] ** 2)
        fraction = (.001 / zf)[:, None]
        xgel = apertures[None, :] + (xf[:, None] - apertures[None, :]) * fraction
        xs = xgel[:, :, None] + (xf[:, None, None] - xgel[:, :, None]) * mid
        zs = .001 + (zf[:, None, None] - .001) * mid
        zs = np.broadcast_to(zs, xs.shape)
        weight = np.broadcast_to((length * (1 - fraction) / quadrature)[:, :, None], xs.shape)
        coords = np.stack([(zs - z0) / dx, (xs - x0) / dx])
        base = ndimage.map_coordinates(1 / seed_c, coords, order=1, mode='nearest')
        times.append(length * fraction / 1540 + (base * weight).sum(-1))
        ux = np.clip((xs - xnodes[0]) / step_x, 0, len(xnodes) - 1 - 1e-7)
        uz = np.clip((zs - znodes[0]) / step_z, 0, len(znodes) - 1 - 1e-7)
        ix, iz = ux.astype(np.int32), uz.astype(np.int32)
        fx, fz = ux - ix, uz - iz
        row = np.broadcast_to(np.arange(len(points) * ns).reshape(len(points), ns, 1), xs.shape)
        rr, cc, vv = [], [], []
        for ox, oz in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            rr.append(row.ravel())
            cc.append(((iz + oz) * len(xnodes) + ix + ox).ravel())
            vv.append((weight * (fx if ox else 1 - fx) * (fz if oz else 1 - fz)).ravel())
        block = sparse.coo_matrix((np.concatenate(vv).astype(np.float32),
                                   (np.concatenate(rr), np.concatenate(cc))),
                                  shape=(len(points) * ns, nq)).tocsr()
        blocks.append(block)
    return sparse.vstack(blocks, format='csr'), np.concatenate(times)


class PhaseData:
    def __init__(self, rf, tx, rx, foci, fs, tref, batch=24):
        self.rf = torch.from_numpy(np.stack([signal.hilbert(rf, axis=-1).real,
                                             signal.hilbert(rf, axis=-1).imag], -1).astype(np.float32))
        self.tx, self.rx, self.foci = tx, rx, foci
        self.fs, self.tref, self.batch = fs, tref, batch
        self.offsets = torch.linspace(-.4e-6, .4e-6, 9)
        self.window = torch.exp(-.5 * (self.offsets / .25e-6) ** 2)
        # Equal and opposite physical aperture translations. The V5 steps are
        # 18 or 19 receiver pitches, including the offset centre transmitter.
        self.shifts = np.rint(np.diff(tx) / np.diff(rx).mean()).astype(int)
        if not np.allclose(np.diff(tx), self.shifts * np.diff(rx).mean(), atol=1e-9):
            raise ValueError('Common-midpoint pairs require exact physical translations')
        self.fixed_weights = []

    def focused(self, tau):
        b, nt, nr = len(tau), len(self.tx), len(self.rx)
        delay = tau[:, :nt, None] + tau[:, None, nt:] + self.tref
        t = (delay[..., None] + self.offsets) * self.fs
        lo = torch.floor(t).long().clamp(0, self.rf.shape[2] - 2)
        frac = (t - lo).clamp(0, 1)[..., None]
        tx = torch.arange(nt)[None, :, None, None]
        rx = torch.arange(nr)[None, None, :, None]
        values = self.rf[tx, rx, lo] * (1 - frac) + self.rf[tx, rx, lo + 1] * frac
        # Identically translated receive subapertures; retain all receivers.
        v = values.clone()
        v[:, :, 1:-1] = .25 * values[:, :, :-2] + .5 * values[:, :, 1:-1] + .25 * values[:, :, 2:]
        return v

    def correlation(self, tau, start):
        values = self.focused(tau)
        foci = self.foci[start:start + len(tau)]
        phases, coherence, energy, valid = [], [], [], []
        for s, shift in enumerate(self.shifts):
            left, right = values[:, s, shift:], values[:, s + 1, :-shift]
            real = (left[..., 0] * right[..., 0] + left[..., 1] * right[..., 1]) * self.window
            imag = (left[..., 1] * right[..., 0] - left[..., 0] * right[..., 1]) * self.window
            gr, gi = real.sum(-1), imag.sum(-1)
            el = (left.square().sum(-1) * self.window).sum(-1)
            er = (right.square().sum(-1) * self.window).sum(-1)
            scale = torch.sqrt(el * er + 1e-30)
            magnitude = torch.sqrt(gr.square() + gi.square() + 1e-30)
            phases.append(1 - gr / magnitude)
            coherence.append((magnitude / (scale + 1e-20)).clamp(0, 1))
            energy.append(scale)
            xp, zp = foci[:, :1], foci[:, 1:2]
            # Keep both Tx/Rx pairs within +/-45 degrees; omit only extreme
            # grazing rays, and record the actual coverage below.
            geometry = ((np.abs(self.tx[s] - xp) < zp) &
                        (np.abs(self.tx[s + 1] - xp) < zp) &
                        (np.abs(self.rx[None, shift:] - xp) < zp) &
                        (np.abs(self.rx[None, :-shift] - xp) < zp))
            geometry[:, :1] = False
            geometry[:, -1:] = False
            valid.append(torch.from_numpy(geometry))
        return [torch.cat(items, 1) for items in [phases, coherence, energy, valid]]

    def fix_weights(self, initial_times):
        coverage = np.zeros(len(self.foci), np.float64)
        for start in range(0, len(self.foci), self.batch):
            tau = torch.from_numpy(initial_times[start:start + self.batch].astype(np.float32))
            with torch.no_grad():
                _, rho, energy, valid = self.correlation(tau, start)
                # Fixed from observed, initially focused echoes. No optimization
                # can reduce its own weight by destroying coherence or energy.
                floor = energy.amax(1, keepdim=True) * 1e-6 + 1e-20
                weight = rho.square() * valid * (energy > floor)
                weight /= weight.sum(1, keepdim=True).clamp_min(1e-12)
                self.fixed_weights.append(weight)
                coverage[start:start + len(tau)] = (weight > 0).sum(1).numpy()
        self.covered = coverage > 0
        self.total = float(self.covered.sum())
        if self.total < .5 * len(self.foci):
            raise RuntimeError('Too few independent common-midpoint pairs over the stated FOV')
        return {'points':len(self.foci),'covered_points':int(self.total),
                'pairs_per_point_min_median_max':np.quantile(coverage, [0,.5,1]).tolist(),
                'weights':'initial observed coherence squared; fixed, per-focus normalized',
                'geometric_mask':'all four rays within 45 degrees; receiver aperture edge samples excluded'}

    def value_gradient_times(self, times, gradient=True):
        total, grad = 0., np.zeros_like(times)
        for batch_id, start in enumerate(range(0, len(self.foci), self.batch)):
            tau = torch.from_numpy(times[start:start + self.batch].astype(np.float32)).requires_grad_(gradient)
            with torch.set_grad_enabled(gradient):
                loss, _, _, _ = self.correlation(tau, start)
                value = (loss * self.fixed_weights[batch_id]).sum() / self.total
                total += float(value.detach())
                if gradient:
                    value.backward()
                    grad[start:start + len(tau)] = tau.grad.numpy()
        return total, grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--sample', default='train_200')
    parser.add_argument('--iterations', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    seed_q = np.load(args.seed / 'parameters.npy')
    seed_m = np.load(args.seed / 'material.npy')
    if seed_q.shape != (3,961,1121) or seed_m.shape != seed_q.shape:
        raise ValueError('Native RF-selected material and physical coordinates required')
    index = json.loads(Path('out/v5/point_tx_3mhz_v1/index.json').read_text())
    meta = index['acquisition']
    row = next(r for r in index['records'] if r['id'] == args.sample)
    rf_path = Path(row['rf'])
    rf = np.load(rf_path)
    tx, rx = np.array(meta['transmit_x_m']), np.array(meta['receiver_x_m'])
    # 1.5-mm focal sampling through 43 mm; the initialization slowness field
    # has 1-mm nodes. The final material retains its native 50-um grid.
    xx, zz = np.meshgrid(np.arange(-.018,.0181,.0015), np.arange(.0025,.0431,.0015))
    foci = np.stack([xx.ravel(),zz.ravel()],1)
    xnodes, znodes = np.arange(-.028,.0281,.001), np.arange(.001,.0451,.001)
    operator, initial_times = rays(foci, np.r_[tx,rx], seed_m[0], xnodes, znodes)
    scale = 80 / 1540 ** 2
    phase = PhaseData(rf, tx, rx, foci, meta['fs_hz'], meta['source_tref_s'][0])
    coverage = phase.fix_weights(initial_times)
    config = {'status':'running','pid':os.getpid(),'sample':args.sample,
              'method':'measured common-midpoint phase initialization; CPU straight rays',
              'source_kind':meta['source_kind'],'transmitters':len(tx),'receivers':len(rx),
              'RF_sha256':sha(rf_path),'seed_parameters_sha256':sha(args.seed/'parameters.npy'),
              'script_sha256':sha(Path(__file__)),'truth_read':False,'new_wave_simulations':0,
              'unknowns':'1-mm nodal slowness correction for INITIALIZATION ONLY; c/Z/alpha native refinement remains necessary',
              'density_scope':'Z held at RF seed; rho changes algebraically as Z/c; alpha fixed',
              'phase_objective':'fixed-weight mean 1-cos(common midpoint phase); independent weak correction TV',
              'selection':'same full stated phase objective, no material truth',
              'iterations':args.iterations,'coverage':coverage,'GPU_memory_GiB':0}
    save_json(args.out/'run.json',config)
    node_shape = (len(znodes),len(xnodes))
    q = np.zeros(node_shape,np.float64)
    logs = []
    best = [float('inf'),q.ravel().copy()]
    def objective(flat):
        nonlocal logs
        current = initial_times + scale*(operator @ flat).reshape(initial_times.shape)
        value, tg = phase.value_gradient_times(current)
        gradient = scale * (operator.T @ tg.ravel())
        u = flat.reshape(node_shape)
        dx,dz = np.diff(u,axis=1)[:-1],np.diff(u,axis=0)[:,:-1]
        root=np.sqrt(dx*dx+dz*dz+.01**2)
        weight=.002/root/root.size
        gv=np.zeros_like(u)
        gx,gz=weight*dx,weight*dz
        gv[:-1,:-1]-=gx+gz;gv[:-1,1:]+=gx;gv[1:,:-1]+=gz
        reg=.002*float(np.mean(root-.01))
        total=value+reg
        if total < best[0]: best[:]=[total,flat.copy()]
        event={'evaluation':len(logs),'phase_loss':value,'regularization':reg,'objective':total,
               'gradient_norm':float(np.linalg.norm(gradient+gv.ravel())),
               'correction_RMS_m_s_approx':float(80*np.sqrt(np.mean(flat**2))),
               'elapsed_seconds':time.time()-start}
        logs.append(event)
        with open(args.out/'evaluations.jsonl','a') as stream: stream.write(json.dumps(event)+'\n')
        save_json(args.out/'progress.json',event)
        print(json.dumps(event),flush=True)
        return total,gradient+gv.ravel()
    result=minimize(objective,q.ravel(),method='L-BFGS-B',jac=True,
                    bounds=[(-1.5,1.5)]*q.size,
                    options={'maxiter':args.iterations,'maxls':12,'maxcor':10,'ftol':1e-9,'gtol':1e-8})
    field=best[1].reshape(node_shape)
    gz,gx=np.meshgrid(np.arange(961)*.00005-.003,np.arange(1121)*.00005-.028,indexing='ij')
    correction=ndimage.map_coordinates(field,np.stack([(gz-.001)/.001,(gx+.028)/.001]),order=1,mode='nearest')*scale
    correction[gz<.001]=0
    material=seed_m.copy()
    material[0]=1/(1/seed_m[0]+correction)
    material[1]=seed_m[1]*seed_m[0]/material[0]
    parameters=seed_q.copy();parameters[0]=(material[0]-1540)/80
    np.save(args.out/'parameters.npy',parameters)
    np.save(args.out/'material.npy',material)
    np.save(args.out/'slowness_correction.npy',field*scale)
    config.update({'status':'complete','optimizer_success':bool(result.success),
                   'optimizer_message':str(result.message),'iterations_done':int(result.nit),
                   'evaluations':len(logs),'initial_phase_objective':logs[0]['objective'],
                   'selected_phase_objective':best[0],'elapsed_seconds':time.time()-start,
                   'not_a_three_parameter_reconstruction':True,
                   'required_next':'separate material evaluation; only a useful initializer warrants full original RF scoring and native joint refinement'})
    save_json(args.out/'run.json',config)


if __name__ == '__main__':
    main()
