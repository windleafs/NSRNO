"""Observed-channel coherence initialization; no material truth or PDE solve.

Snell-ray focusing with known gel and two-dimensional delay-profile tracking.
The result is a propagation initializer, not a local material reconstruction.
"""
from argparse import ArgumentParser
from pathlib import Path
import hashlib, json, os, subprocess, time, traceback
import numpy as np
import torch


def atomic_json(path, value):
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def rays(x, z, elements, c):
    """Fermat time and endpoint derivatives for a 1 mm / 1540 m/s gel."""
    delta = x[:, None] - elements[None]
    distance = delta.abs()
    body = (z[:, None] - .001).clamp_min(1e-7)
    lower = torch.zeros_like(distance)
    upper = torch.full_like(distance, (1 - 1e-6) / max(c, 1540.))
    for _ in range(24):
        p = (lower + upper) * .5
        cg = (1 - (1540. * p).square()).clamp_min(1e-12).sqrt()
        cb = (1 - (c * p).square()).clamp_min(1e-12).sqrt()
        horizontal = .001 * (1540. * p) / cg + body * (c * p) / cb
        low = horizontal < distance
        lower = torch.where(low, p, lower)
        upper = torch.where(low, upper, p)
    p = (lower + upper) * .5
    cg = (1 - (1540. * p).square()).clamp_min(1e-12).sqrt()
    cb = (1 - (c * p).square()).clamp_min(1e-12).sqrt()
    t = .001 / (1540. * cg) + body / (c * cb)
    return t, delta.sign() * p, cb / c


def moments(ray, reference, weight):
    t, gx, gz = ray
    d = (t - reference) * 1e6
    gx, gz = gx * 1e6, gz * 1e6
    den = weight.sum(1).clamp_min(1e-12)
    def mean(v):
        return (v * weight).sum(1) / den
    return {"d": mean(d), "x": mean(gx), "z": mean(gz),
            "xx": mean(gx * gx), "zz": mean(gz * gz),
            "xz": mean(gx * gz), "xd": mean(gx * d),
            "zd": mean(gz * d), "dd": mean(d * d)}


def register(x0, z0, tx, rx, c, wt, wr, tt0, tr0):
    x = x0.clone()
    z = .001 + (z0 - .001) * (c / 1540.)
    if c != 1540.:
        for _ in range(5):
            a = moments(rays(x, z, tx, c), tt0, wt)
            b = moments(rays(x, z, rx, c), tr0, wr)
            hxx = a['xx'] + b['xx'] + 2 * a['x'] * b['x']
            hzz = a['zz'] + b['zz'] + 2 * a['z'] * b['z']
            hxz = a['xz'] + b['xz'] + a['x'] * b['z'] + b['x'] * a['z']
            qx = a['xd'] + b['xd'] + a['x'] * b['d'] + b['x'] * a['d']
            qz = a['zd'] + b['zd'] + a['z'] * b['d'] + b['z'] * a['d']
            damping = (hxx + hzz) * 1e-6
            hxx, hzz = hxx + damping, hzz + damping
            det = (hxx * hzz - hxz.square()).clamp_min(1e-6)
            dx = (-hzz * qx + hxz * qz) / det
            dz = (hxz * qx - hxx * qz) / det
            x = x + dx.clamp(-.002, .002)
            z = (z + dz.clamp(-.002, .002)).clamp_min(.0011)
    rt, rr = rays(x, z, tx, c), rays(x, z, rx, c)
    a, b = moments(rt, tt0, wt), moments(rr, tr0, wr)
    delay_rms = (a['dd'] + b['dd'] + 2 * a['d'] * b['d']).clamp_min(0).sqrt()
    return x, z, rt[0], rr[0], delay_rms


def coherence(analytic, tt, tr, wt, wr, f0=3e6):
    p, nr = tr.shape
    combined = torch.zeros((p, nr), dtype=torch.complex64, device=tr.device)
    energy = torch.zeros(p, device=tr.device)
    nrftime = analytic.shape[-1]
    for shot in range(len(analytic)):
        sample = (tt[:, shot, None] + tr + 4 / f0) * 40e6
        if not bool(((sample >= 0) & (sample <= nrftime - 1)).all()):
            raise ValueError('Focusing would leave the observed RF time interval')
        index = sample.floor().long().clamp(0, nrftime - 2)
        fraction = (sample - index).clamp(0, 1)
        first = torch.gather(analytic[shot], 1, index.T).T
        second = torch.gather(analytic[shot], 1, (index + 1).T).T
        value = first + (second - first) * fraction
        combined += wt[:, shot, None] * value
        energy += wt[:, shot] * (value.abs().square() * wr).sum(1)
    beamsum = (combined * wr).sum(1)
    count = wt.sum(1) * wr.sum(1)
    numerator = beamsum.abs().square()
    joint = numerator / (count * energy).clamp_min(1e-20)
    receive = numerator / (wr.sum(1) * (combined.abs().square() * wr).sum(1)).clamp_min(1e-20)
    power = energy / count.clamp_min(1e-12)
    return joint.clamp(0, 1), receive.clamp(0, 1), power


@torch.no_grad()
def execute(a):
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    actual = {int(row.split(',')[0]): int(row.split(',')[1]) for row in used.splitlines()}
    if actual[a.device] + a.memory_limit_gib * 1024 + 512 > 16 * 1024:
        raise RuntimeError('Insufficient room within the 16 GiB whole-card budget')
    torch.set_num_threads(1)
    torch.cuda.set_device(a.device)
    torch.cuda.set_per_process_memory_fraction(
        a.memory_limit_gib * 2**30 / torch.cuda.get_device_properties(a.device).total_memory, a.device)
    device = f'cuda:{a.device}'
    index_path = Path('out/v5/point_tx_3mhz_v1/index.json')
    index = json.loads(index_path.read_text())
    row = next(r for r in index['records'] if r['id'] == a.sample)
    layout = np.load(index['source_layout'])
    geom = np.load('out/v5/time_triparam_v1/geometry.npz')
    tx = torch.from_numpy(layout['transmit_x_m'].astype(np.float32)).to(device)
    rx = torch.from_numpy(geom['xe'].astype(np.float32)).to(device)
    rf_path = Path(row['rf'])
    rf_np = np.load(rf_path)
    if rf_np.shape != (11, 192, 3601):
        raise ValueError(f'Unexpected point RF layout: {rf_np.shape}')
    raw = torch.from_numpy(rf_np).to(device)
    del rf_np
    ft = torch.fft.fft(raw, dim=-1)
    hilbert = torch.zeros(raw.shape[-1], device=device)
    hilbert[0] = 1
    hilbert[1:(raw.shape[-1] + 1)//2] = 2
    analytic = torch.fft.ifft(ft * hilbert, dim=-1)
    del raw, ft
    xs = torch.arange(-40, 41, device=device) * .0004
    zs = .005 + torch.arange(86, device=device) * .0004
    zz, xx = torch.meshgrid(zs, xs, indexing='ij')
    shape = tuple(xx.shape)
    x0, z0 = xx.flatten(), zz.flatten()
    wt = torch.exp(-.5 * ((x0[:, None] - tx[None]) / (1.5 * z0[:, None])).pow(4))
    wr = torch.exp(-.5 * ((x0[:, None] - rx[None]) / (.75 * z0[:, None])).pow(4))
    tt0, tr0 = rays(x0, z0, tx, 1540.)[0], rays(x0, z0, rx, 1540.)[0]
    cf0, _, p0 = coherence(analytic, tt0, tr0, wt, wr)
    median = p0[p0 > 0].median()
    fixed_weight = (p0 / median.clamp_min(1e-20)).clamp_min(0).pow(.25).clamp(max=1)
    bands = [(5, 12), (12, 25), (25, 39.001)]
    masks = [(z0 * 1000 >= lo - 1e-4) & (z0 * 1000 < hi - 1e-4) for lo, hi in bands]
    speeds = np.arange(1400., 1660.1, 5.)
    rows, cf_maps, rx_maps, positions, delays = [], [], [], [], []
    start = time.time()
    record = {'status': 'running', 'sample': a.sample, 'pid': os.getpid(),
              'truth_used': False, 'propagation_evaluations': 0,
              'role': 'path-average propagation initialization; not local c/rho/alpha maps',
              'trial_c_m_s': speeds.tolist(), 'gel_mm': 1., 'gel_c_m_s': 1540.,
              'shots': 11, 'receivers': 192, 'device': a.device,
              'memory_limit_GiB': a.memory_limit_gib,
              'card_used_MiB_before': actual, 'rf': str(rf_path.resolve()),
              'index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'selection': 'maximize fixed-reference-power-weighted joint CF; three depth classes equal weight',
              'registration': '5 Gauss-Newton steps for weighted two-way Snell delay-profile matching',
              'reference_image_grid': {'x_mm': [-16., 16.], 'z_mm': [5., 39.], 'spacing_mm': .4}}
    atomic_json(a.out/'run.json', record)
    for c in speeds:
        x, z, tt, tr, delay_rms = register(x0, z0, tx, rx, float(c), wt, wr, tt0, tr0)
        joint, receive, power = coherence(analytic, tt, tr, wt, wr)
        scores = [float((joint[mask] * fixed_weight[mask]).sum() /
                        fixed_weight[mask].sum().clamp_min(1e-12)) for mask in masks]
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite coherence score')
        item = {'c_m_s': float(c), 'joint_CF': float(np.mean(scores)),
                'depth_class_CF': scores, 'delay_profile_RMS_us_mean': float(delay_rms.mean()),
                'seconds': time.time() - start,
                'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30}
        rows.append(item)
        cf_maps.append(joint.cpu().numpy().reshape(shape))
        rx_maps.append(receive.cpu().numpy().reshape(shape))
        positions.append(torch.stack([x, z]).cpu().numpy().reshape(2, *shape))
        delays.append(delay_rms.cpu().numpy().reshape(shape))
        atomic_json(a.out/'progress.json', {'completed': len(rows), 'planned': len(speeds), **item})
        print(json.dumps(item), flush=True)
    best = max(rows, key=lambda r: r['joint_CF'])
    peaks = [max(rows, key=lambda r: r['depth_class_CF'][j])['c_m_s'] for j in range(3)]
    np.savez(a.out/'coherence_maps.npz', c_m_s=speeds,
             x_reference_m=xs.cpu().numpy(), z_reference_m=zs.cpu().numpy(),
             joint_CF=np.stack(cf_maps), receive_CF=np.stack(rx_maps),
             registered_xz_m=np.stack(positions), delay_profile_RMS_us=np.stack(delays),
             fixed_reference_power_weight=fixed_weight.cpu().numpy().reshape(shape))
    record.update(status='finished', seconds=time.time()-start, curves=rows,
                  selected_c_m_s=best['c_m_s'], selected_joint_CF=best['joint_CF'],
                  depth_class_peak_c_m_s=peaks,
                  boundary_peak=best['c_m_s'] in (float(speeds[0]), float(speeds[-1])),
                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                  interpretation='No material reconstruction or independent local sound-speed inference')
    atomic_json(a.out/'run.json', record)


def main():
    p = ArgumentParser()
    p.add_argument('--sample', default='train_200')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--memory-limit-gib', type=float, default=4.)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/'run.json').exists():
        raise FileExistsError('Preserve previous focusing runs')
    try:
        execute(a)
    except BaseException:
        atomic_json(a.out/'failure.json', {'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
