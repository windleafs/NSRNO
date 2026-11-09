"""Once-only small synthetic four-channel operator/statistic qualification.

No patient data, material Jacobian, HVP, or patient PML/precision experiments.
"""
from pathlib import Path
import argparse
import ctypes
import json
import os
import time
import numpy as np
import torch
from .native_time import Config
from .native_multi_equation import MultiEquationWorkspace
from .conditional_material_quadratic import right, down
from .pressure_resources import PressureResources
from .low_frequency_0p5 import atomic, digest


def relative(a, b):
    return abs(a-b)/max(abs(a), abs(b), 1e-30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    project = Path.cwd().resolve()
    a.out = a.out.resolve()
    if a.out.exists() or not a.out.is_relative_to(project/'out/v5'):
        raise ValueError('One fresh synthetic operator qualification only')
    old = project/'out/v5/reconstruction_anatomical_score_v1'
    run = json.loads((old/'run.json').read_text())
    workers = json.loads((old/'worker_processes.json').read_text())
    if run['status'] != 'matched_score_actual_stage_finished' or any(
            Path('/proc', str(pid)).exists() for pid in [run['pid'], *workers['worker_pids']]):
        raise RuntimeError('Do not overlap an actual GPU experiment')
    a.out.mkdir()
    monitor = PressureResources(a.out)
    started = time.time()
    atomic(a.out/'run.json', {'status': 'new_synthetic_operator_qualification_running', 'pid': os.getpid(),
        'created_unix_s': started, 'maximum_seconds': 180, 'patient_arrays_read': False,
        'maximum_allocated_GiB': 2, 'per_card_tree_limit_GiB': 16})
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory, 0)
    torch.manual_seed(6111001)
    device = 'cuda:0'
    usage = monitor.sample('synthetic_new_operator_admission')
    if usage[0]['whole_MiB']+2048+768 > usage[0]['total_MiB']:
        raise RuntimeError('No room for the once-only synthetic qualification')
    c = Config(96, 112, 513, 1, 13, 24, 32, 5e-5, 2.5e-9,
        2.6525823848649224e-8, 1.061032953945969e-7, .008899079785215043, .0070595340096893805)
    z, x = np.indices((c.nz, c.nx))
    material = torch.from_numpy(np.stack([1530+35*np.sin(.07*z+.03*x),
        1000+25*np.cos(.06*z-.09*x), .5+.07*np.sin(.04*z+.05*x)]).astype(np.float32)).to(device)
    node = (z >= 40)&(z < 70)&(x >= 34)&(x < 78)
    def sparse(zz, xx):
        indices = np.asarray(zz)*c.nx+np.asarray(xx)
        indices = np.repeat(indices.reshape(-1, 1), 4, axis=1).astype(np.int32)
        weights = np.zeros(indices.shape, np.float32)
        weights[:, 0] = 1
        return torch.from_numpy(indices).to(device), torch.from_numpy(weights).to(device)
    si, sw = sparse([35], [56])
    ri, rw = sparse(np.full(13, 35), np.linspace(32, 80, 13, dtype=int))
    times = torch.arange(c.nt, device=device)*c.dt
    wave = (1e5*torch.exp(-.5*((times-3e-7)/1e-7)**2)*torch.sin(2*np.pi*3e6*times))[:, None].contiguous()
    zero = torch.zeros_like(wave)
    w = MultiEquationWorkspace(material, c, node)
    native, _ = w.physical_forward(wave, si, sw, ri, rw, False)
    explicit0, _ = w.explicit(wave, si, sw, ri, rw)
    zero_error = float(torch.linalg.vector_norm((native-explicit0).double())/torch.linalg.vector_norm(native.double()))
    force = torch.randn((c.nt, 4, c.nz, c.nx), device=device)*.3
    raw_dual = torch.randn((c.nt, c.nr), device=device)
    response, _ = w.explicit(zero, si, sw, ri, rw, force)
    norm, cp, pullback = w.adjoint(raw_dual, ri, rw, True, True)
    lhs = float((response.double()*raw_dual.double()).sum())
    rhs = float((force.double()*pullback.double()).sum())
    transpose_error = relative(lhs, rhs)
    norm_error = relative(norm, float(pullback.double().square().sum()))
    replay, _ = w.from_dual(zero, si, sw, ri, rw, raw_dual, cp)
    explicit, _ = w.explicit(zero, si, sw, ri, rw, pullback)
    replay_error = float(torch.linalg.vector_norm((replay-explicit).double())/
        torch.clamp(torch.linalg.vector_norm(explicit.double()), min=1e-30))
    del force, cp, pullback, w, native, explicit0, response, replay, explicit
    short = Config.from_buffer_copy(bytes(c))
    short.nt = 1
    w = MultiEquationWorkspace(material, short, node)
    forces = torch.randn((2, 4, c.nz, c.nx), device=device)*.3
    source0 = torch.zeros((2, 1), device=device)
    _, s1 = w.explicit(source0[:1], si, sw, ri, rw, forces[:1], True)
    previous = w.other.cpu().numpy().astype(np.float64)
    stats1 = s1.clone()
    short.nt = 2
    _, s2 = w.explicit(source0, si, sw, ri, rw, forces, True)
    following = w.state.cpu().numpy().astype(np.float64)
    vp = w.vp.cpu().numpy().astype(np.float64)
    stats = (s2-stats1).cpu().numpy()
    padded = w.m.cpu().numpy().astype(np.float64)
    k = w.k.cpu().numpy().astype(np.float64)
    masks = [v.cpu().numpy().astype(np.float64) for v in [w.node, w.fx, w.fz]]
    forcing = forces[1].cpu().numpy().astype(np.float64)
    candidate = padded.copy()
    candidate[0, node] *= np.exp((.025*np.sin(.17*z-.11*x))[node])
    candidate[1, node] *= np.exp((.035*np.cos(.13*z+.19*x))[node])
    candidate[2, node] *= np.exp((.1*np.sin(.1*z+.07*x))[node])
    b = (float(c.e1)+float(c.e2))/2
    def coefficients(m):
        h = m[2]*m[0]/1540
        return m[1]*m[0]**2*(1+b*h)**2, h/(1+b*h)**2
    K0, R0 = coefficients(padded)
    K1, R1 = coefficients(candidate)
    dK, dR, drho = K1-K0, R1-R0, candidate[1]-padded[1]
    sx, sz = drho+right(drho), drho+down(drho)
    predicted = sum(np.sum(stats[j+1]*d+.5*stats[j]*d*d)
        for j, d in [(0, dK), (2, dR), (4, sx), (6, sz)])
    ds = (following[3]+following[4])/padded[1]
    f = previous[0]+previous[5]+previous[6]
    ep = following[0]+previous[5]+previous[6]-k[3]*padded[1]*ds
    beta = np.array([float(c.dt)*float(c.e1)/float(c.tau1), float(c.dt)*float(c.e2)/float(c.tau2)])
    memory = [following[5+j]-(1-float(c.dt)/float(tau))*previous[5+j]-k[4+j]*f
        for j, tau in enumerate([c.tau1, c.tau2])]
    def change(r, d, mask):
        return float(np.sum((r*d+.5*d*d)*mask))
    direct = change(ep, -dK*ds, masks[0])
    direct += sum(change(memory[j], -beta[j]*dR*f, masks[0]) for j in range(2))
    direct += change(forcing[1], 770*(vp[0]-previous[1])*sx, masks[1])
    direct += change(forcing[2], 770*(vp[1]-previous[2])*sz, masks[2])
    statistics_error = relative(predicted, direct)
    monitor.sample('synthetic_new_operator_complete')
    passed = zero_error < 1e-7 and transpose_error < 5e-5 and norm_error < 1e-10 and replay_error < 1e-7 and statistics_error < 5e-4
    result = {'status': 'synthetic_four_channel_operator_and_centered_statistics_passed' if passed else 'failed_for_inspection',
        'zero_auxiliary_original_native_relative_error': zero_error,
        'four_channel_transpose_relative_error': transpose_error,
        'adjoint_forcing_norm_relative_error': norm_error, 'checkpoint_replay_relative_error': replay_error,
        'centered_statistics_vs_direct_frozen_equations_relative_error': statistics_error,
        'statistics_predicted_change': predicted, 'direct_equation_change': direct,
        'seconds': time.time()-started, 'patient_arrays_read': False, 'new_patient_derivative_probes': 0,
        'synthetic_full_adjoint_passes': 2, 'synthetic_physical_forward_passes': 1,
        'synthetic_explicit_forward_passes': 5, 'synthetic_auxiliary_replay_forward_passes': 1,
        'two_explicit_checks_use_only_one_and_two_time_steps': True,
        'new_operator_qualification_not_imaging_evidence': True,
        'source_SHA256': {name: digest(Path(__file__).with_name(name)) for name in
            ['native_multi_equation.py', 'native_multi_equation.cu', 'native_source_extension.py',
             'native_source_extension.cu', 'native_time.py', 'native_time.cu', Path(__file__).name]}}
    atomic(a.out/'result.json', result)
    atomic(a.out/'run.json', {'status': result['status'], 'pid': os.getpid(), 'finished_unix_s': time.time(),
        'seconds': result['seconds'], 'patient_arrays_read': False})
    print(json.dumps(result), flush=True)
    monitor.log.close()
    if not passed:
        raise RuntimeError('Inspect the new four-channel implementation before any patient use')


if __name__ == '__main__':
    if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
        raise RuntimeError('Cannot disable THP')
    main()
