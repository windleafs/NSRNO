"""Finish the SAME cached convex CMA inverse by bounded Newton; CPU only."""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import json
import os
import resource
import shutil
import signal
import sys
import time
import traceback
import numpy as np
from scipy import linalg, ndimage, sparse
from .cma_phase_geometry import ray_matrix, effective_speed, nominal_physical_speed
from .low_frequency_0p5 import atomic, digest, SAMPLE, CASE


def bounded_quadratic(H, rhs, initial, lo, hi, deadline, counts):
    """Feasible primal active-set QP; every accepted segment decreases the QP."""
    value = np.clip(initial.copy(), lo, hi)
    active = np.zeros(len(value), np.int8)
    active[value <= lo + 1e-11] = -1
    active[value >= hi - 1e-11] = 1
    for _ in range(4 * len(value)):
        if time.time() >= deadline:
            return value, 'time_budget_in_QP'
        counts['QP_iterations'] += 1
        free = np.flatnonzero(active == 0)
        fixed = np.flatnonzero(active != 0)
        target = value.copy()
        if len(free):
            reduced_rhs = rhs[free].copy()
            if len(fixed):
                reduced_rhs -= H[np.ix_(free, fixed)] @ value[fixed]
            factor = linalg.cho_factor(H[np.ix_(free, free)], lower=True,
                                      check_finite=False, overwrite_a=True)
            counts['Cholesky_factorizations'] += 1
            target[free] = linalg.cho_solve(factor, reduced_rhs, check_finite=False)
        step = target - value
        alpha = 1.
        blocker = None
        lower = np.flatnonzero((active == 0) & (step < -1e-14))
        upper = np.flatnonzero((active == 0) & (step > 1e-14))
        for indices, bound, sign in [(lower, lo, -1), (upper, hi, 1)]:
            if len(indices):
                ratios = (bound - value[indices]) / step[indices]
                k = int(np.argmin(ratios))
                if ratios[k] < alpha:
                    alpha = max(0., float(ratios[k]))
                    blocker = (int(indices[k]), sign)
        value = np.clip(value + alpha * step, lo, hi)
        if blocker is not None:
            index, sign = blocker
            active[index] = sign
            value[index] = lo if sign == -1 else hi
            continue
        gradient = H @ value - rhs
        violation = np.zeros(len(value))
        violation[active == -1] = -gradient[active == -1]
        violation[active == 1] = gradient[active == 1]
        worst = int(np.argmax(violation))
        if violation[worst] <= 1e-10:
            return value, 'QP_KKT_satisfied'
        active[worst] = 0
    return value, 'QP_iteration_limit'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--index', type=Path,
        default=Path('out/v5/matched_existing_3mhz_full90_v1/index.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=120.)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not 60 <= args.max_seconds <= 120:
        raise ValueError('Require explicit CPU only and a <=120 second stage')
    args.parent, args.index, args.out = [p.resolve() for p in
                                        [args.parent, args.index, args.out]]
    project = Path.cwd().resolve()
    if not all(p.is_relative_to(project) for p in [args.parent, args.index, args.out]):
        raise ValueError('All paths must remain in the independent nsrno project')
    if args.out.exists():
        raise RuntimeError('Inspect the same existing refinement; never overwrite')
    parent = json.loads((args.parent / 'run.json').read_text())
    index = json.loads(args.index.read_text())
    if (parent['status'] != 'CPU_CMA_phase_velocity_seed_finished' or
        parent['sample'] != SAMPLE or parent['case'] != CASE or
        parent['index_SHA256'] != digest(args.index)):
        raise ValueError('Require the finished same authorized development inverse')
    if shutil.disk_usage('.').free < 2 * 2**30 + 400 * 2**20 + 20 * 2**20:
        raise RuntimeError('Insufficient reserved output storage')
    args.out.mkdir()
    started = time.time()
    deadline = started + args.max_seconds - 10
    counts = {'QP_iterations': 0, 'Cholesky_factorizations': 0}
    run = {'status': 'CPU_rebuilding_cached_phase_inverse', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': started + args.max_seconds,
        'parent': str(args.parent), 'parent_phase_SHA256': digest(args.parent / 'phase_seed.npz'),
        'parent_seconds': parent['seconds'], 'parent_iterations': parent['optimization_iterations'],
        'index_SHA256': digest(args.index), 'sample': SAMPLE, 'case': CASE,
        'source_SHA256': {name: digest(Path(__file__).parent / name) for name in
                             [Path(__file__).name, 'cma_phase_geometry.py']},
        'GPU_work': False, 'new_acoustic_wave_solves': 0,
        'new_acoustic_material_derivatives': 0, 'new_patient_data': False,
        'RF_rebeamforming': False, 'patient_truth_read_by_optimizer': False,
        'joint_reconstruction_claimed': False, 'objective_and_bounds_unchanged': True}
    atomic(args.out / 'run.json', run)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        with np.load(args.parent / 'phase_seed.npz') as saved:
            cache = {key: saved[key] for key in saved.files}
        xc, zc = cache['x'], cache['z']
        shape = cache['c_effective'].shape
        if shape != (42, 39) or cache['phase'].shape != (30, *shape):
            raise ValueError('Unexpected saved phase geometry')
        zedges = np.linspace(.001, .043, 43)
        rays = {}
        def ray(angle):
            key = round(float(angle), 12)
            if key not in rays:
                rays[key] = ray_matrix(angle, xc, zedges)
            return rays[key]
        blocks, data, weight = [], [], []
        angles = np.deg2rad(parent['target_transmit_angles_deg'])
        scale = 2 * np.pi * parent['source_f0_hz'] / 1540. * .05
        row = 0
        for gamma in np.deg2rad(parent['common_midangles_deg']):
            for phi_a, phi_b in zip(angles[:-1], angles[1:]):
                if abs(phi_a + phi_b - 2 * gamma) < 1e-9:
                    continue
                phase = cache['phase'][row].astype(np.float64).ravel()
                confidence = cache['confidence'][row].astype(np.float64).ravel()
                use = (confidence > 1e-6) & np.isfinite(phase)
                block = scale * ((ray(phi_b) + ray(2 * gamma - phi_b)) / np.cos(phi_b - gamma) -
                                 (ray(phi_a) + ray(2 * gamma - phi_a)) / np.cos(phi_a - gamma))
                blocks.append(block[use]); data.append(phase[use]); weight.append(confidence[use])
                row += 1
        if row != 30:
            raise ValueError('Cached observation pairing changed')
        data, weight = np.concatenate(data), np.concatenate(weight)
        norm = float(np.sum(weight * data**2))
        sqrtw = np.sqrt(weight / norm)
        A = sparse.vstack(blocks, format='csr').multiply(sqrtw[:, None]).tocsr()
        b = sqrtw * data
        del blocks, rays
        n = A.shape[1]
        grid = np.arange(n).reshape(shape)
        origin = grid[:-1, :-1].ravel()
        count = len(origin)
        rows = np.repeat(np.arange(count), 2)
        Gx = sparse.csr_matrix((np.tile([-1., 1.], count),
                (rows, np.stack((origin, grid[:-1, 1:].ravel()), axis=1).ravel())), shape=(count, n))
        Gz = sparse.csr_matrix((np.tile([-1., 1.], count),
                (rows, np.stack((origin, grid[1:, :-1].ravel()), axis=1).ravel())), shape=(count, n))
        gram = (A.T @ A).toarray()
        diagonal = np.diag_indices(n)
        gram[diagonal] += .002 / n
        rhs = np.asarray(A.T @ b)
        constant = .5 * float(b @ b)
        def objective(value, hessian=False):
            dx, dz = Gx @ value, Gz @ value
            radius = np.sqrt(dx**2 + dz**2 + .02**2)
            gv = gram @ value
            loss = .5 * float(value @ gv) - float(rhs @ value) + constant + .02 * float(np.mean(radius - .02))
            gradient = gv - rhs + .02 / count * (Gx.T @ (dx / radius) + Gz.T @ (dz / radius))
            if not hessian:
                return loss, gradient
            # Exact Hessian of the same smoothed isotropic TV, including its mixed term.
            xx = (dz**2 + .02**2) / radius**3
            zz = (dx**2 + .02**2) / radius**3
            xz = -dx * dz / radius**3
            tv_H = (Gx.T @ (Gx.multiply(xx[:, None]) + Gz.multiply(xz[:, None])) +
                    Gz.T @ (Gz.multiply(zz[:, None]) + Gx.multiply(xz[:, None])))
            return loss, gradient, gram + .02 / count * tv_H.toarray()
        lo = float((1540. / effective_speed(1680., index['absorption']) - 1) / .05)
        hi = float((1540. / effective_speed(1400., index['absorption']) - 1) / .05)
        value = np.clip(((1540. / cache['c_effective'].astype(np.float64) - 1) / .05).ravel(), lo, hi)
        first_loss, gradient = objective(value)
        if abs(first_loss - parent['final_phase_objective']) > 2e-6:
            raise ValueError('Cached float32 phase reconstruction changed the parent objective')
        first_kkt = float(np.max(abs(value - np.clip(value - gradient, lo, hi))))
        history = []
        termination = 'Newton_iteration_limit'
        run.update(status='CPU_bounded_Newton_same_phase_objective',
            phase_rows=len(b), unknown_slowness_cells=n,
            initial_phase_objective=first_loss, initial_projected_gradient_inf=first_kkt,
            parent_objective_difference=first_loss-parent['final_phase_objective'])
        atomic(args.out / 'run.json', run)
        for iteration in range(40):
            if time.time() >= deadline:
                termination = 'CPU_budget_at_last_accepted_Newton_step'; break
            loss, gradient, H = objective(value, True)
            kkt = float(np.max(abs(value - np.clip(value - gradient, lo, hi))))
            if kkt <= 1e-8:
                termination = 'projected_gradient_converged'; break
            target, qp_status = bounded_quadratic(H, H @ value - gradient, value,
                                                   lo, hi, deadline, counts)
            direction = target - value
            slope = float(gradient @ direction)
            fraction, accepted = 1., False
            for _ in range(25):
                trial = np.clip(value + fraction * direction, lo, hi)
                trial_loss, _ = objective(trial)
                if slope < 0 and trial_loss <= loss + 1e-4 * fraction * slope + 1e-13:
                    value, accepted = trial, True; break
                fraction *= .5
            history.append({'iteration': iteration + 1, 'before': loss,
                'after': trial_loss if accepted else loss, 'KKT_before': kkt,
                'accepted': accepted, 'fraction': fraction, 'QP_status': qp_status})
            run.update(accepted_Newton_steps=sum(h['accepted'] for h in history),
                current_phase_objective=history[-1]['after'],
                current_projected_gradient_inf=kkt, counts=counts, history=history)
            atomic(args.out / 'run.json', run)
            if not accepted:
                termination = 'no_further_decreasing_feasible_Newton_step'; break
        final_loss, gradient = objective(value)
        final_kkt = float(np.max(abs(value - np.clip(value - gradient, lo, hi))))
        if final_kkt <= 1e-8:
            termination = 'projected_gradient_converged'
        effective = 1540. / (1 + .05 * value.reshape(shape))
        physical = nominal_physical_speed(effective, index['absorption'])
        with np.load('out/v5/time_triparam_v1/geometry.npz') as saved:
            g = {key: saved[key] for key in saved.files}
        nz, nx = json.loads(Path('out/v5/time_triparam_v1/geometry.json').read_text())['shape']
        native_z = g['origin'][1] + np.arange(nz) * g['spacing'][1] - np.mean(g['receiver_coordinates'][:, 1])
        coordinates = np.stack(np.meshgrid((native_z-zc[0])/(zc[1]-zc[0]),
                            (g['x']-xc[0])/(xc[1]-xc[0]), indexing='ij'))
        speed = ndimage.map_coordinates(physical, coordinates, order=1, mode='nearest', prefilter=False)
        edge = np.broadcast_to(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'), (nz, nx))
        material = np.stack((speed, np.full((nz, nx), 1000.), .5+edge)).astype(np.float32)
        material[0, :80] = 1540.; material[1, :80] = 1000.; material[2, :80] = .002+edge[:80]
        np.save(args.out / 'material.npy', material)
        np.savez_compressed(args.out / 'refined_seed.npz', c_nominal=physical, c_effective=effective,
                            normalized_slowness=value.reshape(shape), x=xc, z=zc)
        run.update(status='CPU_CMA_same_objective_refinement_finished',
            finished_unix_s=time.time(), seconds=time.time()-started,
            cumulative_CMA_CPU_seconds=parent['seconds']+time.time()-started,
            final_phase_objective=final_loss, final_projected_gradient_inf=final_kkt,
            termination=termination, optimization_history=history, counts=counts,
            material_SHA256=digest(args.out/'material.npy'),
            physical_c_range=[float(physical.min()), float(physical.max())],
            RSS_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            rho_and_alpha_are_uniform_seeds_not_estimates=True,
            torch_imported='torch' in sys.modules)
        atomic(args.out / 'run.json', run)
        print(json.dumps({k: run[k] for k in ['status', 'seconds', 'initial_phase_objective',
            'final_phase_objective', 'initial_projected_gradient_inf', 'final_projected_gradient_inf',
            'termination', 'physical_c_range', 'RSS_peak_GiB']}), flush=True)
    except BaseException:
        run.update(status='CPU_CMA_refinement_interrupted_or_failed', error=traceback.format_exc(),
                   stopped_unix_s=time.time())
        atomic(args.out / 'run.json', run)
        raise


if __name__ == '__main__':
    main()
