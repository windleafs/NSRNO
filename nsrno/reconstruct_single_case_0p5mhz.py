"""Actual three-field native reconstruction from newly authorized 0.5 MHz RF.

A bounded first frequency stage. This is a new acquisition experiment with
its own resource ledger; no old physical-gradient budget is reset.
"""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
from .low_frequency_native_pool import Pool
from .low_frequency_0p5 import atomic, digest, SAMPLE, CASE
from .native_gn_coordinates import Coordinates
from .interval_lbfgs_adaptive import solve
from .pressure_resources import PressureResources
from .reconstruct_native_gn import source_manifest


class BudgetBoundary(Exception):
    pass


def array(path, value):
    temporary = path.with_name(path.stem + '.partial.npy')
    np.save(temporary, value)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--band-hz', type=float, nargs=2, default=[.2e6, .325e6])
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--maximum-evaluations', type=int, default=14)
    p.add_argument('--max-seconds', type=float, default=3600)
    p.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--cap-gib', type=float, default=10.)
    p.add_argument('--previous-phase', type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('Inspect existing completed evaluations; never restart implicitly')
    if not (1 <= a.iterations <= 24 and 1 <= a.maximum_evaluations <= 36 and
            0 < a.max_seconds <= 10800 and 0 < a.cap_gib <= 10):
        raise ValueError('Exceeded explicitly bounded new-stage resources')
    if not a.devices or len(a.devices) != len(set(a.devices)):
        raise ValueError('Distinct source worker GPUs required')
    if not .1e6 <= a.band_hz[0] < a.band_hz[1] <= 1.25e6:
        raise ValueError('Band must be contained in the acquired retained frequencies')
    dataset = json.loads((a.data / 'index.json').read_text())
    if (dataset['status'] != 'paired_0p5MHz_generation_finished' or
            dataset['case'] != CASE or dataset['sample'] != SAMPLE or
            dataset['completed_wave_solves'] != 44):
        raise ValueError('Require the actual completed single-case paired data')
    if Path('/proc', str(dataset['pid'])).exists():
        raise RuntimeError('Wait for the original generator to exit')
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory',
         '--format=csv,noheader'], text=True).strip()
    if gpu:
        raise RuntimeError('Another GPU experiment is live')
    reserve = 2 * 2**30 + 400 * 2**20
    if shutil.disk_usage('.').free < reserve + 300 * 2**20:
        raise RuntimeError('Archive closed deterministic caches first; keep checkpoint reserve')
    edge = np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    edge = np.broadcast_to(edge, (961, 1121))
    if a.previous_phase is None:
        seed = np.empty((3, 961, 1121), np.float32)
        seed[0] = 1540.
        seed[1] = 1000.
        seed[2] = .002 + edge
        seed[2, 80:] = .5 + edge[80:]
        initialization = 'known homogeneous tissue, fixed known coupling layer and edge attenuation'
    else:
        prior = json.loads((a.previous_phase / 'run.json').read_text())
        if prior['status'] != 'single_case_low_frequency_reconstruction_finished':
            raise ValueError('A previous frequency stage must have actually finished')
        path = a.previous_phase / 'final' / 'material.npy'
        if digest(path) != prior['material_SHA256']:
            raise ValueError('Previous accepted material changed')
        seed = np.load(path)
        initialization = str(path.resolve())
    if seed.shape != (3, 961, 1121) or seed.dtype != np.float32:
        raise ValueError('Require the full native independent three-field initial material')
    co = Coordinates(seed, edge)
    zero = np.zeros_like(seed, dtype=np.float64)
    low, high = co.bounds(trust=(1., 1., 10.))
    a.out = a.out.resolve()
    a.data = a.data.resolve()
    a.out.mkdir()
    for n in ['jobs', 'evaluations', 'checkpoints', 'final']:
        (a.out / n).mkdir()
    hashes = source_manifest()
    for name in [
        Path(__file__).name, 'acquisition_single_case_0p5.py',
        'low_frequency_native_worker.py', 'low_frequency_native_pool.py',
        'low_frequency_0p5.py', 'matrix_free_rf_response.py',
        'interval_lbfgs_adaptive.py', 'native_gn_coordinates.py',
        'pressure_resources.py']:
        hashes[name] = digest(Path(__file__).parent / name)
    for name in hashes:
        shutil.copy2(Path(__file__).parent / name, a.out / name)
    started = time.time()
    run = {
        'status': 'starting_actual_single_case_low_frequency_reconstruction',
        'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': started + a.max_seconds,
        'walltime_limit_checked_at_completed_evaluation_boundary': True,
        'data_root': str(a.data), 'data_index_SHA256': digest(a.data / 'index.json'),
        'sample': SAMPLE, 'case': CASE, 'source_f0_hz': .5e6,
        'band_hz': a.band_hz, 'initialization': initialization,
        'native_shape': list(seed.shape), 'spacing_m': 50e-6,
        'full_FOV_depth_mm': [1, 43], 'shots': '11 plane + 11 point3',
        'receivers': 192, 'record_us': 90,
        'unknowns': 'independent full native c, rho, positive tissue alpha',
        'all_three_fields_free': True, 'all_exterior_tissue_cells_free': True,
        'known_coupling_rows': [0, 79], 'shared_edges_or_smooth_c_constraint': False,
        'truth_read_by_optimizer': False, 'reserved_case_unread': 'Neg_47_Left',
        'new_phase_budget': True, 'old_physical_budget_not_reset': True,
        'maximum_new_complete_native_gradients': a.maximum_evaluations,
        'maximum_new_native_shots': 22 * a.maximum_evaluations,
        'maximum_accepted_updates': a.iterations,
        'completed_native_gradients': 0, 'attempted_native_gradients': 0,
        'accepted_steps': 0, 'devices': a.devices, 'allocator_cap_GiB': a.cap_gib,
        'per_card_limit_GiB': 16, 'source_SHA256': hashes,
        'objective': '.25*sum_family(||selected_complex_RF_error||^2/observed_energy) + .001 TV',
        'solver': 'bounded CPU L-BFGS, complete current native RF and material gradient',
        'preconditioner': 'spatially constant field scaling from the necessary first actual gradient',
        'no_material_Jacobian_HVP_or_identifiability_probes': True,
        'gradient_reused_at_same_initial_point_to_avoid_duplicate_propagation': True,
        'RF_archived_every_evaluation': True,
        'post_hoc_maps_every_accepted_steps': 4,
        'joint_imaging_success_claimed': False}
    atomic(a.out / 'run.json', run)
    array(a.out / 'initial_material.npy', seed)
    monitor = PressureResources(a.out)
    pool = None
    stopped = [0]
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda value, frame: stopped.__setitem__(0, value))
    cache = {}
    metric = None
    accepted = 0
    last_mapped = 0
    stop_reason = None
    completed = 0

    def freeze():
        for name, expected in hashes.items():
            if digest(Path(__file__).parent / name) != expected:
                raise RuntimeError('Frozen source changed: ' + name)
        if digest(a.data / 'index.json') != run['data_index_SHA256']:
            raise RuntimeError('The acquired dataset index changed')

    def actual(u):
        nonlocal completed
        if stopped[0]:
            raise BudgetBoundary('stop_requested_at_completed_evaluation_boundary')
        if completed >= a.maximum_evaluations:
            raise BudgetBoundary('new_stage_complete_gradient_budget')
        if time.time() >= run['deadline_unix_s']:
            raise BudgetBoundary('new_stage_deadline')
        if shutil.disk_usage('.').free < reserve + 40 * 2**20:
            raise BudgetBoundary('disk_reserve_for_atomic_material_checkpoint')
        freeze()
        material = co.material(u)
        run.update(status='actual_low_frequency_RF_and_material_gradient',
                   attempted_native_gradients=completed+1, accepted_steps=accepted)
        atomic(a.out / 'run.json', run)
        array(a.out / 'latest_trial_material.npy', material)
        rf, g = pool.evaluate(material, f'eval{completed+1:03d}')
        completed += 1
        tv, gtv, _, _ = co.regularization(u, .001)
        record = {'evaluation': completed, 'native_RF': rf, 'RF_loss': rf['loss'],
                  'TV': tv, 'objective': rf['loss']+tv,
                  'truth_read': False, 'seconds': time.time()-started}
        atomic(a.out / 'evaluations' / f'eval{completed:03d}.json', record)
        run.update(completed_native_gradients=completed,
                   actual_completed_new_forward_shots=22*completed,
                   actual_completed_new_material_gradient_shots=22*completed)
        atomic(a.out / 'run.json', run)
        return {'lower': rf['loss']+tv, 'upper': rf['loss']+tv,
                'physical_coordinate_gradient': g+gtv,
                'record': record, 'material': material, 'u': u.copy()}

    def saved_maps(folder):
        with (folder / 'maps.log').open('x') as log:
            subprocess.run([
                sys.executable, '-m', 'nsrno.evaluate_time_material',
                str(folder / 'material.npy'), '--sample', SAMPLE,
                '--out', str(folder / 'maps'),
                '--estimate-label', f'0.5 MHz, {accepted} accepted joint steps'],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)

    def checkpoint():
        nonlocal last_mapped
        folder = a.out / 'checkpoints' / f'accepted{accepted:03d}'
        folder.mkdir()
        array(folder / 'material.npy', cache['material'])
        os.link(a.out / 'initial_material.npy', folder / 'stage0_initial_material.npy')
        atomic(folder / 'accepted.json', cache['record'])
        saved_maps(folder)
        last_mapped = accepted

    def callback(x, value, point):
        nonlocal accepted
        accepted = point['accepted_step']
        cache.update(material=value['material'].copy(), u=value['u'].copy(),
                     gradient=value['gradient'].copy(),
                     record={**value['record'], **point})
        array(a.out / 'accepted_material.npy', cache['material'])
        array(a.out / 'accepted_gradient.npy', cache['gradient'].reshape(seed.shape))
        atomic(a.out / 'accepted_point.json', cache['record'])
        run.update(accepted_steps=accepted, current_RF_loss=value['record']['RF_loss'],
                   projected_gradient_relative=point['projected_gradient_relative'])
        atomic(a.out / 'run.json', run)
        print(json.dumps({'stage':'accepted', 'accepted_steps':accepted,
              'completed_native_gradients':completed,
              'RF_loss':value['record']['RF_loss'],
              'peak_Torch_GiB':value['record']['native_RF']['peak_Torch_GiB']}), flush=True)
        if accepted and accepted % 4 == 0:
            checkpoint()

    try:
        pool = Pool(a, monitor)
        first = actual(zero)
        rms = np.sqrt(np.mean(first['physical_coordinate_gradient'][:, co.mask]**2, axis=1))
        floor = max(float(rms.max())*1e-3, 1e-20)
        diagonal = np.maximum(rms, floor) / .2
        metric = np.broadcast_to(np.sqrt(diagonal)[:, None, None], seed.shape).copy()
        metric[:, ~co.mask] = 1.
        run.update(initial_actual_RF_loss=first['record']['RF_loss'],
                   initial_field_gradient_RMS=rms.tolist(),
                   global_coordinate_metric_diagonal=diagonal.tolist())
        atomic(a.out / 'run.json', run)
        first['gradient'] = (first['physical_coordinate_gradient']/metric).reshape(-1)
        reused = [False]

        def oracle(x):
            if not reused[0] and not np.any(x):
                reused[0] = True
                return first
            value = actual(x.reshape(seed.shape)/metric)
            value['gradient'] = (value['physical_coordinate_gradient']/metric).reshape(-1)
            return value

        try:
            _, _, termination = solve(
                oracle, zero.reshape(-1), (low*metric).reshape(-1),
                (high*metric).reshape(-1), iterations=a.iterations,
                history=8, maximum_lines=6, callback=callback, target=None)
            stop_reason = termination['stop']
        except BudgetBoundary as exc:
            stop_reason = str(exc)
            termination = {'stop': stop_reason, 'last_accepted_step': accepted,
                           'rejected_trial_not_selected': True}
        if not cache:
            raise RuntimeError('No complete necessary initial gradient')
        if accepted and accepted != last_mapped:
            checkpoint()
        final = a.out / 'final'
        array(final / 'material.npy', cache['material'])
        os.link(a.out / 'initial_material.npy', final / 'stage0_initial_material.npy')
        atomic(final / 'incumbent_RF.json', cache['record']['native_RF'])
        atomic(final / 'accepted.json', cache['record'])
        atomic(a.out / 'termination.json', termination)
        pool.close()
        pool = None
        if accepted:
            folder = a.out / 'checkpoints' / f'accepted{accepted:03d}'
            for suffix in ['png', 'metrics.json']:
                shutil.copy2(folder / ('maps.'+suffix), final / ('maps.'+suffix))
        else:
            saved_maps(final)
        monitor.sample('single_low_frequency_stage_finished')
        run.update(status='single_case_low_frequency_reconstruction_finished',
                   finished_unix_s=time.time(), seconds=time.time()-started,
                   accepted_steps=accepted, completed_native_gradients=completed,
                   final_RF_loss=cache['record']['RF_loss'], stop_reason=stop_reason,
                   material_SHA256=digest(final / 'material.npy'))
        atomic(a.out / 'run.json', run)
        print(json.dumps({k:run[k] for k in ['status','accepted_steps',
              'completed_native_gradients','initial_actual_RF_loss','final_RF_loss',
              'seconds','material_SHA256']}), flush=True)
    except BaseException:
        run.update(status='single_case_low_frequency_reconstruction_interrupted_or_failed',
                   error=traceback.format_exc(), stopped_unix_s=time.time())
        atomic(a.out / 'run.json', run)
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__ == '__main__':
    main()
