"""New scalar-pressure WRI trajectory after a complete old reconstruction.

This never stops another run, reuses old multipliers, deletes a bank, or reads
material truth. It defaults to one complete outer step followed by native RF
and separate post-hoc maps, so further iteration is a scientific decision.
"""
from pathlib import Path
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import numpy as np
from .conditioned_wave import FrequencyGeometry
from .wri_material import MaterialVariables
from .pressure_material_stencil import material_indices
from .pressure_material_normal import PressureMaterialNormal
from .pressure_worker import run_worker
from .pressure_resources import PressureResources
from .wri_worker import atomic
from .reconstruct_wri import command


GROUPS = {'lf': 180, 'point3': 270}
SOURCES = ['reconstruct_pressure_wri.py', 'pressure_worker.py', 'pressure_resources.py',
    'pressure_schur.py', 'pressure_material_stencil.py', 'pressure_material_normal.py',
    'pressure_receiver_normal.py', 'first_order_wavefield.py', 'conditioned_wave.py',
    'native_compact_weighted_pressure_gpu.py', 'native_compact_weighted_pressure.py',
    'gpu_sparse_lu.py', 'block_wavefield_step.py', 'dense_wave.py', 'frequency_group_data.py',
    'wri_material.py', 'coupled_coordinates.py', 'score_wri_material.py',
    'native_time.py', 'native_time.cu', 'reconstruct_wri.py', 'wri_worker.py',
    'time_acquisition.py', 'v5_observations.py', 'lf_frequency_data.py']


def live(pid):
    path = Path('/proc')/str(pid)/'cmdline'
    return path.exists() and bool(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source(args):
    source = args.source_root.resolve()
    run = json.loads((source/'run.json').read_text())
    if run['status'] != 'revised_step_finished' or live(run['pid']):
        raise RuntimeError('Require a naturally exited, complete source coordinator')
    if run['completed_step'] != args.source_step or run['step'] != args.source_step:
        raise RuntimeError('Source does not match the reviewed complete outer step')
    folder = source/f'step{args.source_step:03d}'
    if json.loads((folder/'complete_sweep.json').read_text())['status'] != 'complete':
        raise RuntimeError('Incomplete material/dual source boundary')
    for path in folder.glob('*_processes.json'):
        if any(live(pid) for pid in json.loads(path.read_text()).get('pids', [])):
            raise RuntimeError('A source GPU child is still live')
    if not (folder/'rf_score.json').exists() or not (folder/'maps.metrics.json').exists():
        raise RuntimeError('Require actual full RF and post-hoc map results before a route change')
    best = json.loads((source/'best_RF.json').read_text())
    material = Path(best['material'])
    if not material.is_absolute():
        material = material.resolve()
    if source not in material.parents or material.name != 'material.npy':
        raise RuntimeError('RF-selected source material is outside this experiment')
    score = material.parent/'rf_score.json'
    rf = json.loads(score.read_text())
    if abs(rf['primary_loss']-best['primary_loss']) > 1e-12 or rf.get('truth_read') is not False:
        raise RuntimeError('RF-only source candidate and saved score disagree')
    if not args.check_only:
        if args.decision is None:
            raise RuntimeError('Require the actual post-step route decision record')
        decision = json.loads(args.decision.read_text())
        if decision['source_step'] != args.source_step or decision['activate_pressure_route'] is not True:
            raise RuntimeError('Pressure route has not been selected from the complete result')
        if decision['source_material_sha256'] != sha(folder/'material.npy'):
            raise RuntimeError('Decision belongs to a different material result')
        if decision['source_RF_sha256'] != sha(folder/'rf_score.json'):
            raise RuntimeError('Decision belongs to a different actual RF result')
    return source, run, material, score


def run_phase(folder, step, mode, assignments, args, monitor, cap, proposal=None, label=None):
    tag = label or mode
    monitor.wait_budget(args.devices, cap, tag+'_admission')
    if mode == 'field':
        info = Path('/proc/meminfo').read_text().splitlines()
        available = int(next(s.split()[1] for s in info if s.startswith('MemAvailable:')))/2**20
        if available < 16*len(args.devices):
            raise RuntimeError('Insufficient host memory for native pressure factors')
    context = mp.get_context('spawn')
    processes = []
    try:
        for rank, device in enumerate(args.devices):
            process = context.Process(target=run_worker, args=(rank, device, assignments[rank],
                folder, step, mode, cap, args.field_cg, .003, proposal, label))
            process.start()
            processes.append(process)
        atomic(folder/f'{tag}_processes.json', {'pids': [p.pid for p in processes],
            'devices': args.devices, 'phase': mode, 'allocator_cap_GiB': cap,
            'parent_pid': os.getpid(), 'truth_read': False})
        while any(p.is_alive() for p in processes):
            monitor.sample(tag)
            if monitor.owned_above_limit:
                raise RuntimeError('This run exceeded the per-card 16 GiB process-tree budget')
            if any(p.exitcode is not None and p.exitcode != 0 for p in processes):
                raise RuntimeError('A pressure worker failed; immutable RHS and warm fields retained')
            time.sleep(20)
        for p in processes:
            p.join()
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError('A pressure worker failed; do not treat partial fields as a complete sweep')
        summaries = [json.loads((folder/f'{tag}_rank{r}.json').read_text()) for r in range(len(processes))]
        expected = {(s, p) for assignment in assignments for s, p in assignment}
        observed = [tuple(x) for row in summaries for x in row['assignments']]
        if len(observed) != 450 or set(observed) != expected:
            raise RuntimeError('Require all 450 frequency tasks exactly once')
        return summaries
    finally:
        # Only freshly spawned children owned by this function are contained.
        # No other project or earlier coordinator is signalled.
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)


def create_banks(root, source, shape, monitor):
    folder = root/'banks'
    folder.mkdir()
    field_bytes = sum(GROUPS.values())*11*int(np.prod(shape))*8
    # X, Y, immutable per-generation R, three worker Grams and merged Gram.
    required = 3*field_bytes + 4*(190+19)*int(np.prod(shape))*8 + 12*2**30
    if shutil.disk_usage(root).free < required:
        raise RuntimeError('Insufficient free NVMe for pressure banks, Gram statistics and reserve')
    for group, count in GROUPS.items():
        for kind in ['X', 'Y', 'R']:
            bank = np.lib.format.open_memmap(folder/f'{group}_{kind}.npy', mode='w+',
                dtype=np.complex64, shape=(count, 11, *shape))
            if kind == 'X':
                previous = np.load(source/'banks'/f'{group}_X.npy', mmap_mode='r')
                if previous.shape != (count, 11, 3, *shape):
                    raise RuntimeError('Source warm-pressure shape differs from the complete native bank')
                for position in range(count):
                    bank[position] = previous[position, :, 0]
                    if position % 16 == 15:
                        monitor.sample('copying_pressure_warm_fields')
                del previous
            bank.flush()
            del bank
        for kind, dtype, bank_shape in [('Ydata', np.complex64, (count, 11, 192)),
                ('D', np.complex64, (count, 11, 192)), ('generation', np.int16, (count,)),
                ('penalty', np.float64, (count,))]:
            bank = np.lib.format.open_memmap(folder/f'{group}_{kind}.npy', mode='w+',
                                             dtype=dtype, shape=bank_shape)
            bank[:] = 0
            bank.flush()
            del bank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-step', type=int, default=4)
    parser.add_argument('--decision', type=Path)
    parser.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--cap-gib', type=float, default=14.5)
    parser.add_argument('--field-cg', type=int, default=64)
    parser.add_argument('--material-cg', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--tv', type=float, default=.001)
    parser.add_argument('--damping', type=float, default=.01)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not args.devices or len(set(args.devices)) != len(args.devices) or len(args.devices) > 3:
        raise ValueError('Use one to three distinct devices')
    if not 0 < args.cap_gib <= 14.5 or args.iterations < 1:
        raise ValueError('Invalid memory or iteration budget')
    source, old_run, seed, seed_score = validate_source(args)
    if args.check_only:
        print(json.dumps({'source_complete': True, 'source_step': args.source_step,
            'source_coordinator_exited': True, 'seed': str(seed), 'seed_SHA256': sha(seed),
            'all450_all11_all192': True, 'bank_writes': False, 'signals_sent': False}), flush=True)
        return
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'run.json').exists() or (root/'banks').exists():
        raise RuntimeError('Existing pressure output; no automatic partial-bank continuation')
    source_lock = open(source/'gpu_coordinator.lock', 'a')
    fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = open(root/'coordinator.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monitor = PressureResources(root)
    run = None
    try:
        # A second positive check under the source coordinator lock.
        source, old_run, seed, seed_score = validate_source(args)
        archive = root/'source'
        archive.mkdir()
        hashes = {}
        for name in SOURCES:
            path = Path(__file__).with_name(name)
            hashes[name] = sha(path)
            shutil.copy2(path, archive/name)
        shutil.copy2(args.decision, root/'activation_decision.json')
        shutil.copy2(source/'run.json', root/'source_run_complete.json')
        run = {'status': 'creating_pressure_banks', 'pid': os.getpid(), 'boot_id': monitor.boot,
            'sample': old_run['sample'], 'sources': list(GROUPS), 'frequency_counts': GROUPS,
            'transmissions': 11, 'receivers': 192, 'native_material_shape': [3, 961, 1121],
            'unknowns': ['density', 'compressibility', 'attenuation_scale'],
            'physical_outputs': ['sound_speed', 'mass_density', 'tissue_attenuation'],
            'devices': args.devices, 'allocator_cap_GiB': args.cap_gib, 'model_per_card_limit_GiB': 16,
            'source_sha256': hashes, 'source_root': str(source), 'source_step': args.source_step,
            'seed_RF_only': str(seed), 'seed_SHA256': sha(seed), 'multipliers': 'new scalar-pressure duals, initially zero',
            'initial_fields': 'pressure-only warm values from complete source; no old multipliers',
            'iterations': args.iterations, 'step': 0, 'created_unix_s': time.time(),
            'physical_equation_changed': False, 'lifted_residual_metric_changed': True,
            'truth_read_by_optimizer': False, 'HF_use': 'native score diagnostic until successful base reconstruction',
            'acquisition_scope': 'existing plane and point 3-MHz data; no claim of plane-only sufficiency',
            'known_approximation': 'frequency equation omits finite-record endpoint terms; native RF score required',
            'joint_reconstruction_success_claimed': False}
        atomic(root/'run.json', run)
        initial = root/'initial'
        initial.mkdir()
        for name in ['material.npy', 'parameters.npy']:
            shutil.copy2(seed.parent/name, initial/name)
        variables = MaterialVariables(np.load(initial/'material.npy'), np.load(initial/'parameters.npy'))
        current = variables.initial.copy()
        np.save(initial/'variables.npy', current)
        rf = json.loads(seed_score.read_text())
        rf.update(reused_saved_full_native_RF=True, source_score=str(seed_score), material_SHA256=sha(seed))
        atomic(initial/'rf_score.json', rf)
        best = {'step': 0, 'material': str(initial/'material.npy'), 'primary_loss': rf['primary_loss']}
        atomic(root/'best_RF.json', best)
        spec = json.loads(Path('out/v5/conditioned_lf_full_v1/index.json').read_text())['spec']
        geometry = FrequencyGeometry(spec)
        indices = material_indices(geometry)
        create_banks(root, source, variables.shape, monitor)
        tasks = [(s, p) for s, count in GROUPS.items() for p in range(count)]
        assignments = [tasks[rank::len(args.devices)] for rank in range(len(args.devices))]
        for step in range(1, args.iterations+1):
            folder = root/f'step{step:03d}'
            folder.mkdir()
            (folder/'candidates').mkdir()
            np.save(folder/'old_variables.npy', current)
            np.save(folder/'old_material.npy', variables.material(current))
            run.update(status='pressure_field_sweep', step=step, phase_started_unix_s=time.time())
            atomic(root/'run.json', run)
            field = run_phase(folder, step, 'field', assignments, args, monitor, args.cap_gib)
            atomic(folder/'field_convergence.json', {'frequencies': 450, 'transmissions_each': 11,
                'unconverged_shots': sum(r['unconverged_shots'] for r in field), 'requested_tolerance': .003,
                'maximum_physical_global_normal_residual': max(r['max_physical_global_normal_residual'] for r in field),
                'maximum_preconditioned_global_normal_residual': max(r['max_preconditioned_global_normal_residual'] for r in field),
                'data_loss': sum(r['data_loss'] for r in field), 'PDE_loss': sum(r['PDE_loss'] for r in field),
                'scope': 'field normal equations; not joint material accuracy'})
            run['status'] = 'pressure_material_statistics'
            atomic(root/'run.json', run)
            run_phase(folder, step, 'statistics', assignments, args, monitor, 8.)
            # Merge on CPU. No pressure LU or receiver basis is resident now.
            merged = {}
            for kind, count in [('gram', 190), ('right', 19)]:
                target = np.lib.format.open_memmap(folder/f'{kind}.npy', mode='w+',
                    dtype=np.float64, shape=(count, *variables.shape))
                for rank in range(len(args.devices)):
                    partial = np.load(folder/f'{kind}_rank{rank}.npy', mmap_mode='r')
                    for i in range(count):
                        if rank == 0:
                            target[i] = partial[i]
                        else:
                            target[i] += partial[i]
                    del partial
                target.flush()
                merged[kind] = target
            run['status'] = 'coupled_pressure_material_update'
            atomic(root/'run.json', run)
            normal = PressureMaterialNormal(variables, indices, merged['gram'], merged['right'])
            candidate_count = 0
            def cost(proposal):
                nonlocal candidate_count
                tag = f'candidate{candidate_count:03d}'
                path = folder/'candidates'/f'{tag}_material.npy'
                np.save(path, variables.material(proposal))
                run.update(status='fixed_pressure_candidate_cost', candidate=candidate_count)
                atomic(root/'run.json', run)
                rows = run_phase(folder, step, 'candidate', assignments, args, monitor, 2.5, path, tag)
                change = sum(row['equation_cost_change'] for row in rows)
                atomic(folder/'candidates'/f'{tag}_cost.json', {'equation_cost_change': change,
                    'frequencies': 450, 'all11_each': True, 'new_propagation': False, 'truth_read': False})
                candidate_count += 1
                return change
            def progress(row):
                atomic(folder/'material_CG_progress.json', row)
                if row['iteration'] % 8 == 0:
                    print(json.dumps({'step': step, 'material_CG': row}), flush=True)
            proposed, record = normal.solve(current, cost, args.tv, args.damping,
                maximum=args.material_cg, callback=progress)
            np.save(folder/'variables.npy', proposed)
            np.save(folder/'material.npy', variables.material(proposed))
            np.save(folder/'parameters.npy', variables.parameters(proposed))
            atomic(folder/'material_update.json', record)
            del normal, merged
            run['status'] = 'pressure_dual_update'
            atomic(root/'run.json', run)
            run_phase(folder, step, 'dual', assignments, args, monitor, 2.5, folder/'material.npy')
            atomic(folder/'complete_sweep.json', {'status': 'complete', 'step': step,
                'frequencies': 450, 'transmissions_each': 11, 'unix_s': time.time(), 'truth_read': False})
            run['status'] = 'full_native_RF_score'
            atomic(root/'run.json', run)
            used = monitor.sample('full_native_RF_admission')
            device = min(args.devices, key=lambda d: used[d]['whole_MiB'])
            monitor.wait_budget([device], 1.1, 'full_native_RF_admission')
            command([sys.executable, '-m', 'nsrno.score_wri_material', str(folder/'material.npy'),
                '--sample', run['sample'], '--out', str(folder/'rf_score.json'),
                '--device', str(device), '--cap-gib', '1.1'], folder/'rf_score.log', monitor, 'full_native_RF')
            rf = json.loads((folder/'rf_score.json').read_text())
            if rf['primary_loss'] < best['primary_loss']:
                best = {'step': step, 'material': str(folder/'material.npy'), 'primary_loss': rf['primary_loss']}
                atomic(root/'best_RF.json', best)
            run['status'] = 'post_hoc_maps_only'
            atomic(root/'run.json', run)
            shutil.copy2(initial/'material.npy', folder/'stage0_initial_material.npy')
            command([sys.executable, '-m', 'nsrno.evaluate_time_material', str(folder/'material.npy'),
                '--sample', run['sample'], '--out', str(folder/'maps'), '--estimate-label', 'Scalar-pressure WRI'],
                folder/'maps.log', monitor, 'post_hoc_maps_only')
            row = {'step': step, 'primary_full_RF_loss': rf['primary_loss'], 'best_RF': best,
                   'material_accepted': record['accepted'], 'truth_read_by_optimizer': False}
            with (root/'history.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
            current = proposed
            run['completed_step'] = step
        run.update(status='planned_pressure_sweeps_finished', best_RF=best,
                   joint_reconstruction_success_claimed=False)
        atomic(root/'run.json', run)
    except BaseException:
        if run is not None:
            run.update(status='failed', error=traceback.format_exc())
            atomic(root/'run.json', run)
        raise
    finally:
        monitor.log.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        fcntl.flock(source_lock, fcntl.LOCK_UN)
        source_lock.close()


if __name__ == '__main__':
    main()
