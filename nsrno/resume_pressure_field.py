"""Explicit partial-field recovery after the recorded block-CG failure.

Completed frequency fields are reused. Unfinished fields use their immutable
R/D generation; no Y_base is added twice. All450 statistics/material/dual/RF
phases retain the original definitions. This entry sends no process signals.
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
from .wri_worker import atomic
from .wri_material import MaterialVariables
from .pressure_material_stencil import material_indices
from .pressure_material_normal import PressureMaterialNormal
from .conditioned_wave import FrequencyGeometry
from .pressure_resources import PressureResources
from .pressure_stable_field import run_stable_field
from .reconstruct_pressure_wri import run_phase, GROUPS
from .reconstruct_wri import command


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def live(pid):
    p = Path('/proc')/str(pid)/'cmdline'
    return p.exists() and bool(p.read_bytes())


def read_records(folder, tag):
    records = {}
    for path in folder.glob(tag+'_rank*.log'):
        for line in path.open():
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if 'block' not in row:
                continue
            key = (row['source'], row['position'])
            b = row['block']
            values = b['physical_global_normal_residual']+b['preconditioned_global_normal_residual']+b['augmented_cost_after']
            if not np.isfinite(values).all():
                raise RuntimeError('Nonfinite committed field log; requires separate recovery')
            if key in records:
                raise RuntimeError('Repeated committed frequency in source logs')
            records[key] = row
    return records


def inspect(root):
    run = json.loads((root/'run.json').read_text())
    if run['status'] != 'failed' or run.get('field_log_tag', 'field') != 'field' or run['step'] != 1:
        raise RuntimeError('Only the original failed first pressure field sweep is recoverable here')
    if run['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise RuntimeError('Source boot changed; no automatic adoption')
    folder = root/'step001'
    pids = [run['pid'], *json.loads((folder/'field_processes.json').read_text())['pids']]
    if any(live(pid) for pid in pids):
        raise RuntimeError('A source process is still live')
    for name in ['material.npy', 'statistics_processes.json', 'complete_sweep.json', 'field_stable_processes.json']:
        if (folder/name).exists():
            raise RuntimeError('Source advanced beyond the original failed field sweep: '+name)
    errors = [json.loads(p.read_text())['error'] for p in folder.glob('field_rank*.failed.json')]
    if not any('Block field curvature lost Hermitian symmetry' in e for e in errors):
        raise RuntimeError('This recovery requires the diagnosed block-curvature exception')
    for name, digest in run['source_sha256'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise RuntimeError('Frozen physical/numerical source changed: '+name)
    for name in ['old_material.npy', 'old_variables.npy']:
        value = np.load(folder/name, mmap_mode='r')
        if value.shape != (3, 961, 1121) or not np.isfinite(value).all():
            raise RuntimeError('Frozen native field-stage material missing')
    records = read_records(folder, 'field')
    tasks = [(s, i) for s, n in GROUPS.items() for i in range(n)]
    if not set(records).issubset(set(tasks)):
        raise RuntimeError('Source contains a wrong frequency key')
    for source, count in GROUPS.items():
        arrays = {kind: np.load(root/'banks'/f'{source}_{kind}.npy', mmap_mode='r')
                  for kind in ['X', 'Y', 'Ydata', 'R', 'D', 'generation', 'penalty']}
        for kind, value in arrays.items():
            shape = ((count, 11, 961, 1121) if kind in ['X', 'Y', 'R'] else
                     (count, 11, 192) if kind in ['Ydata', 'D'] else (count,))
            if value.shape != shape:
                raise RuntimeError('Wrong partial-bank shape: '+source+'_'+kind)
            if kind in ['X', 'Y', 'Ydata', 'R', 'D'] and value.dtype != np.complex64:
                raise RuntimeError('Partial pressure fields must use complex64')
        generation, penalty = arrays['generation'], arrays['penalty']
        if not np.isin(generation, [0, 1]).all() or not np.isfinite(penalty).all():
            raise RuntimeError('Invalid immutable RHS generations or penalties')
        if any(generation[i] != 1 or penalty[i] <= 0 for s, i in records if s == source):
            raise RuntimeError('Completed field lacks its immutable positive-penalty RHS')
        if np.any(arrays['Ydata'][generation == 0] != 0):
            raise RuntimeError('Unstarted first-generation data multipliers are not initial zero')
    weights = Path('out/v5/forward_round8_dense_full/last.pt')
    if weights.stat().st_mtime > run['created_unix_s']:
        raise RuntimeError('Original inverse weights changed after source launch')
    return run, folder, records, [t for t in tasks if t not in records], weights


def remaining_fields(folder, assignments, args, monitor):
    monitor.wait_budget(args.devices, args.cap_gib, 'stable_field_admission')
    context = mp.get_context('spawn')
    processes = []
    try:
        for rank, device in enumerate(args.devices):
            process = context.Process(target=run_stable_field, args=(rank, device, assignments[rank],
                folder, 1, args.cap_gib, args.field_cg, .003))
            process.start()
            processes.append(process)
        atomic(folder/'field_stable_processes.json', {'pids': [p.pid for p in processes],
            'devices': args.devices, 'parent_pid': os.getpid(), 'solver': 'independent batched CG',
            'remaining_frequencies': sum(map(len, assignments)), 'all11_each': True})
        while any(p.is_alive() for p in processes):
            monitor.sample('stable_field')
            if monitor.owned_above_limit or any(p.exitcode is not None and p.exitcode != 0 for p in processes):
                raise RuntimeError('Stable field worker failure; preserve the original RHS ledger')
            time.sleep(20)
        for p in processes:
            p.join()
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError('Stable field worker exited unsuccessfully')
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--field-cg', type=int, default=64)
    p.add_argument('--cap-gib', type=float, default=14.5)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    root = args.out.resolve()
    source, folder, completed, remaining, weights = inspect(root)
    if not 0 < args.cap_gib <= 14.5 or args.field_cg < 1:
        raise ValueError('Invalid numerical field budget')
    if args.check_only:
        print(json.dumps({'source_all_handles_gone': True, 'complete_frequencies_reused': len(completed),
            'remaining_frequencies': len(remaining), 'all450_all11_all192': True,
            'immutable_R_D_and_generations_valid': True, 'bank_writes': False,
            'signals_sent': False}), flush=True)
        return
    lock = open(root/'coordinator.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monitor, run = None, None
    try:
        source, folder, completed, remaining, weights = inspect(root)
        archive = root/'source_stable_field_step001'
        archive.mkdir(exist_ok=False)
        shutil.copy2(root/'run.json', archive/'run_before_stable_recovery.json')
        shutil.copy2(root/'resource_summary.json', archive/'source_resource_summary.json')
        hashes = {}
        names = set(source['source_sha256']) | {'pressure_stable_cg.py', 'pressure_stable_field.py', Path(__file__).name}
        for name in sorted(names):
            path = Path(__file__).with_name(name)
            hashes[name] = sha(path)
            shutil.copy2(path, archive/name)
        frozen_weights = archive/'field_inverse_weights.pt'
        shutil.copy2(weights, frozen_weights)
        checkpoint = {'type': 'partial_pressure_field_RHS_generation', 'status': 'resuming',
            'source_pid': source['pid'], 'boot_id': source['boot_id'], 'source_handles_gone': True,
            'complete_frequency_keys': list(completed), 'remaining_frequency_keys': remaining,
            'old_material_sha256': sha(folder/'old_material.npy'),
            'old_variables_sha256': sha(folder/'old_variables.npy'),
            'source_log_sha256': {x.name: sha(x) for x in folder.glob('field_rank*.log')},
            'source_code_sha256': source['source_sha256'], 'signals_sent': False,
            'R_D_rule': 'gen1 immutable RHS; gen0 original complete zero dual; no Y_base double addition'}
        atomic(folder/'field_partial_resume.json', checkpoint)
        run = dict(source)
        run.setdefault('solver_changes', []).append({'type': 'field_numerical_partial_resume',
            'source_pid': source['pid'], 'pid': os.getpid(), 'unix_s': time.time(),
            'source_sha256': hashes, 'complete_frequencies_reused': len(completed),
            'remaining_frequencies': len(remaining), 'solver': 'independent batched CG with energy curvature',
            'physical_equation_changed': False, 'observations_changed': False,
            'requested_field_tolerance': .003, 'maximum_iterations': args.field_cg})
        run.update(status='pressure_field_sweep', pid=os.getpid(), source_sha256=hashes,
            field_log_tag='field_stable', resumed_complete_field_count=len(completed),
            stable_field_checkpoint=str(frozen_weights), stable_field_checkpoint_sha256=sha(frozen_weights),
            error=None, allocator_cap_GiB=args.cap_gib)
        atomic(root/'run.json', run)
        monitor = PressureResources(root)
        args.devices = source['devices']
        args.sample = source['sample']
        assignments = [remaining[r::len(args.devices)] for r in range(len(args.devices))]
        remaining_fields(folder, assignments, args, monitor)
        stable = read_records(folder, 'field_stable')
        if set(stable) != set(remaining) or set(stable)&set(completed):
            raise RuntimeError('Missing, repeated or reprocessed completed frequency')
        all_records = list(completed.values())+list(stable.values())
        if len(all_records) != 450:
            raise RuntimeError('Require all450 field records before material statistics')
        atomic(folder/'field_convergence.json', {'frequencies': 450, 'transmissions_each': 11,
            'original_frequencies_reused': len(completed), 'stable_frequencies': len(stable),
            'unconverged_shots': sum(sum(a > .003 or b > .003 for a, b in zip(
                x['block']['physical_global_normal_residual'], x['block']['preconditioned_global_normal_residual'])) for x in all_records),
            'maximum_physical_global_normal_residual': max(max(x['block']['physical_global_normal_residual']) for x in all_records),
            'maximum_preconditioned_global_normal_residual': max(max(x['block']['preconditioned_global_normal_residual']) for x in all_records),
            'requested_tolerance': .003, 'scope': 'field normal only; no material recovery claim'})
        checkpoint.update(status='complete_field_sweep', resume_pid=os.getpid())
        atomic(folder/'field_partial_resume.json', checkpoint)
        tasks = [(s, i) for s, n in GROUPS.items() for i in range(n)]
        full_assignments = [tasks[r::len(args.devices)] for r in range(len(args.devices))]
        run['status'] = 'pressure_material_statistics'
        atomic(root/'run.json', run)
        run_phase(folder, 1, 'statistics', full_assignments, args, monitor, 8.)
        initial = root/'initial'
        variables = MaterialVariables(np.load(initial/'material.npy'), np.load(initial/'parameters.npy'))
        current = np.load(folder/'old_variables.npy')
        spec = json.loads(Path('out/v5/conditioned_lf_full_v1/index.json').read_text())['spec']
        indices = material_indices(FrequencyGeometry(spec))
        merged = {}
        for kind, count in [('gram', 190), ('right', 19)]:
            result = np.lib.format.open_memmap(folder/f'{kind}.npy', mode='w+', dtype=np.float64,
                                              shape=(count, *variables.shape))
            for rank in range(len(args.devices)):
                partial = np.load(folder/f'{kind}_rank{rank}.npy', mmap_mode='r')
                for i in range(count):
                    if rank == 0: result[i] = partial[i]
                    else: result[i] += partial[i]
                del partial
            result.flush()
            merged[kind] = result
        run['status'] = 'coupled_pressure_material_update'
        atomic(root/'run.json', run)
        normal = PressureMaterialNormal(variables, indices, merged['gram'], merged['right'])
        counter = 0
        def cost(proposal):
            nonlocal counter
            tag = f'candidate{counter:03d}'
            file = folder/'candidates'/f'{tag}_material.npy'
            np.save(file, variables.material(proposal))
            run.update(status='fixed_pressure_candidate_cost', candidate=counter)
            atomic(root/'run.json', run)
            rows = run_phase(folder, 1, 'candidate', full_assignments, args, monitor, 2.5, file, tag)
            counter += 1
            return sum(x['equation_cost_change'] for x in rows)
        proposed, record = normal.solve(current, cost, maximum=64,
            callback=lambda row: atomic(folder/'material_CG_progress.json', row))
        np.save(folder/'variables.npy', proposed)
        np.save(folder/'material.npy', variables.material(proposed))
        np.save(folder/'parameters.npy', variables.parameters(proposed))
        atomic(folder/'material_update.json', record)
        del normal, merged
        run['status'] = 'pressure_dual_update'
        atomic(root/'run.json', run)
        run_phase(folder, 1, 'dual', full_assignments, args, monitor, 2.5, folder/'material.npy')
        atomic(folder/'complete_sweep.json', {'status': 'complete', 'step': 1, 'frequencies': 450,
            'transmissions_each': 11, 'stable_field_recovery': True, 'unix_s': time.time(), 'truth_read': False})
        run['status'] = 'full_native_RF_score'
        atomic(root/'run.json', run)
        used = monitor.sample('native_RF_admission')
        device = min(args.devices, key=lambda d: used[d]['whole_MiB'])
        monitor.wait_budget([device], 1.1, 'native_RF_admission')
        command([sys.executable, '-m', 'nsrno.score_wri_material', str(folder/'material.npy'),
            '--sample', args.sample, '--out', str(folder/'rf_score.json'), '--device', str(device),
            '--cap-gib', '1.1'], folder/'rf_score.log', monitor, 'native_RF_score')
        rf = json.loads((folder/'rf_score.json').read_text())
        best = json.loads((root/'best_RF.json').read_text())
        if rf['primary_loss'] < best['primary_loss']:
            best = {'step': 1, 'material': str(folder/'material.npy'), 'primary_loss': rf['primary_loss']}
            atomic(root/'best_RF.json', best)
        run['status'] = 'post_hoc_maps_only'
        atomic(root/'run.json', run)
        shutil.copy2(initial/'material.npy', folder/'stage0_initial_material.npy')
        command([sys.executable, '-m', 'nsrno.evaluate_time_material', str(folder/'material.npy'),
            '--sample', args.sample, '--out', str(folder/'maps'), '--estimate-label', 'Scalar-pressure WRI'],
            folder/'maps.log', monitor, 'post_hoc_maps_only')
        with (root/'history.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': 1, 'primary_full_RF_loss': rf['primary_loss'],
                'material_accepted': record['accepted'], 'stable_field_recovery': True,
                'best_RF': best, 'truth_read_by_optimizer': False})+'\n')
        run.update(status='planned_pressure_sweeps_finished', completed_step=1, best_RF=best,
            joint_reconstruction_success_claimed=False)
        atomic(root/'run.json', run)
    except BaseException:
        if run is not None:
            run.update(status='failed', error=traceback.format_exc())
            atomic(root/'run.json', run)
        raise
    finally:
        if monitor is not None: monitor.log.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
