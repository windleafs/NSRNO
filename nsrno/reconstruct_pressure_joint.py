"""One complete simultaneous field/material step from completed pressure WRI.

It requires actual source RF/maps and an agent's result-based route decision.
No active process is stopped or adopted. Existing scalar pressure banks are
continued in place; scratch trial banks are reusable, not archival snapshots.
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
from .wri_material import MaterialVariables
from .coupled_coordinates import dot
from .pressure_joint_direction import JointMaterialDirection
from .pressure_joint_worker import run_joint_worker
from .pressure_resources import PressureResources
from .wri_worker import atomic
from .reconstruct_wri import command


GROUPS = {'lf': 180, 'point3': 270}
NEW_SOURCES = ['reconstruct_pressure_joint.py', 'pressure_joint_worker.py',
               'pressure_joint_direction.py', 'pressure_joint_kernels.py', 'pressure_stable_cg.py']


def live(pid):
    p = Path('/proc')/str(pid)/'cmdline'
    return p.exists() and bool(p.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root, decision=None, check_only=False):
    run = json.loads((root/'run.json').read_text())
    if run['status'] not in ['planned_pressure_sweeps_finished', 'planned_joint_sweeps_finished']:
        raise RuntimeError('Require a completed pressure material/dual/RF/maps boundary')
    if live(run['pid']) or run['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise RuntimeError('Source coordinator must have naturally exited in the current boot')
    step = run['completed_step']
    if step != run['step'] or run['sources'] != list(GROUPS):
        raise RuntimeError('Wrong complete pressure step or acquisition groups')
    previous = root/f'step{step:03d}'
    complete = json.loads((previous/'complete_sweep.json').read_text())
    if complete['status'] != 'complete' or complete['step'] != step:
        raise RuntimeError('Source contains incomplete pressure multipliers')
    for p in previous.glob('*_processes.json'):
        if any(live(pid) for pid in json.loads(p.read_text())['pids']):
            raise RuntimeError('A source pressure worker is still live')
    for name in ['variables.npy', 'material.npy', 'parameters.npy']:
        value = np.load(previous/name, mmap_mode='r')
        if value.shape != (3, 961, 1121) or not np.isfinite(value).all():
            raise RuntimeError('Missing native three-parameter checkpoint: '+name)
    rf = json.loads((previous/'rf_score.json').read_text())
    if not np.isfinite(rf['primary_loss']) or rf.get('truth_read') is not False:
        raise RuntimeError('Source lacks a valid observation-only native RF score')
    if not (previous/'maps.metrics.json').exists():
        raise RuntimeError('Require the source actual post-hoc maps before choosing a route')
    for name, digest in run['source_sha256'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise RuntimeError('Frozen source changed: '+name)
    if 'field_inverse_checkpoint' in run:
        if sha(Path(run['field_inverse_checkpoint'])) != run['field_inverse_checkpoint_sha256']:
            raise RuntimeError('Source field-inverse checkpoint changed')
    if 'stable_field_checkpoint' in run:
        if sha(Path(run['stable_field_checkpoint'])) != run['stable_field_checkpoint_sha256']:
            raise RuntimeError('Source stable field-inverse checkpoint changed')
    for source, count in GROUPS.items():
        for kind in ['X', 'Y', 'R', 'Ydata', 'D', 'generation', 'penalty']:
            value = np.load(root/'banks'/f'{source}_{kind}.npy', mmap_mode='r')
            shape = ((count, 11, 961, 1121) if kind in ['X', 'Y', 'R'] else
                     (count, 11, 192) if kind in ['Ydata', 'D'] else (count,))
            if value.shape != shape:
                raise RuntimeError('Wrong complete bank shape: '+source+'_'+kind)
            if kind in ['X', 'Y', 'R', 'Ydata', 'D'] and value.dtype != np.complex64:
                raise RuntimeError('Complete field banks must use complex64')
            if kind == 'generation' and not np.all(value == step):
                raise RuntimeError('The scalar RHS ledger is not a complete outer generation')
            if kind == 'penalty' and (not np.isfinite(value).all() or not np.all(value > 0)):
                raise RuntimeError('Frozen pressure penalties must remain positive and finite')
    if (root/f'step{step+1:03d}').exists():
        raise RuntimeError('Next-step directory exists; no automatic partial adoption')
    if not check_only:
        if decision is None:
            raise RuntimeError('Require the actual source results and a saved route decision')
        selected = json.loads(decision.read_text())
        if selected['source_step'] != step or selected['activate_joint_route'] is not True:
            raise RuntimeError('The joint route was not selected from this complete result')
        if selected['source_material_sha256'] != sha(previous/'material.npy') or selected['source_RF_sha256'] != sha(previous/'rf_score.json'):
            raise RuntimeError('Route decision does not match the actual source material/RF')
    return run, step, previous


def phase(folder, step, mode, assignments, args, monitor, cap,
          proposal=None, tag=None, line_scale=1.):
    tag = tag or mode
    monitor.wait_budget(args.devices, cap, tag+'_admission')
    if mode in ['field_joint', 'candidate_first']:
        available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                             if s.startswith('MemAvailable:')))/2**20
        if available < 16*len(args.devices):
            raise RuntimeError('Insufficient host memory for native pressure factors')
    context = mp.get_context('spawn')
    processes = []
    try:
        for rank, device in enumerate(args.devices):
            process = context.Process(target=run_joint_worker, args=(rank, device,
                assignments[rank], folder, step, mode, cap, args.field_cg, .003,
                proposal, tag, line_scale))
            process.start()
            processes.append(process)
        atomic(folder/f'{tag}_processes.json', {'pids': [p.pid for p in processes],
            'devices': args.devices, 'phase': mode, 'parent_pid': os.getpid(),
            'allocator_cap_GiB': cap, 'truth_read': False})
        while any(p.is_alive() for p in processes):
            monitor.sample(tag)
            if monitor.owned_above_limit:
                raise RuntimeError('Joint reconstruction exceeded the per-card model 16 GiB budget')
            if any(p.exitcode is not None and p.exitcode != 0 for p in processes):
                raise RuntimeError('A joint worker failed; immutable RHS and trial state retained')
            time.sleep(20)
        for p in processes:
            p.join()
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError('Joint worker exit; no automatic partial continuation')
        rows = [json.loads((folder/f'{tag}_rank{rank}.json').read_text()) for rank in range(len(processes))]
        observed = [tuple(x) for row in rows for x in row['assignments']]
        expected = {x for assignment in assignments for x in assignment}
        if len(observed) != 450 or set(observed) != expected or any(r['status'] != 'complete' for r in rows):
            raise RuntimeError('Require every one of the 450 complete frequency blocks')
        return rows
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=10)


def init_trial_bank(scratch, suffix, shape):
    for source, count in GROUPS.items():
        path = scratch/f'{source}_{suffix}_X.npy'
        if path.exists():
            bank = np.load(path, mmap_mode='r+')
            if bank.shape != (count, 11, *shape) or bank.dtype != np.complex64:
                raise RuntimeError('Unexpected reusable trial-bank header')
        else:
            bank = np.lib.format.open_memmap(path, mode='w+', dtype=np.complex64,
                                              shape=(count, 11, *shape))
        bank.flush()
        del bank


def commit_trial(root, scratch, suffix, folder, proposal, alpha, monitor):
    ready = json.loads((scratch/'candidate_complete.json').read_text())
    if (ready['status'] != 'complete' or ready['step'] != int(folder.name[4:]) or ready['suffix'] != suffix
            or ready['material_sha256'] != sha(proposal/'material.npy')
            or ready['alpha'] != alpha or ready['frequencies'] != 450):
        raise RuntimeError('Wrong complete candidate field for material commit')
    atomic(folder/'field_commit.json', {'status': 'copying_complete_accepted_cache',
        'candidate': str(proposal), 'alpha': alpha, 'suffix': suffix,
        'idempotent_copy': True, 'material_sha256': ready['material_sha256']})
    for source, count in GROUPS.items():
        trial = np.load(scratch/f'{source}_{suffix}_X.npy', mmap_mode='r')
        bank = np.load(root/'banks'/f'{source}_X.npy', mmap_mode='r+')
        for position in range(count):
            bank[position] = trial[position]
            if position % 16 == 15:
                monitor.sample('commit_complete_joint_field')
        bank.flush()
        del trial, bank
    for name in ['variables.npy', 'material.npy', 'parameters.npy']:
        shutil.copy2(proposal/name, folder/name)
    atomic(folder/'field_commit.json', {'status': 'complete_accepted_field_copied',
        'candidate': str(proposal), 'alpha': alpha, 'suffix': suffix,
        'idempotent_copy': True, 'material_sha256': ready['material_sha256']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--decision', type=Path)
    parser.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--cap-gib', type=float, default=12.)
    parser.add_argument('--field-cg', type=int, default=64)
    parser.add_argument('--material-cg', type=int, default=64)
    parser.add_argument('--tv', type=float, default=.001)
    parser.add_argument('--damping', type=float, default=.01)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.out.resolve()
    if not args.devices or len(set(args.devices)) != len(args.devices) or len(args.devices) > 3:
        raise ValueError('Use one to three distinct devices')
    if not 0 < args.cap_gib <= 12.:
        raise ValueError('Leave room below 16 GiB for non-allocator sparse-LU workspaces')
    if args.field_cg < 1 or args.material_cg < 1:
        raise ValueError('Positive field and material iteration budgets required')
    source, completed, previous = validate(root, args.decision, args.check_only)
    if args.check_only:
        print(json.dumps({'complete_pressure_boundary_valid': True, 'step': completed,
            'all450_all11_all192': True, 'bank_writes': False, 'signals_sent': False}), flush=True)
        return
    lock = open(root/'coordinator.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monitor = None
    changed = False
    run = None
    try:
        source, completed, previous = validate(root, args.decision)
        step = completed+1
        variables = MaterialVariables(np.load(root/'initial/material.npy'), np.load(root/'initial/parameters.npy'))
        current = np.load(previous/'variables.npy')
        field_bytes = sum(GROUPS.values())*11*variables.size*8
        scratch = root/'joint_trial_banks'
        new_bytes = sum(field_bytes for suffix in ['first', 'line']
            if not all((scratch/f'{s}_{suffix}_X.npy').exists() for s in GROUPS))
        if shutil.disk_usage(root).free < new_bytes+17*2**30:
            raise RuntimeError('Need worst-case joint trial space plus16GiB and statistic reserve')
        if (scratch/'owner.json').exists():
            owner = json.loads((scratch/'owner.json').read_text())
            if owner['step'] != completed or owner['status'] != 'complete_step':
                raise RuntimeError('Reusable scratch belongs to an incomplete joint step')
        names = sorted(set(source['source_sha256']) | set(NEW_SOURCES))
        hashes = {name: sha(Path(__file__).with_name(name)) for name in names}
        weights = Path(source.get('field_inverse_checkpoint', source.get(
            'stable_field_checkpoint', 'out/v5/forward_round8_dense_full/last.pt')))
        weights_sha = sha(weights)
        folder = root/f'step{step:03d}'
        folder.mkdir()
        (folder/'receiver_gram').mkdir()
        (folder/'receiver_projection').mkdir()
        (folder/'candidates').mkdir()
        archive = root/f'source_joint_step{step:03d}'
        archive.mkdir()
        shutil.copy2(root/'run.json', archive/'source_complete_run.json')
        if (root/'resource_summary.json').exists():
            shutil.copy2(root/'resource_summary.json', archive/'source_resource_summary.json')
        shutil.copy2(args.decision, archive/'route_decision.json')
        for name in names:
            path = Path(__file__).with_name(name)
            shutil.copy2(path, archive/name)
        frozen_weights = archive/'field_inverse_weights.pt'
        shutil.copy2(weights, frozen_weights)
        if sha(frozen_weights) != weights_sha:
            raise RuntimeError('Field-inverse checkpoint changed while freezing it')
        scratch.mkdir(exist_ok=True)
        # At most two reusable fields, receiver Grams and normal maps; old
        # observations and prior experiment artifacts are never deleted.
        atomic(scratch/'owner.json', {'step': step, 'status': 'prepared',
            'source_completed_step': completed, 'reusable_not_archival': True})
        (folder/'trial_banks').symlink_to(scratch, target_is_directory=True)
        for a, b in [('variables.npy', 'old_variables.npy'), ('material.npy', 'old_material.npy')]:
            shutil.copy2(previous/a, folder/b)
        run = dict(source)
        run.setdefault('solver_changes', []).append({'step': step, 'source_pid': source['pid'],
            'pid': os.getpid(), 'unix_s': time.time(), 'source_sha256': hashes,
            'change': 'simultaneous pressure/material direction with same-node receiver-space curvature',
            'physical_equation_changed': False, 'observations_changed': False,
            'frozen_penalties_changed': False, 'off_node_curvature_omitted': True})
        run.update(status='joint_pressure_field_sweep', step=step, pid=os.getpid(),
            source_sha256=hashes, solver_revision='joint_receiver_v1', iterations=step,
            field_log_tag='field_joint', resumed_complete_field_count=0,
            allocator_cap_GiB=args.cap_gib, devices=args.devices,
            field_inverse_checkpoint=str(frozen_weights), field_inverse_checkpoint_sha256=weights_sha,
            actual_solver_settings={'field_solver': 'batched_independent_CG_true_energy',
                'field_CG_max': args.field_cg, 'field_tolerance': .003,
                'material_CG_max': args.material_cg, 'TV_weight': args.tv,
                'damping_relative': args.damping, 'trust_RMS_r_beta_t': [.03, .04, .10],
                'maximum_joint_line_trials': 9},
            joint_reconstruction_success_claimed=False)
        atomic(root/'run.json', run)
        changed = True
        monitor = PressureResources(root)
        tasks = [(s, p) for s, count in GROUPS.items() for p in range(count)]
        assignments = [tasks[r::len(args.devices)] for r in range(len(args.devices))]
        records = phase(folder, step, 'field_joint', assignments, args, monitor, args.cap_gib)
        atomic(folder/'field_convergence.json', {'frequencies': 450, 'transmissions_each': 11,
            'unconverged_shots': sum(r['unconverged_shots'] for r in records),
            'max_physical_normal': max(r['max_physical_normal'] for r in records),
            'max_preconditioned_normal': max(r['max_preconditioned_normal'] for r in records),
            'requested_tolerance': .003, 'scope': 'field solve only, not material RF or recovery'})
        blocks = np.zeros((6, *variables.shape), np.float64)
        gradient = np.zeros((3, *variables.shape), np.float64)
        for record in records:
            with np.load(record['normal_file']) as values:
                blocks += values['blocks']
                gradient += values['gradient']
        np.savez(folder/'joint_normal.npz', blocks=blocks, gradient=gradient)
        run['status'] = 'joint_material_direction'
        atomic(root/'run.json', run)
        system = JointMaterialDirection(variables, blocks, gradient)
        def progress(row):
            atomic(folder/'material_CG_progress.json', row)
        full_proposal, direction_record = system.solve(current, args.tv, args.damping,
            args.material_cg, callback=progress)
        atomic(folder/'material_direction.json', direction_record)
        old_tv, tv_gradient, _, _ = variables.tv(current, args.tv)
        candidates = []
        accepted = None
        del system, blocks, gradient, records
        if not direction_record['zero_material_direction']:
            for line in range(9):
                alpha = .5**line
                candidate = folder/'candidates'/f'line{line:03d}'
                candidate.mkdir()
                proposed = variables.project(current+alpha*(full_proposal-current))
                np.save(candidate/'variables.npy', proposed)
                np.save(candidate/'material.npy', variables.material(proposed))
                np.save(candidate/'parameters.npy', variables.parameters(proposed))
                suffix = 'first' if line == 0 else 'line'
                init_trial_bank(scratch, suffix, variables.shape)
                atomic(scratch/'candidate_complete.json', {'status': 'writing', 'step': step,
                    'alpha': alpha, 'suffix': suffix, 'candidate': str(candidate)})
                run.update(status='joint_candidate_cost', candidate=line, candidate_alpha=alpha)
                atomic(root/'run.json', run)
                mode = 'candidate_first' if line == 0 else 'candidate_scaled'
                rows = phase(folder, step, mode, assignments, args, monitor,
                    args.cap_gib if line == 0 else 2.5, candidate,
                    tag=f'joint_candidate{line:03d}', line_scale=alpha)
                data = sum(r['data_halfcost_change'] for r in rows)
                equation = sum(r['equation_halfcost_change'] for r in rows)
                slope = sum(r['joint_linear_slope'] for r in rows)+dot(tv_gradient, (proposed-current)/.1)
                tv_change = variables.tv(proposed, args.tv)[0]-old_tv
                change = data+equation+tv_change
                actual_rms = np.sqrt(np.mean((proposed-current)[:, variables.unknown]**2, axis=1))
                ok = bool(np.isfinite(change) and slope < 0 and change <= 1e-4*slope
                    and np.all(actual_rms <= np.array([.03, .04, .10])*1.01))
                row = {'alpha': alpha, 'data_cost_change': data, 'equation_cost_change': equation,
                    'TV_change': tv_change, 'joint_objective_change': change,
                    'joint_linear_slope': slope, 'actual_RMS_r_beta_t': actual_rms.tolist(),
                    'accepted': ok, 'frequencies': 450, 'all11_each': True}
                if line == 0:
                    hc = sum(r['conditional_curvature_along_material'] for r in rows)
                    hr = sum(r['receiver_reduced_curvature_along_material'] for r in rows)
                    row.update(conditional_curvature_along_material=hc,
                        receiver_reduced_curvature_along_material=hr,
                        reduced_over_conditional_curvature=hr/max(hc, 1e-300),
                        conditional_material_linear_slope=sum(r['conditional_material_linear_slope'] for r in rows),
                        receiver_projected_material_linear_slope=sum(r['receiver_projected_material_linear_slope'] for r in rows),
                        curvature_diagnostic_extra_wave_solves=0)
                candidates.append(row)
                atomic(candidate/'cost.json', row)
                atomic(scratch/'candidate_complete.json', {'status': 'complete', 'step': step,
                    'alpha': alpha, 'suffix': suffix, 'candidate': str(candidate),
                    'material_sha256': sha(candidate/'material.npy'), 'frequencies': 450})
                if ok:
                    accepted = (candidate, suffix, alpha)
                    break
        run['status'] = 'joint_field_commit'
        atomic(root/'run.json', run)
        if accepted is not None:
            candidate, suffix, alpha = accepted
            commit_trial(root, scratch, suffix, folder, candidate, alpha, monitor)
        else:
            for name in ['variables.npy', 'material.npy', 'parameters.npy']:
                shutil.copy2(previous/name, folder/name)
        atomic(folder/'material_update.json', {'accepted': accepted is not None,
            'line_search': candidates, 'direction': direction_record,
            'same_node_receiver_curvature': True, 'off_node_data_curvature_omitted': True,
            'simultaneous_field_change_in_acceptance': True, 'truth_read': False})
        run['status'] = 'joint_pressure_dual_update'
        atomic(root/'run.json', run)
        phase(folder, step, 'dual_joint', assignments, args, monitor, 2.5, folder)
        atomic(folder/'complete_sweep.json', {'status': 'complete', 'step': step,
            'frequencies': 450, 'transmissions_each': 11, 'unix_s': time.time(),
            'solver_revision': 'joint_receiver_v1', 'truth_read': False})
        run['status'] = 'full_native_RF_score'
        atomic(root/'run.json', run)
        if sha(folder/'material.npy') == sha(previous/'material.npy'):
            rf = json.loads((previous/'rf_score.json').read_text())
            rf.update(material=str(folder/'material.npy'), reused_unchanged_material_RF=True,
                source_RF=str(previous/'rf_score.json'))
            atomic(folder/'rf_score.json', rf)
        else:
            used = monitor.sample('joint_native_RF_admission')
            device = min(args.devices, key=lambda d: used[d]['whole_MiB'])
            monitor.wait_budget([device], 1.1, 'joint_native_RF_admission')
            command([sys.executable, '-m', 'nsrno.score_wri_material', str(folder/'material.npy'),
                '--sample', run['sample'], '--out', str(folder/'rf_score.json'), '--device', str(device),
                '--cap-gib', '1.1'], folder/'rf_score.log', monitor, 'joint_native_RF_score')
            rf = json.loads((folder/'rf_score.json').read_text())
        best = json.loads((root/'best_RF.json').read_text())
        if rf['primary_loss'] < best['primary_loss']:
            best = {'step': step, 'material': str(folder/'material.npy'), 'primary_loss': rf['primary_loss']}
            atomic(root/'best_RF.json', best)
        run['status'] = 'post_hoc_maps_only'
        atomic(root/'run.json', run)
        shutil.copy2(root/'initial/material.npy', folder/'stage0_initial_material.npy')
        command([sys.executable, '-m', 'nsrno.evaluate_time_material', str(folder/'material.npy'),
            '--sample', run['sample'], '--out', str(folder/'maps'), '--estimate-label',
            'Joint pressure/material update'], folder/'maps.log', monitor, 'post_hoc_joint_maps_only')
        row = {'step': step, 'solver_revision': 'joint_receiver_v1',
            'material_accepted': accepted is not None, 'primary_full_RF_loss': rf['primary_loss'],
            'best_RF': best, 'truth_read_by_optimizer': False}
        with (root/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        atomic(scratch/'owner.json', {'step': step, 'status': 'complete_step',
            'reusable_not_archival': True, 'material_sha256': sha(folder/'material.npy')})
        run.update(status='planned_joint_sweeps_finished', completed_step=step,
            best_RF=best, joint_reconstruction_success_claimed=False)
        atomic(root/'run.json', run)
    except BaseException:
        if changed:
            run.update(status='failed', error=traceback.format_exc())
            atomic(root/'run.json', run)
        raise
    finally:
        if monitor is not None:
            monitor.log.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
