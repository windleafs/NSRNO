"""One actual all44 spatial-prior update at the same controlled material start.

Reuses 44 completed ordinary gradients and 44 old data-direction JVPs. Only
three fresh independent-prior directions require native tangents. No new data.
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
from .conditional_patch_pool import ConditionalPatchPool
from .conditional_patch_prior import CombinedIndependentPrior
from .conditional_patch_step import regularization, build_directions, solve_prior_step
from .independent_physical_coordinates import IndependentPhysicalCoordinates
from .pressure_resources import PressureResources
from .low_frequency_0p5 import atomic, digest, SAMPLE, CASE
from .reconstruct_frequency_multilevel import array
from .reconstruct_native_frequency_incremental import same_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--fit-root', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--devices', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--cap-gib', type=float, default=10.)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--max-seconds', type=float, default=4500.)
    args = parser.parse_args()
    project, source = Path.cwd().resolve(), Path(__file__).parent
    args.parent_root, args.fit_root, args.qualification, args.out, args.artifact_root = [p.resolve()
        for p in [args.parent_root, args.fit_root, args.qualification, args.out, args.artifact_root]]
    if (not all(p.is_relative_to(project/'out/v5') for p in [args.parent_root, args.fit_root, args.out]) or
        not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or
        args.out.exists() or args.artifact_root.exists() or args.devices != [0, 1] or
        not 0 < args.cap_gib <= 10 or args.chunk != 128 or args.max_seconds != 4500):
        raise ValueError('Dedicated 33-unit, three-candidate, 4500-second spatial-prior update only')
    previous = json.loads((args.parent_root/'run.json').read_text())
    old_workers = json.loads((args.parent_root/'worker_processes.json').read_text())
    if (previous['status'] != 'full_field_prior_native_stage_finished' or same_process(previous) or
        any(Path(f'/proc/{pid}').exists() for pid in old_workers['worker_pids']) or
        previous['new_reserved_equivalent_units'] != 55 or previous['original44_checks_completed'] != 2):
        raise RuntimeError('Previous independent-value-prior experiment must remain closed')
    fit = json.loads((args.fit_root/'run.json').read_text())
    qualification = json.loads(args.qualification.read_text())
    patch_path = args.fit_root/'products/prior.npz'
    if (fit['status'] != 'TRAIN_conditional_patch_prior_CPU_fit_finished' or same_process(fit) or
        fit['development_arrays_read'] or fit['reserved_arrays_read'] or fit['source_case'] != 'Neg_07_Left' or
        qualification['status'] != 'conditional_patch_synthetic_scaled_log_gradient_CPU_passed' or
        qualification['prior_SHA256'] != digest(patch_path) or
        qualification['source_SHA256'] != digest(source/'conditional_patch_prior.py') or
        fit['prior_SHA256'] != qualification['prior_SHA256']):
        raise RuntimeError('Finished TRAIN-only spatial prior and its fixed CPU gradient qualification required')
    for name, sha in previous['source_SHA256'].items():
        if digest(source/name) != sha:
            raise RuntimeError('Cached physical product source changed: '+name)
    gradient = json.loads((args.parent_root/'jobs/0001_original44_gradient.json').read_text())
    old_model = json.loads((args.parent_root/'jobs/0002_joint_tangents.json').read_text())
    if (not gradient['all44_sources'] or gradient['patient_truth_read'] or
        gradient['actual_work']['completed_adjoint_direction_passes'] != 44 or
        gradient['actual_work']['unfinished_native_phases'] or
        digest(gradient['gradient_path']) != gradient['gradient_SHA256'] or
        old_model['loss'] != previous['initial_RF']):
        raise RuntimeError('The exact original completed gradient and source model are required')
    initial = (args.parent_root/'arrays/initial_material.npy').resolve()
    if digest(initial) != previous['initial_material_SHA256']:
        raise RuntimeError('Controlled comparison start changed')
    plan = json.loads((args.parent_root/'native_plan.json').read_text())
    if plan['reference_material_SHA256'] != digest(initial):
        raise RuntimeError('RF cache is not at the common comparison start')
    old_control_path = args.parent_root/'arrays/control_directions.npy'
    old_physical_path = args.parent_root/'arrays/physical_directions.npy'
    if (digest(old_control_path) != previous['control_direction_SHA256'] or
        digest(old_physical_path) != previous['physical_direction_SHA256']):
        raise RuntimeError('Old data direction changed; do not reuse its products')
    entries = [(name, shot) for name in plan['families'] for shot in range(11)]
    references = plan['reference_RF']
    if sorted((r['family'], r['shot']) for r in references) != sorted(entries):
        raise RuntimeError('RF comparison start lacks all44 true sources')
    for row in references:
        if digest(row['path']) != row['SHA256']:
            raise RuntimeError('Same-material original RF cache changed')
    reused = []
    for index, (family, shot) in enumerate(entries):
        path = args.parent_root/'workers'/f'rank{index%2:02d}'/'arrays/token0002'/f'{family}_{shot:02d}_J03.npy'
        reused.append({'family': family, 'shot': shot, 'path': str(path.resolve()),
                       'SHA256': digest(path), 'direction_index_in_old_pack': 3})
    plan.update(reused_data_JVPs=reused, old_physical_directions=str(old_physical_path.resolve()),
        old_physical_directions_SHA256=digest(old_physical_path),
        spatial_prior_controlled_comparison=True, reused_source_model_SHA256=digest(
            args.parent_root/'jobs/0002_joint_tangents.json'))
    edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    for path, sha in [(Path(plan['native_anchor_material']), previous['native_anchor_material_SHA256']),
        (edge_path, plan['known_edge_SHA256']),
        (Path(plan['data_root'])/'index.json', plan['data_index_SHA256']),
        (Path(plan['high_index']), plan['high_index_SHA256'])]:
        if digest(path) != sha:
            raise RuntimeError('Original acquisition or coordinates changed')
    marginal_path = project/'out/v5/independent_material_marginals_v1/prior.npz'
    if digest(marginal_path) != previous['prior_SHA256']:
        raise RuntimeError('The old independent value prior must remain unchanged')
    reserve = 2*2**30+400*2**20
    if shutil.disk_usage(project).free < reserve+2*2**20 or shutil.disk_usage(Path.home()).free < reserve+800*2**20:
        raise RuntimeError('Retain existing project and HOME disk reserves')
    names = [Path(__file__).name, 'conditional_patch_pool.py', 'conditional_patch_worker.py',
             'conditional_patch_step.py', 'conditional_patch_prior.py']
    hashes = {**previous['source_SHA256'], **{name: digest(source/name) for name in names}}
    args.out.mkdir()
    args.artifact_root.mkdir()
    for name in ['arrays', 'workers', 'candidates', 'final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name, target_is_directory=True)
    (args.out/'jobs').mkdir()
    for name in hashes:
        shutil.copy2(source/name, args.out/name)
    started = time.time()
    deadline = started+args.max_seconds
    run = {'status': 'actual_independent_spatial_prior_gradient_for_new_directions',
        'pid': os.getpid(), 'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': deadline, 'parent_root': str(args.parent_root),
        'artifact_root': str(args.artifact_root), 'source_SHA256': hashes, 'sample': SAMPLE, 'case': CASE,
        'initial_material_SHA256': digest(initial), 'native_anchor_material_SHA256': previous['native_anchor_material_SHA256'],
        'patch_prior_SHA256': digest(patch_path), 'marginal_prior_SHA256': digest(marginal_path),
        'qualification_SHA256': digest(args.qualification), 'new_reserved_equivalent_units': 33,
        'maximum_new_adjoint_passes': 0, 'maximum_new_tangent_passes': 132,
        'maximum_new_tangent_background_forwards': 132, 'maximum_new_candidate_forwards': 132,
        'maximum_outer_updates': 1, 'maximum_candidate_lines': 3,
        'reused_old_ordinary_adjoint_sources': 44, 'reused_old_data_direction_JVPs': 44,
        'reused_gradient_SHA256': gradient['gradient_SHA256'],
        'closed_value_prior_budget': {'reserved_units': 55, 'limit_units': 55,
            'actual_full44_checks': 2, 'maximum_checks': 3,
            'created_unix_s': previous['created_unix_s'], 'deadline_unix_s': previous['deadline_unix_s'],
            'retained_unchanged': True},
        'closed_full_dual_budget': previous['closed_full_dual_budget'],
        'closed_predecessor_budget': previous['closed_predecessor_budget'],
        'cumulative_reserved_units_from_closed_predecessor': 89+134+55+33,
        'reservation_is_conservative_not_actual_thread_total': True,
        'common_start_original44_RF': previous['initial_RF'],
        'other_value_prior_candidate_RF': previous['final_RF'],
        'acceptance_compares_common_start_not_claimed_sequential_RF_decrease': True,
        'physical_log_step_caps': [.02, .04, .12], 'independent_native_material_fields': True,
        'no_material_lowpass': True, 'shape': [3, 961, 1121], 'devices': args.devices,
        'per_card_process_tree_limit_GiB': 16, 'allocator_cap_GiB': args.cap_gib,
        'source_encoding': False, 'all11_sources_per_family': True,
        'new_data_or_frequency': False, 'new_patient_derivative_probes': 0,
        'patient_truth_read_by_optimizer': False, 'reserved_case_unread': 'Neg_47_Left',
        'original44_checks_issued': 0, 'original44_checks_completed': 0,
        'accepted_spatial_prior_update': False, 'joint_reconstruction_success_claimed': False}
    atomic(args.out/'run.json', run)
    def request_stop(number, frame):
        (args.out/'stop_requested.flag').write_text(str(number)+'\n')
    for number in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(number, request_stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    monitor, pool = PressureResources(args.out), None
    def freeze():
        if any(digest(source/name) != sha for name, sha in hashes.items()):
            raise RuntimeError('Frozen executable changed during the actual inverse')
        if time.time() >= deadline-60 or (args.out/'stop_requested.flag').exists():
            raise RuntimeError('Absolute spatial-prior deadline or cooperative stop reached')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Existing disk reserve reached')
    def checkpoint():
        if pool is not None:
            run['actual_native_work'] = pool.actual_work()
        atomic(args.out/'run.json', run)
    try:
        freeze()
        coordinates = IndependentPhysicalCoordinates(np.load(plan['native_anchor_material']), np.load(edge_path))
        control = np.load(args.parent_root/'arrays/initial_control.npy').astype(np.float64)
        material = coordinates.material(control)
        if not np.array_equal(material, np.load(initial)):
            raise RuntimeError('Independent coordinates do not reproduce the exact comparison start')
        z, x = np.ogrid[:961, :1121]
        active = (z >= 80)&(z < 937)&(x >= 24)&(x < 1097)
        physical_z = (np.arange(961)-60)*.05
        physical_x = (np.arange(1121)-560)*.05
        lateral = abs(physical_x) <= 19.1
        roi = (physical_z[:, None] >= 1)&(physical_z[:, None] <= 43)&lateral[None]
        prior = CombinedIndependentPrior(marginal_path, patch_path)
        R0, prior_gradient, parts0 = regularization(coordinates, prior, control, roi)
        RF0 = previous['initial_RF']
        directions, physical, details = build_directions(coordinates, control, prior_gradient, active,
            lateral, np.load(old_control_path, mmap_mode='r')[3], np.load(old_physical_path, mmap_mode='r')[3])
        data_gradient = np.load(gradient['gradient_path'])
        details['data_gradient_directional_products'] = np.einsum('aijk,ijk->a', directions, data_gradient).tolist()
        for name, value in [('initial_control', control), ('initial_material', material),
                            ('initial_prior_gradient', prior_gradient), ('control_directions', directions),
                            ('physical_directions', physical)]:
            array(args.out/'arrays'/f'{name}.npy', value)
        plan['new_physical_directions_SHA256'] = digest(args.out/'arrays/physical_directions.npy')
        atomic(args.out/'native_plan.json', plan)
        atomic(args.out/'direction_details.json', details)
        run.update(initial_regularization=R0, initial_prior_parts=parts0,
            control_direction_SHA256=digest(args.out/'arrays/control_directions.npy'),
            physical_direction_SHA256=plan['new_physical_directions_SHA256'],
            status='actual_all44_JVPs_for_three_new_spatial_prior_directions')
        checkpoint()
        freeze()
        pool = ConditionalPatchPool(args, monitor, plan, deadline)
        model = pool.evaluate(material, 'joint_tangents', args.out/'arrays/physical_directions.npy')
        if model['reused_data_JVPs'] != 44 or model['actual_work']['completed_tangent_direction_passes'] != 132:
            raise RuntimeError('Expected exactly 132 new and 44 reused actual directional products')
        run['status'] = 'CPU_conditional_spatial_prior_selection_with_all44_data_model'
        checkpoint()
        coefficients, optimization = solve_prior_step(model['H'], model['linear'], RF0,
            coordinates, prior, control, roi, directions)
        atomic(args.out/'local_prior_solution.json', optimization)
        run.update(coefficients=coefficients.tolist(), predicted_RF_after_prior_step=optimization['RF_model_after_trust'])
        selected, selected_control, selected_RF, selected_R = None, control, model, R0
        last_evaluated = None
        minimum_drop = 10*abs(gradient['loss']-RF0)
        for line, fraction in enumerate([1., .5, .25], 1):
            freeze()
            proposal = control+fraction*np.einsum('a,aijk->ijk', coefficients, directions)
            if np.max(abs(coordinates.project(proposal, active)-proposal)) > 1e-9:
                raise RuntimeError('Projection altered a measured physical direction combination')
            R, _, parts = regularization(coordinates, prior, proposal, roi)
            if R >= R0-1e-12:
                atomic(args.out/f'skipped_line{line:02d}.json', {'reason': 'no_prior_decrease_before_new_wave',
                    'regularization': R, 'line': line})
                continue
            folder = args.out/'candidates'/f'line{line:02d}'
            folder.mkdir()
            array(folder/'control.npy', proposal)
            array(folder/'material.npy', coordinates.material(proposal))
            run.update(status='actual_all44_spatial_prior_candidate_line', active_line=line,
                active_fraction=fraction, active_regularization=R)
            run['original44_checks_issued'] += 1
            checkpoint()
            trial = pool.evaluate(coordinates.material(proposal), 'original44_forward')
            run['original44_checks_completed'] += 1
            accepted = bool(trial['loss'] < RF0-minimum_drop and
                trial['loss']+parts['TV'] < RF0+parts0['TV'] and R < R0)
            atomic(folder/'actual_native_acceptance.json', {'accepted': accepted,
                'controlled_start_RF': RF0, 'RF_after': trial['loss'], 'prior_before': R0, 'prior_after': R,
                'line': line, 'fraction': fraction, 'parts_before': parts0, 'parts_after': parts,
                'other_value_prior_candidate_RF': previous['final_RF'],
                'also_better_than_other_value_prior_RF': trial['loss'] < previous['final_RF'],
                'material_SHA256': digest(folder/'material.npy'), 'all44_true_sources': True,
                'joint_reconstruction_success_claimed': False})
            last_evaluated = folder
            checkpoint()
            print(json.dumps({'line': line, 'RF': trial['loss'], 'regularization': R,
                              'accepted_spatial_prior_step': accepted}), flush=True)
            if accepted:
                selected, selected_control, selected_RF, selected_R = folder, proposal, trial, R
                run['accepted_spatial_prior_update'] = True
                break
        pool.close()
        pool = None
        final = args.out/'final'
        array(final/'control.npy', selected_control)
        array(final/'material.npy', coordinates.material(selected_control))
        if selected is not None:
            refs = []
            for index, (family, shot) in enumerate(entries):
                path = args.out/'workers'/f'rank{index%2:02d}'/'arrays'/f'token{selected_RF["token"]:04d}'/f'{family}_{shot:02d}.npy'
                refs.append({'family': family, 'shot': shot, 'path': str(path.resolve()), 'SHA256': digest(path)})
        else:
            refs = references
        final_RF = selected_RF['loss'] if selected is not None else RF0
        native_tv = coordinates.regularization(selected_control, .001)[0]
        atomic(final/'actual_native_acceptance.json', {'accepted': selected is not None,
            'actual_native_RF': {'loss': final_RF, 'all44_sources': True, 'source_encoding': False},
            'RF_source_records': refs, 'native_TV': native_tv, 'independent_prior_value': selected_R,
            'material_SHA256': digest(final/'material.npy'), 'full_original44_source_acceptance': True,
            'common_start_RF': RF0, 'other_value_prior_candidate_RF': previous['final_RF'],
            'joint_imaging_success_claimed': False})
        plotted = selected or last_evaluated
        if plotted is not None:
            (plotted/'stage0_initial_material.npy').symlink_to(initial)
            with (plotted/'maps.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material', str(plotted/'material.npy'),
                    '--sample', SAMPLE, '--out', str(plotted/'maps'),
                    '--estimate-label', 'Independent conditional spatial prior'],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log,
                    stderr=subprocess.STDOUT, check=True, timeout=120)
        copied = selected or Path(previous['parent_root'])/'final'
        for suffix in ['png', 'metrics.json']:
            shutil.copy2(copied/f'maps.{suffix}', final/f'maps.{suffix}')
        atomic(final/'maps_provenance.json', {'copied_from_same_material': str(copied),
            'material_SHA256': digest(final/'material.npy'), 'no_duplicate_truth_evaluation': True,
            'if_no_new_acceptance_initial_column_is_historical': selected is None})
        run.update(status='conditional_spatial_prior_native_stage_finished',
            stop_reason='one_separately_budgeted_conditional_spatial_prior_update_finished',
            final_RF=final_RF, final_regularization=selected_R, material_SHA256=digest(final/'material.npy'),
            selected_candidate=str(selected) if selected else None,
            last_evaluated_candidate=str(last_evaluated) if last_evaluated else None,
            seconds=time.time()-started, finished_unix_s=time.time())
        checkpoint()
        print(json.dumps({k: run[k] for k in ['status', 'accepted_spatial_prior_update', 'final_RF',
            'final_regularization', 'material_SHA256', 'original44_checks_completed', 'seconds']}), flush=True)
    except BaseException:
        run.update(status='conditional_spatial_prior_failed_for_inspection', error=traceback.format_exc(),
                   seconds=time.time()-started, all_issued_work_remains_charged=True)
        checkpoint()
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__ == '__main__':
    main()
