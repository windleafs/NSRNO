"""Bounded three-update all44 physical inversion with full-field common descent."""
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
from .projected_common_pool import ProjectedCommonPool
from .projected_common_descent import common_direction
from .conditional_patch_prior import CombinedIndependentPrior
from .conditional_patch_step import regularization
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
    parser.add_argument('--max-seconds', type=float, default=7200.)
    args = parser.parse_args()
    project, source = Path.cwd().resolve(), Path(__file__).parent
    args.parent_root, args.fit_root, args.qualification, args.out, args.artifact_root = [p.resolve()
        for p in [args.parent_root, args.fit_root, args.qualification, args.out, args.artifact_root]]
    if (not all(p.is_relative_to(project/'out/v5') for p in [args.parent_root, args.fit_root, args.out]) or
        not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or
        args.out.exists() or args.artifact_root.exists() or args.devices != [0, 1] or
        not 0 < args.cap_gib <= 10 or args.chunk != 128 or args.max_seconds != 7200):
        raise ValueError('Dedicated 3-update/132-adjoint/12-candidate/7200-second actual trajectory only')
    previous = json.loads((args.parent_root/'run.json').read_text())
    old_workers = json.loads((args.parent_root/'worker_processes.json').read_text())
    if (previous['status'] != 'conditional_spatial_prior_native_stage_finished' or same_process(previous) or
        any(Path(f'/proc/{pid}').exists() for pid in old_workers['worker_pids']) or
        previous['new_reserved_equivalent_units'] != 33 or previous['original44_checks_completed'] != 2):
        raise RuntimeError('Require the exact finished spatial-prior parent, without resetting it')
    qualification = json.loads(args.qualification.read_text())
    if (qualification['status'] != 'weighted_projection_and_common_descent_synthetic_CPU_passed' or
        qualification['source_SHA256'] != digest(source/'projected_common_descent.py') or
        qualification['patient_arrays_read'] or qualification['new_acoustic_solves']):
        raise RuntimeError('Fixed synthetic qualification of the new optimizer required')
    for name, sha in previous['source_SHA256'].items():
        if digest(source/name) != sha:
            raise RuntimeError('Previously used physical, acquisition, or prior source changed: '+name)
    patch_path = args.fit_root/'products/prior.npz'
    marginal_path = project/'out/v5/independent_material_marginals_v1/prior.npz'
    if (digest(patch_path) != previous['patch_prior_SHA256'] or
        digest(marginal_path) != previous['marginal_prior_SHA256']):
        raise RuntimeError('Keep the same already-fitted and qualified independent prior')
    acceptance = json.loads((args.parent_root/'final/actual_native_acceptance.json').read_text())
    initial = (args.parent_root/'final/material.npy').resolve()
    if digest(initial) != previous['material_SHA256'] or acceptance['material_SHA256'] != digest(initial):
        raise RuntimeError('Accepted current material changed')
    RF0 = acceptance['actual_native_RF']['loss']
    if RF0 != previous['final_RF']:
        raise RuntimeError('Accepted initial RF does not match the finished parent')
    plan = json.loads((args.parent_root/'native_plan.json').read_text())
    for key in ['reused_data_JVPs', 'old_physical_directions', 'old_physical_directions_SHA256',
                'new_physical_directions_SHA256', 'reused_source_model_SHA256',
                'spatial_prior_controlled_comparison']:
        plan.pop(key, None)
    references = acceptance['RF_source_records']
    entries = [(family, shot) for family in plan['families'] for shot in range(11)]
    if sorted((r['family'], r['shot']) for r in references) != sorted(entries):
        raise RuntimeError('Initial RF needs all original44 true sources')
    for row in references:
        if digest(row['path']) != row['SHA256']:
            raise RuntimeError('Accepted initial RF cache changed')
    plan.update(reference_RF=references, reference_material_SHA256=digest(initial),
                projected_common_trajectory=True, cached_old_point_derivatives_not_reused=True)
    edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    for path, sha in [(Path(plan['native_anchor_material']), previous['native_anchor_material_SHA256']),
        (edge_path, plan['known_edge_SHA256']), (Path(plan['data_root'])/'index.json', plan['data_index_SHA256']),
        (Path(plan['high_index']), plan['high_index_SHA256'])]:
        if digest(path) != sha:
            raise RuntimeError('Fixed physical acquisition or coordinate map changed')
    reserve = 2*2**30+400*2**20
    if shutil.disk_usage(project).free < reserve+2*2**20 or shutil.disk_usage(Path.home()).free < reserve+1600*2**20:
        raise RuntimeError('Retain project and HOME disk reserves for the full declared trajectory')
    names = [Path(__file__).name, 'projected_common_descent.py', 'projected_common_pool.py']
    hashes = {**previous['source_SHA256'], **{name: digest(source/name) for name in names}}
    args.out.mkdir()
    args.artifact_root.mkdir()
    for name in ['arrays', 'workers', 'steps', 'final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name, target_is_directory=True)
    (args.out/'jobs').mkdir()
    for name in hashes:
        shutil.copy2(source/name, args.out/name)
    atomic(args.out/'native_plan.json', plan)
    started = time.time()
    deadline = started+args.max_seconds
    run = {'status': 'initializing_projected_common_descent_trajectory', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': deadline, 'parent_root': str(args.parent_root),
        'artifact_root': str(args.artifact_root), 'source_SHA256': hashes, 'sample': SAMPLE, 'case': CASE,
        'initial_material_SHA256': digest(initial), 'native_anchor_material_SHA256': previous['native_anchor_material_SHA256'],
        'initial_RF': RF0, 'current_RF': RF0, 'patch_prior_SHA256': digest(patch_path),
        'marginal_prior_SHA256': digest(marginal_path), 'qualification_SHA256': digest(args.qualification),
        'new_reserved_equivalent_units': 33, 'maximum_new_adjoint_passes': 132,
        'maximum_new_gradient_background_forwards': 132, 'maximum_new_candidate_forwards': 528,
        'maximum_new_tangent_passes': 0, 'maximum_outer_updates': 3,
        'maximum_candidates_per_update': 4, 'maximum_full44_candidates': 12,
        'closed_spatial_prior_budget': {'reserved_units': 33, 'actual_full44_checks': 2, 'maximum_checks': 3,
            'created_unix_s': previous['created_unix_s'], 'deadline_unix_s': previous['deadline_unix_s'],
            'retained_unchanged': True},
        'closed_value_prior_budget': previous['closed_value_prior_budget'],
        'closed_full_dual_budget': previous['closed_full_dual_budget'],
        'closed_predecessor_budget': previous['closed_predecessor_budget'],
        'cumulative_reserved_units_from_closed_predecessor': 89+134+55+33+33,
        'reservation_not_actual_thread_total': True, 'prior_or_weight_changed': False,
        'new_data_or_frequency': False, 'new_patient_derivative_probes': 0,
        'old_point_gradient_or_JVP_reused_at_new_material': False,
        'physical_log_caps': [.02, .04, .12], 'line_fractions': [.25, .125, .0625, .03125],
        'independent_native_material_fields': True, 'no_material_lowpass': True,
        'shape': [3, 961, 1121], 'devices': args.devices, 'per_card_process_tree_limit_GiB': 16,
        'allocator_cap_GiB': args.cap_gib, 'source_encoding': False, 'all11_sources_per_family': True,
        'patient_truth_read_by_optimizer': False, 'intermediate_truth_feedback': False,
        'reserved_case_unread': 'Neg_47_Left', 'original44_checks_issued': 0,
        'original44_checks_completed': 0, 'accepted_updates': 0, 'active_step': 0,
        'joint_reconstruction_success_claimed': False, 'accepted_steps': []}
    atomic(args.out/'run.json', run)
    def request_stop(number, frame):
        (args.out/'stop_requested.flag').write_text(str(number)+'\n')
    for number in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(number, request_stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    monitor, pool = PressureResources(args.out), None
    def checkpoint():
        if pool is not None:
            run['actual_native_work'] = pool.actual_work()
        atomic(args.out/'run.json', run)
    def freeze():
        if any(digest(source/name) != sha for name, sha in hashes.items()):
            raise RuntimeError('Frozen executable changed during the actual trajectory')
        if time.time() >= deadline-60 or (args.out/'stop_requested.flag').exists():
            raise RuntimeError('Fixed trajectory deadline or cooperative stop reached')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Existing disk reserve reached')
    def checkpoint_memory_admission():
        available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                             if s.startswith('MemAvailable:')))/2**20
        bank = ((36001+127)//128)*7*961*1121*4/2**30
        if available < 2*bank*len(args.devices)+24:
            raise RuntimeError('Insufficient host space for checkpoints and existing 24GiB headroom')
    try:
        freeze()
        coordinates = IndependentPhysicalCoordinates(np.load(plan['native_anchor_material']), np.load(edge_path))
        control = np.load(args.parent_root/'final/control.npy').astype(np.float64)
        material = coordinates.material(control)
        if not np.array_equal(material, np.load(initial)):
            raise RuntimeError('Coordinates do not exactly reproduce the accepted starting material')
        array(args.out/'arrays/initial_control.npy', control)
        array(args.out/'arrays/initial_material.npy', material)
        z, x = np.ogrid[:961, :1121]
        active = (z >= 80)&(z < 937)&(x >= 24)&(x < 1097)
        physical_z = (np.arange(961)-60)*.05
        physical_x = (np.arange(1121)-560)*.05
        lateral = abs(physical_x) <= 19.1
        roi = (physical_z[:, None] >= 1)&(physical_z[:, None] <= 43)&lateral[None]
        prior = CombinedIndependentPrior(marginal_path, patch_path)
        R, prior_gradient, parts = regularization(coordinates, prior, control, roi)
        if abs(R-previous['final_regularization']) > 1e-9:
            raise RuntimeError('Unchanged prior energy does not match the accepted parent')
        run.update(initial_regularization=R, current_regularization=R, initial_prior_parts=parts,
                   status='admitting_workers_for_projected_common_descent')
        checkpoint()
        checkpoint_memory_admission()
        pool = ProjectedCommonPool(args, monitor, plan, deadline)
        current_RF = RF0
        current_material_path = args.out/'arrays/initial_material.npy'
        current_control_path = args.out/'arrays/initial_control.npy'
        stop_reason = 'declared_three_actual_updates_finished'
        for step in range(1, 4):
            freeze()
            checkpoint_memory_admission()
            folder = args.out/'steps'/f'step{step:02d}'
            folder.mkdir()
            (folder/'start_material.npy').symlink_to(current_material_path.resolve())
            (folder/'start_control.npy').symlink_to(current_control_path.resolve())
            if step > 1:
                R, prior_gradient, parts = regularization(coordinates, prior, control, roi)
            array(folder/'prior_gradient.npy', prior_gradient)
            run.update(status='actual_all44_gradient_for_projected_common_step', active_step=step,
                       current_RF=current_RF, current_regularization=R,
                       current_material_SHA256=digest(current_material_path))
            checkpoint()
            gradient = pool.evaluate(material, 'original44_gradient', step)
            difference = gradient['loss']-current_RF
            allowance = 8*float(np.finfo(np.float32).eps)*max(abs(current_RF), 1e-12)
            comparison = {'cached_RF': current_RF, 'required_gradient_RF': gradient['loss'],
                'difference': difference, 'eight_relative_float32_eps': allowance,
                'numerical_consistency_guard_not_rigorous_error_bound': True}
            atomic(folder/'same_material_RF_comparison.json', comparison)
            if abs(difference) > allowance:
                raise RuntimeError('Inspect required-gradient RF discrepancy without repeating it')
            data_gradient = np.load(gradient['gradient_path'])
            run['status'] = 'CPU_full_field_projected_common_direction'
            checkpoint()
            direction, details = common_direction(coordinates, control, data_gradient, prior_gradient,
                                                   active, lateral)
            array(folder/'direction.npy', direction)
            atomic(folder/'direction_details.json', {**details,
                'data_gradient_SHA256': gradient['gradient_SHA256'],
                'start_material_SHA256': digest(current_material_path)})
            if not details['actual_common_step_available']:
                stop_reason = 'no_resolved_full_field_common_descent_at_current_material'
                run['last_direction_details'] = details
                break
            accepted = False
            minimum_drop = max(10*abs(difference), allowance)
            for line, fraction in enumerate([.25, .125, .0625, .03125], 1):
                freeze()
                proposal = control+fraction*direction
                if np.max(abs(coordinates.project(proposal, active)-proposal)) > 1e-9:
                    raise RuntimeError('Measured direction left the original physical domain')
                candidate_R, _, candidate_parts = regularization(coordinates, prior, proposal, roi)
                target_R = R+1e-4*fraction*details['prior_slope']
                if candidate_R > target_R:
                    atomic(folder/f'skipped_line{line:02d}.json', {'line': line, 'fraction': fraction,
                        'reason': 'prior_Armijo_not_met_before_new_wave', 'prior': candidate_R,
                        'required_prior': target_R})
                    continue
                candidate = folder/f'line{line:02d}'
                candidate.mkdir()
                array(candidate/'control.npy', proposal)
                candidate_material = coordinates.material(proposal)
                array(candidate/'material.npy', candidate_material)
                run.update(status='actual_all44_projected_common_candidate_line', active_line=line,
                    active_fraction=fraction, active_regularization=candidate_R)
                run['original44_checks_issued'] += 1
                checkpoint()
                trial = pool.evaluate(candidate_material, 'original44_forward', step)
                run['original44_checks_completed'] += 1
                target_RF = current_RF+1e-4*fraction*details['RF_slope']
                accepted = bool(trial['loss'] < current_RF-minimum_drop and trial['loss'] <= target_RF and
                    trial['loss']+candidate_parts['TV'] < current_RF+parts['TV'] and candidate_R <= target_R)
                record = {'accepted': accepted, 'outer_step': step, 'line': line, 'fraction': fraction,
                    'RF_before': current_RF, 'RF_after': trial['loss'], 'RF_Armijo_target': target_RF,
                    'prior_before': R, 'prior_after': candidate_R, 'prior_Armijo_target': target_R,
                    'minimum_RF_drop': minimum_drop, 'parts_before': parts, 'parts_after': candidate_parts,
                    'material_SHA256': digest(candidate/'material.npy'), 'all44_true_sources': True,
                    'patient_truth_read_by_optimizer': False, 'joint_reconstruction_success_claimed': False}
                atomic(candidate/'actual_native_acceptance.json', record)
                checkpoint()
                print(json.dumps({'step': step, 'line': line, 'RF': trial['loss'],
                    'prior': candidate_R, 'accepted': accepted}), flush=True)
                if not accepted:
                    continue
                references = []
                for index, (family, shot) in enumerate(entries):
                    path = args.out/'workers'/f'rank{index%2:02d}'/'arrays'/f'token{trial["token"]:04d}'/f'{family}_{shot:02d}.npy'
                    references.append({'family': family, 'shot': shot,
                        'path': str(path.resolve()), 'SHA256': digest(path)})
                current_RF, R, parts = trial['loss'], candidate_R, candidate_parts
                control, material = proposal, candidate_material
                current_material_path, current_control_path = candidate/'material.npy', candidate/'control.npy'
                pool.mark_accepted(step)
                run['accepted_updates'] += 1
                accepted_record = {'outer_step': step, 'candidate': str(candidate),
                    'material_SHA256': record['material_SHA256'], 'RF': current_RF,
                    'regularization': R, 'fraction': fraction}
                run['accepted_steps'].append(accepted_record)
                run.update(current_RF=current_RF, current_regularization=R,
                           current_material_SHA256=record['material_SHA256'])
                atomic(folder/'accepted_RF_cache.json', {'RF_source_records': references,
                    'material_SHA256': record['material_SHA256'], 'actual_native_RF': trial,
                    'native_TV': parts['TV']})
                checkpoint()
                break
            if not accepted:
                stop_reason = 'no_actual_candidate_within_declared_four_line_range'
                break
        pool.close()
        pool = None
        final = args.out/'final'
        array(final/'control.npy', control)
        array(final/'material.npy', material)
        atomic(final/'actual_native_acceptance.json', {'accepted_updates': run['accepted_updates'],
            'actual_native_RF': {'loss': current_RF, 'all44_sources': True, 'source_encoding': False},
            'RF_source_records': references, 'native_TV': parts['TV'], 'independent_prior_value': R,
            'material_SHA256': digest(final/'material.npy'), 'full_original44_source_acceptance': True,
            'joint_imaging_success_claimed': False})
        if run['accepted_updates']:
            (final/'stage0_initial_material.npy').symlink_to(initial)
            with (final/'maps.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material', str(final/'material.npy'),
                    '--sample', SAMPLE, '--out', str(final/'maps'),
                    '--estimate-label', 'Full-field projected common descent'],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log,
                    stderr=subprocess.STDOUT, check=True, timeout=120)
        else:
            for suffix in ['png', 'metrics.json']:
                shutil.copy2(args.parent_root/f'final/maps.{suffix}', final/f'maps.{suffix}')
        atomic(final/'maps_provenance.json', {'new_fixed_material_evaluation': run['accepted_updates'] > 0,
            'copied_from_same_material_if_no_update': str(args.parent_root/'final'),
            'if_no_update_initial_column_is_historical': run['accepted_updates'] == 0,
            'optimizer_uses_no_intermediate_truth_feedback': True})
        run.update(status='projected_common_descent_native_stage_finished', stop_reason=stop_reason,
            final_RF=current_RF, final_regularization=R, material_SHA256=digest(final/'material.npy'),
            seconds=time.time()-started, finished_unix_s=time.time())
        checkpoint()
        print(json.dumps({k: run[k] for k in ['status', 'stop_reason', 'accepted_updates', 'final_RF',
            'final_regularization', 'original44_checks_completed', 'material_SHA256', 'seconds']}), flush=True)
    except BaseException:
        run.update(status='projected_common_descent_failed_for_inspection', error=traceback.format_exc(),
                   seconds=time.time()-started, all_issued_work_remains_charged=True)
        checkpoint()
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__ == '__main__':
    main()
