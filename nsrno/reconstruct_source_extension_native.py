"""Finite corrected source-extension trajectory; preserve original native incumbent."""
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
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
from .source_extension_native_pool import SourceExtensionNativePool
from .encoded_physical_native_pool import EncodedPhysicalNativePool
from .depth_rms_preconditioner import depth_rms_direction
from .independent_physical_coordinates import IndependentPhysicalCoordinates, SCALES
from .low_frequency_0p5 import atomic, digest, SAMPLE, CASE
from .pressure_resources import PressureResources
from .reconstruct_native_frequency_incremental import native_RF_records, same_process
from .reconstruct_native_gn import source_manifest
from .reconstruct_frequency_multilevel import array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--previous-final-maps-reviewed', action='store_true')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--cap-gib', type=float, default=10.)
    parser.add_argument('--chunk', type=int, default=128)
    args = parser.parse_args()
    if (not args.previous_final_maps_reviewed or args.steps != 4 or args.chunk != 128 or
        not 0 < args.cap_gib <= 10 or not 1 <= len(args.devices) <= 3 or
        len(set(args.devices)) != len(args.devices)):
        raise ValueError('Reviewed four-step trajectory under the original resource limits only')
    project = Path.cwd().resolve()
    args.parent_root, args.out, args.artifact_root = [v.resolve() for v in
        [args.parent_root, args.out, args.artifact_root]]
    if (not all(p.is_relative_to(project/'out/v5') for p in [args.parent_root, args.out]) or
        not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts')):
        raise ValueError('Independent nsrno project and same-account artifact volume only')
    if args.out.exists() or args.artifact_root.exists():
        raise RuntimeError('Inspect the same existing experiment; do not reset partial work')
    prior = json.loads((args.parent_root/'run.json').read_text())
    if (prior['status'] != 'bounded_neural_subspace_native_continuation_finished' or
        same_process(prior) or prior['sample'] != SAMPLE or prior['case'] != CASE):
        raise RuntimeError('Require the completed, reviewed bounded predecessor')
    previous_workers = json.loads((args.parent_root/'worker_processes.json').read_text())
    if any(Path(f'/proc/{pid}').exists() for pid in previous_workers['worker_pids']):
        raise RuntimeError('Owned predecessor workers have not exited')
    previous_units = prior['total_original_campaign_work_units']
    unit_limit = prior['original_campaign_work_unit_limit']
    previous_checks = prior['total_original_campaign_original44_checks']
    check_limit = prior['original_campaign_original44_check_limit']
    deadline = prior['deadline_unix_s']
    if previous_units+args.steps*7 > unit_limit or previous_checks+2 > check_limit:
        raise RuntimeError('The original remaining work/check budget cannot cover this trajectory')
    if time.time() >= deadline-2400:
        raise RuntimeError('Insufficient time remains under the original absolute deadline')
    plan = json.loads((args.parent_root/'native_plan.json').read_text())
    original_names = ['native_time.py', 'native_time.cu', 'native_tangent.py', 'native_tangent.cu',
        'time_acquisition.py', 'frequency_multilevel_acquisition.py', 'full90_3mhz_acquisition.py',
        'acquisition_single_case_0p5.py', 'low_frequency_0p5.py', 'matrix_free_rf_response.py',
        'encoded_physical_native_worker.py', 'encoded_physical_native_pool.py',
        'independent_physical_coordinates.py', 'depth_rms_preconditioner.py']
    source = Path(__file__).parent
    for name in original_names:
        if name in prior['source_SHA256'] and digest(source/name) != prior['source_SHA256'][name]:
            raise RuntimeError('Frozen original physical/acquisition source changed: '+name)
    qualification = project/'out/v5/source_extension_ray_algebra_CPU_v1/result.json'
    qualified = json.loads(qualification.read_text())
    if qualified['status'] != 'corrected_reduced_ray_algebra_CPU_passed':
        raise RuntimeError('The necessary new objective algebra check did not pass')
    for name, expected in qualified['source_SHA256'].items():
        if digest(source/name) != expected:
            raise RuntimeError('Qualified new objective implementation changed: '+name)
    if (digest(Path(plan['data_root'])/'index.json') != plan['data_index_SHA256'] or
        digest(Path(plan['high_index'])) != plan['high_index_SHA256']):
        raise RuntimeError('Authorized observed data changed')
    if set(plan['families']) != {'lf', 'point3', 'lf3', 'point3_3mhz'}:
        raise RuntimeError('All four acquired eleven-source families are required')
    initial_path = (args.parent_root/'final/material.npy').resolve()
    if digest(initial_path) != prior['material_SHA256']:
        raise RuntimeError('Original44-accepted starting material changed')
    anchor = Path(plan['native_anchor_material']).resolve()
    if digest(anchor) != prior['native_anchor_material_SHA256']:
        raise RuntimeError('Original coordinate anchor changed')
    edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    if digest(edge_path) != plan['known_edge_SHA256']:
        raise RuntimeError('Fixed known coupling/PML absorption changed')
    references = native_RF_records(args.parent_root, prior, plan['families'])
    for row in references:
        if digest(Path(row['path'])) != row['SHA256']:
            raise RuntimeError('Accepted original44 RF cache changed')
    reserve = 2*2**30+400*2**20
    if (shutil.disk_usage(project).free < reserve+3*2**20 or
        shutil.disk_usage(Path.home()).free < reserve+2*2**30):
        raise RuntimeError('Retain original disk reserves')
    def available_host_GiB():
        return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                        if line.startswith('MemAvailable:')))/2**20
    native_n = 961*1121
    per_worker_CP_GiB = 3*((36001+127)//128)*7*native_n*4/2**30
    if available_host_GiB() < per_worker_CP_GiB*len(args.devices)+24:
        raise RuntimeError('Reserve all three CPU checkpoint banks and 24 GiB shared-server headroom')
    coordinates = IndependentPhysicalCoordinates(np.load(anchor), np.load(edge_path))
    current = np.load(args.parent_root/'latest_control.npy').astype(np.float64)
    z, x = np.ogrid[:961, :1121]
    active = (z >= 80)&(z < 937)&(x >= 24)&(x < 1097)
    lateral = abs((np.arange(1121)-560)*.05) <= 19.1
    if not np.array_equal(coordinates.material(current), np.load(initial_path)):
        raise RuntimeError('Native control does not reproduce the exact accepted starting bytes')
    args.out.mkdir()
    args.artifact_root.mkdir()
    for name in ['jobs', 'steps', 'RF_checks']:
        (args.out/name).mkdir()
    for name in ['workers_wri', 'candidates', 'final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name, target_is_directory=True)
    for name in ['latest_control.npy', 'latest_material.npy', 'best_control.npy', 'best_material.npy']:
        (args.out/name).symlink_to(args.artifact_root/name)
    plan = {**plan, 'reference_RF': references, 'source_extension_initial_material': str(initial_path),
        'source_extension_initial_material_SHA256': prior['material_SHA256'],
        'source_extension_source_encoding': False, 'truth_read': False}
    atomic(args.out/'native_plan.json', plan)
    hashes = source_manifest()
    for name in original_names+[Path(__file__).name, 'native_source_extension.py',
        'native_source_extension.cu', 'source_extension_ray.py', 'source_extension_native_worker.py',
        'source_extension_native_pool.py', 'qualify_source_extension_ray_algebra_CPU.py',
        'pressure_resources.py', 'reconstruct_native_frequency_incremental.py',
        'reconstruct_frequency_multilevel.py']:
        hashes[name] = digest(source/name)
    for name in hashes:
        shutil.copy2(source/name, args.out/name)
    started, stop = time.time(), [0]
    for number in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(number, lambda value, frame: stop.__setitem__(0, int(value)))
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    names = list(plan['families'])
    rng = np.random.default_rng(8041)
    schedule = {name: rng.permutation(11).tolist() for name in names}
    ledger = {'token': 0, 'issued_adjoint_direction_passes': 0,
        'completed_adjoint_direction_passes': 0, 'issued_true_source_contributions': 0,
        'completed_true_source_contributions': 0, 'completed_gradient_jobs': 0,
        'completed_line_value_jobs': 0}
    run = {'status': 'initializing_corrected_source_extension_native_inverse',
        'pid': os.getpid(), 'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'original_campaign_created_unix_s': prior['original_campaign_created_unix_s'],
        'deadline_unix_s': deadline, 'original_campaign_work_unit_limit': unit_limit,
        'original_campaign_original44_check_limit': check_limit,
        'previous_campaign_work_units': previous_units, 'previous_campaign_original44_checks': previous_checks,
        'maximum_steps': args.steps, 'maximum_lines_per_step': 2, 'maximum_new_original44_checks': 2,
        'maximum_new_adjoint_direction_passes_including_replays': args.steps*7*4,
        'total_original_campaign_work_units': previous_units,
        'total_original_campaign_original44_checks': previous_checks,
        'sample': SAMPLE, 'case': CASE, 'parent_root': str(args.parent_root),
        'artifact_root': str(args.artifact_root), 'devices': args.devices,
        'allocator_cap_GiB': args.cap_gib, 'per_card_process_tree_limit_GiB': 16,
        'CPU_checkpoint_banks_per_worker': 3, 'static_CPU_checkpoint_GiB_per_worker': per_worker_CP_GiB,
        'source_SHA256': hashes, 'CPU_objective_qualification_SHA256': digest(qualification),
        'native_anchor_material_SHA256': prior['native_anchor_material_SHA256'],
        'initial_material_SHA256': prior['material_SHA256'], 'source_schedule': schedule,
        'all_eleven_sources_remain_in_full_objective_and_original_RF_checks': True,
        'true_source_subsampling_is_not_discarding_acquisition_angles': True,
        'nonlinear_ray_objective_not_source_encoded': True,
        'patient_truth_read_by_optimizer': False, 'reserved_case_unread': 'Neg_47_Left',
        'independent_three_native_controls': True, 'material_field_lowpass': False,
        'TV_weight': .001, 'physical_log_step_caps': [.004, .006, .05],
        'new_data_or_frequency': False, 'new_patient_derivative_validation_probes': 0,
        'joint_imaging_success_claimed': False, 'original_budgets_not_reset': True,
        'original_forward_propagations': 0, 'original_forward_propagations_issued': 0,
        'original_full_RF_checks': 0, 'accepted_ray_steps': 0, 'accepted_original44_candidates': 0}
    best = current.copy()
    best_folder = args.parent_root/'final'
    acceptance = json.loads((best_folder/'actual_native_acceptance.json').read_text())
    best_RF = acceptance['actual_native_RF']
    if abs(best_RF['loss']-prior['final_actual_native_RF_loss']) > 1e-12:
        raise RuntimeError('Accepted RF record and terminal predecessor disagree')
    best_TV, best_records = coordinates.regularization(best, .001)[0], references
    run.update(initial_original44_RF_loss=best_RF['loss'], initial_TV=best_TV,
        initial_RF_reused_from_same_material_operator_and_acquisition=True,
        last_full_accepted_RF_loss=best_RF['loss'], last_full_accepted_material_SHA256=prior['material_SHA256'])
    calibration, pool, RF_pool = {}, None, None
    monitor = PressureResources(args.out)
    generation, check_index, stop_reason = 0, 0, None
    RF_cache = {prior['material_SHA256']: (best_RF, references)}

    def freeze():
        if any(digest(source/name) != expected for name, expected in hashes.items()):
            raise RuntimeError('Frozen executable changed during actual inverse work')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Original disk reserve reached')

    def checkpoint():
        array(args.artifact_root/'latest_control.npy', current)
        array(args.artifact_root/'latest_material.npy', coordinates.material(current))
        array(args.artifact_root/'best_control.npy', best)
        array(args.artifact_root/'best_material.npy', coordinates.material(best))
        run.update(source_extension_work_ledger=dict(ledger), ray_calibration=calibration,
            total_original_campaign_work_units=previous_units+ledger['issued_adjoint_direction_passes']/4,
            total_original_campaign_original44_checks=previous_checks+run['original_forward_propagations_issued']/44,
            coordinator_peak_host_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        atomic(args.out/'run.json', run)

    def new_pool():
        nonlocal pool, generation
        if available_host_GiB() < per_worker_CP_GiB*len(args.devices)+24:
            raise RuntimeError('Insufficient shared-server host memory for actual checkpoint banks')
        generation += 1
        pool = SourceExtensionNativePool(args, monitor, plan, generation, ledger, calibration)

    def maps(folder):
        (folder/'stage0_initial_material.npy').symlink_to(initial_path)
        with (folder/'maps.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material',
                str(folder/'material.npy'), '--sample', SAMPLE, '--out', str(folder/'maps'),
                '--estimate-label', 'Corrected physical source-extension trajectory'],
                stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                check=True, timeout=120)

    def original_check(step):
        nonlocal pool, RF_pool, check_index, best, best_RF, best_TV, best_folder, best_records
        pool.close()
        pool = None
        folder = args.out/'candidates'/f'step{step:02d}'
        folder.mkdir()
        array(folder/'control.npy', current)
        array(folder/'material.npy', coordinates.material(current))
        sha = digest(folder/'material.npy')
        if sha in RF_cache:
            actual, records = RF_cache[sha]
            reused = True
        else:
            if (run['original_full_RF_checks'] >= 2 or
                previous_checks+run['original_forward_propagations_issued']/44 >= check_limit or
                time.time() >= deadline-240):
                raise RuntimeError('Original complete RF check budget exhausted before a new evaluation')
            check_index += 1
            check_root = args.out/'RF_checks'/f'check{check_index:02d}'
            check_root.mkdir()
            artifact_check = args.artifact_root/f'RF_check{check_index:02d}'
            artifact_check.mkdir()
            (check_root/'workers').symlink_to(artifact_check, target_is_directory=True)
            (check_root/'jobs').mkdir()
            atomic(check_root/'native_plan.json', plan)
            run.update(status='original_all44_source_extension_candidate_acceptance', active_step=step,
                       active_RF_check_root=str(check_root))
            atomic(args.out/'active_job.json', {'kind': 'original44', 'tag': f'step{step:02d}',
                'started_unix_s': time.time(), 'check_root': str(check_root)})
            run['original_forward_propagations_issued'] += 44
            checkpoint()
            RF_args = SimpleNamespace(**{**vars(args), 'out': check_root})
            RF_pool = EncodedPhysicalNativePool(RF_args, monitor, plan)
            workers = json.loads((check_root/'worker_processes.json').read_text())
            atomic(args.out/'worker_processes.json', {**workers, 'kind': 'original_RF',
                'check_root': str(check_root), 'one_GPU_experiment': True})
            actual, _ = RF_pool.evaluate(coordinates.material(current), 'forward',
                                         f'step{step:02d}_original44', save_RF=True)
            records = []
            for family in names:
                for shot in range(11):
                    files = list((check_root/'workers').glob(
                        f'rank*/fine_RF/token{actual["token"]:04d}/{family}_{shot:02d}.npy'))
                    if len(files) != 1:
                        raise RuntimeError('Incomplete actual original44 RF archive')
                    records.append({'family': family, 'shot': shot,
                        'path': str(files[0].resolve()), 'SHA256': digest(files[0])})
            run['original_full_RF_checks'] += 1
            run['original_forward_propagations'] += 44
            RF_cache[sha] = (actual, records)
            RF_pool.close()
            RF_pool = None
            reused = False
        tv = coordinates.regularization(current, .001)[0]
        accepted = bool(actual['loss'] < best_RF['loss'] and actual['loss']+tv < best_RF['loss']+best_TV)
        atomic(folder/'actual_native_acceptance.json', {'step': step, 'accepted': accepted,
            'RF_before': best_RF['loss'], 'actual_native_RF': actual, 'TV_before': best_TV,
            'TV_after': tv, 'RF_source_records': records, 'material_SHA256': sha,
            'RF_cache_reused_for_identical_material': reused,
            'exploratory_path_continues_after_nonacceptance': True,
            'patient_truth_read': False, 'joint_imaging_success_claimed': False})
        maps(folder)
        if accepted:
            best, best_RF, best_TV, best_records, best_folder = current.copy(), actual, tv, records, folder
            run['accepted_original44_candidates'] += 1
            run.update(last_full_accepted_RF_loss=best_RF['loss'], last_full_accepted_material_SHA256=sha)
        checkpoint()
        print(json.dumps({'step': step, 'original44_RF': actual['loss'], 'accepted_native_incumbent': accepted,
                          'material_SHA256': sha, 'goal_achieved_claimed': False}), flush=True)

    try:
        checkpoint()
        new_pool()
        caps = np.array([.004, .006, .05])[:, None, None]/SCALES
        for step in range(1, args.steps+1):
            if stop[0] or time.time() >= deadline-900:
                stop_reason = 'signal' if stop[0] else 'original_time_budget'
                break
            freeze()
            if previous_units+(ledger['issued_adjoint_direction_passes']+28)/4 > unit_limit:
                stop_reason = 'original_adjoint_direction_work_budget'
                break
            sources = {name: schedule[name][step-1] for name in names}
            run.update(status='actual_corrected_source_extension_gradient', active_step=step, active_sources=sources)
            checkpoint()
            result, gradient = pool.evaluate(coordinates.material(current), sources,
                'ray_gradient', f'step{step:02d}_actual_corrected_gradient')
            TV, TV_gradient = coordinates.regularization(current, .001)
            total_gradient = (gradient+TV_gradient)*active[None]
            direction, metric = depth_rms_direction(total_gradient, active, lateral, caps)
            slope = float(np.sum(total_gradient*direction))
            if slope >= 0 or not np.isfinite(slope):
                raise RuntimeError('The independent physical update is not an actual descent direction')
            lines, accepted = [], False
            for fraction in [1., .5]:
                if stop[0] or time.time() >= deadline-600:
                    stop_reason = 'signal' if stop[0] else 'original_time_budget_before_line'
                    break
                proposal = coordinates.project(current+fraction*direction, active)
                change = proposal-current
                actual_slope = float(np.sum(total_gradient*change))
                if (np.max(np.abs(change)/caps) > 1.00001 or actual_slope >= 0):
                    raise RuntimeError('Projection violated independent bounded descent')
                run.update(status='actual_source_extension_line_value', active_line_fraction=fraction)
                checkpoint()
                trial, _ = pool.evaluate(coordinates.material(proposal), sources,
                    'ray_value', f'step{step:02d}_line{len(lines)+1}')
                proposed_TV = coordinates.regularization(proposal, .001)[0]
                ok = bool(trial['loss']+proposed_TV < result['loss']+TV+1e-4*actual_slope)
                lines.append({'fraction': fraction, 'actual_control_slope': actual_slope,
                    'ray_before': result['loss'], 'ray_after': trial['loss'], 'TV_after': proposed_TV,
                    'accepted': ok, 'token': trial['token']})
                if ok:
                    current = proposal.copy()
                    run['accepted_ray_steps'] += 1
                    accepted = True
                    break
            atomic(args.out/'steps'/f'step{step:02d}.json', {'step': step, 'sources': sources,
                'gradient_result_token': result['token'], 'accepted_ray_step': accepted,
                'metric': metric, 'lines': lines, 'patient_truth_read': False,
                'new_material_derivative_validation_probes': 0})
            checkpoint()
            if stop_reason:
                break
            if step % 2 == 0 or step == args.steps:
                original_check(step)
                if step < args.steps and not stop[0] and time.time() < deadline-900:
                    new_pool()
            if stop[0]:
                stop_reason = 'signal'
                break
        stop_reason = stop_reason or 'prespecified_remaining_budget_trajectory_finished'
        if pool is not None:
            pool.close()
            pool = None
        if RF_pool is not None:
            RF_pool.close()
            RF_pool = None
        final = args.out/'final'
        array(final/'material.npy', coordinates.material(best))
        array(final/'control.npy', best)
        atomic(final/'actual_native_acceptance.json', {'accepted': run['accepted_original44_candidates'] > 0,
            'material_SHA256': digest(final/'material.npy'), 'actual_native_RF': best_RF,
            'RF_source_records': best_records, 'native_TV': best_TV,
            'full_original44_source_acceptance': True, 'joint_imaging_success_claimed': False})
        maps(final)
        run.update(status='source_extension_native_trajectory_finished', stop_reason=stop_reason,
            finished_unix_s=time.time(), seconds=time.time()-started,
            original_campaign_elapsed_seconds=time.time()-prior['original_campaign_created_unix_s'],
            final_actual_native_RF_loss=best_RF['loss'], material_SHA256=digest(final/'material.npy'),
            final_actual_maps_still_require_visual_review=True, stopped_signal=stop[0])
        checkpoint()
        print(json.dumps({k: run[k] for k in ['status', 'stop_reason', 'seconds',
            'accepted_ray_steps', 'accepted_original44_candidates', 'total_original_campaign_work_units',
            'total_original_campaign_original44_checks', 'final_actual_native_RF_loss', 'material_SHA256']}), flush=True)
    except BaseException:
        run.update(status='source_extension_native_failed_for_inspection', error=traceback.format_exc(),
                   stopped_unix_s=time.time(), issued_work_remains_charged=True)
        checkpoint()
        raise
    finally:
        if pool is not None:
            pool.close()
        if RF_pool is not None:
            RF_pool.close()
        monitor.log.close()


if __name__ == '__main__':
    main()
