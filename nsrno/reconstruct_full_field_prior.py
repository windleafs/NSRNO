"""One bounded independent-prior update from fresh true all44 native gradients."""
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
from .full_field_prior_pool import FullFieldPriorPool
from .full_field_prior_step import regularization, build_directions, solve_prior_step
from .factorized_material_prior import FactorizedMaterialPrior
from .independent_physical_coordinates import IndependentPhysicalCoordinates
from .pressure_resources import PressureResources
from .low_frequency_0p5 import atomic, digest, SAMPLE, CASE
from .reconstruct_frequency_multilevel import array
from .reconstruct_native_frequency_incremental import same_process


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--prior', type=Path, default=Path('out/v5/independent_material_marginals_v1/prior.npz'))
    p.add_argument('--qualification', type=Path,
                   default=Path('out/v5/factorized_prior_CPU_qualification_v1/result.json'))
    p.add_argument('--devices', type=int, nargs='+', default=[0, 1])
    p.add_argument('--cap-gib', type=float, default=10.)
    p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--max-seconds', type=float, default=4500.)
    args = p.parse_args()
    project = Path.cwd().resolve()
    args.parent_root, args.out, args.artifact_root = [v.resolve() for v in
        [args.parent_root, args.out, args.artifact_root]]
    if (not all(v.is_relative_to(project/'out/v5') for v in [args.parent_root, args.out]) or
        not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or
        args.out.exists() or args.artifact_root.exists() or
        args.devices != [0, 1] or not 0 < args.cap_gib <= 10 or args.chunk != 128 or
        args.max_seconds != 4500):
        raise ValueError('Dedicated one-step, two-card, 55-unit/3-line/4500-second prior experiment only')
    previous = json.loads((args.parent_root/'run.json').read_text())
    workers = json.loads((args.parent_root/'worker_processes.json').read_text())
    if (previous['status'] != 'all44_joint_tangent_stage_finished' or same_process(previous) or
        any(Path(f'/proc/{pid}').exists() for pid in workers['worker_pids']) or
        previous['reserved_campaign_units'] != 134 or previous['original44_checks_completed'] != 1):
        raise RuntimeError('Require the completed source trajectory; its closed budget is not reset')
    qualification = json.loads(args.qualification.read_text())
    source = Path(__file__).parent
    if (qualification['status'] != 'factorized_TRAIN_prior_physical_log_gradient_CPU_passed' or
        qualification['prior_SHA256'] != digest(args.prior) or
        qualification['source_SHA256'] != digest(source/'factorized_material_prior.py')):
        raise RuntimeError('Prior physical-log gradient qualification changed')
    metadata = json.loads(args.prior.with_name('metadata.json').read_text())
    if metadata['source_case'] != 'Neg_07_Left' or metadata['development_material_arrays_read']:
        raise RuntimeError('Independent prior must remain TRAIN-only')
    plan = json.loads((args.parent_root/'native_plan.json').read_text())
    prior_acceptance = json.loads((args.parent_root/'final/actual_native_acceptance.json').read_text())
    initial = (args.parent_root/'final/material.npy').resolve()
    if digest(initial) != previous['material_SHA256']:
        raise RuntimeError('Retained original44 material changed')
    references = prior_acceptance['RF_source_records']
    if sorted((r['family'], r['shot']) for r in references) != sorted(
            (family, shot) for family in plan['families'] for shot in range(11)):
        raise RuntimeError('The accepted cache must contain all44 true sources')
    for row in references:
        if digest(row['path']) != row['SHA256']:
            raise RuntimeError('Accepted source RF cache changed')
    plan = {**plan, 'reference_RF': references,
        'reference_material_SHA256': previous['material_SHA256'],
        'new_material_prior_experiment_no_new_data': True}
    if (digest(Path(plan['data_root'])/'index.json') != plan['data_index_SHA256'] or
        digest(Path(plan['high_index'])) != plan['high_index_SHA256']):
        raise RuntimeError('Authorized observations changed')
    anchor = Path(plan['native_anchor_material'])
    edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    if (digest(anchor) != previous['native_anchor_material_SHA256'] or
        digest(edge_path) != plan['known_edge_SHA256']):
        raise RuntimeError('Original coordinate map or known absorption changed')
    coordinates = IndependentPhysicalCoordinates(np.load(anchor), np.load(edge_path))
    control = np.load(args.parent_root/'final/control.npy').astype(np.float64)
    material = coordinates.material(control)
    if not np.array_equal(material, np.load(initial)):
        raise RuntimeError('Independent control does not reproduce the exact accepted material')
    z, x = np.ogrid[:961, :1121]
    active = (z >= 80)&(z < 937)&(x >= 24)&(x < 1097)
    physical_z = (np.arange(961)-60)*.05
    physical_x = (np.arange(1121)-560)*.05
    lateral = abs(physical_x) <= 19.1
    roi = (physical_z[:, None] >= 1)&(physical_z[:, None] <= 43)&lateral[None]
    prior = FactorizedMaterialPrior(args.prior, tail=.1, temperature=0.)
    R0, gradient_prior, parts0 = regularization(coordinates, prior, control, roi)
    RF0 = prior_acceptance['actual_native_RF']['loss']
    if abs(RF0-previous['final_RF']) > 1e-12:
        raise RuntimeError('Native accepted RF records disagree')
    core_names = ['native_time.py', 'native_time.cu', 'native_tangent.py', 'native_tangent.cu',
        'frequency_multilevel_acquisition.py', 'full90_3mhz_acquisition.py', 'matrix_free_rf_response.py',
        'independent_physical_coordinates.py', 'depth_rms_preconditioner.py', 'low_frequency_0p5.py']
    for name in core_names:
        if digest(source/name) != previous['source_SHA256'][name]:
            raise RuntimeError('Qualified physical or acquisition implementation changed: '+name)
    inherited = json.loads((Path(previous['parent_root'])/'run.json').read_text())['source_SHA256']
    extra_core = ['time_acquisition.py', 'acquisition_single_case_0p5.py']
    for name in extra_core:
        if digest(source/name) != inherited[name]:
            raise RuntimeError('Previously used waveform implementation changed: '+name)
    names = core_names+extra_core+[Path(__file__).name, 'full_field_prior_worker.py', 'full_field_prior_pool.py',
        'full_field_prior_step.py', 'factorized_material_prior.py', 'constrained_prior_saved_model_CPU.py',
        'native_full_source_dual.py', 'pressure_resources.py']
    hashes = {name: digest(source/name) for name in names}
    reserve = 2*2**30+400*2**20
    if shutil.disk_usage(project).free < reserve+2*2**20 or shutil.disk_usage(Path.home()).free < reserve+800*2**20:
        raise RuntimeError('Retain the existing disk reserves')
    available = int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines()
                         if v.startswith('MemAvailable:')))/2**20
    per_worker_bank = ((36001+127)//128)*7*961*1121*4/2**30
    if available < 2*per_worker_bank*len(args.devices)+24:
        raise RuntimeError('Retain two possible CPU checkpoint banks per worker and 24 GiB headroom')
    args.out.mkdir()
    args.artifact_root.mkdir()
    for name in ['arrays', 'workers', 'candidates', 'final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name, target_is_directory=True)
    (args.out/'jobs').mkdir()
    for name in hashes:
        shutil.copy2(source/name, args.out/name)
    atomic(args.out/'native_plan.json', plan)
    array(args.out/'arrays/initial_control.npy', control)
    array(args.out/'arrays/initial_material.npy', material)
    array(args.out/'arrays/initial_prior_gradient.npy', gradient_prior)
    started = time.time()
    deadline = started+args.max_seconds
    def request_stop(number, frame):
        (args.out/'stop_requested.flag').write_text(str(number)+'\n')
    for number in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(number, request_stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    run = {'status': 'initializing_full_field_independent_prior_update', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': deadline, 'parent_root': str(args.parent_root),
        'artifact_root': str(args.artifact_root), 'sample': SAMPLE, 'case': CASE,
        'source_SHA256': hashes, 'native_anchor_material_SHA256': digest(anchor),
        'initial_material_SHA256': digest(initial), 'prior_SHA256': digest(args.prior),
        'prior_qualification_SHA256': digest(args.qualification),
        'initial_RF': RF0, 'initial_regularization': R0, 'initial_prior_parts': parts0,
        'maximum_outer_updates': 1, 'maximum_candidate_lines': 3,
        'maximum_new_actual_adjoint_passes': 44, 'maximum_new_actual_tangent_passes': 176,
        'maximum_new_gradient_and_tangent_background_forward_passes': 220,
        'maximum_new_candidate_forward_passes': 132, 'new_reserved_equivalent_units': 55,
        'reservation_is_conservative_not_measured_actual_work': True,
        'closed_predecessor_budget': previous['closed_predecessor_budget'],
        'closed_full_dual_budget': {'reserved_units': previous['reserved_campaign_units'],
            'limit_units': 136, 'complete_original44_checks': 2, 'original44_limit': 2,
            'deadline_unix_s': previous['deadline_unix_s'], 'retained_unchanged': True},
        'new_experiment_is_independent_material_prior_objective_not_old_budget_resume': True,
        'cumulative_reserved_units_from_closed_predecessor': 89+134+55,
        'prior_weight_sweep': False, 'prior_tail': .1, 'prior_temperature': 0.,
        'objective': 'min independent marginal prior + existing TV, subject to original44 RF nonincrease',
        'physical_log_step_caps': [.02, .04, .12], 'no_material_lowpass': True,
        'independent_native_material_fields': True, 'shape': [3, 961, 1121],
        'devices': args.devices, 'per_card_process_tree_limit_GiB': 16,
        'allocator_cap_GiB': args.cap_gib, 'one_GPU_experiment': True,
        'all11_sources_per_family': True, 'source_encoding': False,
        'patient_truth_read_by_optimizer': False, 'reserved_case_unread': 'Neg_47_Left',
        'new_data_or_frequency': False, '0p15MHz_generated_or_authorized_by_this_run': False,
        'new_patient_derivative_probes': 0, 'all_gradients_and_JVPs_used_in_actual_inverse': True,
        'original44_checks_issued': 0, 'original44_checks_completed': 0,
        'accepted_prior_update': False, 'joint_reconstruction_success_claimed': False}
    pool = None
    monitor = PressureResources(args.out)
    selected, selected_control, selected_RF, selected_R = None, control, prior_acceptance['actual_native_RF'], R0
    last_evaluated = None
    def checkpoint():
        if pool is not None:
            run['actual_native_work'] = pool.actual_work()
        atomic(args.out/'run.json', run)
    def freeze():
        if any(digest(source/name) != sha for name, sha in hashes.items()):
            raise RuntimeError('Frozen executable changed during the actual inverse')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Existing disk reserve reached')
        if time.time() >= deadline-60:
            raise RuntimeError('Separately declared prior deadline reached; all work remains charged')
        if (args.out/'stop_requested.flag').exists():
            raise InterruptedError('Stop requested; retain every completed product without replay')
    def maps(folder, label):
        (folder/'stage0_initial_material.npy').symlink_to(initial)
        with (folder/'maps.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material', str(folder/'material.npy'),
                '--sample', SAMPLE, '--out', str(folder/'maps'), '--estimate-label', label],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log,
                stderr=subprocess.STDOUT, check=True, timeout=120)
    try:
        checkpoint()
        freeze()
        pool = FullFieldPriorPool(args, monitor, plan, deadline)
        run['status'] = 'actual_all44_physical_gradient_for_new_prior_directions'
        checkpoint()
        gradient_result = pool.evaluate(material, 'original44_gradient')
        if abs(gradient_result['loss']-RF0) > 1e-10:
            raise RuntimeError('Same-material all44 objective changed during required gradient computation')
        gradient_data = np.load(gradient_result['gradient_path'])
        directions, physical, direction_details = build_directions(coordinates, control,
            gradient_data, gradient_prior, active, lateral)
        array(args.out/'arrays/control_directions.npy', directions)
        array(args.out/'arrays/physical_directions.npy', physical)
        atomic(args.out/'direction_details.json', direction_details)
        run.update(status='actual_all44_JVPs_for_new_full_field_prior_directions',
            data_gradient_SHA256=gradient_result['gradient_SHA256'],
            control_direction_SHA256=digest(args.out/'arrays/control_directions.npy'),
            physical_direction_SHA256=digest(args.out/'arrays/physical_directions.npy'))
        checkpoint()
        freeze()
        model = pool.evaluate(material, 'joint_tangents', args.out/'arrays/physical_directions.npy')
        run['status'] = 'CPU_prior_selection_constrained_by_all44_data_model'
        checkpoint()
        coefficients, details = solve_prior_step(model['H'], model['linear'], RF0,
            coordinates, prior, control, roi, directions)
        atomic(args.out/'local_prior_solution.json', details)
        run.update(coefficients=coefficients.tolist(),
                   predicted_RF_after_prior_step=details['RF_model_after_trust'])
        for line, fraction in enumerate([1., .5, .25], 1):
            freeze()
            proposal = control+fraction*np.einsum('a,aijk->ijk', coefficients, directions)
            projected = coordinates.project(proposal, active)
            if np.max(abs(projected-proposal)) > 1e-9:
                raise RuntimeError('Physical projection altered the measured direction combination')
            R, _, parts = regularization(coordinates, prior, proposal, roi)
            if R >= R0-1e-12:
                atomic(args.out/f'skipped_line{line:02d}.json', {'line': line,
                    'reason': 'no_prior_decrease_before_any_new_wave', 'regularization': R})
                continue
            folder = args.out/'candidates'/f'line{line:02d}'
            folder.mkdir()
            array(folder/'control.npy', proposal)
            array(folder/'material.npy', coordinates.material(proposal))
            run.update(status='actual_all44_prior_candidate_line', active_line=line,
                active_fraction=fraction, active_regularization=R)
            run['original44_checks_issued'] += 1
            checkpoint()
            trial = pool.evaluate(coordinates.material(proposal), 'original44_forward',
                                  args.out/'arrays/physical_directions.npy')
            run['original44_checks_completed'] += 1
            accepted = bool(trial['loss'] < RF0 and
                trial['loss']+parts['TV'] < RF0+parts0['TV'] and R < R0)
            atomic(folder/'actual_native_acceptance.json', {'accepted': accepted,
                'line': line, 'fraction': fraction, 'RF_before': RF0, 'RF_after': trial['loss'],
                'prior_TV_before': R0, 'prior_TV_after': R, 'parts_before': parts0, 'parts_after': parts,
                'material_SHA256': digest(folder/'material.npy'), 'all44_true_sources': True,
                'joint_reconstruction_success_claimed': False})
            last_evaluated = folder
            checkpoint()
            print(json.dumps({'line': line, 'RF': trial['loss'], 'regularization': R,
                              'accepted_prior_step': accepted}), flush=True)
            if accepted:
                selected, selected_control, selected_RF, selected_R = folder, proposal, trial, R
                run['accepted_prior_update'] = True
                break
        pool.close()
        pool = None
        final = args.out/'final'
        array(final/'control.npy', selected_control)
        array(final/'material.npy', coordinates.material(selected_control))
        if selected is not None:
            refs = []
            entries = [(family, shot) for family in plan['families'] for shot in range(11)]
            for index, (family, shot) in enumerate(entries):
                rank = index % len(args.devices)
                path = args.out/'workers'/f'rank{rank:02d}'/'arrays'/f'token{selected_RF["token"]:04d}'/f'{family}_{shot:02d}.npy'
                refs.append({'family': family, 'shot': shot, 'path': str(path.resolve()), 'SHA256': digest(path)})
            native_tv = regularization(coordinates, prior, selected_control, roi)[2]['TV']
        else:
            refs, native_tv = references, parts0['TV']
        atomic(final/'actual_native_acceptance.json', {'accepted': selected is not None,
            'actual_native_RF': selected_RF, 'RF_source_records': refs, 'native_TV': native_tv,
            'material_SHA256': digest(final/'material.npy'), 'full_original44_source_acceptance': True,
            'independent_prior_value': selected_R, 'joint_imaging_success_claimed': False})
        plotted = selected or last_evaluated
        if plotted is not None:
            maps(plotted, 'Independent-prior joint update')
        if selected is not None:
            for suffix in ['png', 'metrics.json']:
                shutil.copy2(selected/f'maps.{suffix}', final/f'maps.{suffix}')
        else:
            for suffix in ['png', 'metrics.json']:
                shutil.copy2(args.parent_root/f'final/maps.{suffix}', final/f'maps.{suffix}')
        atomic(final/'maps_provenance.json', {
            'copied_from_same_material': str(selected or args.parent_root/'final'),
            'material_SHA256': digest(final/'material.npy'),
            'no_additional_truth_evaluation_for_duplicate_final_material': True})
        run.update(status='full_field_prior_native_stage_finished',
            stop_reason='one_separately_budgeted_full_field_prior_update_finished',
            final_RF=selected_RF['loss'], final_regularization=selected_R,
            material_SHA256=digest(final/'material.npy'), selected_candidate=str(selected) if selected else None,
            last_evaluated_candidate=str(last_evaluated) if last_evaluated else None,
            seconds=time.time()-started, finished_unix_s=time.time())
        checkpoint()
        print(json.dumps({k: run[k] for k in ['status', 'accepted_prior_update', 'final_RF',
            'final_regularization', 'material_SHA256', 'original44_checks_completed', 'seconds']}), flush=True)
    except BaseException:
        run.update(status='full_field_prior_failed_for_inspection', error=traceback.format_exc(),
            issued_work_remains_charged=True, seconds=time.time()-started)
        checkpoint()
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__ == '__main__':
    main()
