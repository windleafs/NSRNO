"""Continue the existing prior update after its overly strict scalar RF guard.

Reuses all 44 completed adjoints, their original counters and absolute deadline.
No completed acoustic propagation is replayed. The original frozen sources stay
unchanged; this continuation is separately fingerprinted in the same run.
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
from .full_field_prior_pool import FullFieldPriorPool
from .full_field_prior_step import regularization, build_directions, solve_prior_step
from .factorized_material_prior import FactorizedMaterialPrior
from .independent_physical_coordinates import IndependentPhysicalCoordinates
from .pressure_resources import PressureResources
from .low_frequency_0p5 import atomic, digest, SAMPLE
from .reconstruct_frequency_multilevel import array
from .reconstruct_native_frequency_incremental import same_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    project = Path.cwd().resolve()
    if root != project/'out/v5/reconstruction_full_field_prior_v1':
        raise ValueError('This continuation applies only to the stopped existing update')
    run = json.loads((root/'run.json').read_text())
    old_workers = json.loads((root/'worker_processes.json').read_text())
    stop = 'Same-material all44 objective changed during required gradient computation'
    if (run['status'] != 'full_field_prior_failed_for_inspection' or stop not in run['error'] or
        same_process(run) or any(Path(f'/proc/{p}').exists() for p in old_workers['worker_pids']) or
        run['original44_checks_issued'] or run['original44_checks_completed'] or
        run['new_reserved_equivalent_units'] != 55 or
        run['deadline_unix_s'] != run['created_unix_s']+4500):
        raise RuntimeError('Not the one completed-gradient stop; do not repeat work or reset its budget')
    snapshot = root/'gradient_completed_stop.json'
    if snapshot.exists() and json.loads(snapshot.read_text()) != run:
        raise RuntimeError('Stopped run changed after the saved CPU-only continuation attempt')
    deadline, started = run['deadline_unix_s'], run['created_unix_s']
    if time.time() >= deadline-60:
        raise RuntimeError('Original prior-stage deadline already reached')
    plan = json.loads((root/'native_plan.json').read_text())
    source = Path(__file__).parent
    for name, sha in run['source_SHA256'].items():
        if digest(source/name) != sha:
            raise RuntimeError('Frozen source changed: '+name)
    gradient_result = json.loads((root/'jobs/0001_original44_gradient.json').read_text())
    work = gradient_result['actual_work']
    if (not gradient_result['all44_sources'] or gradient_result['patient_truth_read'] or
        work['completed_adjoint_direction_passes'] != 44 or
        work['issued_adjoint_direction_passes'] != 44 or
        work['completed_physical_forward_passes'] != 44 or
        work['issued_physical_forward_passes'] != 44 or
        work['issued_tangent_direction_passes'] or work['completed_tangent_direction_passes'] or
        work['unfinished_native_phases'] or (root/'jobs/0002_joint_tangents.json').exists()):
        raise RuntimeError('Require all completed gradients and no begun tangent work')
    if digest(gradient_result['gradient_path']) != gradient_result['gradient_SHA256']:
        raise RuntimeError('Completed aggregate gradient changed')
    entries = sorted((name, shot) for name in plan['families'] for shot in range(11))
    completed = []
    for rank in range(2):
        folder = root/'workers'/f'rank{rank:02d}'
        receipt = json.loads((folder/'gradient_receipt_token0001.json').read_text())
        native = json.loads((folder/'native_work.json').read_text())
        if native['inflight_phase'] is not None or digest(receipt['path']) != receipt['SHA256']:
            raise RuntimeError('Do not resume an incomplete native phase')
        completed.extend(tuple(row) for row in receipt['completed_sources'])
    if sorted(completed) != entries:
        raise RuntimeError('Saved ordinary gradient is missing a true source')
    parent = Path(run['parent_root'])
    acceptance = json.loads((parent/'final/actual_native_acceptance.json').read_text())
    RF0 = acceptance['actual_native_RF']['loss']
    difference = gradient_result['loss']-RF0
    # A case-specific floating-point consistency allowance, not a rigorous
    # propagation-error bound. Exact material/source/acquisition hashes remain
    # mandatory. No new precision experiment is performed to set it.
    allowance = float(np.finfo(np.float32).eps)*abs(RF0)
    if abs(difference) > allowance or RF0 != run['initial_RF']:
        raise RuntimeError('Observed discrepancy exceeds one relative float32 epsilon')
    for path, sha in [(Path(plan['native_anchor_material']), run['native_anchor_material_SHA256']),
                      (Path(plan['data_root'])/'index.json', plan['data_index_SHA256']),
                      (Path(plan['high_index']), plan['high_index_SHA256']),
                      (project/'out/v5/uno_triparam_v1/known_edge_alpha.npy', plan['known_edge_SHA256'])]:
        if digest(path) != sha:
            raise RuntimeError('Original physical inputs changed')
    for row in plan['reference_RF']:
        if digest(row['path']) != row['SHA256']:
            raise RuntimeError('Same-material accepted RF cache changed')
    coordinates = IndependentPhysicalCoordinates(np.load(plan['native_anchor_material']),
        np.load(project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'))
    control = np.load(root/'arrays/initial_control.npy')
    material = coordinates.material(control)
    if (not np.array_equal(material, np.load(root/'arrays/initial_material.npy')) or
        not np.array_equal(material, np.load(parent/'final/material.npy')) or
        digest(parent/'final/material.npy') != run['initial_material_SHA256']):
        raise RuntimeError('Continuation must use the exactly identical accepted material')
    prior_path = project/'out/v5/independent_material_marginals_v1/prior.npz'
    if digest(prior_path) != run['prior_SHA256']:
        raise RuntimeError('Independent TRAIN prior changed')
    prior = FactorizedMaterialPrior(prior_path, tail=.1, temperature=0.)
    gradient_data = np.load(gradient_result['gradient_path'])
    gradient_prior = np.load(root/'arrays/initial_prior_gradient.npy')
    z, x = np.ogrid[:961, :1121]
    active = (z >= 80)&(z < 937)&(x >= 24)&(x < 1097)
    physical_z = (np.arange(961)-60)*.05
    physical_x = (np.arange(1121)-560)*.05
    lateral = abs(physical_x) <= 19.1
    roi = (physical_z[:, None] >= 1)&(physical_z[:, None] <= 43)&lateral[None]
    R0, _, parts0 = regularization(coordinates, prior, control, roi)
    if abs(R0-run['initial_regularization']) > 1e-12:
        raise RuntimeError('The independent prior objective changed')
    saved_direction_paths = [root/'arrays/control_directions.npy', root/'arrays/physical_directions.npy',
                             root/'direction_details.json']
    if all(path.exists() for path in saved_direction_paths):
        if not snapshot.exists():
            raise RuntimeError('Unexplained existing direction products')
        directions, physical = [np.load(path) for path in saved_direction_paths[:2]]
        direction_details = json.loads(saved_direction_paths[2].read_text())
    elif any(path.exists() for path in saved_direction_paths):
        raise RuntimeError('Inspect incomplete CPU direction products without overwriting them')
    else:
        directions, physical, direction_details = build_directions(coordinates, control,
            gradient_data, gradient_prior, active, lateral)
    reserve = 2*2**30+400*2**20
    if (shutil.disk_usage(project).free < reserve+2*2**20 or
        shutil.disk_usage(Path.home()).free < reserve+800*2**20):
        raise RuntimeError('Retain the existing disk reserves')
    if not snapshot.exists():
        atomic(snapshot, run)
        atomic(root/'gradient_completed_worker_processes.json', old_workers)
        atomic(root/'gradient_completed_resource_summary.json',
               json.loads((root/'resource_summary.json').read_text()))
    old_resource = json.loads((root/'gradient_completed_resource_summary.json').read_text())
    continuation_sha = digest(Path(__file__))
    saved_continuation = root/Path(__file__).name
    if saved_continuation.exists() and digest(saved_continuation) != continuation_sha:
        backup = root/'resume_full_field_prior_saved_gradient_CPU_serialization_stop.py'
        if backup.exists():
            raise RuntimeError('A different continuation revision already exists; inspect it')
        shutil.copy2(saved_continuation, backup)
    shutil.copy2(Path(__file__), root/Path(__file__).name)
    hashes = {**run['source_SHA256'], Path(__file__).name: continuation_sha}
    if not saved_direction_paths[0].exists():
        array(saved_direction_paths[0], directions)
        array(saved_direction_paths[1], physical)
        atomic(saved_direction_paths[2], direction_details)
    run.pop('error')
    run.update(status='resuming_saved_all44_gradient_into_required_JVPs', pid=os.getpid(),
        start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        source_SHA256=hashes, continuation_created_unix_s=time.time(),
        no_completed_gradient_repeated=True, original_absolute_deadline_retained=True,
        original_native_counters_retained=True, saved_gradient_SHA256=gradient_result['gradient_SHA256'],
        CPU_serialization_stop_recovered_without_new_wave=True,
        same_material_RF_comparison={'cache_RF': RF0, 'gradient_RF': gradient_result['loss'],
            'difference': difference, 'one_relative_float32_epsilon': allowance,
            'not_a_rigorous_error_bound': True, 'no_new_precision_probe': True},
        data_gradient_SHA256=gradient_result['gradient_SHA256'],
        control_direction_SHA256=digest(root/'arrays/control_directions.npy'),
        physical_direction_SHA256=digest(root/'arrays/physical_directions.npy'))
    args.out, args.devices, args.cap_gib, args.chunk = root, [0, 1], 10., 128
    monitor = PressureResources(root)
    monitor.maximum_owned = {int(k): v for k, v in old_resource['sampled_peak_owned_MiB'].items()}
    monitor.maximum_whole = {int(k): v for k, v in old_resource['sampled_peak_whole_card_MiB'].items()}
    pool = None
    selected, selected_control, selected_RF, selected_R = None, control, acceptance['actual_native_RF'], R0
    last_evaluated = None
    def checkpoint():
        if pool is not None:
            run['actual_native_work'] = pool.actual_work()
        atomic(root/'run.json', run)
    def freeze():
        if any(digest(source/name) != sha for name, sha in hashes.items()):
            raise RuntimeError('Frozen executable changed during the actual inverse')
        if time.time() >= deadline-60 or (root/'stop_requested.flag').exists():
            raise RuntimeError('Original deadline or cooperative stop reached; preserve partial products')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Existing disk reserve reached')
    def request_stop(number, frame):
        (root/'stop_requested.flag').write_text(str(number)+'\n')
    for number in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(number, request_stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        checkpoint()
        freeze()
        pool = FullFieldPriorPool(args, monitor, plan, deadline)
        # Token 1's 44 ordinary gradients are complete and remain charged.
        pool.token = 1
        run['status'] = 'actual_all44_JVPs_for_new_full_field_prior_directions'
        checkpoint()
        model = pool.evaluate(material, 'joint_tangents', root/'arrays/physical_directions.npy')
        run['status'] = 'CPU_prior_selection_constrained_by_all44_data_model'
        checkpoint()
        coefficients, details = solve_prior_step(model['H'], model['linear'], RF0,
            coordinates, prior, control, roi, directions)
        atomic(root/'local_prior_solution.json', details)
        run.update(coefficients=coefficients.tolist(),
            predicted_RF_after_prior_step=details['RF_model_after_trust'])
        for line, fraction in enumerate([1., .5, .25], 1):
            freeze()
            proposal = control+fraction*np.einsum('a,aijk->ijk', coefficients, directions)
            if np.max(abs(coordinates.project(proposal, active)-proposal)) > 1e-9:
                raise RuntimeError('Physical projection altered the measured direction combination')
            R, _, parts = regularization(coordinates, prior, proposal, roi)
            if R >= R0-1e-12:
                atomic(root/f'skipped_line{line:02d}.json', {'line': line,
                    'reason': 'no_prior_decrease_before_any_new_wave', 'regularization': R})
                continue
            folder = root/'candidates'/f'line{line:02d}'
            folder.mkdir()
            array(folder/'control.npy', proposal)
            array(folder/'material.npy', coordinates.material(proposal))
            run.update(status='actual_all44_prior_candidate_line', active_line=line,
                active_fraction=fraction, active_regularization=R)
            run['original44_checks_issued'] += 1
            checkpoint()
            trial = pool.evaluate(coordinates.material(proposal), 'original44_forward',
                                  root/'arrays/physical_directions.npy')
            run['original44_checks_completed'] += 1
            # Do not count a measured same-material scalar discrepancy as a
            # meaningful RF reduction. This margin uses no new propagation.
            minimum_drop = 10*abs(difference)
            accepted = bool(trial['loss'] < RF0-minimum_drop and
                trial['loss']+parts['TV'] < RF0+parts0['TV'] and R < R0)
            atomic(folder/'actual_native_acceptance.json', {'accepted': accepted,
                'line': line, 'fraction': fraction, 'RF_before': RF0, 'RF_after': trial['loss'],
                'minimum_RF_drop_from_observed_scalar_difference': minimum_drop,
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
        final = root/'final'
        array(final/'control.npy', selected_control)
        array(final/'material.npy', coordinates.material(selected_control))
        if selected is not None:
            refs = []
            ordered = [(family, shot) for family in plan['families'] for shot in range(11)]
            for index, (family, shot) in enumerate(ordered):
                rank = index % len(args.devices)
                path = root/'workers'/f'rank{rank:02d}'/'arrays'/f'token{selected_RF["token"]:04d}'/f'{family}_{shot:02d}.npy'
                refs.append({'family': family, 'shot': shot, 'path': str(path.resolve()), 'SHA256': digest(path)})
            native_tv = regularization(coordinates, prior, selected_control, roi)[2]['TV']
        else:
            refs, native_tv = acceptance['RF_source_records'], parts0['TV']
        atomic(final/'actual_native_acceptance.json', {'accepted': selected is not None,
            'actual_native_RF': selected_RF, 'RF_source_records': refs, 'native_TV': native_tv,
            'material_SHA256': digest(final/'material.npy'), 'full_original44_source_acceptance': True,
            'independent_prior_value': selected_R, 'joint_imaging_success_claimed': False})
        plotted = selected or last_evaluated
        if plotted is not None:
            (plotted/'stage0_initial_material.npy').symlink_to((parent/'final/material.npy').resolve())
            with (plotted/'maps.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material', str(plotted/'material.npy'),
                    '--sample', SAMPLE, '--out', str(plotted/'maps'),
                    '--estimate-label', 'Independent-prior joint update'],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log,
                    stderr=subprocess.STDOUT, check=True, timeout=120)
        copied = selected or parent/'final'
        for suffix in ['png', 'metrics.json']:
            shutil.copy2(copied/f'maps.{suffix}', final/f'maps.{suffix}')
        atomic(final/'maps_provenance.json', {'copied_from_same_material': str(copied),
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
