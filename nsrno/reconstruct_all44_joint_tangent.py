"""Reuse native spatial directions, choose the joint step using every true source."""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from .all44_joint_tangent_pool import All44JointPool
from .all44_joint_subspace import box_quadratic
from .independent_physical_coordinates import IndependentPhysicalCoordinates, SCALES
from .depth_rms_preconditioner import depth_rms_direction
from .pressure_resources import PressureResources
from .low_frequency_0p5 import atomic, digest, SAMPLE, CASE
from .reconstruct_frequency_multilevel import array
from .reconstruct_native_frequency_incremental import same_process


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--devices', type=int, nargs='+', default=[0, 1])
    p.add_argument('--cap-gib', type=float, default=10.)
    p.add_argument('--chunk', type=int, default=128)
    args = p.parse_args()
    project = Path.cwd().resolve()
    args.parent_root, args.out, args.artifact_root = [x.resolve() for x in
        [args.parent_root, args.out, args.artifact_root]]
    if (not args.parent_root.is_relative_to(project/'out/v5') or
        not args.out.is_relative_to(project/'out/v5') or
        not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or
        args.out.exists() or args.artifact_root.exists() or not 0 < args.cap_gib <= 10 or args.chunk != 128):
        raise ValueError('A new dedicated nsrno output with inherited fixed limits is required')
    prior = json.loads((args.parent_root/'run.json').read_text())
    if prior['status'] != 'full_source_dual_native_stage_finished' or same_process(prior):
        raise RuntimeError('The same completed predecessor must be inspected before continuing')
    old_workers = json.loads((args.parent_root/'worker_processes.json').read_text())['worker_pids']
    if any(Path(f'/proc/{pid}').exists() for pid in old_workers):
        raise RuntimeError('Predecessor workers have not exited')
    deadline = prior['deadline_unix_s']
    if time.time() > deadline-3600:
        raise RuntimeError('Fewer than sixty minutes remain in the inherited deadline')
    old_passes = prior['full_dual_budget_ledger']['charged_adjoint_direction_passes']
    if old_passes != 360 or prior['original_full_RF_checks'] != 1:
        raise RuntimeError('Do not silently change the remaining 46-unit, one-RF-check budget')
    plan = json.loads((args.parent_root/'native_plan.json').read_text())
    initial = args.parent_root/'final/material.npy'
    if digest(initial) != prior['material_SHA256'] or prior['material_SHA256'] != prior['initial_material_SHA256']:
        raise RuntimeError('Require the retained same-material original44 incumbent')
    control = np.load(args.parent_root/'best_control.npy').astype(np.float64)
    edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    if digest(edge_path) != plan['known_edge_SHA256']:
        raise RuntimeError('Known coupling absorption changed')
    anchor = Path(plan['native_anchor_material'])
    if digest(anchor) != prior['native_anchor_material_SHA256']:
        raise RuntimeError('Original independent coordinate anchor changed')
    coordinates = IndependentPhysicalCoordinates(np.load(anchor), np.load(edge_path))
    material = coordinates.material(control)
    if not np.array_equal(material, np.load(initial)):
        raise RuntimeError('Retained control does not reproduce the exact accepted material')
    if (digest(Path(plan['data_root'])/'index.json') != plan['data_index_SHA256'] or
        digest(Path(plan['high_index'])) != plan['high_index_SHA256']):
        raise RuntimeError('Authorized data changed')
    for row in plan['reference_RF']:
        if digest(row['path']) != row['SHA256']:
            raise RuntimeError('Accepted original44 cache changed')
    source = Path(__file__).parent
    physical_names = ['native_time.py', 'native_time.cu', 'native_tangent.py', 'native_tangent.cu',
        'frequency_multilevel_acquisition.py', 'full90_3mhz_acquisition.py', 'matrix_free_rf_response.py',
        'independent_physical_coordinates.py', 'depth_rms_preconditioner.py', 'low_frequency_0p5.py']
    for name in physical_names:
        if digest(source/name) != prior['source_SHA256'][name]:
            raise RuntimeError('Previously qualified physical implementation changed: '+name)
    new_names = [Path(__file__).name, 'all44_joint_tangent_pool.py',
        'all44_joint_tangent_worker.py', 'all44_joint_subspace.py', 'pressure_resources.py',
        'native_full_source_dual.py']
    hashes = {name: digest(source/name) for name in physical_names+new_names}
    reserve = 2*2**30+400*2**20
    if shutil.disk_usage(project).free < reserve+2*2**20 or shutil.disk_usage(Path.home()).free < reserve+600*2**20:
        raise RuntimeError('Retain the same disk reserves')
    z, x = np.ogrid[:961, :1121]
    active = (z >= 80)&(z < 937)&(x >= 24)&(x < 1097)
    lateral = abs((np.arange(1121)-560)*.05) <= 19.1
    caps = np.array([.004, .006, .05])[:, None, None]/SCALES
    previous = json.loads((args.parent_root/'steps/step01.json').read_text())
    fraction = previous['initial_fraction_from_objective_and_slope']
    previous_delta = np.load(args.parent_root/'candidates/step01/control.npy')-control
    ordinary = previous_delta/fraction
    directions = np.zeros((4, 3, 961, 1121), np.float64)
    for j in range(3):
        directions[j, j] = .5*ordinary[j]
    high_gradient = np.load(args.parent_root/'workers_full/pool01/rank00/tasks/token0002/lf3_04/actual_control_gradient.npy')
    extra, _ = depth_rms_direction(high_gradient, active, lateral, caps)
    for j in range(3):
        base = directions[j, j]
        norm = np.sum(base*base)
        if norm > 0:
            extra[j] -= base*(np.sum(extra[j]*base)/norm)
    extra *= active[None]
    relative = np.max(abs(extra)/caps)
    if relative <= 1e-16:
        raise RuntimeError('The fourth retained spatial direction is not independent')
    directions[3] = .5*extra/relative
    if np.max(np.sum(abs(directions), axis=0)/caps) > 1.00001:
        raise RuntimeError('Coefficient box would exceed independent physical step caps')
    physical = np.empty_like(directions, dtype=np.float32)
    edge = np.broadcast_to(np.load(edge_path), (961, 1121))
    for j, direction in enumerate(directions):
        physical[j] = np.stack([.05*material[0]*direction[0], .10*material[1]*direction[1],
                               .70*(material[2]-edge)*direction[2]])
    total_reserved = old_passes+44*len(directions)
    if total_reserved > 136*4:
        raise RuntimeError('All actual inverse products must fit the same original budget')
    args.out.mkdir()
    args.artifact_root.mkdir()
    for name in ['arrays', 'workers', 'candidate', 'final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name, target_is_directory=True)
    (args.out/'jobs').mkdir()
    for name in hashes:
        shutil.copy2(source/name, args.out/name)
    array(args.out/'arrays/control_directions.npy', directions)
    array(args.out/'arrays/physical_directions.npy', physical)
    atomic(args.out/'native_plan.json', plan)
    started = time.time()
    run = {'status': 'initializing_all44_joint_tangent_update', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': deadline,
        'parent_root': str(args.parent_root), 'artifact_root': str(args.artifact_root),
        'sample': SAMPLE, 'case': CASE, 'source_SHA256': hashes,
        'initial_material_SHA256': digest(initial), 'native_anchor_material_SHA256': digest(anchor),
        'closed_predecessor_budget': prior['closed_predecessor_budget'],
        'full_dual_campaign_work_limit': 136, 'inherited_reserved_units': old_passes/4,
        'charged_new_tangent_equivalent_units': 44*len(directions)/4,
        'reserved_campaign_units': total_reserved/4,
        'reservation_is_conservative_not_measured_actual_work': True,
        'same_budget_and_deadline_not_reset': True,
        'inherited_actual_full_dual_work': prior['actual_new_native_work'],
        'inherited_new_original44_checks': 1, 'maximum_remaining_original44_checks': 1,
        'original44_checks_issued': 0, 'original44_checks_completed': 0,
        'actual_all44_tangent_directions': len(directions),
        'maximum_new_tangent_propagations': 44*len(directions),
        'maximum_new_background_forward_passes_for_JVPs': 44*len(directions),
        'maximum_new_candidate_forward_passes': 44,
        'one_GPU_experiment': True, 'devices': args.devices, 'allocator_cap_GiB': args.cap_gib,
        'per_card_process_tree_limit_GiB': 16, 'independent_three_native_fields': True,
        'native_shape': [3, 961, 1121], 'source_encoding': False,
        'all_eleven_sources_per_family_in_actual_joint_subproblem': True,
        'not_a_full_native_Jacobian_convergence_claim': True,
        'no_new_patient_derivative_probes': True, 'all_JVPs_used_in_actual_joint_update': True,
        'patient_truth_read_by_optimizer': False, 'reserved_case_unread': 'Neg_47_Left',
        'new_data_or_frequency': False, 'joint_reconstruction_success_claimed': False}
    monitor = PressureResources(args.out)
    pool = None
    def checkpoint():
        if pool is not None:
            run['actual_native_work'] = pool.actual_work()
        atomic(args.out/'run.json', run)
    def freeze():
        if any(digest(source/name) != sha for name, sha in hashes.items()):
            raise RuntimeError('Frozen executable changed during the actual inverse update')
    try:
        checkpoint()
        freeze()
        pool = All44JointPool(args, monitor, plan, deadline)
        run['status'] = 'actual_all44_joint_tangent_inverse_products'
        checkpoint()
        result = pool.evaluate(material, 'joint_tangents', args.out/'arrays/physical_directions.npy')
        initial_RF = prior['final_actual_native_RF_loss']
        if abs(result['loss']-initial_RF) > 1e-10:
            raise RuntimeError('Same-material cached full objective changed')
        H, linear = np.asarray(result['H']), np.asarray(result['linear'])
        tv, gt = coordinates.regularization(control, .001)
        tv_linear = np.einsum('aijk,ijk->a', directions, gt)
        coefficients, details = box_quadratic(H, linear+tv_linear)
        q = .5*float(coefficients@H@coefficients)
        trust = min(1., np.sqrt(.25*initial_RF/max(q, 1e-30)))
        coefficients *= trust
        change = np.einsum('a,aijk->ijk', coefficients, directions)
        proposal = coordinates.project(control+change, active)
        actual_change = proposal-control
        if np.max(abs(actual_change)/caps) > 1.00001:
            raise RuntimeError('Native joint proposal exceeded the existing physical caps')
        if np.linalg.norm(actual_change-change) > 1e-8*max(np.linalg.norm(change), 1e-30):
            raise RuntimeError('Projection altered the measured tangent combination; retain products for review')
        predicted = float((linear+tv_linear)@coefficients+.5*coefficients@H@coefficients)
        if predicted >= 0 or not np.isfinite(predicted):
            raise RuntimeError('No descending actual joint candidate in the retained spatial span')
        old_coeff = np.array([2*fraction]*3+[0.])
        run.update(status='actual_all44_joint_candidate_original_RF', coefficients=coefficients.tolist(),
            direction_coefficient_QP=details, linearized_RF_trust_fraction=trust,
            predicted_RF_TV_change=predicted,
            old_rejected_step_full44_linear_change=float(linear@old_coeff),
            old_rejected_step_full44_GN_change=float(linear@old_coeff+.5*old_coeff@H@old_coeff),
            initial_RF=initial_RF, initial_TV=tv, original44_checks_issued=1)
        array(args.out/'candidate/control.npy', proposal)
        array(args.out/'candidate/material.npy', coordinates.material(proposal))
        checkpoint()
        freeze()
        actual = pool.evaluate(coordinates.material(proposal), 'original44_forward', args.out/'arrays/physical_directions.npy')
        run['original44_checks_completed'] = 1
        proposed_tv = coordinates.regularization(proposal, .001)[0]
        accepted = bool(actual['loss'] < initial_RF and actual['loss']+proposed_tv < initial_RF+tv)
        atomic(args.out/'candidate/actual_native_acceptance.json', {'accepted': accepted,
            'RF_before': initial_RF, 'RF_after': actual['loss'], 'TV_before': tv, 'TV_after': proposed_tv,
            'material_SHA256': digest(args.out/'candidate/material.npy'),
            'full44_actual_candidate': True, 'joint_reconstruction_success_claimed': False})
        pool.close()
        pool = None
        final = proposal if accepted else control
        array(args.out/'final/control.npy', final)
        array(args.out/'final/material.npy', coordinates.material(final))
        for folder, label in [(args.out/'candidate', 'All-44 joint update'), (args.out/'final', 'Retained physical solution')]:
            (folder/'stage0_initial_material.npy').symlink_to(initial.resolve())
            with (folder/'maps.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material', str(folder/'material.npy'),
                    '--sample', SAMPLE, '--out', str(folder/'maps'), '--estimate-label', label],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=120)
        run.update(status='all44_joint_tangent_stage_finished', accepted_original_RF=accepted,
            candidate_RF=actual['loss'], final_RF=actual['loss'] if accepted else initial_RF,
            candidate_material_SHA256=digest(args.out/'candidate/material.npy'),
            material_SHA256=digest(args.out/'final/material.npy'),
            seconds=time.time()-started, stop_reason='one_budgeted_all44_joint_candidate_finished')
        checkpoint()
        print(json.dumps({k: run[k] for k in ['status', 'candidate_RF', 'final_RF',
            'accepted_original_RF', 'candidate_material_SHA256', 'reserved_campaign_units', 'seconds']}), flush=True)
    except BaseException:
        run.update(status='all44_joint_tangent_failed_for_inspection', error=traceback.format_exc(),
            issued_work_remains_charged=True, seconds=time.time()-started)
        checkpoint()
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__ == '__main__':
    main()
