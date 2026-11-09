"""One fixed-penalty warm/recycled full44 conditional material continuation."""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable THP')
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
import numpy as np
from .conditional_material_quadratic import ConditionalMaterialQuadratic, native_body
from .independent_physical_coordinates import IndependentPhysicalCoordinates, SCALES
from .multi_equation_continuation_pool import ContinuationPool
from .pressure_resources import PressureResources
from .low_frequency_0p5 import atomic, digest, SAMPLE


def array(path, value):
    temporary = path.with_name(path.stem+'.partial.npy')
    np.save(temporary, value)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--devices', type=int, nargs='+', default=[0, 1])
    p.add_argument('--max-seconds', type=float, default=14400.)
    a = p.parse_args()
    project, source = Path.cwd().resolve(), Path(__file__).parent
    for name in ['parent_root', 'out', 'artifact_root']:
        setattr(a, name, getattr(a, name).resolve())
    if (not a.parent_root.is_relative_to(project/'out/v5') or
            not a.out.is_relative_to(project/'out/v5') or a.out.exists() or
            not a.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or a.artifact_root.exists() or
            a.devices != [0, 1] or a.max_seconds != 14400):
        raise ValueError('Fresh one-outer-step/full44/12totalK/14400s continuation only')
    old = json.loads((a.parent_root/'run.json').read_text())
    previous_workers = json.loads((a.parent_root/'worker_processes.json').read_text())
    allowed = ['actual_multiequation_conditional_stage_finished',
               'actual_recycled_conditional_continuation_stage_finished']
    if (old['status'] not in allowed or not old.get('material_finite_update_generated') or
            not old.get('closed_at_actual_counts') or not old.get('unused_limits_not_reopened')):
        raise RuntimeError('Inspect a completed finite parent trajectory before continuation')
    current_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    identities = {old['pid']: old['start_ticks']}
    identities.update({r['pid']: r['start_ticks'] for r in previous_workers['worker_identities']})
    for pid, ticks in identities.items():
        path = Path('/proc', str(pid), 'stat')
        if old['boot_id'] == current_boot and path.exists():
            now = int(path.read_text().rsplit(')', 1)[1].split()[19])
            if now == ticks:
                raise RuntimeError('The actual parent process is still alive;never overlap stages')
    parent_work = json.loads((a.parent_root/'actual_native_work.json').read_text())
    if parent_work.get('unfinished_native_phases', 0):
        raise RuntimeError('Parent work is partial;do not continue or replay it')
    for kind in ['adjoint_direction_passes', 'auxiliary_forward_passes',
                 'physical_forward_passes', 'tangent_direction_passes']:
        if parent_work.get('issued_'+kind, 0) != parent_work.get('completed_'+kind, 0):
            raise RuntimeError('Parent issued and completed work differ;inspect the original stage')
    op = json.loads((project/'out/v5/native_multi_equation_qualification_v1/result.json').read_text())
    cpu = json.loads((project/'out/v5/conditional_material_quadratic_qualification_v1/result.json').read_text())
    warm_cpu = json.loads((project/'out/v5/warm_recycled_receiver_CPU_qualification_v1/result.json').read_text())
    if (op['status'] != 'synthetic_four_channel_operator_and_centered_statistics_passed' or
            cpu['status'] != 'independent_synthetic_conditional_known_optimum_passed' or
            warm_cpu['status'] != 'independent_warm_currentK_fixed_penalty_CPU_control_path_passed'):
        raise RuntimeError('Use the existing once-only operator/material/control-path qualifications')
    hashes = {**old['source_SHA256'], **op['source_SHA256'], **cpu['source_SHA256'],
              **warm_cpu['source_SHA256']}
    for name in ['multi_equation_continuation_source.py', 'multi_equation_continuation_worker.py',
            'multi_equation_continuation_pool.py', Path(__file__).name]:
        hashes[name] = digest(source/name)
    for name, sha in hashes.items():
        if digest(source/name) != sha:
            raise RuntimeError('An immutable qualified or original source changed: '+name)
    initial_material, initial_control = a.parent_root/'trajectory_final/material.npy', a.parent_root/'trajectory_final/control.npy'
    acceptance = json.loads((a.parent_root/'trajectory_final/actual_native_acceptance.json').read_text())
    initial_RF = float(old['trajectory_full_RF'])
    initial_SHA = old['trajectory_material_SHA256']
    if (digest(initial_material) != initial_SHA or acceptance['material_SHA256'] != initial_SHA or
            acceptance['actual_native_RF']['loss'] != initial_RF):
        raise RuntimeError('Use the parent trajectory and its own all44 physical RF,not another incumbent')
    plan = json.loads((a.parent_root/'native_plan.json').read_text())
    parent_entries = [tuple(v) for v in plan['entries']]
    required_families = ['lf', 'point3', 'lf3', 'point3_3mhz']
    if set(plan['families']) != set(required_families) or len(parent_entries) != 44:
        raise RuntimeError('The same four complete acquisition families are required')
    entries = [(name, shot) for pair in [('lf', 'point3'), ('lf3', 'point3_3mhz')]
               for shot in range(11) for name in pair]
    if sorted(parent_entries) != sorted(entries):
        raise RuntimeError('The parent rank/source map must cover the same original44 sources')
    parent_fields = json.loads((a.parent_root/'actual_wavefield_summary.json').read_text())
    parent_rows = parent_fields['rows']
    if sorted((r['family'], r['shot']) for r in parent_rows) != sorted(entries):
        raise RuntimeError('Parent source fields are incomplete;do not fill them by replay')
    parent_rank = {v: i % 2 for i, v in enumerate(parent_entries)}
    warm_records = []
    for row in parent_rows:
        name, shot = row['family'], row['shot']
        dual_path = a.parent_root/'workers'/f'rank{parent_rank[(name,shot)]:02d}'/'source_fields'/f'{name}_{shot:02d}'/'dual_state.pt'
        penalty = float(row['lambda_squared'])
        if not dual_path.exists() or not np.isfinite(penalty) or penalty <= 0:
            raise RuntimeError('Completed source dual and unchanged positive penalty required')
        warm_records.append({'family': name, 'shot': shot,
            'dual_state_path': str(dual_path.resolve()), 'dual_state_SHA256': digest(dual_path),
            'lambda_squared': penalty, 'parent_completed_inner_actions': row['completed_inner_actions'],
            'parent_operator_material_SHA256': old['initial_material_SHA256'],
            'never_reuse_old_K_images_across_materials': True})
    previous_P = float(parent_fields['joint_wavefield_objective_before_material'])
    previous_deltaQ = float(old['conditional_change'])
    previous_reference = previous_P+previous_deltaQ
    if not np.isfinite([previous_P, previous_deltaQ, previous_reference]).all() or previous_reference <= 0:
        raise RuntimeError('Finite positive fixed-field parent objective reference required')
    refs = acceptance['RF_source_records']
    if len(entries) != 44 or sorted((r['family'], r['shot']) for r in refs) != sorted(entries):
        raise RuntimeError('All44 original source baseline caches required')
    for row in refs:
        if digest(row['path']) != row['SHA256']:
            raise RuntimeError('A completed same-point physical RF prediction changed')
    edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    expected_anchor_SHA = old.get('native_anchor_material_SHA256',
        'd9ce699c4d52a872b2b9f86f34875baf89e2c46a2f09ecc77f67758a8baa0754')
    for path, sha in [(edge_path, plan['known_edge_SHA256']),
            (Path(plan['native_anchor_material']), expected_anchor_SHA),
            (Path(plan['data_root'])/'index.json', plan['data_index_SHA256']),
            (Path(plan['high_index']), plan['high_index_SHA256'])]:
        if digest(path) != sha:
            raise RuntimeError('Original acquisition or known absorption changed')
    index = json.loads((Path(plan['data_root'])/'index.json').read_text())
    eta = index['absorption']['eta_per_unit_alpha']
    high = json.loads(Path(plan['high_index']).read_text())
    if high['absorption'] != index['absorption']:
        raise RuntimeError('The same frozen two-pole material model is required')
    reserve = 2*2**30+400*2**20
    if shutil.disk_usage(project).free < reserve+3*2**20 or shutil.disk_usage(Path.home()).free < reserve+2**30:
        raise RuntimeError('Retain old reserves plus1GiB HOME allowance')
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
        if l.startswith('MemAvailable:')))/2**20
    bank = ((36001+127)//128)*7*961*1121*4/2**30
    if available < 2*bank*len(a.devices)+24:
        raise RuntimeError('Preserve checkpoint banks and24GiB host headroom')
    a.out.mkdir()
    a.artifact_root.mkdir()
    for name in ['arrays', 'workers', 'trajectory_final', 'final']:
        (a.artifact_root/name).mkdir()
        (a.out/name).symlink_to(a.artifact_root/name, target_is_directory=True)
    for name in hashes:
        shutil.copy2(source/name, a.out/name)
    for name, path in [('initial_material', initial_material), ('initial_control', initial_control)]:
        (a.out/'arrays'/f'{name}.npy').symlink_to(path.resolve())
    plan.update(entries=entries, initial_RF_source_records=refs, source_SHA256=hashes,
        initial_material_SHA256=initial_SHA, initial_RF=initial_RF,
        native_anchor_material_SHA256=expected_anchor_SHA,
        new_method='fixed_penalty_warm_recycled_four_channel_conditional_continuation',
        conditional_node_support={'z': [80, 936], 'x': [25, 1096]},
        requested_FOV_unchanged=True, per_source_CG_maximum=12,
        source_penalty_policy='unchanged_completed_parent_lambda_squared',
        warm_source_records=warm_records, previous_wavefield_primal=previous_P,
        previous_conditional_change=previous_deltaQ,
        previous_fixed_field_reference=previous_reference,
        complete_family_per_worker=True, coarse_space_maximum=16)
    plan.pop('gamma', None)
    import hashlib
    operator_keys = {}
    for name, spec in plan['families'].items():
        signature = {'material': initial_SHA, 'family': name, 'spec': spec,
            'data_index': plan['data_index_SHA256'], 'high_index': plan['high_index_SHA256'],
            'node_support': plan['conditional_node_support'], 'reference_speed': 1540.,
            'native_source': hashes['native_multi_equation.cu'],
            'native_wrapper': hashes['native_multi_equation.py'],
            'fixed_metric': 'pressure,momentum_x,momentum_z,common_memory;Pa'}
        operator_keys[name] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    plan['receiver_operator_keys'] = operator_keys
    atomic(a.out/'native_plan.json', plan)
    started = time.time()
    deadline = started+14400
    run = {'status': 'initializing_recycled_conditional_continuation_stage', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': deadline, 'statistics_deadline_unix_s': deadline-1200,
        'max_seconds': 14400, 'parent_root': str(a.parent_root), 'source_SHA256': hashes,
        'initial_RF': initial_RF, 'initial_material_SHA256': initial_SHA,
        'native_anchor_material_SHA256': expected_anchor_SHA,
        'initial_material_from_parent_trajectory': True,
        'same_source_penalty_across_material_steps': True,
        'necessary_current_K_on_old_y_included_in_12_actions': True,
        'coarse_space_maximum': 16, 'complete_family_per_worker': True,
        'parent_wavefield_primal': previous_P,
        'parent_conditional_change': previous_deltaQ,
        'parent_fixed_field_reference': previous_reference,
        'fixed_field_reference_not_a_rigorous_roundoff_certificate': True,
        'parent_actual_native_work_retained': parent_work,
        'parent_actual_native_work_SHA256': digest(a.parent_root/'actual_native_work.json'),
        'parent_wavefield_summary_SHA256': digest(a.parent_root/'actual_wavefield_summary.json'),
        'parent_native_plan_SHA256': digest(a.parent_root/'native_plan.json'),
        'new_control_CPU_qualification_only_not_patient_acceleration_proof': True,
        'maximum_outer_material_updates': 1, 'maximum_inner_actions_per_source': 12,
        'maximum_new_state_adjoint_passes_including_replays': 1144,
        'maximum_new_auxiliary_forward_passes': 572, 'maximum_new_physical_forward_passes': 44,
        'new_material_acoustic_gradient_evaluations': 0, 'new_patient_JVPs': 0,
        'previous_cumulative_reserved_units': 432, 'directional_equivalents_not_defined_for_this_method': True,
        'not_whole_thread_actual_cost': True, 'all_old_budgets_closed_and_unchanged': True,
        'source_count': 44, 'completed_wavefield_sources': 0, 'source_encoding': False,
        'completed_conditional_material_updates': 0, 'physical_candidate_checks': 0,
        'patient_truth_read_by_optimizer': False, 'intermediate_truth_feedback': False,
        'reserved_case_unread': 'Neg_47_Left', 'new_data_or_frequency': False,
        'requested_FOV_mm': [1, 43], 'independent_native_fields': True, 'material_lowpass': False,
        'devices': a.devices, 'one_GPU_experiment': True,
        'per_card_process_tree_limit_GiB': 16, 'allocator_cap_GiB': 10,
        'joint_reconstruction_success_claimed': False}
    atomic(a.out/'run.json', run)
    def stop(number, frame):
        (a.out/'stop_requested.flag').write_text(str(number)+'\n')
    for number in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(number, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    monitor, pool = PressureResources(a.out), None
    def checkpoint():
        if pool is not None:
            run['actual_native_work'] = pool.actual_work()
        atomic(a.out/'run.json', run)
    def freeze():
        if any(digest(source/name) != sha for name, sha in hashes.items()):
            raise RuntimeError('A frozen actual/qualified source changed')
        if time.time() >= deadline-60 or (a.out/'stop_requested.flag').exists():
            raise RuntimeError('Original absolute multiequation deadline or cooperative stop reached')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Original disk reserves reached')
    try:
        initial = np.load(initial_material)
        pool = ContinuationPool(a, monitor, deadline, freeze)
        ready_RF = sum(r['initial_RF'] for r in pool.ready)
        if abs(ready_RF-initial_RF) > 1e-10:
            raise RuntimeError('Same-point all44 cached objective changed;no wavefield actions should be replayed')
        run.update(status='actual_full44_four_channel_wavefield_solve', cached_baseline_RF=ready_RF,
            same_point_cache_objective_difference=ready_RF-initial_RF)
        checkpoint()
        rows = pool.evaluate('conditional_statistics', initial)
        statistics = np.zeros((8, 961, 1121), np.float64)
        sources = []
        for row in rows:
            if digest(row['statistics_path']) != row['statistics_SHA256']:
                raise RuntimeError('Completed allsource conditional aggregate changed')
            statistics += np.load(row['statistics_path'])
            sources.extend(row['rows'])
        if sorted((r['family'], r['shot']) for r in sources) != sorted(entries):
            raise RuntimeError('Do not fit materials from an incomplete source set')
        array(a.out/'arrays/all44_centered_statistics.npy', statistics)
        penalty = sum(r['weighted_source_penalty'] for r in sources)
        summary = {'source_count': 44, 'rows': sources,
            'all_inner_converged': all(r['actual_inner_converged'] for r in sources),
            'maximum_actual_relative_normal_residual': max(r['actual_relative_normal_residual'] for r in sources),
            'maximum_relative_primal_dual_gap': max(r['relative_primal_dual_gap'] for r in sources),
            'total_augmented_RF': sum(r['augmented_RF'] for r in sources),
            'total_weighted_source_penalty': penalty,
            'joint_wavefield_objective_before_material': sum(r['wavefield_primal'] for r in sources),
            'joint_wavefield_dual': sum(r['wavefield_dual'] for r in sources),
            'parent_fixed_field_reference': previous_reference,
            'same_parent_source_penalties': True}
        current_P = summary['joint_wavefield_objective_before_material']
        current_D = summary['joint_wavefield_dual']
        summary.update(current_primal_minus_parent_reference=current_P-previous_reference,
            current_dual_minus_parent_reference=current_D-previous_reference,
            relative_aggregate_gap=max(0.,current_P-current_D)/max(abs(current_P),abs(current_D),1e-30),
            comparison_is_not_material_imaging_or_rigorous_roundoff_certificate=True)
        atomic(a.out/'actual_wavefield_summary.json', summary)
        run.update(status='CPU_finite_conditional_material_estimation', completed_wavefield_sources=44,
            all_wavefield_inner_converged=summary['all_inner_converged'],
            maximum_actual_inner_relative_residual=summary['maximum_actual_relative_normal_residual'],
            total_augmented_RF=summary['total_augmented_RF'], total_source_penalty=penalty,
            total_wavefield_primal=current_P, total_wavefield_dual=current_D,
            current_primal_minus_parent_reference=current_P-previous_reference,
            current_dual_minus_parent_reference=current_D-previous_reference)
        checkpoint()
        freeze()
        edge = np.broadcast_to(np.load(edge_path), (961, 1121)).astype(np.float64)
        solver = ConditionalMaterialQuadratic(initial, edge, statistics, native_body(), eta,
            normalization=max(initial_RF/2, penalty, 1e-12))
        candidate, details = solver.optimize(max_iterations=250, max_seconds=min(600, deadline-time.time()-420))
        if details['conditional_change'] < -penalty-1e-4*max(penalty, 1e-30):
            raise RuntimeError('Conditional energy fell below its actual source-energy lower bound')
        details.update(joint_extended_objective_after_material=current_P+details['conditional_change'],
            parent_fixed_field_reference=previous_reference,
            joint_extended_change_from_parent_reference=current_P+details['conditional_change']-previous_reference,
            same_unchanged_source_penalties=True,
            not_original_physical_RF_objective=True)
        atomic(a.out/'conditional_material_result.json', details)
        del statistics, solver
        freeze()
        changed = details['conditional_change'] < 0 and not np.array_equal(candidate, initial)
        trajectory = a.out/'trajectory_final'
        if changed:
            array(trajectory/'material.npy', candidate)
            anchor = np.load(plan['native_anchor_material'])
            coordinates = IndependentPhysicalCoordinates(anchor, edge)
            control = np.stack([np.log(candidate[0].astype(np.float64)/anchor[0])/.05,
                np.log(candidate[1].astype(np.float64)/anchor[1])/.10,
                np.log(np.maximum(candidate[2].astype(np.float64)-edge, 1e-30)/
                    np.maximum(anchor[2].astype(np.float64)-edge, 1e-30))/.70])
            if not np.array_equal(coordinates.material(control), candidate):
                raise RuntimeError('Finite conditional material/control provenance disagrees')
            array(trajectory/'control.npy', control)
            run.update(status='actual_original_full44_physical_conditional_candidate',
                completed_conditional_material_updates=1, physical_candidate_checks=1,
                conditional_change=details['conditional_change'], candidate_material_SHA256=digest(trajectory/'material.npy'))
            checkpoint()
            scores = pool.evaluate('physical_forward', candidate)
            candidate_RF = sum(r['loss'] for r in scores)
            candidate_refs = [v for r in scores for v in r['RF_source_records']]
        else:
            (trajectory/'material.npy').symlink_to(initial_material.resolve())
            (trajectory/'control.npy').symlink_to(initial_control.resolve())
            candidate_RF, candidate_refs = initial_RF, refs
        worker_ids = [p.pid for p in pool.workers]
        pool.close()
        run['actual_native_work'] = pool.actual_work()
        pool = None
        if any(Path('/proc', str(pid)).exists() for pid in worker_ids):
            raise RuntimeError('Owned wavefield workers must be gone before any truth evaluation')
        trajectory_sha = digest(trajectory/'material.npy')
        improved = candidate_RF < old['final_RF']-8*np.finfo(np.float32).eps*old['final_RF']
        atomic(trajectory/'actual_native_acceptance.json', {'actual_native_RF': {'loss': candidate_RF, 'all44_sources': True},
            'RF_source_records': candidate_refs, 'material_SHA256': trajectory_sha,
            'conditional_objective_decreased': changed, 'physical_RF_improved': bool(improved),
            'allowed_nonmonotone_physical_RF_trajectory': True, 'joint_imaging_success_claimed': False})
        final = a.out/'final'
        best = trajectory if improved else a.parent_root/'final'
        for name in ['material.npy', 'control.npy']:
            (final/name).symlink_to((best/name).resolve())
        if improved:
            shutil.copy2(trajectory/'actual_native_acceptance.json', final/'actual_native_acceptance.json')
        else:
            shutil.copy2(a.parent_root/'final/actual_native_acceptance.json', final/'actual_native_acceptance.json')
        run.update(status='conditional_trajectory_and_global_incumbent_fixed_before_posthoc',
            trajectory_full_RF=candidate_RF, trajectory_material_SHA256=trajectory_sha,
            final_RF=candidate_RF if improved else old['final_RF'], material_SHA256=digest(final/'material.npy'),
            global_full_RF_improved=bool(improved), material_finite_update_generated=bool(changed))
        checkpoint()
        if changed:
            (trajectory/'stage0_initial_material.npy').symlink_to(initial_material.resolve())
            with (trajectory/'maps.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material', str(trajectory/'material.npy'),
                    '--sample', SAMPLE, '--out', str(trajectory/'maps'), '--estimate-label', 'Fixed penalty continuation: conditional material'],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
            with (trajectory/'increment.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'nsrno.posthoc_fixed_material_increment', '--candidate', str(trajectory),
                    '--initial-material', str(initial_material), '--initial-control', str(initial_control)],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        else:
            for suffix in ['png', 'metrics.json']:
                shutil.copy2(a.parent_root/f'trajectory_final/maps.{suffix}', trajectory/f'maps.{suffix}')
        for suffix in ['png', 'metrics.json']:
            shutil.copy2(best/f'maps.{suffix}', final/f'maps.{suffix}')
        atomic(a.out/'maps_provenance.json', {'trajectory_new_fixed_material_evaluation': bool(changed),
            'final_same_as_trajectory': bool(improved), 'otherwise_final_copied_historical_incumbent': not improved,
            'patient_truth_never_used_in_candidate_selection': True})
        run.update(status='actual_recycled_conditional_continuation_stage_finished', finished_unix_s=time.time(),
            joint_extended_objective_after_material=details['joint_extended_objective_after_material'],
            joint_extended_change_from_parent_reference=details['joint_extended_change_from_parent_reference'],
            seconds=time.time()-started, CPU_maximum_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            closed_at_actual_counts=True, unused_limits_not_reopened=True)
        checkpoint()
        print(json.dumps({k: run[k] for k in ['status', 'trajectory_full_RF', 'final_RF', 'material_SHA256',
            'global_full_RF_improved', 'material_finite_update_generated', 'all_wavefield_inner_converged', 'seconds']}), flush=True)
    except BaseException:
        run.update(status='recycled_conditional_continuation_failed_for_inspection', error=traceback.format_exc(),
            seconds=time.time()-started, all_issued_work_retained=True)
        checkpoint()
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__ == '__main__':
    main()
