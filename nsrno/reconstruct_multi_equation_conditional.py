"""One full44 multi-equation wavefield/conditional-material reconstruction."""
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
from .multi_equation_pool import MultiEquationPool
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
    if (a.parent_root != project/'out/v5/reconstruction_anatomical_score_v1' or
            not a.out.is_relative_to(project/'out/v5') or a.out.exists() or
            not a.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or a.artifact_root.exists() or
            a.devices != [0, 1] or a.max_seconds != 14400):
        raise ValueError('Fresh one-outer-step/full44/12CG/14400s actual experiment only')
    old = json.loads((a.parent_root/'run.json').read_text())
    previous_workers = json.loads((a.parent_root/'worker_processes.json').read_text())
    if (old['status'] != 'matched_score_actual_stage_finished' or
            old['material_SHA256'] != '8f957125798f2d1d7ca5b527d6003f0992604cb30a2449bf58e46611eac07983' or
            any(Path('/proc', str(pid)).exists() for pid in [old['pid'], *previous_workers['worker_pids']])):
        raise RuntimeError('The actual fullRF-selected predecessor must remain terminal')
    op = json.loads((project/'out/v5/native_multi_equation_qualification_v1/result.json').read_text())
    cpu = json.loads((project/'out/v5/conditional_material_quadratic_qualification_v1/result.json').read_text())
    oq = json.loads((project/'out/v5/native_multi_equation_qualification_v1/run.json').read_text())
    if (op['status'] != 'synthetic_four_channel_operator_and_centered_statistics_passed' or
            cpu['status'] != 'independent_synthetic_conditional_known_optimum_passed' or
            Path('/proc', str(oq['pid'])).exists()):
        raise RuntimeError('Both new implementations must be once qualified,finished and gone')
    hashes = {**old['source_SHA256'], **op['source_SHA256'], **cpu['source_SHA256']}
    for name in ['multi_equation_source_solve.py', 'multi_equation_worker.py', 'multi_equation_pool.py',
            Path(__file__).name]:
        hashes[name] = digest(source/name)
    for name, sha in hashes.items():
        if digest(source/name) != sha:
            raise RuntimeError('An immutable qualified or original source changed: '+name)
    initial_material, initial_control = a.parent_root/'final/material.npy', a.parent_root/'final/control.npy'
    acceptance = json.loads((a.parent_root/'final/actual_native_acceptance.json').read_text())
    if digest(initial_material) != old['material_SHA256'] or acceptance['actual_native_RF']['loss'] != old['final_RF']:
        raise RuntimeError('The complete same-point physical incumbent changed')
    plan = json.loads((a.parent_root/'native_plan.json').read_text())
    entries = [(name, shot) for name in plan['families'] for shot in range(11)]
    refs = acceptance['RF_source_records']
    if len(entries) != 44 or sorted((r['family'], r['shot']) for r in refs) != sorted(entries):
        raise RuntimeError('All44 original source baseline caches required')
    for row in refs:
        if digest(row['path']) != row['SHA256']:
            raise RuntimeError('A completed same-point physical RF prediction changed')
    edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    coordinate_receipt = json.loads((Path(old['parent_root'])/'run.json').read_text())
    for path, sha in [(edge_path, plan['known_edge_SHA256']),
            (Path(plan['native_anchor_material']), coordinate_receipt['native_anchor_material_SHA256']),
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
        initial_material_SHA256=old['material_SHA256'], initial_RF=old['final_RF'],
        new_method='four_residual_channels_and_centered_conditional_materials',
        conditional_node_support={'z': [80, 936], 'x': [25, 1096]},
        requested_FOV_unchanged=True, per_source_CG_maximum=12, gamma=.1)
    atomic(a.out/'native_plan.json', plan)
    started = time.time()
    deadline = started+14400
    run = {'status': 'initializing_actual_multiequation_conditional_stage', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': deadline, 'statistics_deadline_unix_s': deadline-1200,
        'max_seconds': 14400, 'parent_root': str(a.parent_root), 'source_SHA256': hashes,
        'initial_RF': old['final_RF'], 'initial_material_SHA256': old['material_SHA256'],
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
        pool = MultiEquationPool(a, monitor, deadline, freeze)
        ready_RF = sum(r['initial_RF'] for r in pool.ready)
        if abs(ready_RF-old['final_RF']) > 1e-10:
            raise RuntimeError('Same-point all44 cached objective changed;no wavefield actions should be replayed')
        run.update(status='actual_full44_four_channel_wavefield_solve', cached_baseline_RF=ready_RF,
            same_point_cache_objective_difference=ready_RF-old['final_RF'])
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
            'joint_wavefield_objective_before_material': sum(r['wavefield_primal'] for r in sources)}
        atomic(a.out/'actual_wavefield_summary.json', summary)
        run.update(status='CPU_finite_conditional_material_estimation', completed_wavefield_sources=44,
            all_wavefield_inner_converged=summary['all_inner_converged'],
            maximum_actual_inner_relative_residual=summary['maximum_actual_relative_normal_residual'],
            total_augmented_RF=summary['total_augmented_RF'], total_source_penalty=penalty)
        checkpoint()
        freeze()
        edge = np.broadcast_to(np.load(edge_path), (961, 1121)).astype(np.float64)
        solver = ConditionalMaterialQuadratic(initial, edge, statistics, native_body(), eta,
            normalization=max(old['final_RF']/2, penalty, 1e-12))
        candidate, details = solver.optimize(max_iterations=250, max_seconds=min(600, deadline-time.time()-420))
        if details['conditional_change'] < -penalty-1e-4*max(penalty, 1e-30):
            raise RuntimeError('Conditional energy fell below its actual source-energy lower bound')
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
            candidate_RF, candidate_refs = old['final_RF'], refs
        worker_ids = [p.pid for p in pool.workers]
        pool.close()
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
                    '--sample', SAMPLE, '--out', str(trajectory/'maps'), '--estimate-label', 'Four residual channels: conditional material'],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
            with (trajectory/'increment.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'nsrno.posthoc_fixed_material_increment', '--candidate', str(trajectory),
                    '--initial-material', str(initial_material), '--initial-control', str(initial_control)],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        else:
            for suffix in ['png', 'metrics.json']:
                shutil.copy2(a.parent_root/f'final/maps.{suffix}', trajectory/f'maps.{suffix}')
        for suffix in ['png', 'metrics.json']:
            shutil.copy2(best/f'maps.{suffix}', final/f'maps.{suffix}')
        atomic(a.out/'maps_provenance.json', {'trajectory_new_fixed_material_evaluation': bool(changed),
            'final_same_as_trajectory': bool(improved), 'otherwise_final_copied_historical_incumbent': not improved,
            'patient_truth_never_used_in_candidate_selection': True})
        run.update(status='actual_multiequation_conditional_stage_finished', finished_unix_s=time.time(),
            seconds=time.time()-started, CPU_maximum_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            closed_at_actual_counts=True, unused_limits_not_reopened=True)
        checkpoint()
        print(json.dumps({k: run[k] for k in ['status', 'trajectory_full_RF', 'final_RF', 'material_SHA256',
            'global_full_RF_improved', 'material_finite_update_generated', 'all_wavefield_inner_converged', 'seconds']}), flush=True)
    except BaseException:
        run.update(status='multiequation_conditional_stage_failed_for_inspection', error=traceback.format_exc(),
            seconds=time.time()-started, all_issued_work_retained=True)
        checkpoint()
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__ == '__main__':
    main()
