"""One matched actual material update with a fixed denoiser score guide.

Zero new adjoints/JVPs, at most three full44 candidate forwards, 1800 seconds.
"""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable THP')
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
from .cdf_continuation_pool import CDFContinuationPool
from .frozen_score_guidance import FrozenScoreGuidance
from .independent_physical_coordinates import IndependentPhysicalCoordinates
from .pressure_resources import PressureResources
from .low_frequency_0p5 import atomic, digest, SAMPLE
from .reconstruct_frequency_multilevel import array


class MatchedForwardPool(CDFContinuationPool):
    def actual_work(self):
        total = super().actual_work()
        if (total.get('issued_adjoint_direction_passes', 0) != 0 or
                total.get('issued_tangent_direction_passes', 0) != 0 or
                total.get('issued_physical_forward_passes', 0) > 132):
            raise RuntimeError('The distinct matched budget is 132forward/0adjoint/0JVP')
        return total

    def evaluate(self, material, kind, objective, point, final_score=False):
        if kind != 'forward' or objective != 'full_joint' or point != 1 or final_score:
            raise RuntimeError('No derivative replay, auxiliary scores or extra directions')
        return super().evaluate(material, kind, objective, point, final_score)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--devices', type=int, nargs='+', default=[0, 1])
    p.add_argument('--max-seconds', type=float, default=1800.)
    args = p.parse_args()
    project, source = Path.cwd().resolve(), Path(__file__).parent
    for name in ['inputs', 'out', 'artifact_root']:
        setattr(args, name, getattr(args, name).resolve())
    if (not args.inputs.is_relative_to(project/'out/v5') or
            not args.out.is_relative_to(project/'out/v5') or args.out.exists() or
            not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or
            args.artifact_root.exists() or args.devices != [0, 1] or args.max_seconds != 1800):
        raise ValueError('Fresh single-direction/3candidate/132forward/1800s matched stage only')
    prepared = json.loads((args.inputs/'run.json').read_text())
    if (prepared['status'] != 'actual_fixed_model_score_CPU_inputs_prepared' or
            Path('/proc', str(prepared['pid'])).exists() or
            not prepared['direction_details']['actual_common_step_available']):
        raise RuntimeError('Prepared common direction and finished CPU process required')
    parent = Path(prepared['parent_root'])
    old = json.loads((parent/'run.json').read_text())
    closed = json.loads((parent/'scientific_boundary_CPU_finalization.json').read_text())
    workers = json.loads((parent/'worker_processes.json').read_text())
    if (closed['status'] != 'completed_saved_boundary_CPU_finalization' or
            digest(parent/'run.json') != closed['native_run_SHA256'] or
            any(Path('/proc', str(pid)).exists() for pid in [old['pid'], *workers['worker_pids']])):
        raise RuntimeError('Original trajectory must remain closed and unchanged')
    hashes = dict(prepared['source_SHA256'])
    hashes[Path(__file__).name] = digest(source/Path(__file__).name)
    for name, sha in hashes.items():
        if digest(source/name) != sha:
            raise RuntimeError('Frozen actual inputs or physical source changed: '+name)
    for key in ['material', 'data_gradient', 'score_guide', 'direction', 'neural', 'cdf']:
        if digest(prepared[key+'_path']) != prepared[key+'_SHA256']:
            raise RuntimeError('A fixed matched scientific product changed: '+key)
    plan = json.loads((parent/'native_plan.json').read_text())
    edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    for path, sha in [(Path(plan['native_anchor_material']), old['native_anchor_material_SHA256']),
            (edge_path, plan['known_edge_SHA256']),
            (Path(plan['data_root'])/'index.json', plan['data_index_SHA256']),
            (Path(plan['high_index']), plan['high_index_SHA256'])]:
        if digest(path) != sha:
            raise RuntimeError('Original acquisition/coordinate product changed')
    expected = sorted((family, shot) for family in plan['families'] for shot in range(11))
    refs = prepared['RF_source_records']
    if len(refs) != 44 or sorted((r['family'], r['shot']) for r in refs) != expected:
        raise RuntimeError('Only the completed all44 same-point baseline is allowed')
    for row in refs:
        if digest(row['path']) != row['SHA256']:
            raise RuntimeError('The same-point required-gradient RF cache changed')
    reserve = 2*2**30+400*2**20
    if shutil.disk_usage(project).free < reserve+3*2**20 or shutil.disk_usage(Path.home()).free < reserve+600*2**20:
        raise RuntimeError('Retain original reserves plus600MiB for this separate stage')
    args.out.mkdir()
    args.artifact_root.mkdir()
    for name in ['arrays', 'workers', 'lines', 'final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name, target_is_directory=True)
    (args.out/'jobs').mkdir()
    for name in hashes:
        shutil.copy2(source/name, args.out/name)
    atomic(args.out/'native_plan.json', plan)
    for name, path in [('initial_material', prepared['material_path']), ('initial_control', prepared['control_path']),
            ('completed_data_gradient', prepared['data_gradient_path']), ('score_guide', prepared['score_guide_path']),
            ('direction', prepared['direction_path'])]:
        (args.out/'arrays'/f'{name}.npy').symlink_to(Path(path).resolve())
    started = time.time()
    deadline = started+1800
    details = prepared['direction_details']
    fraction = min(1., .25*prepared['initial_RF']/(-details['RF_slope']))
    run = {'status': 'initializing_matched_score_actual_update', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': deadline, 'max_seconds': 1800,
        'source_SHA256': hashes, 'parent_root': str(parent), 'inputs_root': str(args.inputs),
        'initial_material_SHA256': prepared['material_SHA256'], 'reused_gradient_SHA256': prepared['data_gradient_SHA256'],
        'neural_SHA256': prepared['neural_SHA256'], 'CDF_SHA256': prepared['cdf_SHA256'],
        'reused_completed_adjoint_passes': 44, 'initial_RF': prepared['initial_RF'],
        'current_RF': prepared['initial_RF'], 'parent_global_RF': closed['final_RF'],
        'maximum_actual_directions': 1, 'maximum_full44_candidates': 3,
        'maximum_new_physical_forwards': 132, 'maximum_new_adjoint_passes': 0, 'maximum_new_tangent_passes': 0,
        'new_reserved_directional_equivalent_units': 0, 'pure_forward_cost_is_not_zero_compute': True,
        'previous_cumulative_reserved_units': 432, 'not_whole_thread_actual_cost': True,
        'original_CDF_budget_closed_and_unchanged': True, 'closed_parent_actual_work': closed['original_actual_work'],
        'closed_parent_created_unix_s': old['created_unix_s'], 'closed_parent_deadline_unix_s': old['deadline_unix_s'],
        'candidate_checks_issued': 0, 'candidate_checks_completed': 0, 'accepted_updates': 0,
        'first_fraction': fraction, 'direction_details': details, 'initial_guide_parts': prepared['initial_guide_parts'],
        'neural_guide_is_not_known_scalar_energy': True, 'network_Jacobian_used': False,
        'patient_truth_read_by_optimizer': False, 'intermediate_truth_feedback': False,
        'new_patient_derivative_probes': 0, 'new_data_or_frequency': False, 'reserved_case_unread': 'Neg_47_Left',
        'independent_native_fields': True, 'no_material_lowpass': True, 'one_GPU_experiment': True,
        'devices': args.devices, 'per_card_process_tree_limit_GiB': 16, 'allocator_cap_GiB': 10,
        'joint_reconstruction_success_claimed': False}
    atomic(args.out/'run.json', run)
    def stop(number, frame):
        (args.out/'stop_requested.flag').write_text(str(number)+'\n')
    for number in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(number, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    monitor, pool = PressureResources(args.out), None
    def checkpoint():
        if pool is not None:
            run['actual_native_work'] = pool.actual_work()
        atomic(args.out/'run.json', run)
    def freeze(final=False, inflight=False):
        if any(digest(source/name) != sha for name, sha in hashes.items()):
            raise RuntimeError('A frozen source changed during the matched stage')
        if time.time() >= deadline-60 or (args.out/'stop_requested.flag').exists():
            raise RuntimeError('Absolute matched deadline or cooperative stop reached')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Original disk reserves reached')
    try:
        coordinates = IndependentPhysicalCoordinates(np.load(plan['native_anchor_material']), np.load(edge_path))
        control = np.load(prepared['control_path']).astype(np.float64)
        if not np.array_equal(coordinates.material(control), np.load(prepared['material_path'])):
            raise RuntimeError('The fixed matched material must be reproduced exactly')
        direction = np.load(prepared['direction_path'])
        prior = FrozenScoreGuidance(prepared['neural_path'], prepared['cdf_path'])
        z, x = np.ogrid[:961, :1121]
        active = (z >= 80)&(z < 937)&(x >= 24)&(x < 1097)
        F0 = prepared['initial_RF']
        parts = prepared['initial_guide_parts']
        minimum_drop = float(8*np.finfo(np.float32).eps*max(F0, 1e-12))
        selected_material, selected_control = Path(prepared['material_path']), Path(prepared['control_path'])
        current_RF, current_parts = F0, parts
        stop_reason = 'no_accepted_fixed_model_score_candidate'
        freeze()
        for line in range(1, 4):
            feasible = False
            for backtrack in range(8):
                freeze()
                proposal = control+fraction*direction
                if np.max(abs(coordinates.project(proposal, active)-proposal)) > 1e-9:
                    raise RuntimeError('The qualified step box left the original physical domain')
                run.update(status='CPU_signed_score_line_work', active_line=line, active_fraction=fraction)
                checkpoint()
                work, work_details = prior.signed_line_work(coordinates, control, direction, fraction, parts)
                target_work = 1e-4*fraction*details['prior_slope']
                if work <= target_work:
                    feasible = True
                    break
                atomic(args.out/f'guide_skip_l{line:02d}_b{backtrack:02d}.json', {
                    'fraction': fraction, 'signed_guide_work': work, 'required_work': target_work,
                    'no_wave_issued': True, 'work_details': work_details})
                fraction *= .5
            if not feasible or deadline-time.time() < 360:
                stop_reason = 'no_feasible_guide_line_or_remaining_full44_time'
                break
            folder = args.out/'lines'/f'line{line:02d}'
            folder.mkdir()
            array(folder/'control.npy', proposal)
            candidate_material = coordinates.material(proposal)
            array(folder/'material.npy', candidate_material)
            atomic(folder/'signed_guide_work.json', work_details)
            if pool is None:
                pool = MatchedForwardPool(args, monitor, plan, deadline, freeze)
            run.update(status='actual_full44_matched_score_candidate', active_line=line, active_fraction=fraction)
            run['candidate_checks_issued'] += 1
            checkpoint()
            trial = pool.evaluate(candidate_material, 'forward', 'full_joint', 1)
            run['candidate_checks_completed'] += 1
            after = work_details['parts_after']
            accepted = bool(trial['loss'] < F0-minimum_drop and
                trial['loss'] <= F0+1e-4*fraction*details['RF_slope'] and
                trial['loss']+after['TV'] < F0+parts['TV'] and work <= target_work)
            curvature = 2*(trial['loss']-F0-fraction*details['RF_slope'])/(fraction*fraction)
            record = {'accepted': accepted, 'fraction': fraction, 'RF_before': F0, 'RF_after': trial['loss'],
                'minimum_RF_drop': minimum_drop, 'signed_guide_work': work, 'required_work': target_work,
                'parts_after': after, 'measured_directional_secant_curvature': curvature,
                'material_SHA256': digest(folder/'material.npy'), 'new_adjoint_passes': 0, 'new_JVPs': 0,
                'patient_truth_read': False, 'not_a_known_prior_NLL_Armijo_test': True}
            atomic(folder/'actual_native_acceptance.json', record)
            checkpoint()
            print(json.dumps(record), flush=True)
            if accepted:
                selected_material, selected_control = folder/'material.npy', folder/'control.npy'
                current_RF, current_parts, refs = trial['loss'], after, trial['RF_source_records']
                run.update(accepted_updates=1, current_RF=current_RF, current_material_SHA256=record['material_SHA256'])
                stop_reason = 'single_matched_fixed_model_score_update_accepted'
                checkpoint()
                break
            fraction = min(.5*fraction, .9*(-details['RF_slope'])/curvature) if curvature > 0 else .5*fraction
        if pool is not None:
            ids = [p.pid for p in pool.workers]
            pool.close()
            if any(Path('/proc', str(pid)).exists() for pid in ids):
                raise RuntimeError('Wait for owned source workers before any truth evaluation')
            pool = None
        final = args.out/'final'
        (final/'material.npy').symlink_to(selected_material.resolve())
        (final/'control.npy').symlink_to(selected_control.resolve())
        atomic(final/'actual_native_acceptance.json', {'accepted_updates': run['accepted_updates'],
            'actual_native_RF': {'loss': current_RF, 'all44_sources': True}, 'RF_source_records': refs,
            'native_TV': current_parts['TV'], 'material_SHA256': digest(final/'material.npy'),
            'same_frozen_model_matched_start': True, 'joint_imaging_success_claimed': False})
        run.update(status='matched_material_fixed_before_posthoc_evaluation', final_RF=current_RF,
            material_SHA256=digest(final/'material.npy'), stop_reason=stop_reason,
            beats_parent_global_RF=current_RF < closed['final_RF'])
        checkpoint()
        (final/'stage0_initial_material.npy').symlink_to(Path(prepared['material_path']).resolve())
        with (final/'maps.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material', str(final/'material.npy'),
                '--sample', SAMPLE, '--out', str(final/'maps'), '--estimate-label', 'Fixed model: direct score guide'],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        with (final/'increment.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'nsrno.posthoc_fixed_material_increment', '--candidate', str(final),
                '--initial-material', prepared['material_path'], '--initial-control', prepared['control_path']],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        run.update(status='matched_score_actual_stage_finished', seconds=time.time()-started,
            finished_unix_s=time.time(), original_budget_unchanged=True)
        checkpoint()
        print(json.dumps({k: run[k] for k in ['status', 'stop_reason', 'accepted_updates', 'final_RF',
            'beats_parent_global_RF', 'material_SHA256', 'candidate_checks_completed', 'seconds']}), flush=True)
    except BaseException:
        run.update(status='matched_score_actual_stage_failed_for_inspection', error=traceback.format_exc(),
            seconds=time.time()-started, all_issued_work_remains_charged=True)
        checkpoint()
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__ == '__main__':
    main()
