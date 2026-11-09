"""Prepared bounded encoded native inversion; accept only unchanged all44-source RF."""
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
from .encoded_physical_native_pool import EncodedPhysicalNativePool
from .depth_rms_preconditioner import depth_rms_direction
from .independent_physical_coordinates import IndependentPhysicalCoordinates as Coordinates, SCALES
from .low_frequency_0p5 import atomic, digest, SAMPLE, CASE
from .pressure_resources import PressureResources
from .reconstruct_native_gn import source_manifest
from .reconstruct_frequency_multilevel import array
from .reconstruct_native_frequency_incremental import native_RF_records, same_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--initial-material', type=Path)
    parser.add_argument('--initial-prediction-record', type=Path)
    parser.add_argument('--previous-final-maps-reviewed', action='store_true')
    parser.add_argument('--blocks', type=int, default=1)
    parser.add_argument('--steps-per-block', type=int, default=8)
    parser.add_argument('--max-seconds', type=float, default=1500.)
    parser.add_argument('--seed', type=int, default=8036)
    parser.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--cap-gib', type=float, default=10.)
    parser.add_argument('--chunk', type=int, default=128)
    args = parser.parse_args()
    if (not args.previous_final_maps_reviewed or args.blocks!=1 or
        not 1<=args.steps_per_block<=8 or not 600<=args.max_seconds<=1500 or
        not 0<args.cap_gib<=10 or not 1<=len(args.devices)<=3 or
        len(set(args.devices))!=len(args.devices)):
        raise ValueError('Require prior actual map review and the prespecified bounded phase')
    args.anchor_root, args.out, args.artifact_root = [p.resolve() for p in
                                       [args.anchor_root, args.out, args.artifact_root]]
    project = Path.cwd().resolve()
    if not all(p.is_relative_to(project/'out/v5') for p in [args.anchor_root, args.out]):
        raise ValueError('Code and experiment records must remain inside independent nsrno')
    if not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts'):
        raise ValueError('Use only the same account dedicated artifact volume')
    if args.out.exists() or args.artifact_root.exists():
        raise RuntimeError('Inspect existing same experiment; never reset or overwrite')
    chart_root = project/'out/v5/coefficient_chart_inverse_v1'
    chart = json.loads((chart_root/'run.json').read_text())
    finalization = json.loads((project/'out/v5/coefficient_chart_inverse_finalization_v1/run.json').read_text())
    if (chart['status']!='coefficient_chart_training_finished' or same_process(chart) or
        finalization['status']!='coefficient_chart_sequential_inference_and_CPU_maps_finished' or
        same_process(finalization)):
        raise RuntimeError('Current neural training and its fixed evaluation must both finish first')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                               '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Another GPU experiment remains active')
    prior = json.loads((args.anchor_root/'run.json').read_text())
    if (prior['status'] not in ['native_frequency_incremental_stage_finished',
            'encoded_native_depth_stage_finished'] or same_process(prior) or
        prior['sample']!=SAMPLE or prior['case']!=CASE):
        raise ValueError('Require a completed matched native physical incumbent')
    parent_material = args.anchor_root/'final/material.npy'
    if digest(parent_material)!=prior['material_SHA256']:
        raise ValueError('Finished physical incumbent changed')
    if (args.initial_material is None)!=(args.initial_prediction_record is None):
        raise ValueError('A new neural material requires its saved inference record')
    initial_record_SHA = None
    if args.initial_material is None:
        anchor = parent_material
    else:
        args.initial_material = args.initial_material.resolve()
        args.initial_prediction_record = args.initial_prediction_record.resolve()
        for path in [args.initial_material, args.initial_prediction_record]:
            if not path.is_relative_to(project/'out/v5'):
                raise ValueError('Only the saved actual same-project neural prediction')
        prediction_record = json.loads(args.initial_prediction_record.read_text())
        if (prediction_record['sample']!=SAMPLE or
            prediction_record['material_SHA256']!=digest(args.initial_material) or
            prediction_record['checkpoint_SHA256']!=chart['checkpoint_SHA256'] or
            prediction_record['development_truth_arrays_read']):
            raise ValueError('Actual neural initial state identity changed or used development truth')
        initial_record_SHA = digest(args.initial_prediction_record)
        anchor = args.initial_material
    parent_plan = json.loads((args.anchor_root/'native_plan.json').read_text())
    data_root = Path(parent_plan['data_root'])
    high_index = Path(parent_plan['high_index'])
    if (digest(data_root/'index.json')!=parent_plan['data_index_SHA256'] or
        digest(high_index)!=parent_plan['high_index_SHA256']):
        raise ValueError('Matched existing RF data changed')
    physical_names = ['native_time.py', 'native_time.cu', 'time_acquisition.py',
        'frequency_multilevel_acquisition.py', 'full90_3mhz_acquisition.py',
        'acquisition_single_case_0p5.py', 'low_frequency_0p5.py', 'matrix_free_rf_response.py']
    for name in physical_names:
        if prior['source_SHA256'].get(name)!=digest(Path(__file__).parent/name):
            raise ValueError('Original physical source/receiver operator changed: '+name)
    families = parent_plan['families']
    if (set(families)!={'lf', 'point3', 'lf3', 'point3_3mhz'} or
        families['lf3']['bands_hz']!=[[1500000., 2000000.], [2000000., 2600000.],
                                    [2600000., 3300000.], [3300000., 4500000.]] or
        families['point3_3mhz']['bands_hz']!=families['lf3']['bands_hz']):
        raise ValueError('Retain all full3MHz270 valid frequencies and matched four families')
    # A changed material MUST receive a new original all44-source baseline.
    references = (native_RF_records(args.anchor_root, prior, families)
                  if args.initial_material is None else [])
    reserve = 2*2**30+400*2**20
    if (shutil.disk_usage(project).free<reserve+5*2**20 or
        shutil.disk_usage(Path.home()).free<reserve+350*2**20):
        raise RuntimeError('Reserve bounded original-source RF and atomic controls without deleting data')
    meminfo = Path('/proc/meminfo').read_text().splitlines()
    available_kib = int(next(line.split()[1] for line in meminfo if line.startswith('MemAvailable:')))
    if available_kib < 12*2**20:
        raise RuntimeError('Reserve host source caches and full independent controls before workers')
    edge_path = Path('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    if digest(edge_path)!=parent_plan['known_edge_SHA256']:
        raise ValueError('Known edge absorption changed')
    coordinates = Coordinates(np.load(anchor), np.load(edge_path))
    z, x = np.ogrid[:961, :1121]
    mask = (z>=80)&(z<937)&(x>=24)&(x<1097)
    lateral_ROI = np.abs((np.arange(1121)-560)*50e-6)<=.0191+1e-12
    # Exact convex physical-domain projection occurs on every proposal.
    # Bounds on c and c*rho must remain coupled in these independent controls.
    args.out.mkdir()
    args.artifact_root.mkdir(parents=True)
    for name in ['jobs', 'steps']:
        (args.out/name).mkdir()
    for name in ['workers', 'blocks', 'final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name, target_is_directory=True)
    for name in ['latest_control.npy', 'latest_material.npy']:
        (args.out/name).symlink_to(args.artifact_root/name)
    plan = {**parent_plan, 'reference_RF':references,
        'native_anchor_material':str(anchor), 'native_anchor_material_SHA256':digest(anchor),
        'truth_read':False}
    atomic(args.out/'native_plan.json', plan)
    hashes = source_manifest()
    for name in physical_names+[Path(__file__).name, 'encoded_physical_native_worker.py',
        'encoded_physical_native_pool.py', 'encoded_native_acquisition.py', 'depth_rms_preconditioner.py',
        'independent_physical_coordinates.py', 'reconstruct_native_frequency_incremental.py',
        'reconstruct_frequency_multilevel.py']:
        hashes[name] = digest(Path(__file__).parent/name)
    for name in hashes:
        shutil.copy2(Path(__file__).parent/name, args.out/name)
    started = time.time()
    deadline = started+args.max_seconds
    run = {'status':'initializing_encoded_independent_physical', 'pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s':started, 'deadline_unix_s':deadline,
        'sample':SAMPLE, 'case':CASE, 'parent_root':str(args.anchor_root),
        'artifact_root':str(args.artifact_root), 'source_SHA256':hashes, 'families':families,
        'native_anchor_material_SHA256':digest(anchor), 'seed':args.seed,
        'physical_predecessor_material_SHA256':prior['material_SHA256'],
        'initial_prediction_record_SHA256':initial_record_SHA,
        'baseline_RF_reused_from_identical_material_cache':args.initial_material is None,
        'maximum_blocks':args.blocks, 'maximum_steps_per_block':args.steps_per_block,
        'maximum_encoded_gradient_jobs':args.blocks*args.steps_per_block,
        'maximum_encoded_adjoint_propagations':4*args.blocks*args.steps_per_block,
        'maximum_encoded_lines_per_step':3,
        'maximum_original_full44_checks':args.blocks+int(args.initial_material is not None),
        'encoded_steps_accepted':0, 'blocks_checked':0, 'blocks_accepted':0,
        'allocator_cap_GiB':args.cap_gib, 'per_card_process_tree_limit_GiB':16,
        'devices':args.devices, 'native_shape':[3, 961, 1121], 'native_spacing_m':50e-6,
        'known_coupling_rows':80, 'independent_c_logrho_log_tissuealpha_controls':True,
        'TV_fields':['log c', 'log rho', 'log positive tissue alpha'],
        'original_c_Z_alpha_physical_bounds_retained':True,
        'proposal_projection':'exact scaled-Euclidean projection onto original convex physical domain',
        'TV_weight':.001, 'physical_log_step_caps':[.004, .006, .05],
        'metric_sigma_rows':40., 'metric_relative_RMS_floor':.02,
        'material_fields_smoothed':False, 'all44_original_RF_acceptance_required':True,
        'encoded_float32_rounding_parity_not_claimed':True,
        'first_actual_gradient_includes_setup_and_counts_in_phase_budget':True,
        'parent_native_work_counters':{k:prior.get(k) for k in [
            'native_adjoint_source_shots', 'native_forward_only_source_shots',
            'encoded_adjoint_propagations', 'encoded_forward_propagations',
            'original_forward_propagations', 'gradient_jobs', 'seconds']},
        'parent_earlier_counters_as_recorded':prior.get('parent_counters_as_recorded',
            prior.get('parent_native_work_counters')),
        'completed_chart_updates':chart['completed_steps'],
        'chart_cumulative_optimizer_work':chart['cumulative_optimizer_work'],
        'all_old_phase_budgets_retained':True,
        'new_observations_or_cases':False, 'patient_truth_read_by_optimizer':False,
        'reserved_case_unread':'Neg_47_Left', 'new_material_derivative_probes':0,
        'joint_imaging_success_claimed':False}
    atomic(args.out/'run.json', run)
    monitor = PressureResources(args.out)
    pool = None
    stop = [0]
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for value in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(value, lambda number, frame:stop.__setitem__(0, int(number)))
    rng = np.random.default_rng(args.seed)
    best = np.zeros_like(coordinates.m)
    current = best.copy()
    best_folder = None
    best_paths = references
    rejected = 0
    native_candidate_accepted = False

    def freeze():
        if any(digest(Path(__file__).parent/name)!=sha for name, sha in hashes.items()):
            raise RuntimeError('Frozen implementation changed')
        if digest(anchor)!=plan['native_anchor_material_SHA256']:
            raise RuntimeError('Immutable incumbent changed')
        if (args.initial_prediction_record is not None and
            digest(args.initial_prediction_record)!=initial_record_SHA):
            raise RuntimeError('Frozen neural inference record changed')

    def checkpoint():
        array(args.artifact_root/'latest_control.npy', current)
        array(args.artifact_root/'latest_material.npy', coordinates.material(current))
        run.update(**pool.counts, sampler_state=rng.bit_generator.state)
        run['issued_propagations'] = dict(pool.attempted)
        atomic(args.out/'run.json', run)

    def maps(folder, label):
        with (folder/'maps.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material',
                str(folder/'material.npy'), '--sample', SAMPLE, '--out', str(folder/'maps'),
                '--estimate-label', label], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES':''}, check=True, timeout=120)

    def original_RF_records(token):
        rows = []
        for name in families:
            for shot in range(11):
                paths = list((args.out/'workers').glob(
                    f'rank*/fine_RF/token{token:04d}/{name}_{shot:02d}.npy'))
                if len(paths)!=1:
                    raise RuntimeError('Missing original source RF after full actual measurement')
                rows.append({'family':name, 'shot':shot, 'path':str(paths[0].resolve()),
                             'SHA256':digest(paths[0])})
        return rows

    try:
        pool = EncodedPhysicalNativePool(args, monitor, plan)
        run['status'] = ('reading_SHA_checked_original_full_RF_baseline'
                        if args.initial_material is None else 'measuring_new_neural_initial_original_all44_RF')
        atomic(args.out/'run.json', run)
        if args.initial_material is None:
            best_RF, _ = pool.evaluate(None, 'cached_forward', 'original_cached_baseline')
            if abs(best_RF['loss']-prior['final_actual_native_RF_loss'])>1e-10:
                raise RuntimeError('Original full-source cached baseline differs from completed predecessor')
        else:
            best_RF, _ = pool.evaluate(coordinates.material(best), 'forward',
                                       'new_neural_initial_original_all44', save_RF=True)
            best_paths = original_RF_records(best_RF['token'])
        best_TV = coordinates.regularization(best, .001)[0]
        run.update(native_anchor_RF_loss=best_RF['loss'], native_anchor_TV=best_TV)
        checkpoint()
        for block in range(args.blocks):
            if stop[0] or time.time()>=deadline-360:
                break
            current = best.copy()
            normalized_caps = np.array([.004, .006, .05])[:, None, None]*.5**rejected/SCALES
            completed = 0
            for step in range(args.steps_per_block):
                if stop[0] or time.time()>=deadline-360:
                    break
                freeze()
                tag = f'block{block+1:02d}_step{step+1:02d}'
                codes = {name:rng.choice([-1, 1], size=11).tolist() for name in families}
                run.update(status='actual_encoded_native_gradient', active_block=block+1,
                    active_step=step+1, active_encoding=codes)
                atomic(args.out/'run.json', run)
                rf, gradient = pool.evaluate(coordinates.material(current), 'encoded_gradient',
                                             tag+'_gradient', codes)
                tv, tv_gradient = coordinates.regularization(current, .001)
                gradient += tv_gradient
                gradient[:, ~mask] = 0.
                direction, metric = depth_rms_direction(gradient, mask, lateral_ROI,
                                                       normalized_caps)
                lines = []
                accepted = False
                for line, fraction in enumerate([1., .5, .25], 1):
                    if stop[0] or time.time()>=deadline-180:
                        break
                    proposal = coordinates.project(current+fraction*direction, mask)
                    slope = float(np.sum(gradient*(proposal-current), dtype=np.float64))
                    if slope>=0:
                        lines.append({'fraction':fraction, 'slope':slope, 'propagated':False})
                        continue
                    run['status'] = 'same_encoding_actual_line_search'
                    atomic(args.out/'run.json', run)
                    trial, _ = pool.evaluate(coordinates.material(proposal), 'encoded_forward',
                                              tag+f'_line{line:02d}', codes)
                    proposed_TV = coordinates.regularization(proposal, .001)[0]
                    ok = trial['loss']+proposed_TV<=rf['loss']+tv+1e-4*slope
                    lines.append({'fraction':fraction, 'slope':slope, 'RF_before':rf['loss'],
                        'encoded_RF_after':trial['loss'], 'TV_after':proposed_TV,
                        'accepted':bool(ok), 'token':trial['token']})
                    if ok:
                        current = proposal
                        accepted = True
                        run['encoded_steps_accepted'] += 1
                        break
                completed += 1
                atomic(args.out/'steps'/(tag+'.json'), {'block':block+1, 'step':step+1,
                    'encoding':codes, 'metric':metric, 'lines':lines, 'accepted':accepted,
                    'full_original_RF_acceptance_still_pending':True, 'truth_read':False})
                checkpoint()
                del gradient, tv_gradient, direction
            if np.array_equal(current, best) or stop[0] or time.time()>=deadline-150:
                break
            folder = args.out/'blocks'/f'block{block+1:02d}'
            folder.mkdir()
            array(folder/'material.npy', coordinates.material(current))
            array(folder/'control.npy', current)
            (folder/'stage0_initial_material.npy').symlink_to(anchor)
            run.update(status='unchanged_original_all44_RF_acceptance', active_block=block+1)
            atomic(args.out/'run.json', run)
            actual, _ = pool.evaluate(coordinates.material(current), 'forward',
                                      f'block{block+1:02d}_original_full44', save_RF=True)
            proposed_TV = coordinates.regularization(current, .001)[0]
            ok = (actual['loss']<best_RF['loss'] and
                  actual['loss']+proposed_TV<best_RF['loss']+best_TV)
            run['blocks_checked'] += 1
            rows = original_RF_records(actual['token'])
            record = {'accepted':bool(ok), 'RF_before':best_RF['loss'], 'actual_native_RF':actual,
                'TV_before':best_TV, 'TV_after':proposed_TV, 'completed_encoded_steps':completed,
                'material_SHA256':digest(folder/'material.npy'), 'RF_source_records':rows,
                'unchanged_original_source_receiver_operator':True, 'patient_truth_read':False}
            atomic(folder/'actual_native_acceptance.json', record)
            if ok:
                best, best_RF, best_TV = current.copy(), actual, proposed_TV
                best_folder, best_paths = folder, rows
                native_candidate_accepted = True
                run['blocks_accepted'] += 1
                maps(folder, 'Accepted encoded native depth update')
            else:
                rejected += 1
                current = best.copy()
            checkpoint()
        final = args.out/'final'
        (final/'stage0_initial_material.npy').symlink_to(anchor)
        if best_folder is None:
            (final/'material.npy').symlink_to(anchor)
        else:
            os.link(best_folder/'material.npy', final/'material.npy')
        record = {'accepted':native_candidate_accepted, 'material_SHA256':digest(final/'material.npy'),
            'actual_native_RF':best_RF, 'native_TV':best_TV, 'RF_source_records':best_paths,
            'full_original44_source_acceptance':True, 'joint_imaging_success_claimed':False}
        atomic(final/'actual_native_acceptance.json', record)
        if best_folder is not None:
            for suffix in ['.png', '.metrics.json']:
                shutil.copy2(best_folder/('maps'+suffix), final/('maps'+suffix))
        else:
            maps(final, 'Unchanged original native incumbent')
        run.update(status='encoded_independent_physical_stage_finished', finished_unix_s=time.time(),
            seconds=time.time()-started, **pool.counts,
            native_candidate_accepted=native_candidate_accepted,
            final_actual_native_RF_loss=best_RF['loss'],
            material_SHA256=digest(final/'material.npy'),
            separate_CPU_post_hoc_maps_saved=True, actual_image_view_still_required=True,
            stopped_signal=stop[0], old_budgets_not_reset=True)
        atomic(args.out/'run.json', run)
        print(json.dumps({k:run[k] for k in ['status', 'seconds', 'encoded_gradient_jobs',
            'encoded_adjoint_propagations', 'original_full_RF_checks', 'blocks_accepted',
            'final_actual_native_RF_loss', 'material_SHA256']}), flush=True)
    except BaseException:
        run.update(status='encoded_native_depth_failed_for_inspection', error=traceback.format_exc(),
                   stopped_unix_s=time.time())
        if pool is not None:
            run.update(**pool.counts)
            run['issued_propagations'] = dict(pool.attempted)
        atomic(args.out/'run.json', run)
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__=='__main__':
    main()
