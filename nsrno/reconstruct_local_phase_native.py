"""Bounded actual three-field inversion with local-phase and original RF fidelity."""
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
from .local_phase_native_pool import LocalPhaseNativePool
from .depth_rms_preconditioner import depth_rms_direction
from .native_gn_coordinates import Coordinates, SCALES
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
    parser.add_argument('--previous-final-maps-reviewed', action='store_true')
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--steps-per-block', type=int, default=8)
    parser.add_argument('--max-seconds', type=float, default=2400.)
    parser.add_argument('--seed', type=int, default=8037)
    parser.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--cap-gib', type=float, default=10.)
    parser.add_argument('--chunk', type=int, default=128)
    args = parser.parse_args()
    if (not args.previous_final_maps_reviewed or not 1<=args.blocks<=2 or
        not 1<=args.steps_per_block<=8 or not 600<=args.max_seconds<=2400 or
        not 0<args.cap_gib<=10 or not 1<=len(args.devices)<=3 or
        len(set(args.devices))!=len(args.devices)):
        raise ValueError('Require previous actual map review and the fixed bounded design')
    args.anchor_root, args.out, args.artifact_root = [p.resolve() for p in
                                      [args.anchor_root, args.out, args.artifact_root]]
    project = Path.cwd().resolve()
    if not all(p.is_relative_to(project/'out/v5') for p in [args.anchor_root, args.out]):
        raise ValueError('Keep experiment inside independent nsrno')
    if not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts'):
        raise ValueError('Use same-account dedicated artifact volume')
    if args.out.exists() or args.artifact_root.exists():
        raise RuntimeError('Inspect existing same run; never reset or overwrite')
    prior = json.loads((args.anchor_root/'run.json').read_text())
    if (prior['status']!='encoded_native_depth_stage_finished' or same_process(prior) or
        prior['sample']!=SAMPLE or prior['case']!=CASE):
        raise ValueError('Require completed actual original native predecessor')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                               '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Another GPU experiment is active')
    anchor = args.anchor_root/'final/material.npy'
    if digest(anchor)!=prior['material_SHA256']:
        raise ValueError('Predecessor final material changed')
    parent_plan = json.loads((args.anchor_root/'native_plan.json').read_text())
    data_root, high_index = Path(parent_plan['data_root']), Path(parent_plan['high_index'])
    if (digest(data_root/'index.json')!=parent_plan['data_index_SHA256'] or
        digest(high_index)!=parent_plan['high_index_SHA256']):
        raise ValueError('Fixed matched observations changed')
    physical_names = ['native_time.py', 'native_time.cu', 'time_acquisition.py',
        'frequency_multilevel_acquisition.py', 'full90_3mhz_acquisition.py',
        'acquisition_single_case_0p5.py', 'low_frequency_0p5.py', 'matrix_free_rf_response.py']
    for name in physical_names:
        if prior['source_SHA256'].get(name)!=digest(Path(__file__).parent/name):
            raise ValueError('Original physical operator changed: '+name)
    families = parent_plan['families']
    high_bands = [[1500000., 2000000.], [2000000., 2600000.],
                  [2600000., 3300000.], [3300000., 4500000.]]
    if (set(families)!={'lf', 'point3', 'lf3', 'point3_3mhz'} or
        families['lf3']['bands_hz']!=high_bands or
        families['point3_3mhz']['bands_hz']!=high_bands or
        any(len(spec['bands_hz'])!=4 for spec in families.values())):
        raise ValueError('All valid dense RF frequencies and four original families required')
    references = native_RF_records(args.anchor_root, prior, families)
    reserve = 2*2**30+400*2**20
    if (shutil.disk_usage(project).free<reserve+5*2**20 or
        shutil.disk_usage(Path.home()).free<reserve+350*2**20):
        raise RuntimeError('Retain disk reserve without deleting unique data')
    meminfo = Path('/proc/meminfo').read_text().splitlines()
    available_kib = int(next(line.split()[1] for line in meminfo if line.startswith('MemAvailable:')))
    if available_kib < 12*2**20:
        raise RuntimeError('Reserve host phase caches and full independent controls')
    edge_path = Path('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    if digest(edge_path)!=parent_plan['known_edge_SHA256']:
        raise ValueError('Known edge absorption changed')
    coordinates = Coordinates(np.load(anchor), np.load(edge_path))
    z, x = np.ogrid[:961, :1121]
    mask = (z>=80)&(z<937)&(x>=24)&(x<1097)
    lateral_ROI = np.abs((np.arange(1121)-560)*50e-6)<=.0191+1e-12
    lower, upper = coordinates.bounds(trust=(1., 1., 10.))
    lower[:, ~mask] = upper[:, ~mask] = 0.
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
        'truth_read':False, 'nonlinear_source_encoding':False}
    atomic(args.out/'native_plan.json', plan)
    hashes = source_manifest()
    for name in physical_names+[Path(__file__).name, 'local_phase_native_worker.py',
        'local_phase_native_pool.py', 'frequency_phase_increment.py',
        'depth_rms_preconditioner.py', 'native_gn_coordinates.py',
        'reconstruct_native_frequency_incremental.py', 'reconstruct_frequency_multilevel.py']:
        hashes[name] = digest(Path(__file__).parent/name)
    for name in hashes:
        shutil.copy2(Path(__file__).parent/name, args.out/name)
    started = time.time()
    deadline = started+args.max_seconds
    run = {'status':'initializing_local_phase_native', 'pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s':started, 'deadline_unix_s':deadline,
        'sample':SAMPLE, 'case':CASE, 'parent_root':str(args.anchor_root),
        'artifact_root':str(args.artifact_root), 'source_SHA256':hashes, 'families':families,
        'native_anchor_material_SHA256':digest(anchor), 'seed':args.seed,
        'maximum_blocks':args.blocks, 'maximum_steps_per_block':args.steps_per_block,
        'maximum_gradient_jobs':args.blocks*args.steps_per_block,
        'maximum_native_adjoint_source_shots':4*args.blocks*args.steps_per_block,
        'maximum_lines_per_step':3, 'maximum_original_full_checks':args.blocks,
        'steps_accepted':0, 'blocks_checked':0, 'blocks_accepted':0,
        'allocator_cap_GiB':args.cap_gib, 'per_card_process_tree_limit_GiB':16,
        'devices':args.devices, 'native_shape':[3, 961, 1121], 'native_spacing_m':50e-6,
        'known_coupling_rows':80, 'independent_c_logZ_log_tissuealpha_controls':True,
        'objective':'0.2 RF/RF0 + 0.8 local_phase/local_phase0 + old TV',
        'TV_weight':.001, 'physical_log_step_caps':[.004, .006, .05],
        'full_acceptance_RF_upper_multiple_of_anchor':1.2,
        'full_acceptance_requires_phase_and_hybrid_decrease':True,
        'uniform_actual_source_per_family':True, 'sample_importance_factor':11,
        'nonlinear_phase_source_encoded':False,
        'phase_cache_and_phase_autograd_device':'CPU with differentiable receiving-coefficient copy',
        'material_fields_smoothed':False, 'new_observations_or_cases':False,
        'baseline_is_SHA_checked_existing_original44_RF':True,
        'parent_counters_as_recorded':{k:prior.get(k) for k in [
            'encoded_gradient_jobs', 'encoded_adjoint_propagations',
            'encoded_forward_propagations', 'original_forward_propagations', 'seconds',
            'parent_native_work_counters', 'parent_earlier_counters_as_recorded',
            'completed_chart_updates', 'chart_cumulative_optimizer_work']},
        'patient_truth_read_by_optimizer':False, 'reserved_case_unread':'Neg_47_Left',
        'new_material_derivative_probes':0, 'joint_imaging_success_claimed':False}
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
    best_folder, best_paths, rejected = None, references, 0

    def freeze():
        if any(digest(Path(__file__).parent/name)!=sha for name, sha in hashes.items()):
            raise RuntimeError('Frozen implementation changed')
        if digest(anchor)!=plan['native_anchor_material_SHA256']:
            raise RuntimeError('Immutable incumbent changed')

    def checkpoint():
        array(args.artifact_root/'latest_control.npy', current)
        array(args.artifact_root/'latest_material.npy', coordinates.material(current))
        run.update(**pool.counts, sampler_state=rng.bit_generator.state,
                   issued_propagations=dict(pool.attempted))
        atomic(args.out/'run.json', run)

    def maps(folder):
        with (folder/'maps.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material',
                str(folder/'material.npy'), '--sample', SAMPLE, '--out', str(folder/'maps'),
                '--estimate-label', 'Local phase plus RF actual native reconstruction'],
                stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES':''}, check=True, timeout=120)

    def original_RF_records(token):
        rows = []
        for name in families:
            for shot in range(11):
                paths = list((args.out/'workers').glob(
                    f'rank*/fine_RF/token{token:04d}/{name}_{shot:02d}.npy'))
                if len(paths)!=1:
                    raise RuntimeError('Missing original source RF after full actual evaluation')
                rows.append({'family':name, 'shot':shot, 'path':str(paths[0].resolve()),
                             'SHA256':digest(paths[0])})
        return rows

    try:
        pool = LocalPhaseNativePool(args, monitor, plan)
        run['status'] = 'measuring_local_phase_on_existing_original44_RF'
        atomic(args.out/'run.json', run)
        best_RF, _ = pool.evaluate(None, 'cached_forward', 'cached_original44_new_objective')
        if abs(best_RF['RF']-prior['final_actual_native_RF_loss'])>1e-10:
            raise RuntimeError('Cached original RF objective changed')
        if min(best_RF['RF'], best_RF['phase'])<=1e-10:
            raise RuntimeError('Initial nonzero data scales required')
        pool.normalization = {'RF':best_RF['RF'], 'phase':best_RF['phase']}
        best_RF['loss'] = 1.
        best_TV = coordinates.regularization(best, .001)[0]
        run.update(normalization=pool.normalization, native_anchor_RF_loss=best_RF['RF'],
                   native_anchor_phase_loss=best_RF['phase'], native_anchor_TV=best_TV)
        checkpoint()
        for block in range(args.blocks):
            if stop[0] or time.time()>=deadline-360:
                break
            current = best.copy()
            caps = np.array([.004, .006, .05])[:, None, None]*.5**rejected/SCALES
            completed = 0
            for step in range(args.steps_per_block):
                if stop[0] or time.time()>=deadline-360:
                    break
                freeze()
                tag = f'block{block+1:02d}_step{step+1:02d}'
                shots = {name:int(rng.integers(11)) for name in families}
                run.update(status='actual_local_phase_native_gradient', active_block=block+1,
                           active_step=step+1, active_source_sample=shots)
                atomic(args.out/'run.json', run)
                measured, gradient = pool.evaluate(coordinates.material(current), 'gradient',
                                                   tag+'_gradient', shots)
                tv, tv_gradient, _, _ = coordinates.regularization(current, .001)
                gradient += tv_gradient
                gradient[:, ~mask] = 0.
                direction, metric = depth_rms_direction(gradient, mask, lateral_ROI, caps)
                lines, accepted = [], False
                for line, fraction in enumerate([1., .5, .25], 1):
                    if stop[0] or time.time()>=deadline-180:
                        break
                    proposal = np.clip(current+fraction*direction, lower, upper)
                    slope = float(np.sum(gradient*(proposal-current), dtype=np.float64))
                    if slope>=0:
                        lines.append({'fraction':fraction, 'slope':slope, 'propagated':False})
                        continue
                    run['status'] = 'same_actual_source_local_phase_line_search'
                    atomic(args.out/'run.json', run)
                    trial, _ = pool.evaluate(coordinates.material(proposal), 'sample_forward',
                                              tag+f'_line{line:02d}', shots)
                    proposed_TV = coordinates.regularization(proposal, .001)[0]
                    ok = trial['loss']+proposed_TV<=measured['loss']+tv+1e-4*slope
                    lines.append({'fraction':fraction, 'slope':slope,
                        'hybrid_before':measured['loss'], 'hybrid_after':trial['loss'],
                        'RF_before':measured['RF'], 'RF_after':trial['RF'],
                        'phase_before':measured['phase'], 'phase_after':trial['phase'],
                        'TV_after':proposed_TV, 'accepted':bool(ok), 'token':trial['token']})
                    if ok:
                        current, accepted = proposal, True
                        run['steps_accepted'] += 1
                        break
                completed += 1
                atomic(args.out/'steps'/(tag+'.json'), {'block':block+1, 'step':step+1,
                    'sampled_actual_shots':shots, 'metric':metric, 'lines':lines,
                    'accepted':accepted, 'full_original44_acceptance_still_pending':True,
                    'truth_read':False})
                checkpoint()
                del gradient, tv_gradient, direction
            if np.array_equal(current, best) or stop[0] or time.time()>=deadline-150:
                break
            folder = args.out/'blocks'/f'block{block+1:02d}'
            folder.mkdir()
            array(folder/'material.npy', coordinates.material(current))
            array(folder/'control.npy', current)
            (folder/'stage0_initial_material.npy').symlink_to(anchor)
            run.update(status='original_all44_local_phase_and_RF_acceptance', active_block=block+1)
            atomic(args.out/'run.json', run)
            actual, _ = pool.evaluate(coordinates.material(current), 'forward',
                                      f'block{block+1:02d}_original_full44', save_RF=True)
            proposed_TV = coordinates.regularization(current, .001)[0]
            ok = (actual['loss']+proposed_TV<best_RF['loss']+best_TV and
                  actual['phase']<best_RF['phase'] and
                  actual['RF']<=1.2*pool.normalization['RF'])
            run['blocks_checked'] += 1
            rows = original_RF_records(actual['token'])
            record = {'accepted':bool(ok), 'hybrid_before':best_RF['loss'],
                'phase_before':best_RF['phase'], 'RF_before':best_RF['RF'],
                'actual_native_RF':actual, 'TV_before':best_TV, 'TV_after':proposed_TV,
                'completed_actual_gradient_jobs':completed, 'material_SHA256':digest(folder/'material.npy'),
                'RF_source_records':rows, 'unchanged_original_source_receiver_operator':True,
                'patient_truth_read':False, 'joint_imaging_success_claimed':False}
            atomic(folder/'actual_native_acceptance.json', record)
            if ok:
                best, best_RF, best_TV = current.copy(), actual, proposed_TV
                best_folder, best_paths = folder, rows
                run['blocks_accepted'] += 1
                maps(folder)
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
        atomic(final/'actual_native_acceptance.json', {'accepted':best_folder is not None,
            'material_SHA256':digest(final/'material.npy'), 'actual_native_RF':best_RF,
            'native_TV':best_TV, 'RF_source_records':best_paths,
            'full_original44_source_acceptance':True, 'joint_imaging_success_claimed':False})
        if best_folder is not None:
            for suffix in ['.png', '.metrics.json']:
                shutil.copy2(best_folder/('maps'+suffix), final/('maps'+suffix))
        else:
            maps(final)
        run.update(status='local_phase_native_stage_finished', finished_unix_s=time.time(),
            seconds=time.time()-started, **pool.counts,
            native_candidate_accepted=best_folder is not None,
            final_hybrid_data_loss=best_RF['loss'], final_actual_native_RF_loss=best_RF['RF'],
            final_local_phase_loss=best_RF['phase'], material_SHA256=digest(final/'material.npy'),
            separate_CPU_post_hoc_maps_saved=True, actual_image_view_still_required=True,
            stopped_signal=stop[0], old_budgets_not_reset=True)
        atomic(args.out/'run.json', run)
        print(json.dumps({k:run[k] for k in ['status', 'seconds', 'gradient_jobs',
            'native_adjoint_source_shots', 'original_full_checks', 'blocks_accepted',
            'final_hybrid_data_loss', 'final_actual_native_RF_loss', 'final_local_phase_loss',
            'material_SHA256']}), flush=True)
    except BaseException:
        run.update(status='local_phase_native_failed_for_inspection', error=traceback.format_exc(),
                   stopped_unix_s=time.time())
        if pool is not None:
            run.update(**pool.counts, issued_propagations=dict(pool.attempted))
        atomic(args.out/'run.json', run)
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__=='__main__':
    main()
