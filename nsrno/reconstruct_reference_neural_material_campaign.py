"""Continue one reference-neural inverse state, with original44 block acceptance."""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import hashlib
import io
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
import torch
from .encoded_physical_native_pool import EncodedPhysicalNativePool
from .independent_physical_coordinates import IndependentPhysicalCoordinates, SCALES
from .reference_neural_material_fields import ReferenceNeuralMaterialFields
from .neural_parameter_search import SharedLayerRMS, restore_parameters
from .low_frequency_0p5 import atomic, digest, SAMPLE, CASE
from .pressure_resources import PressureResources
from .reconstruct_frequency_multilevel import array
from .reconstruct_native_frequency_incremental import same_process, native_RF_records
from .reconstruct_native_gn import source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--previous-final-maps-reviewed', action='store_true')
    parser.add_argument('--blocks', type=int, default=12)
    parser.add_argument('--block-steps', type=int, default=8)
    parser.add_argument('--max-seconds', type=float, default=14400.)
    parser.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--cap-gib', type=float, default=10.)
    parser.add_argument('--chunk', type=int, default=128)
    args = parser.parse_args()
    if (not args.previous_final_maps_reviewed or args.block_steps != 8 or
        not 1 <= args.blocks <= 12 or not 1200 <= args.max_seconds <= 14400 or
        not 0 < args.cap_gib <= 10 or not 1 <= len(args.devices) <= 3 or
        len(set(args.devices)) != len(args.devices)):
        raise ValueError('Reviewed predecessor and bounded continuing campaign required')
    project = Path.cwd().resolve()
    for name in ['parent_root', 'out', 'artifact_root']:
        setattr(args, name, getattr(args, name).resolve())
    if (not all(p.is_relative_to(project/'out/v5') for p in [args.parent_root, args.out]) or
        not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts')):
        raise ValueError('Independent nsrno and dedicated same-account artifacts only')
    if args.out.exists() or args.artifact_root.exists():
        raise RuntimeError('Inspect existing run; never reset work')
    prior = json.loads((args.parent_root/'run.json').read_text())
    if (prior['status'] != 'reference_neural_material_stage_finished' or same_process(prior) or
        prior['sample'] != SAMPLE or prior['case'] != CASE):
        raise RuntimeError('Require finished, reviewed, same-case reference network')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                               '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU experiment admission is not empty')
    parent_plan = json.loads((args.parent_root/'native_plan.json').read_text())
    initial_material = (args.parent_root/'final/material.npy').resolve()
    anchor = Path(parent_plan['native_anchor_material']).resolve()
    if (digest(initial_material) != prior['material_SHA256'] or
        digest(anchor) != prior['native_anchor_material_SHA256'] or
        digest(args.parent_root/'checkpoint.pt') != prior['checkpoint_SHA256']):
        raise RuntimeError('Accepted material, original coordinate anchor, or checkpoint changed')
    physical_names = ['native_time.py', 'native_time.cu', 'time_acquisition.py',
        'frequency_multilevel_acquisition.py', 'full90_3mhz_acquisition.py',
        'acquisition_single_case_0p5.py', 'low_frequency_0p5.py', 'matrix_free_rf_response.py',
        'encoded_native_acquisition.py', 'encoded_physical_native_worker.py',
        'encoded_physical_native_pool.py', 'independent_physical_coordinates.py',
        'independent_neural_material_fields.py', 'reference_neural_material_fields.py']
    for name in physical_names:
        if prior['source_SHA256'].get(name) != digest(Path(__file__).parent/name):
            raise RuntimeError('Original physical or neural implementation changed: '+name)
    if (digest(Path(parent_plan['data_root'])/'index.json') != parent_plan['data_index_SHA256'] or
        digest(parent_plan['high_index']) != parent_plan['high_index_SHA256']):
        raise RuntimeError('Fixed authorized observations changed')
    families = parent_plan['families']
    if (set(families) != {'lf', 'point3', 'lf3', 'point3_3mhz'} or
        any(len(spec['bands_hz']) != 4 for spec in families.values()) or
        families['lf3']['bands_hz'][-1][1] != 4.5e6):
        raise RuntimeError('Complete original44 and dense effective bands required')
    references = native_RF_records(args.parent_root, prior, families)
    edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    if digest(edge_path) != parent_plan['known_edge_SHA256']:
        raise RuntimeError('Known absorption changed')
    reserve = 2*2**30+400*2**20
    if (shutil.disk_usage(project).free < reserve+5*2**20 or
        shutil.disk_usage(Path.home()).free < reserve+1024*2**20):
        raise RuntimeError('Preserve disk reserve and unique RF data')
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:')))
    if available < 16*2**20:
        raise RuntimeError('Insufficient host activation and wave-worker memory')
    torch.set_num_threads(8)
    coordinates = IndependentPhysicalCoordinates(np.load(anchor), np.load(edge_path))
    z, x = np.ogrid[:961, :1121]
    mask = (z >= 80) & (z < 937) & (x >= 24) & (x < 1097)
    model = ReferenceNeuralMaterialFields(coordinates, mask, prior['seed'])
    inherited = torch.load(args.parent_root/'checkpoint.pt', map_location='cpu', weights_only=False)
    if (inherited['initial_material_SHA256'] != prior['native_anchor_material_SHA256'] or
        inherited['control_material_SHA256'] != prior['material_SHA256'] or
        not inherited['depth_frozen']):
        raise RuntimeError('Checkpoint does not describe accepted neural state')
    missing, unexpected = model.load_state_dict(inherited['neural_state'], strict=False)
    if set(missing) != {'absolute', 'active'} or unexpected:
        raise RuntimeError('Neural weights, fixed noise, reference, or depth metric missing')
    model.depth_frozen = True
    with torch.no_grad():
        current = model().numpy().copy()
    restored_material = coordinates.material(current)
    buffer = io.BytesIO()
    np.save(buffer, restored_material)
    restored_sha = hashlib.sha256(buffer.getvalue()).hexdigest()
    if restored_sha != prior['material_SHA256'] or not np.array_equal(
            restored_material, np.load(initial_material)):
        raise RuntimeError('Restored neural physical material differs from accepted bytes')
    del buffer, restored_material
    rng = np.random.default_rng()
    rng.bit_generator.state = inherited['sampler_state']
    optimizer = SharedLayerRMS(model)
    del inherited
    args.out.mkdir()
    args.artifact_root.mkdir(parents=True)
    for name in ['jobs', 'steps']:
        (args.out/name).mkdir()
    for name in ['workers', 'blocks', 'final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name, target_is_directory=True)
    for name in ['latest_control.npy', 'latest_material.npy', 'checkpoint.pt', 'best_checkpoint.pt']:
        (args.out/name).symlink_to(args.artifact_root/name)
    plan = {**parent_plan, 'reference_RF': references, 'truth_read': False,
        'new_original44_baseline_required': False,
        'campaign_initial_material': str(initial_material),
        'campaign_initial_material_SHA256': prior['material_SHA256'],
        'restored_material_SHA256': restored_sha}
    atomic(args.out/'native_plan.json', plan)
    hashes = source_manifest()
    for name in physical_names+[Path(__file__).name, 'neural_parameter_search.py',
        'reconstruct_frequency_multilevel.py', 'reconstruct_native_frequency_incremental.py']:
        hashes[name] = digest(Path(__file__).parent/name)
    for name in hashes:
        shutil.copy2(Path(__file__).parent/name, args.out/name)
    started, stop = time.time(), [0]
    deadline = started+args.max_seconds
    run = {'status': 'initializing_reference_neural_campaign', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'deadline_unix_s': deadline, 'sample': SAMPLE, 'case': CASE,
        'parent_root': str(args.parent_root), 'artifact_root': str(args.artifact_root),
        'source_SHA256': hashes, 'families': families, 'seed': prior['seed'],
        'native_anchor_material_SHA256': prior['native_anchor_material_SHA256'],
        'campaign_initial_material_SHA256': prior['material_SHA256'],
        'restored_physical_material_SHA256': restored_sha,
        'resumed_checkpoint_SHA256': prior['checkpoint_SHA256'],
        'fixed_random_noise_reference_depth_and_source_rng_restored': True,
        'new_original44_baseline_required': False,
        'maximum_encoded_gradient_jobs': args.blocks*args.block_steps,
        'maximum_encoded_adjoint_propagations': 4*args.blocks*args.block_steps,
        'maximum_encoded_lines_per_step': 3, 'maximum_original_full44_checks': args.blocks,
        'block_steps': args.block_steps, 'blocks_completed': 0, 'blocks_accepted': 0,
        'accepted_encoded_parameter_steps_including_later_rollbacks': 0,
        'full_accepted_path_neural_steps': 0, 'allocator_cap_GiB': args.cap_gib,
        'per_card_process_tree_limit_GiB': 16, 'devices': args.devices,
        'neural_config': model.config(), 'neural_network_device': 'CPU',
        'TV_weight': .001, 'physical_log_step_caps': [.004, .006, .05],
        'parameter_optimizer': 'positive layer RMS shared across fields; one common step',
        'new_observations_or_cases': False, 'patient_truth_read_by_optimizer': False,
        'reserved_case_unread': 'Neg_47_Left', 'new_material_derivative_probes': 0,
        'preceding_neural_v2_work': {k: prior.get(k) for k in ['encoded_gradient_jobs',
            'encoded_adjoint_propagations', 'encoded_forward_propagations',
            'original_forward_propagations', 'seconds', 'final_actual_native_RF_loss']},
        'earlier_work_retained_in_parent_run_not_reset': True,
        'joint_imaging_success_claimed': False}
    atomic(args.out/'run.json', run)
    monitor, pool = PressureResources(args.out), None
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for value in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(value, lambda number, frame: stop.__setitem__(0, int(number)))
    best = current.copy()
    best_parameters = model.parameter_snapshot()
    best_optimizer = optimizer.state_dict()
    best_folder = args.parent_root/'final'
    cap_factor, rejected_blocks, small_gain_blocks, failed_steps = 1., 0, 0, 0
    stop_reason = None

    def freeze():
        if any(digest(Path(__file__).parent/name) != sha for name, sha in hashes.items()):
            raise RuntimeError('Frozen implementation changed')
        if digest(anchor) != prior['native_anchor_material_SHA256']:
            raise RuntimeError('Original coordinate anchor changed')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Disk reserve reached')

    def save_checkpoint(path, full_accepted):
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()
                 if k.startswith('fields.') or k == 'depth_scale'}
        temporary = path.with_suffix('.partial.pt')
        torch.save({'neural_state': state, 'depth_frozen': model.depth_frozen,
            'initial_material_SHA256': prior['native_anchor_material_SHA256'],
            'control_material_SHA256': digest(args.artifact_root/'latest_material.npy'),
            'source_SHA256': hashes, 'sampler_state': rng.bit_generator.state,
            'optimizer': optimizer.state_dict(), 'full_original44_accepted': full_accepted,
            'campaign_completed_work': dict(pool.counts), 'next_cap_factor': cap_factor}, temporary)
        temporary.replace(path)

    def checkpoint(full_accepted=False):
        array(args.artifact_root/'latest_control.npy', current)
        array(args.artifact_root/'latest_material.npy', coordinates.material(current))
        save_checkpoint(args.artifact_root/'checkpoint.pt', full_accepted)
        if full_accepted:
            save_checkpoint(args.artifact_root/'best_checkpoint.pt', True)
        run.update(**pool.counts, issued_propagations=dict(pool.attempted),
            sampler_state=rng.bit_generator.state, next_physical_cap_factor=cap_factor,
            coordinator_peak_host_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        atomic(args.out/'run.json', run)

    def RF_records(token):
        records = []
        for family in families:
            for shot in range(11):
                paths = list((args.out/'workers').glob(
                    f'rank*/fine_RF/token{token:04d}/{family}_{shot:02d}.npy'))
                if len(paths) != 1:
                    raise RuntimeError('Incomplete original44 RF persistence')
                records.append({'family': family, 'shot': shot, 'path': str(paths[0].resolve()),
                                'SHA256': digest(paths[0])})
        return records

    def maps(folder):
        # Separate post-hoc process; no metrics or truth are returned to the optimizer.
        with (folder/'maps.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material',
                str(folder/'material.npy'), '--sample', SAMPLE, '--out', str(folder/'maps'),
                '--estimate-label', 'Continuing independent reference neural materials + native RF'],
                stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                check=True, timeout=120)

    try:
        pool = EncodedPhysicalNativePool(args, monitor, plan)
        run['status'] = 'reading_restored_original_all44_RF_baseline'
        atomic(args.out/'run.json', run)
        best_RF, _ = pool.evaluate(None, 'cached_forward', 'restored_original44_baseline')
        if abs(best_RF['loss']-prior['final_actual_native_RF_loss']) > 1e-10:
            raise RuntimeError('SHA-checked cached baseline changed')
        best_records = references
        best_TV = coordinates.regularization(best, .001)[0]
        run.update(campaign_initial_RF_loss=best_RF['loss'], campaign_initial_TV=best_TV)
        checkpoint(full_accepted=True)
        normalized_caps = np.array([.004, .006, .05])[:, None, None]/SCALES
        for block in range(1, args.blocks+1):
            if stop[0] or time.time() >= deadline-400:
                stop_reason = 'signal' if stop[0] else 'time_budget'
                break
            block_steps_accepted, block_steps_done = 0, 0
            for within in range(1, args.block_steps+1):
                if stop[0] or time.time() >= deadline-400:
                    break
                freeze()
                step = pool.counts['encoded_gradient_jobs']+1
                codes = {family: rng.choice([-1, 1], size=11).tolist() for family in families}
                run.update(status='actual_continuing_neural_native_gradient', active_block=block,
                    active_step=step, active_block_step=within, active_encoding=codes)
                atomic(args.out/'run.json', run)
                measured, gradient = pool.evaluate(coordinates.material(current), 'encoded_gradient',
                    f'step{step:03d}_actual_gradient', codes)
                block_steps_done += 1
                tv, tv_gradient = coordinates.regularization(current, .001)
                gradient += tv_gradient
                gradient[:, ~mask] = 0.
                model.zero_grad(set_to_none=True)
                control = model()
                state_difference = float(np.max(abs(control.detach().numpy()-current)))
                if state_difference > 2e-7:
                    raise RuntimeError('Neural and propagated control states differ')
                (control*torch.from_numpy(gradient)).sum().backward()
                snapshot = model.parameter_snapshot()
                direction, direction_info = optimizer.direction(model)
                with torch.no_grad():
                    old_raw = model.raw_control().numpy()
                    model.set_parameter_step(snapshot, direction, np.ones(3))
                    raw_difference = model.raw_control().numpy()-old_raw
                    restore_parameters(model, snapshot)
                    ratio = float(np.max(abs(raw_difference)/(normalized_caps*cap_factor)))
                    eta = min(1., .95/max(ratio, 1e-30))
                    candidate_ready = False
                    for _ in range(6):
                        model.set_parameter_step(snapshot, direction, np.full(3, eta))
                        proposal = model().numpy()
                        ratio = float(np.max(abs(proposal-current)/(normalized_caps*cap_factor)))
                        slope = float(np.sum(gradient*(proposal-current), dtype=np.float64))
                        if ratio <= 1.001 and slope < 0:
                            candidate_ready = True
                            break
                        eta *= min(.5, .95/max(ratio, 1e-30))
                    restore_parameters(model, snapshot)
                del control, raw_difference, old_raw
                lines, accepted, fraction = [], False, 1.
                if candidate_ready:
                    for line in range(1, 4):
                        if stop[0] or time.time() >= deadline-350:
                            break
                        with torch.no_grad():
                            model.set_parameter_step(snapshot, direction, np.full(3, eta), fraction)
                            proposal = model().numpy()
                        slope = float(np.sum(gradient*(proposal-current), dtype=np.float64))
                        if slope >= 0:
                            fraction *= .5
                            continue
                        run['status'] = 'same_encoding_continuing_neural_RF_search'
                        atomic(args.out/'run.json', run)
                        trial, _ = pool.evaluate(coordinates.material(proposal), 'encoded_forward',
                            f'step{step:03d}_line{line:02d}', codes)
                        proposed_tv = coordinates.regularization(proposal, .001)[0]
                        ok = trial['loss']+proposed_tv <= measured['loss']+tv+1e-4*slope
                        lines.append({'fraction': fraction, 'common_parameter_step': eta*fraction,
                            'actual_control_slope': slope, 'encoded_RF_before': measured['loss'],
                            'encoded_RF_after': trial['loss'], 'TV_after': proposed_tv,
                            'accepted': bool(ok), 'token': trial['token']})
                        if ok:
                            current, accepted = proposal.copy(), True
                            block_steps_accepted += 1
                            run['accepted_encoded_parameter_steps_including_later_rollbacks'] += 1
                            break
                        remainder = trial['loss']+proposed_tv-measured['loss']-tv-slope
                        fraction *= float(np.clip(-.5*slope/remainder if remainder > 0 else .5, .1, .5))
                if accepted:
                    failed_steps = 0
                else:
                    restore_parameters(model, snapshot)
                    cap_factor *= .5
                    failed_steps += 1
                atomic(args.out/'steps'/f'step{step:03d}.json', {'step': step, 'block': block,
                    'encoding': codes, 'CPU_descent_candidate_ready': candidate_ready,
                    'common_initial_parameter_step': eta, 'parameter_metric': direction_info,
                    'network_control_state_difference': state_difference, 'lines': lines,
                    'accepted_encoded_step': accepted, 'next_physical_cap_factor': cap_factor,
                    'original_full44_acceptance_pending': True, 'truth_read': False})
                checkpoint()
                del gradient, tv_gradient, direction, snapshot
                if failed_steps >= 3:
                    stop_reason = 'three_consecutive_no_accepted_descent_steps'
                    break
            if block_steps_done == 0:
                stop_reason = 'signal' if stop[0] else 'time_budget'
                break
            if not np.array_equal(current, best) and not stop[0] and time.time() < deadline-300:
                folder = args.out/'blocks'/f'block{block:02d}'
                folder.mkdir()
                array(folder/'material.npy', coordinates.material(current))
                array(folder/'control.npy', current)
                (folder/'stage0_initial_material.npy').symlink_to(initial_material)
                run.update(status='original_all44_continuing_neural_acceptance', active_block=block)
                atomic(args.out/'run.json', run)
                actual, _ = pool.evaluate(coordinates.material(current), 'forward',
                    f'block{block:02d}_original44', save_RF=True)
                proposed_tv = coordinates.regularization(current, .001)[0]
                accepted_full = bool(actual['loss'] < best_RF['loss'] and
                    actual['loss']+proposed_tv < best_RF['loss']+best_TV)
                records = RF_records(actual['token'])
                gain = float((best_RF['loss']-actual['loss'])/best_RF['loss'])
                atomic(folder/'actual_native_acceptance.json', {'accepted': accepted_full,
                    'block': block, 'steps_attempted': block_steps_done,
                    'encoded_steps_accepted': block_steps_accepted,
                    'RF_before': best_RF['loss'], 'actual_native_RF': actual,
                    'original_RF_relative_gain': gain, 'TV_before': best_TV, 'TV_after': proposed_tv,
                    'material_SHA256': digest(folder/'material.npy'), 'RF_source_records': records,
                    'patient_truth_read': False, 'joint_imaging_success_claimed': False})
                save_checkpoint(folder/'network.pt', accepted_full)
                maps(folder)
                run['blocks_completed'] += 1
                if accepted_full:
                    best, best_RF, best_TV, best_records = current.copy(), actual, proposed_tv, records
                    best_parameters, best_optimizer = model.parameter_snapshot(), optimizer.state_dict()
                    best_folder = folder
                    rejected_blocks = 0
                    small_gain_blocks = small_gain_blocks+1 if gain < .01 else 0
                    run['blocks_accepted'] += 1
                    run['full_accepted_path_neural_steps'] += block_steps_accepted
                    run['last_full_accepted_RF_loss'] = best_RF['loss']
                    run['last_full_accepted_material_SHA256'] = digest(folder/'material.npy')
                    checkpoint(full_accepted=True)
                else:
                    restore_parameters(model, best_parameters)
                    optimizer.load_state_dict(best_optimizer)
                    current = best.copy()
                    cap_factor *= .5
                    rejected_blocks += 1
                    small_gain_blocks = 0
                    checkpoint(full_accepted=True)
                run.update(consecutive_rejected_blocks=rejected_blocks,
                    consecutive_small_gain_accepted_blocks=small_gain_blocks)
                atomic(args.out/'run.json', run)
                print(json.dumps({'block': block, 'full44_accepted': accepted_full,
                    'original_RF_loss': actual['loss'], 'RF_relative_gain': gain,
                    'best_RF_loss': best_RF['loss'], 'completed_G': pool.counts['encoded_gradient_jobs'],
                    'seconds': time.time()-started}), flush=True)
                if rejected_blocks >= 3:
                    stop_reason = 'three_consecutive_rejected_full44_blocks'
                if small_gain_blocks >= 3:
                    stop_reason = 'three_consecutive_accepted_blocks_below_one_percent_RF_gain'
            else:
                restore_parameters(model, best_parameters)
                optimizer.load_state_dict(best_optimizer)
                current = best.copy()
                checkpoint(full_accepted=True)
                stop_reason = stop_reason or ('signal' if stop[0] else 'no_changed_block_or_time_budget')
            if stop[0] or stop_reason:
                stop_reason = stop_reason or 'signal'
                break
        stop_reason = stop_reason or 'maximum_additional_gradient_budget'
        restore_parameters(model, best_parameters)
        optimizer.load_state_dict(best_optimizer)
        current = best.copy()
        checkpoint(full_accepted=True)
        final = args.out/'final'
        (final/'material.npy').symlink_to((best_folder/'material.npy').resolve())
        (final/'stage0_initial_material.npy').symlink_to(initial_material)
        for suffix in ['.png', '.metrics.json']:
            shutil.copy2(best_folder/('maps'+suffix), final/('maps'+suffix))
        atomic(final/'actual_native_acceptance.json', {'accepted': run['blocks_accepted'] > 0,
            'material_SHA256': digest(final/'material.npy'), 'actual_native_RF': best_RF,
            'native_TV': best_TV, 'RF_source_records': best_records,
            'full_original44_source_acceptance': True, 'joint_imaging_success_claimed': False})
        run.update(status='reference_neural_convergence_campaign_finished', stop_reason=stop_reason,
            finished_unix_s=time.time(), seconds=time.time()-started, **pool.counts,
            final_actual_native_RF_loss=best_RF['loss'], material_SHA256=digest(final/'material.npy'),
            checkpoint_SHA256=digest(args.artifact_root/'checkpoint.pt'),
            best_checkpoint_SHA256=digest(args.artifact_root/'best_checkpoint.pt'),
            separate_CPU_post_hoc_maps_saved=True, actual_image_view_still_required=True,
            stopped_signal=stop[0], old_budgets_not_reset=True)
        atomic(args.out/'run.json', run)
        print(json.dumps({k: run[k] for k in ['status', 'stop_reason', 'seconds',
            'encoded_gradient_jobs', 'blocks_completed', 'blocks_accepted',
            'final_actual_native_RF_loss', 'material_SHA256']}), flush=True)
    except BaseException:
        run.update(status='reference_neural_campaign_failed_for_inspection',
            error=traceback.format_exc(), stopped_unix_s=time.time())
        if pool is not None:
            run.update(**pool.counts, issued_propagations=dict(pool.attempted))
        atomic(args.out/'run.json', run)
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__ == '__main__':
    main()
