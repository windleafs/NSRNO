"""Independent CPU neural materials, actual native encoded RF, and full44 acceptance."""
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
import numpy as np
import torch
from .encoded_physical_native_pool import EncodedPhysicalNativePool
from .independent_physical_coordinates import IndependentPhysicalCoordinates, SCALES
from .independent_neural_material_fields import IndependentNeuralMaterialFields
from .low_frequency_0p5 import atomic, digest, SAMPLE, CASE
from .pressure_resources import PressureResources
from .reconstruct_frequency_multilevel import array
from .reconstruct_native_frequency_incremental import same_process
from .reconstruct_native_gn import source_manifest


UNIFORM_SHA256 = '7d8d47327648a6313fc8f3914a8429a00fa4b6efc6916274add97eb45225f459'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--uniform-seed', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--previous-final-maps-reviewed', action='store_true')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--max-seconds', type=float, default=1200.)
    parser.add_argument('--seed', type=int, default=8039)
    parser.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--cap-gib', type=float, default=10.)
    parser.add_argument('--chunk', type=int, default=128)
    args = parser.parse_args()
    if (not args.previous_final_maps_reviewed or not 1<=args.steps<=4 or
        not 600<=args.max_seconds<=1200 or not 0<args.cap_gib<=10 or
        not 1<=len(args.devices)<=3 or len(set(args.devices))!=len(args.devices)):
        raise ValueError('Actual map review and fixed neural-material phase budget required')
    project = Path.cwd().resolve()
    for name in ['parent_root', 'uniform_seed', 'out', 'artifact_root']:
        setattr(args, name, getattr(args, name).resolve())
    if (not all(p.is_relative_to(project/'out/v5') for p in
                [args.parent_root, args.uniform_seed, args.out]) or
        not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts')):
        raise ValueError('Independent nsrno and same-account dedicated artifact volume only')
    if args.out.exists() or args.artifact_root.exists():
        raise RuntimeError('Inspect same existing run; do not reset or overwrite')
    prior = json.loads((args.parent_root/'run.json').read_text())
    if (prior['status']!='encoded_independent_physical_interpolated_stage_finished' or
        same_process(prior) or prior['sample']!=SAMPLE or prior['case']!=CASE):
        raise RuntimeError('Require completed authorized same-case physical predecessor')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                               '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Another GPU experiment is active')
    qualification = project/'out/v5/neural_material_CPU_qualification_v1/result.json'
    qualified = json.loads(qualification.read_text())
    if qualified['status']!='CPU_neural_material_shape_projection_chain_passed':
        raise RuntimeError('The one necessary CPU qualification has not passed')
    if digest(args.uniform_seed)!=UNIFORM_SHA256:
        raise ValueError('Only the saved known uniform initial material, never development truth')
    parent_plan = json.loads((args.parent_root/'native_plan.json').read_text())
    physical_names = ['native_time.py', 'native_time.cu', 'time_acquisition.py',
        'frequency_multilevel_acquisition.py', 'full90_3mhz_acquisition.py',
        'acquisition_single_case_0p5.py', 'low_frequency_0p5.py', 'matrix_free_rf_response.py',
        'encoded_native_acquisition.py', 'encoded_physical_native_worker.py',
        'encoded_physical_native_pool.py', 'independent_physical_coordinates.py']
    for name in physical_names:
        if prior['source_SHA256'].get(name)!=digest(Path(__file__).parent/name):
            raise ValueError('Original physical/receiver/coordinate implementation changed: '+name)
    if (digest(Path(parent_plan['data_root'])/'index.json')!=parent_plan['data_index_SHA256'] or
        digest(parent_plan['high_index'])!=parent_plan['high_index_SHA256']):
        raise ValueError('Authorized fixed acquisition changed')
    families = parent_plan['families']
    if (set(families)!={'lf', 'point3', 'lf3', 'point3_3mhz'} or
        any(len(spec['bands_hz'])!=4 for spec in families.values()) or
        families['lf3']['bands_hz'][-1][1]!=4.5e6):
        raise ValueError('All original44 sources and complete valid dense bands required')
    edge_path = Path('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    if digest(edge_path)!=parent_plan['known_edge_SHA256']:
        raise ValueError('Known fixed absorption changed')
    reserve = 2*2**30+400*2**20
    if (shutil.disk_usage(project).free<reserve+5*2**20 or
        shutil.disk_usage(Path.home()).free<reserve+400*2**20):
        raise RuntimeError('Retain original disk reserve without deleting unique data')
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:')))
    if available<16*2**20:
        raise RuntimeError('Reserve host neural activations and original native workers')
    torch.set_num_threads(8)
    coordinates = IndependentPhysicalCoordinates(np.load(args.uniform_seed), np.load(edge_path))
    z, x = np.ogrid[:961, :1121]
    mask = (z>=80)&(z<937)&(x>=24)&(x<1097)
    lateral = np.abs((np.arange(1121)-560)*50e-6)<=.0191+1e-12
    model = IndependentNeuralMaterialFields(coordinates, mask, args.seed)
    with torch.no_grad():
        current = model().numpy()
    if np.any(current!=0):
        raise RuntimeError('Neural zero-head state must equal the exact uniform physical initial')
    args.out.mkdir()
    args.artifact_root.mkdir(parents=True)
    for name in ['jobs', 'steps']:
        (args.out/name).mkdir()
    for name in ['workers', 'candidate', 'final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name, target_is_directory=True)
    for name in ['latest_control.npy', 'latest_material.npy', 'checkpoint.pt']:
        (args.out/name).symlink_to(args.artifact_root/name)
    plan = {**parent_plan, 'reference_RF':[], 'native_anchor_material':str(args.uniform_seed),
        'native_anchor_material_SHA256':UNIFORM_SHA256, 'truth_read':False,
        'new_original44_baseline_required':True}
    atomic(args.out/'native_plan.json', plan)
    hashes = source_manifest()
    for name in physical_names+[Path(__file__).name, 'independent_neural_material_fields.py',
        'qualify_neural_material_CPU.py', 'reconstruct_frequency_multilevel.py',
        'reconstruct_native_frequency_incremental.py']:
        hashes[name] = digest(Path(__file__).parent/name)
    for name in hashes:
        shutil.copy2(Path(__file__).parent/name, args.out/name)
    started, stop = time.time(), [0]
    deadline = started+args.max_seconds
    run = {'status':'initializing_independent_neural_material', 'pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s':started, 'deadline_unix_s':deadline, 'sample':SAMPLE, 'case':CASE,
        'parent_root':str(args.parent_root), 'artifact_root':str(args.artifact_root),
        'source_SHA256':hashes, 'families':families, 'seed':args.seed,
        'native_anchor_material_SHA256':UNIFORM_SHA256, 'new_original44_baseline_required':True,
        'previous_best_physical_RF':prior['final_actual_native_RF_loss'],
        'previous_best_material_SHA256':prior['material_SHA256'],
        'maximum_encoded_gradient_jobs':args.steps, 'maximum_encoded_adjoint_propagations':4*args.steps,
        'maximum_encoded_lines_per_step':3, 'maximum_original_full44_checks':2,
        'accepted_neural_parameter_steps':0, 'allocator_cap_GiB':args.cap_gib,
        'per_card_process_tree_limit_GiB':16, 'devices':args.devices,
        'neural_config':model.config(), 'neural_network_device':'CPU',
        'CPU_qualification_result_SHA256':digest(qualification),
        'network_control_state_tolerance':2e-7,
        'uniform_initial_contains_no_development_anatomy':True,
        'not_a_neural_forward_surrogate_success_claim':True, 'TV_weight':.001,
        'physical_log_step_caps':[.004, .006, .05],
        'new_observations_or_cases':False, 'patient_truth_read_by_optimizer':False,
        'reserved_case_unread':'Neg_47_Left', 'new_material_derivative_probes':0,
        'prior_phase_and_ancestry_counters':{k:prior.get(k) for k in [
            'same_original_phase_completed_work', 'prior_phase_work_before_line_repair',
            'parent_native_work_counters', 'parent_earlier_counters_as_recorded']},
        'joint_imaging_success_claimed':False}
    atomic(args.out/'run.json', run)
    monitor = PressureResources(args.out)
    pool = None
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for value in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(value, lambda number, frame:stop.__setitem__(0, int(number)))
    rng = np.random.default_rng(args.seed)
    best = current.copy()
    best_parameters = model.parameter_snapshot()
    accepted_final = False

    def freeze():
        if any(digest(Path(__file__).parent/name)!=sha for name, sha in hashes.items()):
            raise RuntimeError('Frozen executable changed')
        if digest(args.uniform_seed)!=UNIFORM_SHA256:
            raise RuntimeError('Immutable uniform initial changed')

    def checkpoint():
        array(args.artifact_root/'latest_control.npy', current)
        array(args.artifact_root/'latest_material.npy', coordinates.material(current))
        state = {k:v.detach().cpu() for k,v in model.state_dict().items()
                 if k.startswith('fields.') or k=='depth_scale'}
        temporary = args.artifact_root/'checkpoint.partial.pt'
        torch.save({'neural_state':state, 'depth_frozen':model.depth_frozen,
            'uniform_initial_SHA256':UNIFORM_SHA256, 'source_SHA256':hashes,
            'control_material_SHA256':digest(args.artifact_root/'latest_material.npy'),
            'sampler_state':rng.bit_generator.state}, temporary)
        temporary.replace(args.artifact_root/'checkpoint.pt')
        run.update(**pool.counts, issued_propagations=dict(pool.attempted),
            sampler_state=rng.bit_generator.state,
            coordinator_peak_host_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        atomic(args.out/'run.json', run)

    def RF_records(token):
        records = []
        for family in families:
            for shot in range(11):
                paths = list((args.out/'workers').glob(
                    f'rank*/fine_RF/token{token:04d}/{family}_{shot:02d}.npy'))
                if len(paths)!=1:
                    raise RuntimeError('Missing original source RF after actual full44 evaluation')
                records.append({'family':family, 'shot':shot, 'path':str(paths[0].resolve()),
                                'SHA256':digest(paths[0])})
        return records

    def maps(folder):
        with (folder/'maps.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material',
                str(folder/'material.npy'), '--sample', SAMPLE, '--out', str(folder/'maps'),
                '--estimate-label', 'Independent CPU neural materials + native physical RF'],
                stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES':''},
                check=True, timeout=120)

    try:
        pool = EncodedPhysicalNativePool(args, monitor, plan)
        run['status'] = 'measuring_uniform_neural_initial_original_all44_RF'
        atomic(args.out/'run.json', run)
        best_RF, _ = pool.evaluate(coordinates.material(best), 'forward',
                                  'uniform_neural_initial_original44', save_RF=True)
        best_records = RF_records(best_RF['token'])
        best_TV = coordinates.regularization(best, .001)[0]
        run.update(native_anchor_RF_loss=best_RF['loss'], native_anchor_TV=best_TV)
        checkpoint()
        normalized_caps = np.array([.004, .006, .05])[:, None, None]/SCALES
        cap_factor = 1.
        for step in range(args.steps):
            if stop[0] or time.time()>=deadline-350:
                break
            freeze()
            codes = {family:rng.choice([-1, 1], size=11).tolist() for family in families}
            run.update(status='actual_neural_material_native_gradient', active_step=step+1,
                       active_encoding=codes)
            atomic(args.out/'run.json', run)
            measured, gradient = pool.evaluate(coordinates.material(current), 'encoded_gradient',
                                              f'step{step+1:02d}_actual_gradient', codes)
            if not model.depth_frozen:
                run['fixed_depth_metric'] = model.freeze_depth_metric(gradient, lateral)
            tv, tv_gradient = coordinates.regularization(current, .001)
            gradient += tv_gradient
            gradient[:, ~mask] = 0.
            model.zero_grad(set_to_none=True)
            control = model()
            state_difference = float(np.max(abs(control.detach().numpy()-current)))
            if state_difference>2e-7:
                raise RuntimeError('Network state no longer matches actual propagated material')
            (control*torch.from_numpy(gradient)).sum().backward()
            snapshot = model.parameter_snapshot()
            direction = model.unit_negative_parameter_gradient()
            field_steps = np.ones(3, np.float64)
            with torch.no_grad():
                model.set_parameter_step(snapshot, direction, field_steps)
                raw = model.raw_control().numpy()
                model.set_parameter_step(snapshot, direction, field_steps, fraction=0.)
                old_raw = model.raw_control().numpy()
                maxima = np.max(np.abs(raw-old_raw), axis=(1, 2))
                for j in range(3):
                    field_steps[j] *= min(1., float(normalized_caps[j, 0, 0]*cap_factor)/
                                         max(float(maxima[j]), 1e-30))
                cpu_scale = 1.
                candidate_ready = False
                for _ in range(6):
                    model.set_parameter_step(snapshot, direction, field_steps, cpu_scale)
                    proposal = model().numpy()
                    ratio = float(np.max(np.abs(proposal-current)/(normalized_caps*cap_factor)))
                    slope = float(np.sum(gradient*(proposal-current), dtype=np.float64))
                    if ratio<=1.001 and slope<0:
                        candidate_ready = True
                        field_steps *= cpu_scale
                        break
                    cpu_scale *= min(.5, .95/max(ratio, 1e-30))
                model.set_parameter_step(snapshot, direction, field_steps, fraction=0.)
            del control
            lines, accepted = [], False
            fraction = 1.
            if candidate_ready:
                for line in range(1, 4):
                    if stop[0] or time.time()>=deadline-180:
                        break
                    with torch.no_grad():
                        model.set_parameter_step(snapshot, direction, field_steps, fraction)
                        proposal = model().numpy()
                    slope = float(np.sum(gradient*(proposal-current), dtype=np.float64))
                    if slope>=0:
                        fraction *= .5
                        continue
                    run['status'] = 'same_encoding_actual_neural_parameter_RF_search'
                    atomic(args.out/'run.json', run)
                    trial, _ = pool.evaluate(coordinates.material(proposal), 'encoded_forward',
                        f'step{step+1:02d}_line{line:02d}', codes)
                    proposed_tv = coordinates.regularization(proposal, .001)[0]
                    ok = trial['loss']+proposed_tv<=measured['loss']+tv+1e-4*slope
                    lines.append({'fraction':fraction, 'actual_control_slope':slope,
                        'RF_before':measured['loss'], 'encoded_RF_after':trial['loss'],
                        'TV_after':proposed_tv, 'accepted':bool(ok), 'token':trial['token']})
                    if ok:
                        current, accepted = proposal, True
                        run['accepted_neural_parameter_steps'] += 1
                        break
                    remainder = trial['loss']+proposed_tv-measured['loss']-tv-slope
                    fraction *= float(np.clip(-.5*slope/remainder if remainder>0 else .5, .1, .5))
            if not accepted:
                model.set_parameter_step(snapshot, direction, field_steps, fraction=0.)
                cap_factor *= .5
            atomic(args.out/'steps'/f'step{step+1:02d}.json', {'step':step+1,
                'encoding':codes, 'candidate_ready_after_CPU_parameter_scaling':candidate_ready,
                'neural_parameter_field_steps':field_steps.tolist(), 'CPU_scale':cpu_scale,
                'network_control_state_difference':state_difference,
                'lines':lines, 'accepted':accepted, 'next_physical_cap_factor':cap_factor,
                'original_full44_acceptance_still_pending':True, 'truth_read':False})
            checkpoint()
            del gradient, tv_gradient, direction, snapshot
        if not np.array_equal(current, best) and not stop[0] and time.time()<deadline-150:
            folder = args.out/'candidate'
            array(folder/'material.npy', coordinates.material(current))
            array(folder/'control.npy', current)
            (folder/'stage0_initial_material.npy').symlink_to(args.uniform_seed)
            run['status'] = 'original_all44_neural_material_acceptance'
            atomic(args.out/'run.json', run)
            actual, _ = pool.evaluate(coordinates.material(current), 'forward',
                                     'neural_candidate_original44', save_RF=True)
            proposed_tv = coordinates.regularization(current, .001)[0]
            accepted_final = bool(actual['loss']<best_RF['loss'] and
                actual['loss']+proposed_tv<best_RF['loss']+best_TV)
            records = RF_records(actual['token'])
            atomic(folder/'actual_native_acceptance.json', {'accepted':accepted_final,
                'RF_before':best_RF['loss'], 'actual_native_RF':actual,
                'TV_before':best_TV, 'TV_after':proposed_tv,
                'material_SHA256':digest(folder/'material.npy'), 'RF_source_records':records,
                'patient_truth_read':False, 'joint_imaging_success_claimed':False})
            # Fixed candidate post-hoc maps are informative even if fullRF rejects it.
            maps(folder)
            if accepted_final:
                best, best_RF, best_TV, best_records = current.copy(), actual, proposed_tv, records
                best_parameters = model.parameter_snapshot()
        final = args.out/'final'
        if accepted_final:
            os.link(args.out/'candidate/material.npy', final/'material.npy')
            for suffix in ['.png', '.metrics.json']:
                shutil.copy2(args.out/'candidate'/('maps'+suffix), final/('maps'+suffix))
        else:
            (final/'material.npy').symlink_to(args.uniform_seed)
        (final/'stage0_initial_material.npy').symlink_to(args.uniform_seed)
        atomic(final/'actual_native_acceptance.json', {'accepted':accepted_final,
            'material_SHA256':digest(final/'material.npy'), 'actual_native_RF':best_RF,
            'native_TV':best_TV, 'RF_source_records':best_records,
            'full_original44_source_acceptance':True, 'joint_imaging_success_claimed':False})
        model.set_parameter_step(best_parameters, [[torch.zeros_like(p) for p in f.parameters()]
            for f in model.fields], np.zeros(3))
        current = best
        checkpoint()
        if not accepted_final:
            maps(final)
        run.update(status='independent_neural_material_stage_finished', finished_unix_s=time.time(),
            seconds=time.time()-started, **pool.counts, native_candidate_accepted=accepted_final,
            final_actual_native_RF_loss=best_RF['loss'], material_SHA256=digest(final/'material.npy'),
            checkpoint_SHA256=digest(args.artifact_root/'checkpoint.pt'),
            separate_CPU_post_hoc_maps_saved=True, actual_image_view_still_required=True,
            stopped_signal=stop[0], old_budgets_not_reset=True)
        atomic(args.out/'run.json', run)
        print(json.dumps({k:run[k] for k in ['status', 'seconds', 'encoded_gradient_jobs',
            'encoded_adjoint_propagations', 'original_full_RF_checks',
            'accepted_neural_parameter_steps', 'native_candidate_accepted',
            'final_actual_native_RF_loss', 'material_SHA256']}), flush=True)
    except BaseException:
        run.update(status='neural_material_failed_for_inspection', error=traceback.format_exc(),
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
