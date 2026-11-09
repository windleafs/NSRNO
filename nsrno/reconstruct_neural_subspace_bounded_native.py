"""Bounded coupled neural GN with exact reuse of the saved first inverse model."""
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
from .neural_subspace_native_pool import NeuralSubspaceNativePool
from .neural_subspace_coordinates import NeuralParameterChart, physical_TV_hessian
from .bounded_neural_subspace import solve_bounded_subspace
from .neural_subspace_saved_model import load_saved_step, matches_saved_state
from .independent_physical_coordinates import IndependentPhysicalCoordinates, SCALES
from .reference_neural_material_fields import ReferenceNeuralMaterialFields
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
    parser.add_argument('--outer-steps', type=int, default=6)
    parser.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--cap-gib', type=float, default=10.)
    parser.add_argument('--chunk', type=int, default=128)
    args = parser.parse_args()
    if (not args.previous_final_maps_reviewed or not 1 <= args.outer_steps <= 8 or
        not 0 < args.cap_gib <= 10 or not 1 <= len(args.devices) <= 3 or
        len(set(args.devices)) != len(args.devices)):
        raise ValueError('Reviewed neural continuation with fixed resource limits required')
    project = Path.cwd().resolve()
    for name in ['parent_root', 'out', 'artifact_root']:
        setattr(args, name, getattr(args, name).resolve())
    if (not all(p.is_relative_to(project/'out/v5') for p in [args.parent_root, args.out]) or
        not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts')):
        raise ValueError('Independent project and same-account dedicated artifact volume only')
    if args.out.exists() or args.artifact_root.exists():
        raise RuntimeError('Inspect same existing run; do not reset work')
    prior = json.loads((args.parent_root/'run.json').read_text())
    if (prior['status'] != 'neural_subspace_native_continuation_finished' or same_process(prior)
            or prior['sample'] != SAMPLE or prior['case'] != CASE):
        raise RuntimeError('Finish and review the same neural campaign predecessor')
    old_workers = json.loads((args.parent_root/'worker_processes.json').read_text())
    if any(Path(f'/proc/{pid}').exists() for pid in old_workers['worker_pids']):
        raise RuntimeError('Owned predecessor workers have not exited')
    # Shared-server foreign jobs are handled by PressureResources memory admission.
    # Their presence alone is not permission to stop them or a reason to duplicate ours.
    deadline = prior['deadline_unix_s']
    total_limit = prior['original_campaign_work_unit_limit']
    previous_units = prior['total_original_campaign_work_units']
    original_start = prior['original_campaign_created_unix_s']
    ancestor, seen, old_original_checks = args.parent_root, set(), 0
    while True:
        if ancestor in seen:
            raise RuntimeError('Cyclic inverse lineage')
        seen.add(ancestor)
        record = json.loads((ancestor/'run.json').read_text())
        if record.get('original_forward_propagations', 0) % 44:
            raise RuntimeError('Inspect incomplete original-source lineage work')
        old_original_checks += record.get('original_forward_propagations', 0)//44
        if abs(record['created_unix_s']-original_start) < 1e-6:
            original_check_limit = record['maximum_original_full44_checks']
            break
        ancestor = Path(record['parent_root'])
    remaining_checks = original_check_limit-old_original_checks
    maximum_outer = min(args.outer_steps, remaining_checks, 1+(total_limit-previous_units)//7)
    CPU_solution_root = project/'out/v5/neural_subspace_bounded_saved_model_CPU_v1'
    CPU_solution = json.loads((CPU_solution_root/'result.json').read_text())
    if CPU_solution['source_root'] != str(args.parent_root) or CPU_solution['source_step'] != 1:
        raise RuntimeError('Saved CPU solution belongs to another actual inverse model')
    if CPU_solution['code_SHA256'] != digest(Path(__file__).parent/'bounded_neural_subspace.py'):
        raise RuntimeError('Saved bounded-model solver changed')
    for name, expected in CPU_solution['source_model_SHA256'].items():
        if digest(args.parent_root/name) != expected:
            raise RuntimeError('Saved native first-step model changed')
    if maximum_outer < 1 or time.time() >= deadline-1200:
        raise RuntimeError('Original campaign time/work budget has insufficient remaining capacity')
    parent_plan = json.loads((args.parent_root/'native_plan.json').read_text())
    initial_material = (args.parent_root/'final/material.npy').resolve()
    anchor = Path(parent_plan['native_anchor_material']).resolve()
    qualification = project/'out/v5/neural_subspace_CPU_qualification_v1/result.json'
    qualified = json.loads(qualification.read_text())
    if qualified['status'] != 'CPU_flat_neural_JVP_VJP_and_independent_TV_Hessian_passed':
        raise RuntimeError('The one necessary new CPU chain qualification has not passed')
    for name, expected in qualified['source_SHA256'].items():
        if digest(Path(__file__).parent/name) != expected:
            raise RuntimeError('Qualified CPU coordinate implementation changed')
    if (digest(initial_material) != prior['material_SHA256'] or
        digest(anchor) != prior['native_anchor_material_SHA256'] or
        digest(args.parent_root/'checkpoint.pt') != prior['checkpoint_SHA256']):
        raise RuntimeError('Accepted neural state or immutable physical coordinate anchor changed')
    original_names = ['native_time.py', 'native_time.cu', 'native_tangent.py', 'native_tangent.cu',
        'time_acquisition.py', 'frequency_multilevel_acquisition.py', 'full90_3mhz_acquisition.py',
        'acquisition_single_case_0p5.py', 'low_frequency_0p5.py', 'matrix_free_rf_response.py',
        'encoded_native_acquisition.py', 'encoded_physical_native_worker.py',
        'encoded_physical_native_pool.py', 'independent_physical_coordinates.py',
        'independent_neural_material_fields.py', 'reference_neural_material_fields.py']
    for name in original_names:
        if prior['source_SHA256'].get(name) != digest(Path(__file__).parent/name):
            raise RuntimeError('Original physical/neural operator changed: '+name)
    if (digest(Path(parent_plan['data_root'])/'index.json') != parent_plan['data_index_SHA256'] or
        digest(parent_plan['high_index']) != parent_plan['high_index_SHA256']):
        raise RuntimeError('Authorized observations changed')
    families = parent_plan['families']
    if (set(families) != {'lf', 'point3', 'lf3', 'point3_3mhz'} or
        any(len(v['bands_hz']) != 4 for v in families.values()) or
        families['lf3']['bands_hz'][-1][1] != 4.5e6):
        raise RuntimeError('All44 sources and complete dense valid bands required')
    references = native_RF_records(args.parent_root, prior, families)
    edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    if digest(edge_path) != parent_plan['known_edge_SHA256']:
        raise RuntimeError('Known fixed absorption changed')
    reserve = 2*2**30+400*2**20
    if (shutil.disk_usage(project).free < reserve+5*2**20 or
        shutil.disk_usage(Path.home()).free < reserve+maximum_outer*400*2**20):
        raise RuntimeError('Retain data reserve and saved actual inverse products')
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:')))
    if available < 32*2**20:
        raise RuntimeError('Reserve host checkpoints and CPU neural activations')
    torch.set_num_threads(8)
    coordinates = IndependentPhysicalCoordinates(np.load(anchor), np.load(edge_path))
    z, x = np.ogrid[:961, :1121]
    mask = (z >= 80) & (z < 937) & (x >= 24) & (x < 1097)
    model = ReferenceNeuralMaterialFields(coordinates, mask, prior['seed'])
    inherited = torch.load(args.parent_root/'checkpoint.pt', map_location='cpu', weights_only=False)
    if (not inherited['full_original44_accepted'] or not inherited['depth_frozen'] or
        inherited['initial_material_SHA256'] != prior['native_anchor_material_SHA256'] or
        inherited['control_material_SHA256'] != prior['material_SHA256']):
        raise RuntimeError('Require the final original44-accepted neural checkpoint')
    missing, unexpected = model.load_state_dict(inherited['neural_state'], strict=False)
    if set(missing) != {'absolute', 'active'} or unexpected:
        raise RuntimeError('Incomplete original neural state')
    model.depth_frozen = True
    with torch.no_grad():
        current = model().numpy().copy()
    material = coordinates.material(current)
    buffer = io.BytesIO()
    np.save(buffer, material)
    restored_sha = hashlib.sha256(buffer.getvalue()).hexdigest()
    if restored_sha != prior['material_SHA256'] or not np.array_equal(material, np.load(initial_material)):
        raise RuntimeError('Restored neural weights no longer produce the accepted physical bytes')
    del buffer, material
    rng = np.random.default_rng()
    rng.bit_generator.state = inherited['sampler_state']
    inherited_optimizer = inherited.get('optimizer', inherited.get('inherited_first_order_optimizer'))
    inherited_cap = float(inherited.get('next_cap_factor', 1.))
    del inherited
    chart = NeuralParameterChart(model)
    args.out.mkdir()
    args.artifact_root.mkdir(parents=True)
    for name in ['jobs', 'steps']:
        (args.out/name).mkdir()
    for name in ['workers', 'candidates', 'final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name, target_is_directory=True)
    for name in ['latest_control.npy', 'latest_material.npy', 'checkpoint.pt', 'best_checkpoint.pt']:
        (args.out/name).symlink_to(args.artifact_root/name)
    plan = {**parent_plan, 'reference_RF': references,
        'neural_subspace_initial_material': str(initial_material),
        'neural_subspace_initial_material_SHA256': restored_sha, 'truth_read': False,
        'new_original44_baseline_required': False}
    atomic(args.out/'native_plan.json', plan)
    hashes = source_manifest()
    for name in original_names+[Path(__file__).name, 'neural_subspace_native_worker.py',
        'neural_subspace_native_pool.py', 'neural_subspace_coordinates.py',
        'qualify_neural_subspace_CPU.py', 'bounded_neural_subspace.py', 'neural_subspace_saved_model.py',
        'reconstruct_frequency_multilevel.py',
        'reconstruct_native_frequency_incremental.py']:
        hashes[name] = digest(Path(__file__).parent/name)
    for name in hashes:
        shutil.copy2(Path(__file__).parent/name, args.out/name)
    started, stop = time.time(), [0]
    run = {'status': 'initializing_bounded_same_neural_subspace_continuation', 'pid': os.getpid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s': started, 'original_campaign_created_unix_s': original_start,
        'deadline_unix_s': deadline, 'sample': SAMPLE, 'case': CASE, 'seed': prior['seed'],
        'parent_root': str(args.parent_root), 'artifact_root': str(args.artifact_root),
        'source_SHA256': hashes, 'families': families, 'devices': args.devices,
        'native_anchor_material_SHA256': prior['native_anchor_material_SHA256'],
        'initial_accepted_material_SHA256': restored_sha,
        'resumed_checkpoint_SHA256': prior['checkpoint_SHA256'],
        'CPU_chain_qualification_SHA256': digest(qualification),
        'fixed_noise_reference_depth_and_source_rng_restored': True,
        'new_original44_baseline_required': False, 'original_campaign_work_unit_limit': total_limit,
        'previous_campaign_work_units': previous_units,
        'maximum_new_outer_steps': maximum_outer, 'maximum_basis_dimension': 6,
        'original_campaign_original44_check_limit': original_check_limit,
        'previous_campaign_original44_checks': old_original_checks,
        'maximum_new_original44_checks': remaining_checks,
        'reuse_saved_first_model': True, 'reused_native_GN_products': 0,
        'reused_gradient_only_jobs': 0,
        'HVP_work_units_each_at_least_one': True,
        'prior_solver_complete_work': {k: prior.get(k) for k in [
            'encoded_gradient_jobs', 'encoded_adjoint_propagations', 'encoded_forward_propagations',
            'original_forward_propagations', 'encoded_hvp_jobs', 'encoded_tangent_propagations',
            'encoded_hvp_adjoint_propagations', 'total_original_campaign_work_units',
            'issued_propagations', 'seconds',
            'accepted_encoded_parameter_steps_including_later_rollbacks', 'full_accepted_path_neural_steps']},
        'original_ancestry_remains_in_parent_not_reset': True,
        'accepted_original44_steps': 0, 'encoded_steps_accepted_including_full44_rejections': 0,
        'TV_weight': .001, 'physical_log_step_caps': [.004, .006, .05],
        'allocator_cap_GiB': args.cap_gib, 'per_card_process_tree_limit_GiB': 16,
        'neural_network_device': 'CPU', 'neural_config': model.config(),
        'patient_truth_read_by_optimizer': False, 'reserved_case_unread': 'Neg_47_Left',
        'new_data_or_frequency': False, 'new_patient_derivative_validation_probes': 0,
        'joint_imaging_success_claimed': False}
    atomic(args.out/'run.json', run)
    monitor, pool = PressureResources(args.out), None
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for number in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(number, lambda value, frame: stop.__setitem__(0, int(value)))
    best, best_vector = current.copy(), chart.vector()
    best_folder = args.parent_root/'final'
    cap_factor = min(1., inherited_cap)
    gradient_only_cache_used = False
    rejections = int(prior.get('consecutive_rejected_blocks', 0))
    small_gains = int(prior.get('consecutive_small_gain_accepted_blocks', 0))
    failed_steps = 0
    stop_reason = None

    def freeze():
        if any(digest(Path(__file__).parent/name) != expected for name, expected in hashes.items()):
            raise RuntimeError('Frozen executable changed')
        if digest(anchor) != prior['native_anchor_material_SHA256']:
            raise RuntimeError('Immutable original physical coordinate anchor changed')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Original disk reserve reached')

    def save_network(path, full_accepted):
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()
                 if k.startswith('fields.') or k == 'depth_scale'}
        temporary = path.with_suffix('.partial.pt')
        torch.save({'neural_state': state, 'depth_frozen': True,
            'initial_material_SHA256': prior['native_anchor_material_SHA256'],
            'control_material_SHA256': digest(args.artifact_root/'latest_material.npy'),
            'source_SHA256': hashes, 'sampler_state': rng.bit_generator.state,
            'inherited_first_order_optimizer': inherited_optimizer,
            'full_original44_accepted': full_accepted, 'new_solver_counts': dict(pool.counts),
            'previous_campaign_work_units': previous_units,
            'original_deadline_unix_s': deadline, 'next_cap_factor': cap_factor}, temporary)
        temporary.replace(path)

    def checkpoint(full_accepted=False):
        array(args.artifact_root/'latest_control.npy', current)
        array(args.artifact_root/'latest_material.npy', coordinates.material(current))
        save_network(args.artifact_root/'checkpoint.pt', full_accepted)
        if full_accepted:
            save_network(args.artifact_root/'best_checkpoint.pt', True)
        worker_counts = {}
        for file in (args.out/'workers').glob('rank*/completed_propagations.json'):
            d = json.loads(file.read_text())
            for key, value in d.items():
                if key.endswith('_propagations'):
                    worker_counts[key] = worker_counts.get(key, 0)+value
        run.update(**pool.counts, issued_propagations=dict(pool.attempted),
            worker_completed_source_counts=worker_counts, sampler_state=rng.bit_generator.state,
            total_original_campaign_original44_checks=old_original_checks+pool.counts['original_full_RF_checks'],
            total_original_campaign_work_units=previous_units+
                pool.counts['encoded_gradient_jobs']+pool.counts['encoded_hvp_jobs'],
            coordinator_peak_host_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        atomic(args.out/'run.json', run)

    def RF_records(token):
        records = []
        for family in families:
            for shot in range(11):
                files = list((args.out/'workers').glob(
                    f'rank*/fine_RF/token{token:04d}/{family}_{shot:02d}.npy'))
                if len(files) != 1:
                    raise RuntimeError('Incomplete original44 RF persistence')
                records.append({'family': family, 'shot': shot, 'path': str(files[0].resolve()),
                                'SHA256': digest(files[0])})
        return records

    def maps(folder):
        with (folder/'maps.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material',
                str(folder/'material.npy'), '--sample', SAMPLE, '--out', str(folder/'maps'),
                '--estimate-label', 'Independent neural materials + coupled native subspace GN'],
                stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                check=True, timeout=120)

    try:
        pool = NeuralSubspaceNativePool(args, monitor, plan)
        best_RF, _ = pool.evaluate(None, 'cached_forward', 'restored_original44_baseline')
        if abs(best_RF['loss']-prior['final_actual_native_RF_loss']) > 1e-10:
            raise RuntimeError('SHA-checked original44 RF cache changed')
        best_TV, best_records = coordinates.regularization(best, .001)[0], references
        run.update(initial_original44_RF_loss=best_RF['loss'], initial_TV=best_TV)
        checkpoint(full_accepted=True)
        normalized_caps = np.array([.004, .006, .05])[:, None, None]/SCALES
        for step in range(1, maximum_outer+1):
            if stop[0] or time.time() >= deadline-1000:
                stop_reason = 'signal' if stop[0] else 'original_time_budget'
                break
            freeze()
            folder = args.out/'candidates'/f'step{step:02d}'
            folder.mkdir()
            cached_full_model = step == 1
            cached_gradient_only = (step > 1 and not gradient_only_cache_used and
                matches_saved_state(args.parent_root/'candidates/step02',
                                    coordinates.material(current), chart.vector()))
            if cached_full_model or cached_gradient_only:
                saved = load_saved_step(args.parent_root, 1 if cached_full_model else 2,
                                        chart, coordinates, require_matrix=cached_full_model)
                current, base = saved['control'], saved['base']
                directions, images = saved['directions'], saved['images']
                gradient, slopes = saved['gradient'], saved['slopes']
                codes, measured, basis_info = saved['codes'], saved['measured'], saved['basis']
                tv, tv_gradient = coordinates.regularization(current, .001)
                run.update(status='reusing_saved_actual_GN_model' if cached_full_model else
                    'reusing_saved_actual_gradient_and_neural_basis', active_step=step,
                    active_encoding=codes, working_base_material_SHA256=saved['source_material_SHA256'])
                if cached_full_model:
                    run['reused_native_GN_products'] += len(images)
                else:
                    gradient_only_cache_used = True
                    run['reused_gradient_only_jobs'] += 1
                atomic(folder/'derivative_reuse.json', {k: saved[k] for k in
                    ['source_root', 'source_step', 'source_material_SHA256', 'source_file_SHA256']} |
                    {'matrix_reused': cached_full_model, 'all_source_work_already_counted_in_parent': True})
                atomic(folder/'source_actual_gradient_receipt.json', {
                    'loss': measured['loss'], 'token': measured['token'], 'encoding': codes,
                    'source_receipt': measured['reused_from'], 'not_a_new_native_gradient_job': True})
                atomic(args.out/'run.json', run)
            else:
                codes = {family: rng.choice([-1, 1], size=11).tolist() for family in families}
                run.update(status='actual_bounded_neural_subspace_gradient', active_step=step, active_encoding=codes,
                    working_base_material_SHA256=digest(args.artifact_root/'latest_material.npy'))
                atomic(args.out/'run.json', run)
                measured, gradient = pool.evaluate(coordinates.material(current), 'encoded_gradient',
                    f'step{step:02d}_actual_gradient', codes)
                tv, tv_gradient = coordinates.regularization(current, .001)
                gradient += tv_gradient
                gradient[:, ~mask] = 0.
                base, directions, images, slopes, basis_info = chart.actual_gradient_subspace(gradient)
            array(folder/'start_control.npy', current)
            array(folder/'start_material.npy', coordinates.material(current))
            if digest(folder/'start_material.npy') != run['working_base_material_SHA256']:
                raise RuntimeError('Actual derivative working material identity changed')
            array(folder/'actual_objective_gradient.npy', gradient)
            with torch.no_grad():
                difference = float(np.max(abs(model().numpy()-current)))
            if difference > 2e-7:
                raise RuntimeError('Subspace working state differs from actual derivative base')
            array(folder/'neural_parameter_basis.npy', directions.numpy())
            array(folder/'control_basis.npy', images)
            array(folder/'base_neural_parameters.npy', base.numpy())
            dimension = len(images)
            if cached_full_model:
                matrix = saved['matrix'].copy()
                RF_matrix, TV_matrix = saved['RF_matrix'].copy(), saved['TV_matrix'].copy()
                products, complete = saved['products'], True
                atomic(folder/'subspace_products.json', {'basis': basis_info, 'encoding': codes,
                    'completed_columns': dimension, 'all_native_products_reused_without_repropagation': True,
                    'source_root': saved['source_root'], 'source_step': saved['source_step'],
                    'product_summaries': products, 'partial_matrix': matrix.tolist(),
                    'partial_RF_matrix': RF_matrix.tolist(), 'partial_TV_matrix': TV_matrix.tolist()})
            else:
                matrix = np.zeros((dimension, dimension), np.float64)
                RF_matrix, TV_matrix = np.zeros_like(matrix), np.zeros_like(matrix)
                products = []
                complete = True
                for column in range(dimension):
                    if stop[0] or time.time() >= deadline-350:
                        complete = False
                        break
                    freeze()
                    run.update(status='actual_neural_subspace_GN_product', active_basis_column=column+1,
                        actual_basis_dimension=dimension)
                    atomic(args.out/'run.json', run)
                    info, product = pool.hvp(coordinates.material(current), images[column], codes,
                        f'step{step:02d}_GN_column{column+1:02d}')
                    array(folder/f'actual_RF_product{column+1:02d}.npy', product)
                    RF_matrix[:, column] = np.einsum('kizx,izx->k', images, product, optimize=True)
                    TV_product = physical_TV_hessian(coordinates, current, images[column], .001)
                    TV_matrix[:, column] = np.einsum('kizx,izx->k', images, TV_product, optimize=True)
                    matrix[:, column] = RF_matrix[:, column]+TV_matrix[:, column]
                    products.append(info)
                    atomic(folder/'subspace_products.json', {'basis': basis_info, 'encoding': codes,
                        'completed_columns': column+1, 'purpose': 'construct the actual inverse update model',
                        'candidate_solve_still_pending': True,
                        'product_summaries': products, 'partial_matrix': matrix.tolist(),
                        'partial_RF_matrix': RF_matrix.tolist(), 'partial_TV_matrix': TV_matrix.tolist()})
                    checkpoint(full_accepted=True)
                    del product, TV_product
            if not complete:
                stop_reason = 'signal' if stop[0] else 'original_time_budget_inside_GN'
                break
            skew = float(np.linalg.norm(matrix-matrix.T)/max(np.linalg.norm(matrix), 1e-30))
            symmetric = (matrix+matrix.T)/2
            diagonal = np.maximum(np.diag(symmetric), max(float(np.diag(symmetric).max()), 1e-30)*1e-12)
            scaling = np.sqrt(diagonal)
            scaled = symmetric/scaling[:, None]/scaling[None]
            eigenvalues = np.linalg.eigvalsh(scaled)
            if skew > 5e-3 or eigenvalues[0] < -5e-3*max(eigenvalues[-1], 1e-30):
                raise RuntimeError(f'Actual coupled GN matrix is inconsistent: skew={skew}, eigen={eigenvalues}')
            ready, damping = False, .01
            coefficient, predicted = np.zeros(dimension), 0.
            QP_certificate = None
            for damping_attempt in range(5):
                try:
                    if cached_full_model and damping_attempt == 0 and abs(cap_factor-1.) < 1e-12:
                        coefficient = np.load(CPU_solution_root/'bounded_coefficients.npy')
                        QP_certificate = CPU_solution['certificate']
                    else:
                        coefficient, QP_certificate = solve_bounded_subspace(symmetric, slopes,
                            images, normalized_caps*cap_factor, damping=damping)
                except np.linalg.LinAlgError:
                    damping *= 10
                    continue
                linear_slope = float(slopes@coefficient)
                predicted = linear_slope+.5*float(coefficient@symmetric@coefficient)
                if not np.isfinite(coefficient).all() or linear_slope >= 0 or predicted >= 0:
                    damping *= 10
                    continue
                delta = (torch.from_numpy(coefficient.astype(np.float32))[:, None]*directions).sum(0)
                linear = np.einsum('k,kizx->izx', coefficient, images, optimize=True)
                ratio = float(np.max(abs(linear)/(normalized_caps*cap_factor)))
                if ratio > 1+5e-6:
                    raise RuntimeError('Bounded coefficient vector violates its actual linear field caps')
                eta = 1.
                for shrink in range(10):
                    chart.install(base+eta*delta)
                    with torch.no_grad():
                        proposal = model().numpy().copy()
                    ratio = float(np.max(abs(proposal-current)/(normalized_caps*cap_factor)))
                    slope = float(np.sum(gradient*(proposal-current)))
                    if ratio <= 1.001 and slope < 0:
                        ready = True
                        break
                    eta *= .5 if slope >= 0 else min(.95, .95/max(ratio, 1e-30))
                chart.install(base)
                if ready:
                    break
                damping *= 10
            lines, encoded_accepted, fraction = [], False, 1.
            if ready:
                for line in range(1, 4):
                    if stop[0] or time.time() >= deadline-300:
                        break
                    chart.install(base+eta*fraction*delta)
                    with torch.no_grad():
                        proposal = model().numpy().copy()
                    slope = float(np.sum(gradient*(proposal-current)))
                    if slope >= 0 or np.array_equal(coordinates.material(proposal), coordinates.material(current)):
                        fraction *= .5
                        continue
                    run['status'] = 'actual_neural_subspace_RF_search'
                    atomic(args.out/'run.json', run)
                    trial, _ = pool.evaluate(coordinates.material(proposal), 'encoded_forward',
                        f'step{step:02d}_line{line:02d}', codes)
                    candidate_TV = coordinates.regularization(proposal, .001)[0]
                    ok = trial['loss']+candidate_TV <= measured['loss']+tv+1e-4*slope
                    lines.append({'fraction': fraction, 'common_neural_parameter_scale': eta*fraction,
                        'actual_control_slope': slope, 'encoded_RF_before': measured['loss'],
                        'encoded_RF_after': trial['loss'], 'TV_after': candidate_TV,
                        'total_quadratic_predicted_change_at_same_parameter_scale':
                            eta*fraction*float(slopes@coefficient)+.5*(eta*fraction)**2*
                            float(coefficient@symmetric@coefficient),
                        'accepted': bool(ok), 'token': trial['token']})
                    if ok:
                        current, encoded_accepted = proposal.copy(), True
                        run['encoded_steps_accepted_including_full44_rejections'] += 1
                        break
                    remainder = trial['loss']+candidate_TV-measured['loss']-tv-slope
                    fraction *= float(np.clip(-.5*slope/remainder if remainder > 0 else .5, .1, .5))
            atomic(args.out/'steps'/f'step{step:02d}.json', {'step': step, 'basis': basis_info,
                'encoding': codes, 'coupled_matrix': matrix.tolist(),
                'RF_matrix': RF_matrix.tolist(), 'TV_matrix': TV_matrix.tolist(),
                'matrix_relative_skew': skew,
                'diagonally_scaled_eigenvalues': eigenvalues.tolist(),
                'damping': damping, 'coefficients': coefficient.tolist(),
                'bounded_QP_certificate': QP_certificate,
                'saved_full_model_reused': cached_full_model, 'saved_gradient_only_reused': cached_gradient_only,
                'encoded_search_base_is_saved_earlier_material': cached_full_model,
                'linear_predicted_change_before_caps': predicted,
                'CPU_descent_candidate_ready': ready, 'lines': lines,
                'encoded_step_accepted': encoded_accepted,
                'full_original44_acceptance_pending': encoded_accepted, 'truth_read': False})
            if not encoded_accepted:
                chart.install(best_vector)
                current = best.copy()
                cap_factor *= .5
                failed_steps += 1
                checkpoint(full_accepted=True)
                if failed_steps >= 3:
                    stop_reason = 'three_no_accepted_subspace_descent_steps'
                    break
                continue
            failed_steps = 0
            checkpoint()
            array(folder/'control.npy', current)
            array(folder/'material.npy', coordinates.material(current))
            (folder/'stage0_initial_material.npy').symlink_to(initial_material)
            save_network(folder/'network.pt', False)
            if stop[0] or time.time() >= deadline-250:
                chart.install(best_vector)
                current = best.copy()
                checkpoint(full_accepted=True)
                stop_reason = 'signal' if stop[0] else 'original_time_budget_before_full44'
                break
            run['status'] = 'original_all44_neural_subspace_acceptance'
            atomic(args.out/'run.json', run)
            actual, _ = pool.evaluate(coordinates.material(current), 'forward',
                f'step{step:02d}_original44', save_RF=True)
            candidate_TV = coordinates.regularization(current, .001)[0]
            accepted = bool(actual['loss'] < best_RF['loss'] and
                            actual['loss']+candidate_TV < best_RF['loss']+best_TV)
            records = RF_records(actual['token'])
            gain = float((best_RF['loss']-actual['loss'])/best_RF['loss'])
            atomic(folder/'actual_native_acceptance.json', {'step': step, 'accepted': accepted,
                'RF_before': best_RF['loss'], 'actual_native_RF': actual,
                'RF_relative_gain': gain, 'TV_before': best_TV, 'TV_after': candidate_TV,
                'material_SHA256': digest(folder/'material.npy'), 'RF_source_records': records,
                'patient_truth_read': False, 'joint_imaging_success_claimed': False})
            maps(folder)
            if accepted:
                best, best_vector, best_folder = current.copy(), chart.vector(), folder
                best_RF, best_TV, best_records = actual, candidate_TV, records
                rejections = 0
                small_gains = small_gains+1 if gain < .01 else 0
                run['accepted_original44_steps'] += 1
                save_network(folder/'network.pt', True)
                run.update(last_full_accepted_RF_loss=best_RF['loss'],
                    last_full_accepted_material_SHA256=digest(folder/'material.npy'))
            else:
                chart.install(best_vector)
                current = best.copy()
                cap_factor *= .5
                rejections += 1
                small_gains = 0
            run.update(consecutive_rejected_blocks=rejections,
                consecutive_small_gain_accepted_blocks=small_gains)
            checkpoint(full_accepted=True)
            print(json.dumps({'step': step, 'full44_accepted': accepted,
                'candidate_RF': actual['loss'], 'best_RF': best_RF['loss'], 'relative_gain': gain,
                'original_campaign_work_units': run['total_original_campaign_work_units'],
                'new_solver_seconds': time.time()-started}), flush=True)
            if rejections >= 3 or small_gains >= 3:
                stop_reason = 'three_rejected_full44_steps' if rejections >= 3 else 'three_sub_one_percent_full44_gains'
                break
            del images, directions, gradient, tv_gradient
        stop_reason = stop_reason or 'original_original44_or_remaining_work_budget'
        chart.install(best_vector)
        current = best.copy()
        checkpoint(full_accepted=True)
        final = args.out/'final'
        (final/'material.npy').symlink_to((best_folder/'material.npy').resolve())
        (final/'stage0_initial_material.npy').symlink_to(initial_material)
        for suffix in ['.png', '.metrics.json']:
            shutil.copy2(best_folder/('maps'+suffix), final/('maps'+suffix))
        atomic(final/'actual_native_acceptance.json', {'accepted': run['accepted_original44_steps'] > 0,
            'material_SHA256': digest(final/'material.npy'), 'actual_native_RF': best_RF,
            'native_TV': best_TV, 'RF_source_records': best_records,
            'full_original44_source_acceptance': True, 'joint_imaging_success_claimed': False})
        run.update(status='bounded_neural_subspace_native_continuation_finished', stop_reason=stop_reason,
            finished_unix_s=time.time(), seconds=time.time()-started,
            original_campaign_elapsed_seconds=time.time()-original_start,
            final_actual_native_RF_loss=best_RF['loss'], material_SHA256=digest(final/'material.npy'),
            checkpoint_SHA256=digest(args.artifact_root/'checkpoint.pt'),
            final_maps_reused_from_identical_parent_material=run['accepted_original44_steps'] == 0,
            final_metrics_initial_reference_is_parent_metrics_reference=run['accepted_original44_steps'] == 0,
            separate_CPU_post_hoc_maps_saved=True, actual_image_view_still_required=True,
            stopped_signal=stop[0], original_budgets_not_reset=True)
        atomic(args.out/'run.json', run)
        print(json.dumps({k: run[k] for k in ['status', 'stop_reason', 'seconds',
            'encoded_gradient_jobs', 'encoded_hvp_jobs', 'accepted_original44_steps',
            'total_original_campaign_work_units', 'final_actual_native_RF_loss', 'material_SHA256']}), flush=True)
    except BaseException:
        run.update(status='bounded_neural_subspace_native_failed_for_inspection',
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
