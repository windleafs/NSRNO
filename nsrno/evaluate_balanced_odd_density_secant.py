"""Fixed-final-weight actual reconstructions and training-secant accounting.

Development truth is not read by this module. Complete native predictions are
saved before separate CPU post-hoc map evaluation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from .conditional_secant_uno import ConditionalSecantUNO
from .relative_phase_secant_uno import RelativePhaseSecantUNO
from .balanced_odd_density_secant_uno import BalancedOddDensitySecantUNO
from .central_secant_data import TrainingPairs, array, tissue_label, rf_input
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_proposal(prediction, baseline_roi, baseline_material, edge):
    # Extend a correction, preserving the supplied native baseline (including fine scatter).
    delta = torch.from_numpy((prediction - baseline_roi).astype(np.float32))[None]
    up = F.interpolate(delta, size=(861, 765), mode='bilinear', align_corners=True)
    full = F.pad(up, (178, 178, 60, 40), mode='replicate')[0].numpy()
    x = (np.arange(1121) - 560) * 50e-6
    window = np.cos(np.clip((abs(x) - .0191) / .006, 0, 1) * np.pi / 2) ** 2
    raw = baseline_material.astype(np.float64) + full * window[None, None, :]
    material = raw.copy()
    material[0] = np.clip(material[0], 1400, 1680)
    impedance = np.clip(material[0] * material[1], 1540e3 * np.exp(-.25), 1540e3 * np.exp(.25))
    material[1] = impedance / material[0]
    material[2] = edge + np.clip(material[2] - edge, .5 * np.exp(-5.6), .5 * np.exp(1.4))
    material[:, :80] = baseline_material[:, :80]
    raw[:, :80] = baseline_material[:, :80]
    return material.astype(np.float32), {
        'native_independent_log_c_log_Z_log_alpha_physical_box': True,
        'fraction_projected_per_field': np.mean(abs(raw - material) > 1e-9, axis=(1, 2)).tolist(),
        'supplied_native_baseline_preserved_plus_neural_correction': True,
        'unknown_exterior_is_not_fixed_as_known_water': True,
        'correction_lateral_taper_mm': [19.1, 25.1],
        'new_wave_solves': 0}


def predict(model, rf_base, rf_target, base_roi, context, n, dt, scale, device):
    signal, rms = rf_input(rf_base, rf_target, n, dt, scale)
    tensors = [torch.from_numpy(value[None]).to(device)
               for value in (signal, base_roi, context)]
    with torch.no_grad():
        result = model(*tensors, torch.tensor([rms], dtype=torch.float32, device=device))
    return result['prediction'][0].cpu().numpy(), {
        'RF_residual_RMS_in_training_global_units': rms,
        'log_gain': result['log_gain'][0].cpu().tolist(),
        'predicted_amplitude_scaled': result['amplitude_scaled'][0].cpu().tolist()}


def save_immutable_input_snapshot(path, value, previous):
    # Reuse only a byte-compatible input array from an already finished
    # comparison. New prediction arrays are always written independently.
    if previous is not None and previous.exists():
        original = np.load(previous, mmap_mode='r')
        if original.dtype == value.dtype and original.shape == value.shape and np.array_equal(original, value):
            os.link(previous, path)
            return True
    np.save(path, value)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--comparison-root', type=Path)
    a = p.parse_args()
    training = json.loads((a.train_root / 'run.json').read_text())
    if training['status'] != 'conditional_secant_training_finished':
        raise RuntimeError('Wait for actual fixed-budget training completion')
    if Path('/proc', str(training['pid'])).exists():
        raise RuntimeError('Training process still exists')
    if digest(a.train_root / 'weights.pt') != training['weights_SHA256']:
        raise RuntimeError('Final fixed weight file changed')
    for name, expected in training['source_SHA256'].items():
        if digest(Path(__file__).parent / name) != expected:
            raise RuntimeError('Frozen trained model source changed: ' + name)
    if a.out.exists():
        raise RuntimeError('Preserve existing evaluation')
    index_path = Path(training['index'])
    if digest(index_path) != training['index_SHA256']:
        raise RuntimeError('Prepared conditioning index changed')
    index = json.loads(index_path.read_text())
    a.out.mkdir()
    run = {'status': 'starting_actual_conditional_reconstruction', 'pid': os.getpid(),
           'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
           'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
           'created_unix_s': time.time(), 'training_root': str(a.train_root.resolve()),
           'weights_SHA256': training['weights_SHA256'], 'weight_selection': 'fixed final EMA',
           'device': a.device, 'allocator_cap_GiB': 8, 'per_card_limit_GiB': 16,
           'development_truth_used_as_input': False, 'reserved_case_unread': 'Neg_47_Left',
           'new_wave_solves': 0, 'joint_imaging_success_claimed': False,
           'source_SHA256': digest(__file__)}
    atomic(a.out / 'run.json', run); monitor = PressureResources(a.out)
    try:
        monitor.wait_budget([a.device], 8, 'conditional_secant_evaluation')
        torch.set_num_threads(1); torch.cuda.set_device(a.device)
        torch.cuda.set_per_process_memory_fraction(
            8 * 2**30 / torch.cuda.get_device_properties(a.device).total_memory, a.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = f'cuda:{a.device}'
        model = BalancedOddDensitySecantUNO().to(device)
        weights = torch.load(a.train_root / 'weights.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(weights['ema']); model.eval()
        comparison_model = None
        comparison_weights_SHA256 = None
        if a.comparison_root is not None:
            old = json.loads((a.comparison_root / 'run.json').read_text())
            if old['status'] != 'conditional_secant_training_finished':
                raise RuntimeError('Comparison model not finished')
            if digest(a.comparison_root / 'weights.pt') != old['weights_SHA256']:
                raise RuntimeError('Previous frozen weight changed')
            comparison_model = RelativePhaseSecantUNO().to(device)
            comparison_state = torch.load(a.comparison_root / 'weights.pt', map_location='cpu', weights_only=False)
            comparison_model.load_state_dict(comparison_state['ema']); comparison_model.eval()
            comparison_weights_SHA256 = old['weights_SHA256']
            run['comparison_previous_weights_SHA256'] = comparison_weights_SHA256
            run['same_input_common_baseline_comparison'] = True
        edge = np.broadcast_to(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'), (961, 1121))
        reference_context = np.load(index['reference_context'])
        reference_roi = np.load(index['reference_ROI'])
        reference_native = np.empty((3, 961, 1121), np.float32)
        reference_native[0] = 1540.; reference_native[1] = 1000.; reference_native[2] = .002 + edge
        uniform = reference_native.copy(); uniform[2, 80:] = .5 + edge[80:]
        scale = index['training_RF_component_RMS_scale']
        outputs = []; started = time.time()
        run['status'] = 'saving_actual_development_predictions'; atomic(a.out / 'run.json', run)

        def save(sample, name, prediction, baseline_roi, baseline_native, information):
            folder = a.out / name; folder.mkdir()
            material, projection = native_proposal(prediction, baseline_roi, baseline_native, edge)
            np.save(folder / 'prediction.npy', prediction.astype(np.float32))
            np.save(folder / 'material.npy', material)
            previous = a.comparison_root / 'evaluation' / name if a.comparison_root is not None else None
            reused_baseline = save_immutable_input_snapshot(folder / 'stage0_initial_material.npy', baseline_native,
                previous / 'stage0_initial_material.npy' if previous is not None else None)
            reused_uniform = save_immutable_input_snapshot(folder / 'uniform_initial_material.npy', uniform,
                previous / 'uniform_initial_material.npy' if previous is not None else None)
            record = {'sample': sample, 'name': name, 'material_SHA256': digest(folder / 'material.npy'),
                      'prediction_SHA256': digest(folder / 'prediction.npy'),
                      'inference_inputs_use_development_truth': False,
                      'verified_immutable_input_snapshot_reuse': {'baseline': reused_baseline, 'uniform': reused_uniform},
                      'input_baseline_is_actual_estimate_or_known_reference': True,
                      'weights_SHA256': training['weights_SHA256'], **information, **projection}
            atomic(folder / 'prediction.json', record)
            outputs.append({'sample': sample, 'name': name, 'folder': str(folder.resolve()), **record})
            run['completed_actual_predictions'] = len(outputs); atomic(a.out / 'run.json', run)
            monitor.sample('conditional_actual_prediction_saved')

        for sample in ['train_200', 'val_059']:
            row = next(r for r in index['records']
                       if r['id'] == sample and r['role'] != 'train' and r['valid_samples'] == 541)
            measured = np.load(row['raw_rf'])
            prediction, information = predict(model, np.zeros_like(measured), measured,
                reference_roi, reference_context, row['valid_samples'], row['sample_dt_s'], scale, device)
            save(sample, sample + '_reference', prediction, reference_roi, reference_native,
                 {**information, 'observed_RF': row['raw_rf'], 'baseline_RF': 'known zero scattered reference'})
        current = json.loads(Path(index['native_current_inference']).read_text())
        if digest(current['baseline_material']) != current['baseline_material_SHA256']:
            raise RuntimeError('Saved native estimate changed')
        base_roi = np.load(current['baseline_ROI']); context = np.load(current['baseline_context'])
        baseline_native = np.load(current['baseline_material'])
        rf_base = np.load(current['baseline_RF']); measured = np.load(current['observed_RF'])
        prediction, information = predict(model, rf_base, measured, base_roi, context,
            current['valid_samples'], current['sample_dt_s'], scale, device)
        save(current['sample'], current['sample'] + '_native_current', prediction, base_roi, baseline_native,
             {**information, 'baseline_material_SHA256': current['baseline_material_SHA256'],
              'baseline_RF_evaluation': current['baseline_RF_evaluation'],
              'baseline_RF_token': current['baseline_RF_token'],
              'RF_clocks': {k: current[k] for k in ['valid_samples', 'sample_dt_s', 'selected_frequency_count']}})
        atomic(a.out / 'actual_predictions.json', {'outputs': outputs,
               'development_truth_arrays_read': False, 'new_wave_solves': 0})
        # Training-secant metrics diagnose the learned mapping only.
        # All contexts and labels below are explicitly from Neg_07_Left.
        run['status'] = 'fixed_training_secant_assessment'; atomic(a.out / 'run.json', run)
        pairs = TrainingPairs(index); details = []
        for kind, baseline, target in pairs.evaluation_pairs():
            if baseline['role'] != 'train' or target['role'] != 'train':
                raise RuntimeError('Training contexts only in common-baseline assessment')
            base_roi = tissue_label(baseline, pairs.edge)
            exact_target = tissue_label(target, pairs.edge)
            prediction, info = predict(model, array(baseline['raw_rf']), array(target['raw_rf']),
                base_roi, array(baseline['context']), target['valid_samples'],
                target['sample_dt_s'], scale, device)
            changed = {'pure_c': 0, 'pure_rho': 1, 'alpha': 2}[kind]
            exact = (exact_target - base_roi)[:, 5:].astype(np.float64)
            update = (prediction - base_roi)[:, 5:].astype(np.float64)
            wanted = exact[changed].ravel()

            def response_metrics(estimate):
                value = estimate[changed].ravel()
                denominator = float(np.linalg.norm(wanted))
                return {'relative_RMS_error': float(np.linalg.norm(value - wanted) / max(denominator, 1e-30)),
                        'RMS_gain': float(np.linalg.norm(value) / max(denominator, 1e-30)),
                        'direction_cosine': float(np.dot(value, wanted) /
                            max(np.linalg.norm(value) * denominator, 1e-30))}
            metrics = response_metrics(update)
            previous_metrics = None
            if comparison_model is not None:
                old_prediction, _ = predict(comparison_model, array(baseline['raw_rf']), array(target['raw_rf']),
                    base_roi, array(baseline['context']), target['valid_samples'],
                    target['sample_dt_s'], scale, device)
                previous_metrics = response_metrics((old_prediction - base_roi)[:, 5:].astype(np.float64))
            details.append({'kind': kind, 'pair_group': target['pair_group'],
                'baseline_id': baseline['id'], 'target_id': target['id'],
                'target_pair_sign': target['pair_sign'], 'valid_samples': target['valid_samples'],
                'changed_parameter': changed, **metrics,
                'all_parameter_update_RMS': np.sqrt(np.mean(update ** 2, axis=(1, 2))).tolist(),
                'comparison_previous_model_on_identical_input': previous_metrics,
                'baseline_is_same_unperturbed_material_for_both_signs': True, **info})
        atomic(a.out / 'training_secants.json', {
            'details': details, 'not_independent_patient_reconstruction': True,
            'common_unperturbed_baseline': True,
            'comparison_previous_weights_SHA256': comparison_weights_SHA256,
            'previous_model_same_input_groups': {kind: {key: float(np.mean([r['comparison_previous_model_on_identical_input'][key] for r in details if r['kind'] == kind])) for key in ['relative_RMS_error','direction_cosine','RMS_gain']} for kind in pairs.groups} if comparison_model is not None else None,
            'groups': {kind: {'count': len([r for r in details if r['kind'] == kind]),
                             'mean_relative_RMS_error': float(np.mean([r['relative_RMS_error'] for r in details if r['kind'] == kind])),
                             'mean_direction_cosine': float(np.mean([r['direction_cosine'] for r in details if r['kind'] == kind])),
                             'mean_RMS_gain': float(np.mean([r['RMS_gain'] for r in details if r['kind'] == kind]))}
                       for kind in pairs.groups}})
        run.update(status='conditional_actual_predictions_and_training_secants_finished',
                   finished_unix_s=time.time(), seconds=time.time() - started,
                   actual_native_predictions=len(outputs), training_secant_directions=len(details),
                   CPU_post_hoc_native_map_evaluation_still_required=True,
                   peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30)
        monitor.sample('conditional_secant_evaluation_finished'); atomic(a.out / 'run.json', run)
        print(json.dumps(run), flush=True)
    except BaseException:
        run.update(status='failed', error=traceback.format_exc(), failed_unix_s=time.time())
        atomic(a.out / 'run.json', run); raise
    finally:
        monitor.log.close()


if __name__ == '__main__':
    main()
