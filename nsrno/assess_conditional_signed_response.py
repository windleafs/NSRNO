"""Separate even/odd residual dependence of a frozen conditional inverse.

Residual reversal is a synthetic network-input control, not a new physical RF
measurement or a material-derivative experiment. Development truth is absent.
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
from .conditional_secant_uno import ConditionalSecantUNO
from .conditional_secant_data import TrainingPairs, array, tissue_label, rf_input
from .evaluate_conditional_secant import native_proposal
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    a = p.parse_args()
    training = read(a.train_root / 'run.json')
    previous = read(a.train_root / 'evaluation/run.json')
    if training['status'] != 'conditional_secant_training_finished' or previous['status'] != 'conditional_actual_predictions_and_training_secants_finished':
        raise RuntimeError('Complete fixed-weight predictions first')
    if any(Path('/proc', str(r['pid'])).exists() for r in [training, previous]):
        raise RuntimeError('Previous GPU processes still exist')
    if digest(a.train_root / 'weights.pt') != training['weights_SHA256']:
        raise RuntimeError('Frozen weight changed')
    if a.out.exists():
        raise RuntimeError('Preserve existing assessment')
    a.out.mkdir()
    index = read(training['index'])
    run = {'status': 'starting_signed_residual_assessment', 'pid': os.getpid(),
           'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
           'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
           'created_unix_s': time.time(), 'device': a.device, 'weights_SHA256': training['weights_SHA256'],
           'allocator_cap_GiB': 8, 'per_card_limit_GiB': 16,
           'synthetic_RF_control_not_physical_observation': True,
           'development_truth_read': False, 'new_wave_solves': 0, 'new_material_derivatives': 0,
           'reserved_case_unread': 'Neg_47_Left', 'source_SHA256': digest(__file__)}
    atomic(a.out / 'run.json', run); monitor = PressureResources(a.out)
    try:
        monitor.wait_budget([a.device], 8, 'signed_residual_assessment')
        torch.set_num_threads(1); torch.cuda.set_device(a.device)
        torch.cuda.set_per_process_memory_fraction(
            8 * 2**30 / torch.cuda.get_device_properties(a.device).total_memory, a.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = f'cuda:{a.device}'
        model = ConditionalSecantUNO().to(device)
        state = torch.load(a.train_root / 'weights.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['ema']); model.eval()
        scales = np.asarray([80., 100., .2], np.float64)[:, None, None]

        def components(base_RF, target_RF, base_roi, context, n, dt):
            signal, rms = rf_input(base_RF, target_RF, n, dt, index['training_RF_component_RMS_scale'])
            inputs = [torch.from_numpy(v[None]).to(device) for v in [signal, base_roi, context]]
            r = torch.tensor([rms], dtype=torch.float32, device=device)
            with torch.no_grad():
                plus = model(*inputs, r)['delta_scaled'][0].cpu().numpy().astype(np.float64)
                inputs[0] = inputs[0].clone()
                inputs[0][:, :, 2:4] *= -1
                minus = model(*inputs, r)['delta_scaled'][0].cpu().numpy().astype(np.float64)
            return (plus - minus) / 2, (plus + minus) / 2, plus

        started = time.time()
        current = read(index['native_current_inference'])
        base_roi = np.load(current['baseline_ROI']); context = np.load(current['baseline_context'])
        odd, even, original = components(np.load(current['baseline_RF']), np.load(current['observed_RF']),
            base_roi, context, current['valid_samples'], current['sample_dt_s'])
        prediction = (base_roi.astype(np.float64) + scales * odd).astype(np.float32)
        edge = np.broadcast_to(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'), (961, 1121))
        baseline_native = np.load(current['baseline_material'])
        if digest(current['baseline_material']) != current['baseline_material_SHA256']:
            raise RuntimeError('Known current estimate changed')
        folder = a.out / 'train_200_native_odd'; folder.mkdir()
        material, projection = native_proposal(prediction, base_roi, baseline_native, edge)
        np.save(folder / 'material.npy', material)
        np.save(folder / 'prediction.npy', prediction)
        np.save(folder / 'odd_delta_scaled.npy', odd.astype(np.float32))
        np.save(folder / 'even_delta_scaled.npy', even.astype(np.float32))
        np.save(folder / 'stage0_initial_material.npy', baseline_native)
        uniform = np.empty_like(baseline_native)
        uniform[0] = 1540; uniform[1] = 1000; uniform[2] = .5 + edge
        uniform[:, :80] = baseline_native[:, :80]
        np.save(folder / 'uniform_initial_material.npy', uniform)
        e2 = np.mean(even[:, 5:] ** 2, axis=(1, 2))
        o2 = np.mean(odd[:, 5:] ** 2, axis=(1, 2))
        information = {'sample': current['sample'], 'material_SHA256': digest(folder / 'material.npy'),
                       'even_fraction_of_symmetric_output_energy': (e2 / np.maximum(e2 + o2, 1e-30)).tolist(),
                       'odd_scaled_RMS': np.sqrt(o2).tolist(), 'even_scaled_RMS': np.sqrt(e2).tolist(),
                       'development_truth_used': False, 'actual_measured_RF_residual_used_for_odd_prediction': True,
                       'counterfactual_internal_call_is_not_physical_RF': True, **projection}
        atomic(folder / 'prediction.json', information)
        monitor.sample('actual_odd_prediction_saved')
        pairs = TrainingPairs(index); details = []
        for kind, groups in pairs.groups.items():
            changed = {'pure_c': 0, 'pure_rho': 1, 'alpha': 2}[kind]
            for members in groups:
                for baseline, target in [members, members[::-1]]:
                    base_roi = tissue_label(baseline, pairs.edge)
                    target_roi = tissue_label(target, pairs.edge)
                    odd, even, original = components(array(baseline['raw_rf']), array(target['raw_rf']),
                        base_roi, array(baseline['context']), target['valid_samples'], target['sample_dt_s'])
                    exact = ((target_roi - base_roi) / scales)[:, 5:]
                    metrics = {}
                    for name, value in [('original', original), ('odd', odd), ('even', even)]:
                        wanted, estimate = exact[changed].ravel(), value[changed, 5:].ravel()
                        norm = np.linalg.norm(wanted); enorm = np.linalg.norm(estimate)
                        metrics[name] = {'relative_RMS_error': float(np.linalg.norm(estimate - wanted) / max(norm, 1e-30)),
                                         'direction_cosine': float(np.dot(estimate, wanted) / max(norm * enorm, 1e-30)),
                                         'RMS_gain': float(enorm / max(norm, 1e-30))}
                    e2 = float(np.mean(even[changed, 5:] ** 2))
                    o2 = float(np.mean(odd[changed, 5:] ** 2))
                    details.append({'kind': kind, 'pair_group': baseline['pair_group'],
                        'baseline_id': baseline['id'], 'target_id': target['id'],
                        'valid_samples': target['valid_samples'],
                        'even_fraction_of_symmetric_output_energy': e2 / max(e2 + o2, 1e-30),
                        'metrics': metrics})
        groups = {}
        for kind in pairs.groups:
            rows = [r for r in details if r['kind'] == kind]
            groups[kind] = {'count': len(rows),
                'mean_even_fraction_of_symmetric_output_energy': float(np.mean(
                    [r['even_fraction_of_symmetric_output_energy'] for r in rows])),
                **{name: {key: float(np.mean([r['metrics'][name][key] for r in rows]))
                          for key in ['relative_RMS_error', 'direction_cosine', 'RMS_gain']}
                   for name in ['original', 'odd', 'even']}}
        result = {'groups': groups, 'details': details, 'actual_odd_prediction': information,
                  'finite_residual_even_terms_can_be_legitimate_nonlinearity': True,
                  'not_a_global_proof_of_a_baseline_shortcut': True,
                  'synthetic_control_not_a_physical_material_test': True,
                  'new_wave_solves': 0, 'new_material_derivatives': 0}
        atomic(a.out / 'signed_response.json', result)
        run.update(status='signed_residual_assessment_finished', finished_unix_s=time.time(),
                   seconds=time.time() - started, assessed_training_directions=len(details),
                   actual_odd_material_SHA256=information['material_SHA256'])
        monitor.sample('signed_residual_assessment_finished')
        atomic(a.out / 'run.json', run)
        print(json.dumps({'run': run, 'groups': groups, 'actual_odd_prediction': information}), flush=True)
    except BaseException:
        run.update(status='failed', error=traceback.format_exc(), failed_unix_s=time.time())
        atomic(a.out / 'run.json', run); raise
    finally:
        monitor.log.close()


if __name__ == '__main__':
    main()
