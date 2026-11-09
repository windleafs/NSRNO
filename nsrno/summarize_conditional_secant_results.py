"""CPU post-hoc complete-map assessment after fixed conditional inverse predictions."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np
from .native_gn_json import atomic
from .v5_materials import evaluation_material
from .evaluate_conditional_secant import native_proposal


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run = read(a.evaluation / 'run.json')
    if run['status'] != 'conditional_actual_predictions_and_training_secants_finished':
        raise RuntimeError('Complete actual predictions first')
    if Path('/proc', str(run['pid'])).exists():
        raise RuntimeError('Wait for prediction evaluator to exit')
    if a.out.exists():
        raise RuntimeError('Preserve existing analysis')
    predictions = read(a.evaluation / 'actual_predictions.json')
    if predictions['development_truth_arrays_read']:
        raise RuntimeError('Do not use development truth as input')
    training = read(Path(run['training_root']) / 'run.json')
    if training['weights_SHA256'] != run['weights_SHA256']:
        raise RuntimeError('Final fixed weight lineage changed')
    index = read(training['index'])
    edge = np.broadcast_to(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'), (961, 1121))
    edge_roi = np.load(index['ROI_known_edge'])
    mean_roi = np.load(index['training_mean_material']).copy()
    mean_roi[2] -= edge_roi
    mean_roi[:, :5] = np.asarray([1540., 1000., .002], np.float32)[:, None, None]
    ref_roi = np.load(index['reference_ROI'])
    ref_native = np.empty((3, 961, 1121), np.float32)
    ref_native[0] = 1540; ref_native[1] = 1000; ref_native[2] = .002 + edge
    mean_native, _ = native_proposal(mean_roi, ref_roi, ref_native, edge)
    z = (np.arange(961) - 60) * .05
    x = (np.arange(1121) - 560) * .05
    fov = (abs(x)[None, :] <= 19.1) & (z[:, None] >= 1) & (z[:, None] <= 43)
    depth_masks = {f'{lo}-{hi}': fov & (z[:, None] >= lo) & (z[:, None] < hi)
                   for lo, hi in [(1, 5), (5, 12), (12, 25), (25, 43)]}
    names = ['c_m_s', 'density_kg_m3', 'alpha_dB_cm_MHz1p1']

    def stats(material, target, mask):
        values = {}
        for j, name in enumerate(names):
            estimate = material[j][mask].astype(np.float64)
            actual = target[j][mask].astype(np.float64)
            values[name] = {'rmse': float(np.sqrt(np.mean((estimate - actual) ** 2))),
                            'mae': float(np.mean(abs(estimate - actual))),
                            'correlation': float(np.corrcoef(estimate, actual)[0, 1])
                            if estimate.std() > 1e-10 and actual.std() > 1e-10 else None}
        return values

    results = []
    a.out.mkdir()
    for row in predictions['outputs']:
        folder = Path(row['folder'])
        if digest(folder / 'material.npy') != row['material_SHA256']:
            raise RuntimeError('Prediction material changed')
        if row['inference_inputs_use_development_truth']:
            raise RuntimeError('Invalid inference inputs')
        if not (folder / 'maps.png').exists():
            with (folder / 'maps.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'nsrno.evaluate_time_material',
                    str(folder / 'material.npy'), '--sample', row['sample'],
                    '--out', str(folder / 'maps'),
                    '--estimate-label', row['name'] + ', fixed final EMA'],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log,
                    stderr=subprocess.STDOUT, check=True)
        target = evaluation_material(row['sample'])
        material = np.load(folder / 'material.npy')
        baseline = np.load(folder / 'stage0_initial_material.npy')
        uniform = np.load(folder / 'uniform_initial_material.npy')
        values = {'name': row['name'], 'sample': row['sample'], 'folder': str(folder.resolve()),
                  'material_SHA256': row['material_SHA256'], 'inference': row,
                  'prediction': stats(material, target, fov),
                  'actual_input_baseline': stats(baseline, target, fov),
                  'uniform_comparison': stats(uniform, target, fov),
                  'training_mean_comparison': stats(mean_native, target, fov),
                  'prediction_by_depth': {name: stats(material, target, mask)
                                          for name, mask in depth_masks.items()},
                  'all_three_RMSE_better_than_uniform_and_training_mean':
                      all(stats(material, target, fov)[key]['rmse'] <
                          min(stats(uniform, target, fov)[key]['rmse'],
                              stats(mean_native, target, fov)[key]['rmse'])
                          for key in names),
                  'native_RF_not_evaluated': True, 'truth_only_post_hoc': True}
        values['actual_update_RMS_in_FOV'] = np.sqrt(np.mean(
            (material[:, fov].astype(np.float64) - baseline[:, fov]) ** 2, axis=1)).tolist()
        results.append(values)
    report = {'observed_unix_s': time.time(), 'evaluation_root': str(a.evaluation.resolve()),
              'weights_SHA256': run['weights_SHA256'], 'training_run': training,
              'predictions': results, 'training_secant_metrics': read(a.evaluation / 'training_secants.json'),
              'GPU_work': False, 'new_wave_solves': 0, 'new_material_derivatives': 0,
              'patient_truth_read_only_after_actual_prediction_files_saved': True,
              'native_full_FOV_mm': {'x': [-19.1, 19.1], 'z': [1, 43]},
              'native_grid': [3, 961, 1121], 'reserved_case_unread': 'Neg_47_Left',
              'joint_imaging_success_claimed': False,
              'visual_review_and_physical_RF_still_required_for_any_positive_candidate': True,
              'source_SHA256': digest(__file__)}
    atomic(a.out / 'comparison.json', report)
    print(json.dumps({'predictions': [{k: row[k] for k in [
        'name', 'prediction', 'actual_input_baseline', 'uniform_comparison',
        'training_mean_comparison', 'all_three_RMSE_better_than_uniform_and_training_mean']}
        for row in results], 'training_secant_groups': report['training_secant_metrics']['groups'],
        'joint_imaging_success_claimed': False, 'new_waves': 0}), flush=True)


if __name__ == '__main__':
    main()
