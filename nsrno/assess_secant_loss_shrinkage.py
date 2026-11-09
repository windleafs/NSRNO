"""CPU sufficient-statistic diagnosis of material-loss gain shrinkage.

Uses already-saved final neural predictions, not new network evaluations.
The conclusion is restricted to equal-weight independent-parameter tasks.
It does not include reference-to-material or cross-material training tasks.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import json
import os
import time
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=32)
def label(path):
    return np.load(path).astype(np.float64)


def quadratic_summary(groups):
    result = {}
    for name in ('component_relative', 'joint_vector_relative'):
        moments = [np.mean([[r[name]['A'], r[name]['B'], r[name]['C'],
                            r[name]['unknown_cross_dot_bound']] for r in rows], axis=0)
                   for rows in groups.values()]
        a, b, c, bound = np.mean(moments, axis=0)
        optimum = b / a if a > 0 else 0.
        positive = max(0., optimum)
        result[name] = {
            'coefficient_A': float(a), 'coefficient_B': float(b),
            'coefficient_C': float(c), 'unknown_cross_dot_bound': float(bound),
            'unconstrained_global_density_gain': float(optimum),
            'nonnegative_global_density_gain': float(positive),
            'unconstrained_gain_interval_from_missing_off_target_dot_products':
                [float((b - bound) / a), float((b + bound) / a)] if a > 0 else [0., 0.],
            'density_component_loss_at_zero': float(c),
            'density_component_loss_at_original_gain': float(a - 2*b + c),
            'density_component_loss_at_nonnegative_optimum': float(
                a * positive**2 - 2*b*positive + c),
            'scaled_mean_active_density_RMS_gain': float(
                positive * np.mean([r['original_active_RMS_gain']
                                    for r in groups['pure_rho']])),
            'equal_weight_per_kind': True,
            'quadratic_definition': 'A*t^2 - 2*B*t + C',
            'overall_three_field_loss_would_divide_this_component_by_three': True,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--models', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run this existing-statistic diagnosis with CUDA disabled')
    if args.out.exists():
        raise RuntimeError('Do not overwrite an existing assessment')
    started = time.time()
    index = json.loads(args.index.read_text())
    records = {(r['id'], r['valid_samples']): r for r in index['records']
               if r['role'] == 'train'}
    if {r['case'] for r in records.values()} != {'Neg_07_Left'}:
        raise ValueError('Training partition changed')
    scale = np.asarray([80., 100., .2])
    floor_squared = np.asarray([5/80, 5/100, .02/.2])**2
    model_results = []
    for root in args.models:
        run = json.loads((root / 'run.json').read_text())
        if run['status'] != 'conditional_secant_training_finished':
            raise RuntimeError('Only completed fixed-weight predictions are admitted')
        stats_path = root / 'evaluation/training_secants.json'
        stats = json.loads(stats_path.read_text())
        if not stats.get('common_unperturbed_baseline'):
            raise ValueError('Predictions do not share the unperturbed baseline')
        groups = {k: [] for k in ['pure_c', 'pure_rho', 'alpha']}
        checks = []; off_targets = []; multipliers = {'pure_c': [], 'alpha': []}
        for r in stats['details']:
            n = r['valid_samples']
            base = records[(r['baseline_id'], n)]
            target = records[(r['target_id'], n)]
            y = (label(target['label']) - label(base['label']))[:, 5:] / scale[:, None, None]
            y_rms = np.sqrt(np.mean(y**2, axis=(1, 2)))
            v_rms = np.asarray(r['all_parameter_update_RMS']) / scale
            changed = r['changed_parameter']
            dot_changed = v_rms[changed] * y_rms[changed] * r['direction_cosine']
            reproduced = (v_rms[changed]**2 + y_rms[changed]**2 - 2*dot_changed) / y_rms[changed]**2
            checks.append(abs(reproduced - r['relative_RMS_error']**2))
            if checks[-1] > 1e-5:
                raise ValueError('Saved response statistics do not match the training labels')
            dot = dot_changed if changed == 1 else 0.
            unknown_bound = 0. if changed == 1 else v_rms[1] * y_rms[1]
            off_targets.append(float(y_rms[1]) if changed != 1 else 0.)
            denominators = {
                'component_relative': max(y_rms[1]**2, floor_squared[1]),
                'joint_vector_relative': max(float(np.sum(y_rms**2)),
                                             float(np.max(floor_squared))),
            }
            item = {'target_id': r['target_id'], 'valid_samples': n,
                    'kind': r['kind'], 'original_active_RMS_gain': r['RMS_gain']}
            for name, denominator in denominators.items():
                item[name] = {'A': float(v_rms[1]**2 / denominator),
                              'B': float(dot / denominator),
                              'C': float(y_rms[1]**2 / denominator),
                              'unknown_cross_dot_bound': float(unknown_bound / denominator)}
            groups[r['kind']].append(item)
            if r['kind'] in multipliers:
                multipliers[r['kind']].append(
                    denominators['joint_vector_relative'] /
                    denominators['component_relative'])
        model_results.append({
            'root': str(root.resolve()), 'weights_SHA256': run['weights_SHA256'],
            'saved_statistics_SHA256': digest(stats_path),
            'group_counts': {k: len(v) for k, v in groups.items()},
            'maximum_saved_RMS_identity_error': max(checks),
            'maximum_unobserved_density_target_RMS_in_scaled_units': max(off_targets),
            'density_false_update_weight_ratio_component_over_joint_vector':
                {k: {'median': float(np.median(v)), 'minimum': float(np.min(v)),
                     'maximum': float(np.max(v))} for k, v in multipliers.items()},
            'frozen_direction_global_gain_diagnosis': quadratic_summary(groups),
            'per_kind_density_quadratic_at_original_gain':
                {k: {name: np.mean([[r[name]['A'], r[name]['B'], r[name]['C']]
                                    for r in values], axis=0).tolist()
                     for name in denominators}
                 for k, values in groups.items()},
        })
    result = {
        'status': 'existing_secant_loss_shrinkage_assessed',
        'observed_unix_s': time.time(), 'seconds': time.time() - started,
        'index_SHA256': digest(args.index), 'source_SHA256': digest(__file__),
        'scope': 'frozen density proposals; equal weights across independent c/rho/alpha tasks',
        'not_the_complete_training_objective': True,
        'cross_material_and_reference_tasks_missing': True,
        'density_only_global_gain_cannot_repair_incorrect_spatial_direction': True,
        'new_network_evaluations': 0, 'new_wave_solves': 0,
        'new_material_derivatives': 0, 'GPU_work': False,
        'development_material_truth_read': False, 'reserved_case_unread': 'Neg_47_Left',
        'models': model_results,
    }
    args.out.mkdir()
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
