"""CPU-only accounting of saved original-RF reconstruction and actual material maps."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from .native_gn_json import atomic


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_trajectory(root, last_step):
    trajectory = []
    for step in range(last_step + 1):
        point = read(root / 'evaluations' / f'accepted{step:03d}.json')
        rf = point['native_RF_quadratic']
        groups = {s: {'raw_RF_objective': 0., 'shots': []} for s in ['lf', 'point3']}
        for path in (root / 'workers').glob(f"rank*/token{rf['token']:04d}_RF.json"):
            result = read(path)
            if result['token'] != rf['token'] or result['kind'] != 'gradient':
                raise RuntimeError('Unexpected original RF evaluation')
            for row in result['shots']:
                group = groups[row['source']]
                if row['shot'] in group['shots']:
                    raise RuntimeError('Duplicated source/shot')
                group['shots'].append(row['shot'])
                group['raw_RF_objective'] += row['RF_loss']
        for source, group in groups.items():
            if sorted(group['shots']) != list(range(11)):
                raise RuntimeError('Missing complete source objective')
            geometric = rf['groups'][source]['geometry_loss'] / 2
            if abs(group['raw_RF_objective'] - geometric) > 1e-6 * max(abs(geometric), 1e-20):
                raise RuntimeError('Original RF and geometric window accounting disagree')
        if abs(sum(g['raw_RF_objective'] for g in groups.values()) - rf['loss']) > 1e-11:
            raise RuntimeError('Original RF source objectives do not sum to full objective')
        trajectory.append({'accepted_step': step, 'evaluation': point['evaluation'],
                           'token': rf['token'], 'raw_RF': rf['loss'], 'TV': point['TV'],
                           'total_computed_objective': point['approximate_objective'],
                           'groups': groups})
    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root / 'step001'
    if args.out.exists():
        raise RuntimeError('Preserve existing analysis')
    run = read(root / 'run.json')
    if run['status'] != 'native_RF_L2_control_finished':
        raise RuntimeError('Wait for the actual completed original-RF control')
    update = read(folder / 'material_update.json')
    metrics = read(folder / 'maps.metrics.json')
    if not (folder / 'maps.png').exists():
        raise RuntimeError('Actual complete material maps are missing')
    if sha(folder / 'material.npy') != update['material_SHA256']:
        raise RuntimeError('Returned material hash changed')
    if sha(folder / 'stage0_initial_material.npy') != run['source_material_SHA256']:
        raise RuntimeError('Initial material changed')
    source_root = Path(run['source_root'])
    if sha(source_root / 'step001/material.npy') != run['source_material_SHA256']:
        raise RuntimeError('Protected physical baseline changed')
    initial_RF = read(root / 'source_original_RF.json')
    returned_RF = read(folder / 'incumbent_RF.json')
    if returned_RF['loss'] != update['native_data_after']:
        raise RuntimeError('Returned RF lineage disagrees')
    trajectory = source_trajectory(root, update['accepted_steps'])
    if trajectory[-1]['raw_RF'] != returned_RF['loss']:
        raise RuntimeError('Accepted trajectory does not select the returned RF')
    initial = np.load(folder / 'stage0_initial_material.npy')
    material = np.load(folder / 'material.npy')
    if initial.shape != (3, 961, 1121) or material.shape != initial.shape:
        raise ValueError('Full native independent fields required')
    delta = material.astype(np.float64) - initial
    z = (np.arange(961) - 60) * .05
    x = (np.arange(1121) - 560) * .05
    roi = (abs(x)[None, :] <= 19.1) & (z[:, None] >= 1) & (z[:, None] <= 43)
    unknown = np.broadcast_to(z[:, None] >= 1, roi.shape)
    masks = {
        'full_FOV_1_to_43mm': roi,
        'lateral_outside_FOV': (abs(x)[None, :] > 19.1) & (z[:, None] >= 1) & (z[:, None] <= 43),
        'deeper_than_43mm': (abs(x)[None, :] <= 19.1) & (z[:, None] > 43),
        'outside_side_and_depth': (abs(x)[None, :] > 19.1) & (z[:, None] > 43),
    }
    total = np.sum(delta[:, unknown] ** 2, axis=1)
    regions = {}
    for name, mask in masks.items():
        energy = np.sum(delta[:, mask] ** 2, axis=1)
        regions[name] = {
            'physical_update_RMS': np.sqrt(np.mean(delta[:, mask] ** 2, axis=1)).tolist(),
            'fraction_of_actual_squared_update': np.divide(energy, total, out=np.zeros_like(energy), where=total > 0).tolist(),
        }
    weights = np.array([.1, .2, .25, .3, .15])
    windows = {}
    contributions = np.zeros(5)
    for source in ['lf', 'point3']:
        before = np.array(initial_RF['groups'][source]['geometry_window_relative_RMS'])
        after = np.array(returned_RF['groups'][source]['geometry_window_relative_RMS'])
        change = .25 * weights * (before ** 2 - after ** 2)
        contributions += change
        windows[source] = {'initial_relative_RMS': before.tolist(),
                           'returned_relative_RMS': after.tolist(),
                           'raw_RF_objective_decrease_contribution': change.tolist()}
    if abs(float(contributions.sum()) - (initial_RF['loss'] - returned_RF['loss'])) > 1e-7:
        raise RuntimeError('Saved original RF window statistics disagree')
    report = {
        'observed_unix_s': time.time(), 'root': str(root), 'checkpoint': str(folder),
        'summary_source_SHA256': sha(Path(__file__)),
        'returned_material_SHA256': sha(folder / 'material.npy'),
        'initial_material_SHA256': run['source_material_SHA256'],
        'initial_is_completed_GN_result_not_uniform_or_training_mean': True,
        'material_update': update, 'original_RF_result': returned_RF,
        'full_checkpoint_maps': metrics,
        'initial_GN_maps': read(source_root / 'step001/maps.metrics.json'),
        'physical_update_regions_from_pilot_start': regions,
        'raw_RF_windows': windows, 'accepted_source_objective_trajectory': trajectory,
        'raw_RF_initial': initial_RF['loss'], 'raw_RF_returned': returned_RF['loss'],
        'raw_RF_decrease_by_geometry_window': contributions.tolist(),
        'new_wave_solves_training_or_inference': False, 'GPU_work': False,
        'patient_truth_arrays_read': False, 'reserved_case_unread': 'Neg_47_Left',
        'joint_imaging_success_claimed': False,
        'geometry_arrival_windows_do_not_uniquely_determine_scatterer_depth': True,
        'comparison_with_constant_metric_transport_changes_only_data_objective_and_corresponding_gradient': True,
        'data_objectives_are_computed_floating_point_values_not_interval_error_certificates': True,
    }
    args.out.mkdir()
    atomic(args.out / 'comparison.json', report)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    titles = ['Sound-speed change (m/s)', 'Density change (kg/m³)', 'Tissue-attenuation change']
    for j, ax in enumerate(axes):
        scale = max(float(np.percentile(abs(delta[j][unknown]), 99)), 1e-12)
        image = ax.imshow(delta[j], extent=[x[0], x[-1], z[-1], z[0]], aspect='equal',
                          interpolation='none', cmap='RdBu_r', vmin=-scale, vmax=scale)
        ax.add_patch(Rectangle((-19.1, 1), 38.2, 42, fill=False, color='black'))
        ax.set(xlabel='Lateral (mm)', ylabel='Depth (mm)', title=titles[j])
        fig.colorbar(image, ax=ax, shrink=.75)
    fig.suptitle('Accepted material change from the identical physical baseline', fontsize=10)
    fig.savefig(args.out / 'physical_update_location.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    steps = [p['accepted_step'] for p in trajectory]
    for source, color in [('lf', '#176b9b'), ('point3', '#cd7c15')]:
        values = np.array([p['groups'][source]['raw_RF_objective'] for p in trajectory])
        axes[0].plot(steps, values / values[0], 'o-', label=source, color=color)
    values = np.array([p['raw_RF'] for p in trajectory])
    axes[0].plot(steps, values / values[0], 's--', label='combined', color='black')
    axes[0].axhline(1, color='gray', lw=.8)
    axes[0].set(xlabel='Accepted material updates', ylabel='RF objective / initial objective',
                title='Original weighted RF objective')
    axes[0].set_xticks(steps)
    axes[0].grid(alpha=.2); axes[0].legend(fontsize=8)
    windows_labels = ['1-5', '5-12', '12-25', '25-43', '>43']
    axes[1].bar(windows_labels, contributions / initial_RF['loss'])
    axes[1].axhline(0, color='black', lw=.8)
    axes[1].set(xlabel='Geometric-arrival windows (mm)', ylabel='RF decrease / initial total RF',
                title='Saved weighted RF decrease by arrival window')
    axes[1].grid(axis='y', alpha=.2)
    fig.savefig(args.out / 'source_objectives.png', dpi=170); plt.close(fig)
    print(json.dumps({'checkpoint': str(folder), 'accepted_steps': update['accepted_steps'],
                      'raw_RF_initial': initial_RF['loss'], 'raw_RF_returned': returned_RF['loss'],
                      'full_maps': {k: v['reconstruction'] for k, v in metrics['maps'].items()},
                      'physical_update_regions': regions, 'no_new_waves': True,
                      'joint_imaging_success_claimed': False}), flush=True)


if __name__ == '__main__':
    main()
