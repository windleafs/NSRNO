"""Summarize saved transport reconstruction fields on CPU, without new waves."""
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


def source_trajectory(root, run, last_step):
    trajectory = []
    for step in range(last_step + 1):
        point = read(root / 'evaluations' / f'accepted{step:03d}.json')
        rf = point['native_RF_transport']
        archive = Path(run['previous_transport_root']) if point.get('restored_from_complete_previous_evaluation') else root
        sums = {source: {'lower': 0., 'upper': 0., 'shots': []} for source in ['lf', 'point3']}
        for path in (archive / 'workers').glob(f"rank*/token{rf['token']:04d}_RF.json"):
            result = read(path)
            if result['token'] != rf['token']:
                raise RuntimeError('Unexpected source objective token')
            for shot in result['shots']:
                group = sums[shot['source']]
                if shot['shot'] in group['shots'] or not shot['all_inner_converged']:
                    raise RuntimeError('Duplicate or incomplete source objective')
                group['shots'].append(shot['shot'])
                group['lower'] += shot['lower']
                group['upper'] += shot['upper']
        for source, group in sums.items():
            if sorted(group['shots']) != list(range(11)):
                raise RuntimeError('Missing complete source objective for accepted point')
            group['transport_midpoint'] = .5 * (group['lower'] + group['upper'])
            group['raw_RF_objective'] = rf['groups'][source]['geometry_loss'] / 2
        if abs(sum(group['transport_midpoint'] for group in sums.values()) - rf['loss']) > 1e-10:
            raise RuntimeError('Source objective does not sum to the saved full objective')
        trajectory.append({'accepted_step': step, 'evaluation': point['evaluation'],
                           'source_log_root': str(archive), 'token': rf['token'],
                           'transport': rf['loss'], 'raw_RF': rf['raw_RF_loss'],
                           'total_lower': point['lower'], 'total_upper': point['upper'],
                           'groups': sums})
    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.checkpoint == 'final':
        folder = root / 'step001'
    else:
        folder = root / 'physical_checkpoints' / args.checkpoint
        if folder.resolve().parent != root / 'physical_checkpoints':
            raise ValueError('Invalid checkpoint path')
    if args.out.exists():
        raise RuntimeError('Preserve existing analysis')
    run = read(root / 'run.json')
    if run.get('child_pid') and Path(run.get('map_folder', '')) == folder:
        raise RuntimeError('Wait for checkpoint maps to finish')
    if args.checkpoint == 'final' and run['status'] != 'native_transport_pilot_finished':
        raise RuntimeError('Final reconstruction is not finished')
    update = read(folder / 'material_update.json')
    metrics = read(folder / 'maps.metrics.json')
    if not (folder / 'maps.png').exists():
        raise RuntimeError('Checkpoint maps have not been produced')
    if sha(folder / 'material.npy') != update['material_SHA256']:
        raise RuntimeError('Checkpoint material hash changed')
    if sha(folder / 'stage0_initial_material.npy') != run['source_material_SHA256']:
        raise RuntimeError('Physical starting material does not match the protected source')
    initial_RF = read(root / 'source_original_RF.json')
    source_root = Path(run['source_root'])
    if sha(source_root / 'step001/material.npy') != run['source_material_SHA256']:
        raise RuntimeError('Protected source material changed')
    source_maps = read(source_root / 'step001/maps.metrics.json')
    if args.checkpoint == 'final':
        transport = read(folder / 'incumbent_transport.json')
        if transport['raw_RF_loss'] != update['native_data_after']:
            raise RuntimeError('Final raw RF lineage disagrees')
    else:
        transport = update['native_RF_transport']
        if transport['raw_RF_loss'] != update['original_RF_loss']:
            raise RuntimeError('Checkpoint raw RF lineage disagrees')
    initial = np.load(folder / 'stage0_initial_material.npy')
    material = np.load(folder / 'material.npy')
    if initial.shape != (3, 961, 1121) or material.shape != initial.shape:
        raise ValueError('Full native fields required')
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
    for source in ('lf', 'point3'):
        before = np.array(initial_RF['groups'][source]['geometry_window_relative_RMS'])
        after = np.array(transport['groups'][source]['geometry_window_relative_RMS'])
        change = .25 * weights * (before ** 2 - after ** 2)
        contributions += change
        windows[source] = {
            'initial_relative_RMS': before.tolist(),
            'returned_relative_RMS': after.tolist(),
            'raw_RF_objective_decrease_contribution': change.tolist(),
        }
    decrease = initial_RF['loss'] - transport['raw_RF_loss']
    if abs(float(contributions.sum()) - decrease) > 1e-7:
        raise RuntimeError('Saved raw RF window statistics disagree')
    last_step = update.get('accepted_step', update.get('accepted_steps'))
    trajectory = source_trajectory(root, run, last_step)
    report = {
        'observed_unix_s': time.time(), 'root': str(root), 'checkpoint': str(folder),
        'summary_source_SHA256': sha(Path(__file__)),
        'returned_material_SHA256': sha(folder / 'material.npy'),
        'initial_material_SHA256': run['source_material_SHA256'],
        'initial_is_completed_GN_result_not_uniform_or_training_mean': True,
        'material_update': update, 'native_RF_transport': transport,
        'full_checkpoint_maps': metrics, 'initial_GN_maps': source_maps,
        'physical_update_regions_from_pilot_start': regions, 'raw_RF_windows': windows,
        'accepted_source_objective_trajectory': trajectory,
        'raw_RF_initial': initial_RF['loss'], 'raw_RF_returned': transport['raw_RF_loss'],
        'raw_RF_decrease_by_geometry_window': contributions.tolist(),
        'new_wave_solves_training_or_inference': False, 'GPU_work': False,
        'patient_truth_arrays_read': False, 'joint_imaging_success_claimed': False,
        'reserved_case_unread': 'Neg_47_Left',
        'geometry_arrival_windows_do_not_uniquely_determine_scatterer_depth': True,
        'candidate_changes_objective_gradient_refresh_and_optimizer_together': True,
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
    fig.suptitle(f'{args.checkpoint}: total accepted material change from the transport pilot start', fontsize=10)
    fig.savefig(args.out / 'physical_update_location.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    steps = [point['accepted_step'] for point in trajectory]
    for source, color in [('lf', '#176b9b'), ('point3', '#cd7c15')]:
        for ax, key in [(axes[0], 'transport_midpoint'), (axes[1], 'raw_RF_objective')]:
            values = np.array([point['groups'][source][key] for point in trajectory])
            ax.plot(steps, values / values[0], 'o-', label=source, color=color)
    for ax, key in [(axes[0], 'transport'), (axes[1], 'raw_RF')]:
        values = np.array([point[key] for point in trajectory])
        ax.plot(steps, values / values[0], 's--', label='combined', color='black', markersize=4)
        ax.axhline(1, color='gray', lw=.8)
        ax.set(xlabel='Accepted material updates', ylabel='Objective / objective at the initial material')
        ax.set_xticks(steps)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_title('Vector-transport data objective')
    axes[1].set_title('Original weighted RF objective')
    fig.savefig(args.out / 'source_objectives.png', dpi=170)
    plt.close(fig)
    print(json.dumps({
        'checkpoint': str(folder), 'accepted_steps': update.get('accepted_step', update.get('accepted_steps')),
        'raw_RF_initial': initial_RF['loss'], 'raw_RF_returned': transport['raw_RF_loss'],
        'full_maps': metrics['maps'], 'physical_update_regions': regions,
        'no_new_waves': True, 'joint_imaging_success_claimed': False,
    }), flush=True)


if __name__ == '__main__':
    main()
