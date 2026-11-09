"""CPU-only decomposition of the saved first accepted transport step's RF."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json

import numpy as np
import torch
from scipy.signal import hilbert

from .v5_observations import plane_observation, canonical_id
from .rf_representation import decode
from .rf_depth_weights import arrival_bins, WEIGHTS, NAMES
from .native_gn_json import atomic


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(root, source, shot):
    matches = list((root / 'workers').glob(f'rank*/initial_RF/{source}_{shot:02d}.npy'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one complete archived RF for {source}, shot {shot}: {matches}')
    array = np.load(matches[0])
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise RuntimeError('Finite float32 archived predictions required')
    return array, {'path': str(matches[0]), 'SHA256': sha(matches[0])}


def observation(source, sample):
    geometry_file = Path('out/v5/time_triparam_v1/geometry.npz')
    with np.load(geometry_file) as data:
        geometry = {k: data[k] for k in ['x', 'xe']}
    if source == 'point3':
        index_file = Path('out/v5/point_tx_3mhz_v1/index.json')
        index = read(index_file)
        row = next(r for r in index['records'] if r['id'] == canonical_id(sample))
        observed = np.load(row['rf'])
        with np.load(index['source_layout']) as layout:
            positions = layout['transmit_x_m'].copy()
        angles = np.arange(len(positions))
        provenance = {'RF': row['rf'], 'RF_SHA256': sha(Path(row['rf'])),
                      'index': str(index_file), 'index_SHA256': sha(index_file)}
    else:
        row = plane_observation(source, sample)
        observed = decode(torch.from_numpy(np.load(row['iq']))).numpy()
        index_file = Path('out/v5/low_frequency_3mhz_v1/index.json')
        angles = np.array(read(index_file)['acquisition']['angles_deg'])
        positions = None
        provenance = {'IQ': row['iq'], 'IQ_SHA256': sha(Path(row['iq'])),
                      'geometry_angle_index': str(index_file), 'angle_index_SHA256': sha(index_file)}
    if len(angles) != 11 or observed.shape[:2] != (11, 192):
        raise ValueError('Complete 11-source, 192-receiver acquisition required')
    acquisition = SimpleNamespace(source=source, geometry=geometry, observed=observed,
                                  f0=3e6, angles=angles, source_positions=positions)
    return observed, arrival_bins(acquisition), provenance


def source_transport(root, source):
    lower = upper = 0.
    seen = set()
    for path in (root / 'workers').glob('rank*/token0001_RF.json'):
        record = read(path)
        if record['token'] != 1:
            raise RuntimeError('Unexpected archive token')
        for row in record['shots']:
            if row['source'] != source:
                continue
            if row['shot'] in seen or not row['all_inner_converged']:
                raise RuntimeError('Duplicate or incomplete saved source objective')
            seen.add(row['shot'])
            lower += row['lower']
            upper += row['upper']
    if seen != set(range(11)):
        raise RuntimeError('Incomplete source transport objective archive')
    return {'lower': lower, 'upper': upper, 'midpoint': .5 * (lower + upper)}


def cosine(cross, prediction_energy, observed_energy):
    return np.divide(cross, np.sqrt(prediction_energy * observed_energy),
                     out=np.zeros_like(cross), where=prediction_energy * observed_energy > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--accepted', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    accepted = args.accepted.resolve()
    if args.out.exists():
        raise RuntimeError('Preserve existing RF analysis')
    old_run = read(baseline / 'run.json')
    new_run = read(accepted / 'run.json')
    if old_run['sample'] != 'train_200' or new_run['sample'] != old_run['sample']:
        raise ValueError('This analysis is restricted to the already used train_200')
    if Path(new_run['previous_transport_root']).resolve() != baseline:
        raise RuntimeError('The archives are not the actual continuation pair')
    before = read(baseline / 'evaluations/eval001.json')
    after = read(accepted / 'evaluations/accepted001.json')
    if after['native_RF_transport']['tag'] != 'transport_eval004' or after['accepted_step'] != 1:
        raise RuntimeError('The new token1 archive is not the first accepted step')
    if old_run['source_material_SHA256'] != new_run['source_material_SHA256']:
        raise RuntimeError('Physical starting material changed')
    groups = {}
    reconstructed_before = reconstructed_after = 0.
    for source in ['lf', 'point3']:
        observed, bins, observation_provenance = observation(source, old_run['sample'])
        observed_energy = np.zeros(5)
        predicted_energy = np.zeros((2, 5))
        cross = np.zeros((2, 5))
        envelope_prediction_energy = np.zeros((2, 5))
        envelope_observed_energy = np.zeros(5)
        envelope_cross = np.zeros((2, 5))
        error_squared = np.zeros((2, 5))
        provenance = []
        for shot in range(11):
            old, old_prov = archive(baseline, source, shot)
            new, new_prov = archive(accepted, source, shot)
            data = observed[shot]
            if old.shape != data.shape or new.shape != data.shape:
                raise RuntimeError('Archived full RF shape disagrees with observation')
            data64 = data.astype(np.float64)
            data_envelope = abs(hilbert(data64, axis=-1))
            for k in range(5):
                mask = bins[shot] == k
                observed_energy[k] += np.sum(data64[mask] ** 2)
                envelope_observed_energy[k] += np.sum(data_envelope[mask] ** 2)
            for level, prediction in enumerate([old, new]):
                pred64 = prediction.astype(np.float64)
                envelope = abs(hilbert(pred64, axis=-1))
                difference = (prediction - data).astype(np.float64)
                for k in range(5):
                    mask = bins[shot] == k
                    predicted_energy[level, k] += np.sum(pred64[mask] ** 2)
                    cross[level, k] += np.sum(pred64[mask] * data64[mask])
                    error_squared[level, k] += np.sum(difference[mask] ** 2)
                    envelope_prediction_energy[level, k] += np.sum(envelope[mask] ** 2)
                    envelope_cross[level, k] += np.sum(envelope[mask] * data_envelope[mask])
            provenance.append({'shot': shot, 'baseline': old_prov, 'accepted1': new_prov})
        coefficient = .25 * np.array(WEIGHTS) / observed_energy
        objective = np.sum(error_squared * coefficient, axis=1)
        reconstructed_before += objective[0]
        reconstructed_after += objective[1]
        for level, expected in enumerate([before, after]):
            expected_error = expected['native_RF_transport']['groups'][source]['window_error_squared']
            if not np.allclose(error_squared[level], expected_error, rtol=2e-7, atol=1e-8):
                raise RuntimeError('Saved RF predictions do not reproduce the recorded original RF loss')
        energy_change = coefficient * (predicted_energy[1] - predicted_energy[0])
        cross_change = -2 * coefficient * (cross[1] - cross[0])
        groups[source] = {
            'windows': NAMES, 'observation': observation_provenance, 'RF_archives': provenance,
            'observed_energy': observed_energy.tolist(), 'prediction_energy': predicted_energy.tolist(),
            'RF_cross_product': cross.tolist(), 'raw_RF_objectives': objective.tolist(),
            'prediction_RMS_over_observed_RMS': np.sqrt(predicted_energy / observed_energy).tolist(),
            'signed_RF_cosine': cosine(cross, predicted_energy, observed_energy).tolist(),
            'analytic_envelope_cosine': cosine(envelope_cross, envelope_prediction_energy,
                                               envelope_observed_energy).tolist(),
            'raw_RF_increase_prediction_energy_component': energy_change.tolist(),
            'raw_RF_increase_cross_product_component': cross_change.tolist(),
            'algebraic_decomposition_error': float(np.sum(energy_change + cross_change) -
                                                  (objective[1] - objective[0])),
            'baseline_source_transport': source_transport(baseline, source),
            'accepted1_source_transport': source_transport(accepted, source),
        }
    report = {
        'baseline': str(baseline), 'accepted': str(accepted), 'accepted_step': 1,
        'source_SHA256': sha(Path(__file__)), 'groups': groups,
        'raw_RF_reconstructed_before': reconstructed_before, 'raw_RF_reconstructed_after': reconstructed_after,
        'raw_RF_recorded_before': before['original_RF_loss'], 'raw_RF_recorded_after': after['original_RF_loss'],
        'new_RF_wave_solves_training_or_gradients': False, 'GPU_work': False,
        'material_arrays_including_truth_not_read': True, 'reserved_case_unread': 'Neg_47_Left',
        'geometry_arrival_windows_do_not_uniquely_localize_scatterers': True,
        'signed_and_envelope_cosines_do_not_uniquely_separate_phase_from_shape': True,
        'joint_reconstruction_achieved': False,
    }
    args.out.mkdir()
    atomic(args.out / 'comparison.json', report)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    names = ['Before 5 mm', '5-12 mm', '12-25 mm', '25-43 mm', 'Later / off-midpoint']
    x = np.arange(5)
    colors = {'lf': '#176b9b', 'point3': '#cd7c15'}
    for source, row in groups.items():
        for level, style, label in [(0, 'o--', 'baseline'), (1, 's-', 'accepted1')]:
            for ax, key in [(axes[0, 0], 'prediction_RMS_over_observed_RMS'),
                            (axes[0, 1], 'signed_RF_cosine'),
                            (axes[1, 0], 'analytic_envelope_cosine')]:
                ax.plot(x, row[key][level], style, color=colors[source],
                        label=f'{source}: {label}', markersize=4)
    axes[0, 0].set_title('Predicted amplitude / observed amplitude')
    axes[0, 1].set_title('Signed RF waveform cosine')
    axes[1, 0].set_title('Analytic-envelope cosine')
    axes[0, 0].set_yscale('log')
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xticks(x, names, rotation=20)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    positions = np.arange(10)
    energy = np.concatenate([groups[s]['raw_RF_increase_prediction_energy_component'] for s in ['lf', 'point3']])
    alignment = np.concatenate([groups[s]['raw_RF_increase_cross_product_component'] for s in ['lf', 'point3']])
    axes[1, 1].bar(positions - .16, energy, .32, label='prediction-energy change', color='#a05b3d')
    axes[1, 1].bar(positions + .16, alignment, .32, label='cross-product change', color='#366e9f')
    axes[1, 1].axhline(0, color='gray', lw=.8)
    axes[1, 1].set_xticks(positions, [f'{s} w{k + 1}' for s in ['LF', 'point'] for k in range(5)], rotation=45)
    axes[1, 1].set_title('Components of original weighted RF objective increase')
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(axis='y', alpha=.2)
    fig.suptitle('Saved first accepted transport step: arrival windows are geometric proxies, not scatterer depths',
                 fontsize=10)
    fig.savefig(args.out / 'RF_decomposition.png', dpi=170)
    plt.close(fig)
    print(json.dumps({
        'raw_RF_before': reconstructed_before, 'raw_RF_after': reconstructed_after,
        'sources': {s: {k: r[k] for k in ['raw_RF_objectives', 'prediction_RMS_over_observed_RMS',
                      'signed_RF_cosine', 'analytic_envelope_cosine',
                      'raw_RF_increase_prediction_energy_component', 'raw_RF_increase_cross_product_component',
                      'baseline_source_transport', 'accepted1_source_transport']}
                    for s, r in groups.items()},
        'new_waves_or_gradients': False,
    }), flush=True)


if __name__ == '__main__':
    main()
