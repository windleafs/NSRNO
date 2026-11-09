"""Analyze fixed estimates using already generated post-hoc RMSE, without truth arrays.

This module must never be imported by an optimizer. No wave propagation or
material derivative is evaluated. The correction cosine follows the squared
error identity, not a new read of patient truth.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from .low_frequency_0p5 import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--initial-material', type=Path, required=True)
    p.add_argument('--initial-control', type=Path, required=True)
    args = p.parse_args()
    folder = args.candidate.resolve()
    output = folder/'actual_spatial_update_analysis.json'
    if output.exists():
        print(output.read_text(), flush=True)
        return
    # Existing map metrics establish that this exact candidate was fixed first.
    metrics = json.loads((folder/'maps.metrics.json').read_text())
    material = np.load(folder/'material.npy').astype(np.float64)
    initial = np.load(args.initial_material).astype(np.float64)
    control = np.load(folder/'control.npy').astype(np.float64)
    control_initial = np.load(args.initial_control).astype(np.float64)
    if material.shape != (3, 961, 1121) or initial.shape != material.shape:
        raise ValueError('Full independent native fields required')
    z = (np.arange(961)-60)*.05
    x = (np.arange(1121)-560)*.05
    ix = abs(x) <= 19.1
    names = ['c_m_s', 'density_kg_m3', 'alpha_dB_cm_MHz1p1']
    result = {'status': 'fixed_material_increment_from_estimates_and_saved_RMSE',
        'candidate_material_SHA256': digest(folder/'material.npy'),
        'initial_material_SHA256': digest(args.initial_material),
        'patient_truth_arrays_read': False, 'new_acoustic_solves': 0,
        'new_material_derivative_probes': 0, 'directions': {}, 'rows': {},
        'lateral_variation_is_not_correct_anatomy': True}
    for region, iz in [('full', (z >= 1)&(z <= 43)), ('25-43', (z >= 25)&(z < 43))]:
        result['directions'][region] = {}
        for j, name in enumerate(names):
            change = (material[j]-initial[j])[iz][:, ix]
            update = float(np.sqrt(np.mean(change**2)))
            saved = metrics['maps'][name]
            if region != 'full':
                saved = saved['by_depth'][region]
            r0 = saved['initial']['rmse']
            r1 = saved['reconstruction']['rmse']
            cosine = ((r0*r0+update*update-r1*r1)/(2*r0*update)
                      if update > 1e-30 and r0 > 0 else None)
            if cosine is not None and abs(cosine) > 1+1e-5:
                raise RuntimeError('Estimate increment and saved evaluation baseline disagree')
            result['directions'][region][name] = {'before_RMSE': r0, 'after_RMSE': r1,
                'after_correlation': saved['reconstruction']['correlation'],
                'actual_update_RMS': update, 'cosine_with_required_truth_correction': cosine}
    iz = (z >= 1)&(z <= 43)
    for j, name in enumerate(names):
        delta = (control[j]-control_initial[j])[iz][:, ix]
        energy = float(np.mean(delta**2))
        lateral = float(np.mean((delta-delta.mean(axis=1, keepdims=True))**2))
        result['rows'][name] = {'actual_increment_control_RMS': float(np.sqrt(energy)),
            'lateral_variation_energy_fraction': lateral/energy if energy > 0 else None}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
