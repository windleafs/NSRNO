"""Independent post-hoc material error in the frozen surface and target body.

Never imported by an optimizer. No propagation, Jacobian or PML evaluation.
This checks a structural frozen-region assumption after the candidate is fixed.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from .v5_materials import evaluation_material
from .low_frequency_0p5 import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--sample', default='train_200')
    args = p.parse_args()
    folder = args.candidate.resolve()
    output = folder/'posthoc_fixed_surface_error.metrics.json'
    if output.exists():
        print(output.read_text(), flush=True)
        return
    if not (folder/'maps.metrics.json').exists():
        raise RuntimeError('The fixed candidate must already have independent post-hoc map evaluation')
    m = np.load(folder/'material.npy').astype(np.float64)
    target = evaluation_material(args.sample).astype(np.float64)
    z = (np.arange(961)-60)*.05
    x = (np.arange(1121)-560)*.05
    names = ['c_m_s', 'density_kg_m3', 'alpha_dB_cm_MHz1p1']
    result = {'status': 'independent_fixed_surface_material_error_CPU',
        'material_SHA256': digest(folder/'material.npy'), 'sample': args.sample,
        'evaluation_only': True, 'truth_read_by_separate_CPU_evaluator': True,
        'truth_not_read_or_imported_by_optimizer': True, 'new_wave_propagations': 0,
        'new_material_derivative_probes': 0, 'frozen_surface_native_rows': [0, 79], 'regions': {}}
    for label, lo, hi in [('coupling_water', -3., 0.), ('frozen_0_to_1mm', 0., 1.),
                          ('unknown_1_to_2mm', 1., 2.), ('unknown_2_to_43mm', 2., 43.000001)]:
        region = (abs(x)[None] <= 19.1)&(z[:, None] >= lo)&(z[:, None] < hi)
        result['regions'][label] = {name: {'RMSE': float(np.sqrt(np.mean((m[j, region]-target[j, region])**2))),
            'estimate_mean': float(m[j, region].mean()), 'target_mean': float(target[j, region].mean())}
            for j, name in enumerate(names)}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
