"""One necessary CPU chain-rule check before adopting the TRAIN-only prior."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from .factorized_material_prior import FactorizedMaterialPrior
from .low_frequency_0p5 import atomic, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise RuntimeError('Inspect the same qualification; do not repeat a completed check')
    metadata = json.loads(args.prior.with_name('metadata.json').read_text())
    if (metadata['source_case'] != 'Neg_07_Left' or metadata['development_material_arrays_read'] or
        digest(args.prior) != metadata['marginal_prior_SHA256']):
        raise RuntimeError('Existing independent TRAIN marginal provenance changed')
    args.out.mkdir(parents=True)
    start = time.time()
    rng = np.random.default_rng(3912)
    shape = (11, 13)
    mask = rng.uniform(size=shape) > .25
    edge = rng.uniform(.002, .03, size=shape)
    base = np.stack([rng.uniform(1480., 1590., size=shape),
        rng.uniform(940., 1080., size=shape), rng.uniform(.3, .8, size=shape)])
    scale = np.array([.05, .1, .7])[:, None, None]
    direction = rng.normal(0., .1, size=base.shape)
    prior = FactorizedMaterialPrior(args.prior)
    def material(t):
        m = base*np.exp(t*scale*direction)
        m[2] += edge
        return m
    value, gradient = prior.energy_and_scaled_log_gradient(material(0.), edge, mask)
    analytic = float(np.sum(gradient*direction))
    rows = []
    for step in [.003, .001, .0003, .0001]:
        plus = prior.energy_and_scaled_log_gradient(material(step), edge, mask)[0]
        minus = prior.energy_and_scaled_log_gradient(material(-step), edge, mask)[0]
        finite = (plus-minus)/(2*step)
        rows.append({'step': step, 'finite_difference': finite,
            'relative_error': abs(finite-analytic)/max(abs(analytic), 1e-12)})
    if rows[-1]['relative_error'] > 1e-5 or np.any(gradient[:, ~mask] != 0):
        raise RuntimeError('New independent prior physical-log chain rule failed')
    result = {'status': 'factorized_TRAIN_prior_physical_log_gradient_CPU_passed',
        'seconds': time.time()-start, 'prior_SHA256': digest(args.prior),
        'source_SHA256': digest(Path(__file__).with_name('factorized_material_prior.py')),
        'qualification_source_SHA256': digest(__file__), 'analytic_directional_derivative': analytic,
        'finite_differences': rows, 'patient_arrays_read': False, 'new_training': False,
        'new_wave_propagations': 0, 'CUDA_used': False,
        'not_an_imaging_or_patient_identifiability_test': True}
    atomic(args.out/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
