"""One synthetic CPU derivative qualification before this new energy is used."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from .conditional_patch_prior import ConditionalPatchPrior


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    project = Path.cwd().resolve()
    fit, out = args.fit_root.resolve(), args.out.resolve()
    if not fit.is_relative_to(project/'out/v5') or not out.is_relative_to(project/'out/v5'):
        raise ValueError('Same-project fitted TRAIN prior only')
    if out.exists():
        raise FileExistsError('Read existing qualification; do not repeat it')
    run = json.loads((fit/'run.json').read_text())
    prior_path = fit/'products/prior.npz'
    source_path = Path(__file__).with_name('conditional_patch_prior.py')
    if (run['status'] != 'TRAIN_conditional_patch_prior_CPU_fit_finished' or
        run['development_arrays_read'] or run['reserved_arrays_read'] or
        digest(prior_path) != run['prior_SHA256'] or
        digest(source_path) != run['source_SHA256']['conditional_patch_prior.py']):
        raise RuntimeError('TRAIN model or source changed before qualification')
    started = time.time()
    prior = ConditionalPatchPrior(prior_path)
    rng = np.random.default_rng(20261108)
    z, x = np.mgrid[:67, :69]
    u = np.empty((3, 67, 69))
    for field in range(3):
        model = prior.models[field, 1]
        center = float(model.center_mean[np.argmax(model.weight)])
        u[field] = center+.025*np.sin(.19*z+.11*x)+.015*rng.normal(size=z.shape)
    edge = np.full(z.shape, .03)
    material = np.stack([1540.+80.*u[0], 1000.+100.*u[1],
                         edge+.5*np.exp(.7*u[2])])
    roi = np.ones(z.shape, bool)
    baseline, gradient = prior.energy_and_scaled_log_gradient(material, edge, roi)
    def energy(control):
        changed = np.stack([material[0]*np.exp(.05*control[0]),
            material[1]*np.exp(.1*control[1]),
            edge+(material[2]-edge)*np.exp(.7*control[2])])
        return prior.energy_and_scaled_log_gradient(changed, edge, roi)[0]
    rows = []
    for field in range(3):
        norm = float(np.linalg.norm(gradient[field]))
        if not np.isfinite(norm) or norm < 1e-7:
            raise RuntimeError('Synthetic derivative direction is uninformative')
        direction = np.zeros_like(gradient)
        direction[field] = gradient[field]/norm
        epsilon = 1e-5
        finite = (energy(epsilon*direction)-energy(-epsilon*direction))/(2*epsilon)
        analytic = float(np.sum(gradient*direction))
        relative = abs(finite-analytic)/max(abs(finite), abs(analytic), 1e-12)
        rows.append({'field': field, 'analytic': analytic, 'finite_difference': finite,
                     'relative_error': relative, 'epsilon': epsilon})
    passed = all(r['relative_error'] < 2e-4 for r in rows)
    result = {'status': 'conditional_patch_synthetic_scaled_log_gradient_CPU_passed' if passed
              else 'conditional_patch_synthetic_scaled_log_gradient_CPU_failed',
        'seconds': time.time()-started, 'synthetic_shape': list(material.shape),
        'baseline_conditional_energy': baseline, 'checks': rows,
        'maximum_relative_error': max(r['relative_error'] for r in rows),
        'source_SHA256': digest(source_path), 'qualification_SHA256': digest(__file__),
        'prior_SHA256': digest(prior_path), 'patient_arrays_read': False,
        'CUDA_used': False, 'new_acoustic_solves': 0, 'new_patient_derivative_probes': 0,
        'all_three_patch_scales_and_overlap_used': True}
    out.mkdir()
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('New spatial-prior gradient failed; do not use it in a physical inverse')


if __name__ == '__main__':
    main()
