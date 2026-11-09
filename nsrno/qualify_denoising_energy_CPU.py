"""One synthetic CPU derivative check of the newly introduced prior energy."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from .independent_denoising_prior import IndependentMaterialDenoisers, denoising_energy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('Do not repeat a completed new-energy check')
    started = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(6110901)
    model = IndependentMaterialDenoisers().double().eval().requires_grad_(False)
    rng = np.random.default_rng(6110901)
    checks = []
    for f in range(3):
        value = torch.from_numpy(rng.normal(size=(1, 1, 40, 48))).requires_grad_()
        direction = torch.from_numpy(rng.normal(size=value.shape))
        direction /= direction.square().mean().sqrt()
        sigma = [.05, .2, .6][f]
        energy = denoising_energy(model.fields[f], value, sigma, f)
        gradient, = torch.autograd.grad(energy, value)
        analytic = float((gradient*direction).sum())
        eps = 1e-5
        finite = float((denoising_energy(model.fields[f], value+eps*direction, sigma, f)
                      -denoising_energy(model.fields[f], value-eps*direction, sigma, f))/(2*eps))
        error = abs(analytic-finite)/max(abs(analytic), abs(finite), 1e-12)
        if error > 2e-6:
            raise RuntimeError('Full denoiser chain derivative did not match the explicit energy')
        checks.append({'field': f, 'scale': f, 'sigma': sigma, 'analytic': analytic,
            'finite_difference': finite, 'relative_error': error})
    identities = [{id(v) for v in field.parameters()} for field in model.fields]
    independent = not any(identities[i]&identities[j] for i in range(3) for j in range(i))
    assert independent
    source = Path(__file__).with_name('independent_denoising_prior.py')
    result = {'status': 'new_denoising_residual_energy_synthetic_CPU_derivative_passed',
        'source_SHA256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'seconds': time.time()-started, 'checks': checks,
        'maximum_relative_error': max(c['relative_error'] for c in checks),
        'field_parameters_independent': independent, 'patient_arrays_read': False,
        'new_acoustic_solves': 0, 'new_material_derivative_probes': 0, 'CUDA_used': False}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
