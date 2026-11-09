"""One necessary CPU check of the new reduced-objective correction, no waves."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import torch
from .source_extension_ray import ray_scalars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('Read the existing qualification instead of repeating it')
    started = time.time()
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(68117)
    def random(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=rng)
    A0 = random(9, 9)
    A0 = A0.T@A0+3*torch.eye(9, dtype=torch.float64)
    matrices, P, E = random(3, 9, 9), random(4, 9), random(9, 3)
    b, d = random(9), random(4)
    theta = torch.tensor([.12, -.07, .09], dtype=torch.float64, requires_grad=True)
    A = A0+(theta[:, None, None]*matrices).sum(0)
    physical = torch.linalg.solve(A, b)
    r = d-P@physical
    dual = torch.linalg.solve(A.T, P.T@r)
    rr, aa = r.square().sum(), (E.T@dual).square().sum()
    _, alpha, calibration = ray_scalars(float(rr.detach()), float(aa.detach()))
    lam2, norm = calibration['lambda_squared'], calibration['normalization']
    loss = norm*rr.square()/(aa+lam2*rr)
    automatic, = torch.autograd.grad(loss, theta)
    with torch.no_grad():
        augmented = torch.linalg.solve(A, b+alpha*(E@(E.T@dual)))
        gy = r-P@(augmented-physical)-lam2*alpha*r
        J0 = torch.stack([-P@torch.linalg.solve(A, M@physical) for M in matrices], 1)
        Jaug = torch.stack([-P@torch.linalg.solve(A, M@augmented) for M in matrices], 1)
        corrected = -2*alpha*norm*(Jaug.T@r+J0.T@gy)
        omitted = -2*alpha*norm*(Jaug.T@r)
        denominator = max(float(automatic.norm()), 1e-20)
        relative = float((corrected-automatic).norm())/denominator
        omission_error = float((omitted-automatic).norm())/denominator
    if relative > 1e-11 or omission_error < 1e-3 or torch.cuda.is_initialized():
        raise RuntimeError('Reduced-ray algebra qualification failed')
    files = ['source_extension_ray.py', 'native_source_extension.py',
             'native_source_extension.cu', Path(__file__).name]
    source = Path(__file__).parent
    result = {'status': 'corrected_reduced_ray_algebra_CPU_passed',
        'seconds': time.time()-started, 'relative_gradient_error': relative,
        'relative_error_if_required_correction_omitted': omission_error,
        'three_independent_synthetic_parameters': True, 'synthetic_matrix_size': 9,
        'patient_data_read': False, 'new_acoustic_propagations': 0,
        'new_patient_material_derivative_probes': 0, 'CUDA_initialized': False,
        'scope': 'new reduced objective algebra only; not CUDA execution or reconstruction',
        'source_SHA256': {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
