"""One CPU qualification of weighted full dual, resume, and material derivative."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import time
import torch
from .full_source_dual_cg import solve_dual, FourierReceiverMetric, inner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('Read the existing CPU result; do not repeat the same qualification')
    started = time.time()
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(8042)
    def random(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=rng)
    weights = torch.tensor([.7, 1.1, 0.], dtype=torch.float64)
    active = weights > 0
    root = weights[active].sqrt()[None]
    B = random(12, 12)
    S = B.T@B+.2*torch.eye(12, dtype=torch.float64)
    rhs = torch.complex(random(3, 3), random(3, 3))
    def flat(value):
        return torch.view_as_real((value[:, active]*root).contiguous()).reshape(-1)
    def unflat(value):
        result = torch.zeros((3, 3), dtype=torch.complex128)
        result[:, active] = torch.view_as_complex(value.reshape(3, 2, 2).contiguous())/root
        return result
    calls = []
    def action(value, index):
        if index in calls:
            raise RuntimeError('Resume repeated a previously completed action')
        calls.append(index)
        return unflat(S@flat(value))
    metric = FourierReceiverMetric(random(3, 3).abs()+.5)
    first = solve_dual(rhs, weights, .3, action, metric, max_actions=2, relative_gap=1e-18)
    if first['converged']:
        raise RuntimeError('Synthetic resume case did not exercise an incomplete inner solve')
    stream = io.BytesIO()
    torch.save(first['state'], stream)
    stream.seek(0)
    saved = torch.load(stream, map_location='cpu', weights_only=False)
    solved = solve_dual(rhs, weights, .3, action, metric, max_actions=32,
                        relative_gap=1e-18, state=saved)
    oracle = torch.linalg.solve(S+.3*torch.eye(12, dtype=torch.float64), flat(rhs))
    error = float((flat(solved['y'])-oracle).norm()/oracle.norm())
    if not solved['converged'] or error > 1e-8 or calls != list(range(len(calls))):
        raise RuntimeError('Weighted full-dual/resume qualification failed')

    # A separate part of the same qualification checks the new full-space
    # variable-projection material derivative against an exact dense solve.
    A0 = random(9, 9)
    A0 = A0.T@A0+4*torch.eye(9, dtype=torch.float64)
    M, P, E, b, observed = random(3, 9, 9), random(4, 9), random(9, 3), random(9), random(4)
    theta = torch.tensor([.11, -.08, .05], dtype=torch.float64, requires_grad=True)
    A = A0+(theta[:, None, None]*M).sum(0)
    u = torch.linalg.solve(A, b)
    r = observed-P@u
    w = torch.tensor([.7, 1.2, .7, 1.2], dtype=torch.float64)
    G = P@torch.linalg.solve(A, E)
    K = G@G.T*w[None]
    with torch.no_grad():
        lam2 = float(.1*(r*w*(K@r)).sum()/(r*w*r).sum())
    y_exact = torch.linalg.solve(K+lam2*torch.eye(4, dtype=torch.float64), r)
    objective = (r*w*y_exact).sum()
    automatic, = torch.autograd.grad(objective, theta)
    with torch.no_grad():
        def small_action(value, index):
            return ((K@value.real.reshape(-1)).reshape(2, 2)).to(torch.complex128)
        small = solve_dual(r.reshape(2, 2).to(torch.complex128), w[:2], lam2,
            small_action, FourierReceiverMetric(torch.ones(2, 2)), max_actions=12, relative_gap=1e-20)
        y = small['y'].real.reshape(-1)
        dual_field = torch.linalg.solve(A.T, P.T@(w*y))
        augmented = torch.linalg.solve(A, b+E@(E.T@dual_field))
        J = torch.stack([-P@torch.linalg.solve(A, v@augmented) for v in M], 1)
        formula = -2*(J.T@(w*y))
        gradient_error = float((automatic-formula).norm()/automatic.norm())
    if not small['converged'] or gradient_error > 1e-9 or torch.cuda.is_initialized():
        raise RuntimeError('Full-dual material derivative qualification failed')
    names = ['full_source_dual_cg.py', 'source_dual_receiver_metric.py', Path(__file__).name]
    result = {'status': 'weighted_full_dual_resume_and_three_parameter_formula_CPU_passed',
        'seconds': time.time()-started, 'receiver_CG_actions': solved['action_count'],
        'weighted_solution_relative_error': error, 'relative_duality_gap': solved['bounds']['relative_quality_gap'],
        'resume_repeated_actions': 0, 'three_parameter_gradient_relative_error': gradient_error,
        'patient_data_read': False, 'new_acoustic_propagations': 0,
        'new_patient_material_derivative_probes': 0, 'CUDA_initialized': False,
        'source_SHA256': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in names},
        'scope': 'synthetic linear algebra only, not native reconstruction or physical identifiability'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
