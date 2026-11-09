"""One algebra check for the new saved-product residual-minimizing solver."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from .full_source_dual_cg import FourierReceiverMetric
from .full_source_dual_minres import solve_dual
from .low_frequency_0p5 import atomic, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('Do not repeat an already issued qualification')
    args.out.mkdir()
    torch.set_num_threads(1)
    started = time.time()
    rng = np.random.default_rng(473)
    nr, nf = 3, 2
    n = 2*nr*nf
    root = rng.standard_normal((n, n))
    H = root.T@root+.2*np.eye(n)
    weights = torch.tensor([.3, 1.7], dtype=torch.float64)
    shape = (nr, nf)
    w = np.sqrt(np.tile(weights.numpy(), nr))
    ww = np.concatenate([w, w])
    rhs = torch.from_numpy(rng.standard_normal(shape)+1j*rng.standard_normal(shape))
    metric = FourierReceiverMetric(torch.from_numpy(np.exp(rng.normal(size=shape))))
    lam = .07
    issued = []
    folder = args.out/'actions'
    folder.mkdir()
    def vector(p):
        p = p.numpy().ravel()
        return np.concatenate([p.real, p.imag])
    def tensor(v):
        return torch.from_numpy((v[:nr*nf]+1j*v[nr*nf:]).reshape(shape))
    def action(p, index):
        if index in issued:
            raise RuntimeError('A saved toy action was repeated')
        issued.append(index)
        result = tensor((H@(vector(p)*ww))/ww)
        path = folder/f'action{index:03d}.npz'
        np.savez(path, p=p.numpy(), Kp=result.numpy())
        atomic(path.with_suffix('.json'), {'direction_SHA256': hashlib.sha256(p.numpy().tobytes()).hexdigest(),
                                         'array_SHA256': digest(path)})
        return result
    first = solve_dual(rhs, weights, lam, action, metric, max_actions=3, relative_gap=1e-12)
    done = solve_dual(rhs, weights, lam, action, metric, max_actions=n,
                      relative_gap=1e-12, state=first['state'], saved_actions=folder)
    exact = tensor(np.linalg.solve(H+lam*np.eye(n), vector(rhs)*ww)/ww)
    error = float(torch.linalg.vector_norm(done['y']-exact)/torch.linalg.vector_norm(exact))
    residuals = [r['dual_residual_squared'] for r in done['state']['history']]
    if (not done['converged'] or error > 1e-9 or issued != list(range(done['action_count'])) or
        any(b > a+1e-11 for a, b in zip(residuals, residuals[1:]))):
        raise RuntimeError('Full-space residual minimization or saved-action reuse failed')
    source = Path(__file__).parent
    result = {'status': 'saved_product_full_residual_minimization_CPU_passed',
        'seconds': time.time()-started, 'source_SHA256': {name: digest(source/name) for name in
            ['full_source_dual_minres.py', Path(__file__).name]},
        'relative_solution_error': error, 'relative_gap': done['bounds']['relative_quality_gap'],
        'actions_before_resume': first['action_count'], 'total_actions': done['action_count'],
        'repeated_actions': 0, 'complete_receiver_residual_monotone': True,
        'patient_truth_read': False, 'new_acoustic_propagations': 0, 'CUDA_used': False}
    atomic(args.out/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
