"""One known-solution CPU check of a non-complex-linear recycled source solver."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from .recycled_receiver_solver import ReceiverCoarseSpace, solve_shifted
from .low_frequency_0p5 import atomic, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('Do not repeat a completed new solver qualification')
    a.out.mkdir(parents=True)
    started = time.time()
    torch.set_num_threads(1)
    rng = np.random.default_rng(6111101)
    n = 24
    orthogonal, _ = np.linalg.qr(rng.normal(size=(2*n, 2*n)))
    matrix = (orthogonal*np.logspace(-2, 2, 2*n))@orthogonal.T
    def pack(x):
        return torch.from_numpy((x[:n]+1j*x[n:]).copy())
    def unpack(x):
        return np.concatenate((x.numpy().real, x.numpy().imag))
    def K(x, index):
        return pack(matrix@unpack(x))
    space = ReceiverCoarseSpace('synthetic_fixed_real_operator', maximum=16)
    results = []
    previous, previous_K = None, None
    first = rng.normal(size=2*n)
    for index, shift in enumerate([.03, .07, .15]):
        rhs_real = first+.12*index*rng.normal(size=2*n)
        rhs = pack(rhs_real)
        solved = solve_shifted(K, rhs, shift, space, space.key, initial=previous,
            initial_K=previous_K, maximum_actions=120, relative_tolerance=1e-10)
        truth = np.linalg.solve(matrix+shift*np.eye(2*n), rhs_real)
        estimate = unpack(solved['x'])
        error = np.linalg.norm(estimate-truth)/np.linalg.norm(truth)
        residual = np.linalg.norm((matrix+shift*np.eye(2*n))@estimate-rhs_real)/np.linalg.norm(rhs_real)
        results.append({'shift': shift, 'relative_solution_error': float(error),
            'independent_matrix_relative_residual': float(residual), 'actual_matrix_actions': solved['actions'],
            'coarse_dimension_used': solved['coarse_dimension_used'], 'next_dimension': solved['next_coarse_dimension']})
        previous, previous_K = solved['x'], solved['accumulated_Kx']
    passed = all(r['relative_solution_error'] < 5e-8 and r['independent_matrix_relative_residual'] < 5e-8 for r in results)
    result = {'status': 'independent_real_linear_multishift_recycled_solver_CPU_passed' if passed else 'failed_for_inspection',
        'rows': results, 'seconds': time.time()-started, 'patient_arrays_read': False,
        'new_acoustic_propagations': 0, 'new_patient_derivative_probes': 0,
        'GPU_used': False, 'does_not_prove_patient_acceleration_or_imaging': True,
        'source_SHA256': {name: digest(Path(__file__).with_name(name)) for name in
            ['recycled_receiver_solver.py', Path(__file__).name]}}
    atomic(a.out/'result.json', result)
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('Inspect new real-linear recycled solver before deployment')


if __name__ == '__main__':
    main()
