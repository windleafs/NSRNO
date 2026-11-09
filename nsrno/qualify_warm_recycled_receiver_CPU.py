"""Once-only independent CPU check of the new warm/current-K control path."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import numpy as np
import torch
from .recycled_receiver_solver import ReceiverCoarseSpace
from .warm_recycled_receiver_solver import solve_warm_recycled


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This new control-path check is CPU-only')
    if a.out.exists():
        raise RuntimeError('Keep the one actual qualification;never overwrite/replay it')
    torch.set_num_threads(1)
    started = time.time()
    rng = np.random.default_rng(611091)
    M = rng.normal(size=(10, 10))
    K = M.T@M+np.eye(10)*.5
    old_K = K+np.diag(np.linspace(.2, 1.4, 10))
    shift, key = .07, 'independent-new-material-real-linear-K'
    space = ReceiverCoarseSpace(key, maximum=6)

    def pack(v):
        return torch.from_numpy(np.ascontiguousarray(v[::2]+1j*v[1::2]))

    def unpack(v):
        z = v.numpy()
        return np.stack((z.real, z.imag), 1).ravel()

    rows = []
    for source in range(2):
        rhs_real = rng.normal(size=10)
        rhs = pack(rhs_real)
        warm = pack(np.linalg.solve(old_K+shift*np.eye(10), rhs_real))
        exact = np.linalg.solve(K+shift*np.eye(10), rhs_real)
        indices, checkpoints = [], []

        def action(v, index):
            if not indices and not torch.equal(v, warm):
                raise RuntimeError('The first necessary action must recompute current K on old y')
            indices.append(index)
            return pack(K@unpack(v))

        def checkpoint(packet):
            checkpoints.append({'actions': packet['completed_actions'],
                'lambda_squared': packet['lambda_squared']})

        result = solve_warm_recycled(action, rhs, shift, space, key, warm,
            maximum_actions=12, relative_tolerance=1e-10, checkpoint=checkpoint)
        x = unpack(result['x'])
        error = float(np.linalg.norm(x-exact)/np.linalg.norm(exact))
        residual = float(np.linalg.norm((K+shift*np.eye(10))@x-rhs_real)/np.linalg.norm(rhs_real))
        if (indices != list(range(1, len(indices)+1)) or len(indices) > 12 or
                len(indices) != result['actual_K_actions'] or
                result['fixed_lambda_squared'] != shift or not checkpoints or
                any(c['lambda_squared'] != shift for c in checkpoints) or
                checkpoints[-1]['actions'] != len(indices) or
                max(error, residual) > 3e-9):
            raise RuntimeError('New warm-start, fixed shift, known solution or action budget failed')
        rows.append({'source': source, 'actual_current_K_actions': len(indices),
            'inner_CG_actions': result['inner_CG_actions'], 'relative_solution_error': error,
            'independent_matrix_relative_residual': residual,
            'coarse_dimension_used': result['coarse_dimension_used'],
            'maximum_actions_including_seed': 12,
            'requested_tolerance_met': bool(result['converged']),
            'current_K_recomputed_old_material_y': True})

    refused_before_action = [True]
    def forbidden_action(value, index):
        refused_before_action[0] = False
        raise AssertionError('No action may precede operator-key admission')
    try:
        solve_warm_recycled(forbidden_action, rhs, shift, space, 'wrong-material-key', warm)
    except RuntimeError as exc:
        if 'Never reuse receiver images' not in str(exc):
            raise
    else:
        raise RuntimeError('Wrong operator key was accepted')
    if not refused_before_action[0]:
        raise RuntimeError('An action was issued before key validation')

    result = {'status': 'independent_warm_currentK_fixed_penalty_CPU_control_path_passed',
        'seconds': time.time()-started, 'rows': rows,
        'wrong_key_refused_before_action': True, 'patient_truth_read': False,
        'patient_RF_read': False, 'new_wave_solves': 0, 'GPU_work': False,
        'not_patient_acceleration_or_imaging_evidence': True,
        'source_SHA256': {n: digest(Path(__file__).parent/n) for n in
            ['recycled_receiver_solver.py', 'warm_recycled_receiver_solver.py', Path(__file__).name]}}
    a.out.mkdir(parents=True)
    (a.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

