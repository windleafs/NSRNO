"""Minimize the full receiver residual using already completed native K actions."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from .full_source_dual_cg import dual_bounds
from .low_frequency_0p5 import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    task, out = args.task_root.resolve(), args.out.resolve()
    project = Path.cwd().resolve()
    if not task.is_relative_to(Path.home()/'nsrno_artifacts'):
        raise ValueError('Only the dedicated same-account source-task artifact volume')
    if not out.is_relative_to(Path.home()/'nsrno_artifacts'):
        raise ValueError('CPU candidate output must remain on the dedicated artifact volume')
    if out.exists():
        result = out/'result.json'
        if result.exists():
            print(result.read_text(), flush=True)
            return
        raise RuntimeError('Inspect partial same CPU calculation; do not repeat')
    completed = json.loads((task/'result.json').read_text())
    work = json.loads((task/'native_work.json').read_text())
    if completed['actual_native_inner_converged'] or work['inflight_phase'] is not None:
        raise RuntimeError('Only a finished, still-incomplete source task should be refined')
    started = time.time()
    torch.set_num_threads(1)
    state = torch.load(task/'dual_state.pt', map_location='cpu', weights_only=False)
    rhs, weights, lam2 = state['rhs'].numpy(), state['weights'].numpy(), state['lambda_squared']
    files = sorted((task/'operator_actions').glob('action*.json'))
    if len(files) != state['action_count']:
        raise RuntimeError('Saved completed action count changed')
    directions, products, receipts = [], [], []
    for file in files:
        row = json.loads(file.read_text())
        array = file.with_suffix('.npz')
        if digest(array) != row['array_SHA256']:
            raise RuntimeError('A saved native K action changed')
        with np.load(array) as data:
            p, Kp = data['p'].copy(), data['Kp'].copy()
        if hashlib.sha256(p.tobytes()).hexdigest() != row['direction_SHA256']:
            raise RuntimeError('Saved action direction changed')
        if p.shape != rhs.shape or Kp.shape != rhs.shape:
            raise RuntimeError('Wrong full receiver data shape')
        directions.append(p)
        products.append(Kp)
        receipts.append({'path': str(file), 'SHA256': digest(file), 'array_SHA256': row['array_SHA256']})
    weight = np.broadcast_to(weights[None], rhs.shape).ravel()
    root = np.sqrt(weight)
    P = np.stack([p.ravel() for p in directions], 1)
    KP = np.stack([p.ravel() for p in products], 1)
    A = (KP+lam2*P)*root[:, None]
    b = rhs.ravel()*root
    matrix = np.concatenate([A.real, A.imag], 0)
    target = np.concatenate([b.real, b.imag], 0)
    scales = np.linalg.norm(matrix, axis=0)
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise RuntimeError('Invalid already-computed dual directions')
    normalized = matrix/scales[None]
    scaled, residuals, rank, singular = np.linalg.lstsq(normalized, target, rcond=1e-10)
    coefficients = scaled/scales
    y = (P@coefficients).reshape(rhs.shape)
    Ky = (KP@coefficients).reshape(rhs.shape)
    before = dual_bounds(state['rhs'], state['y'], state['Ky'], lam2, state['weights'])
    after = dual_bounds(state['rhs'], torch.from_numpy(y), torch.from_numpy(Ky), lam2, state['weights'])
    norms = np.sqrt((weight[:, None]*(P.conj()*P).real).sum(0))
    gram = ((P.conj()*weight[:, None]).T@KP).real/np.outer(norms, norms)
    skew = float(np.linalg.norm(gram-gram.T)/max(2*np.linalg.norm(gram), 1e-300))
    out.mkdir()
    np.savez(out/'candidate.npz', y=y, Ky=Ky, coefficients=coefficients)
    result = {'status': 'saved_full_dual_residual_minimum_CPU_candidate', 'source_task': str(task),
        'seconds': time.time()-started, 'saved_K_actions_reused': len(files), 'real_subspace_rank': int(rank),
        'column_scaled_residual_matrix_condition': float(singular[0]/singular[-1]),
        'normalized_source_action_Gram_relative_skew': skew,
        'before': before, 'after': after, 'coefficients': coefficients.tolist(),
        'source_action_receipts': receipts, 'candidate_SHA256': digest(out/'candidate.npz'),
        'new_acoustic_propagations': 0, 'new_patient_material_derivative_probes': 0,
        'patient_truth_read': False, 'native_state_not_modified': True,
        'actual_native_inner_convergence_still_requires_reconstruction': True,
        'subspace_residual_is_evaluated_in_the_complete_existing_receiver_space': True}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    compact = {k: result[k] for k in ['status', 'source_task', 'seconds', 'saved_K_actions_reused',
        'real_subspace_rank', 'column_scaled_residual_matrix_condition',
        'normalized_source_action_Gram_relative_skew', 'before', 'after', 'new_acoustic_propagations',
        'native_state_not_modified', 'actual_native_inner_convergence_still_requires_reconstruction']}
    print(json.dumps(compact), flush=True)


if __name__ == '__main__':
    main()
