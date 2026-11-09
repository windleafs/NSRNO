"""Full receiver residual minimization with reusable native operator products.

All coefficients are real because the complex receiver representation encodes
a real Hilbert space. Only new independent directions request native K actions.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from .full_source_dual_cg import dual_bounds, inner
from .low_frequency_0p5 import digest


def load_products(folder, count, rhs):
    directions, products = [], []
    for index in range(count):
        path = Path(folder)/f'action{index:03d}.npz'
        row = json.loads(path.with_suffix('.json').read_text())
        if digest(path) != row['array_SHA256']:
            raise RuntimeError('Saved native product changed')
        with np.load(path) as saved:
            p, Kp = saved['p'].copy(), saved['Kp'].copy()
        if (hashlib.sha256(p.tobytes()).hexdigest() != row['direction_SHA256'] or
            p.shape != rhs.shape or Kp.shape != rhs.shape):
            raise RuntimeError('Saved source direction changed')
        directions.append(p)
        products.append(Kp)
    return directions, products


def residual_minimum(rhs, weights, lam, directions, products):
    """CPU least squares on stored full data, not an observation projection."""
    shape = rhs.shape
    P = np.stack([p.ravel() for p in directions], 1)
    KP = np.stack([p.ravel() for p in products], 1)
    root = np.sqrt(np.broadcast_to(weights[None], shape).ravel())
    B = (KP+lam*P)*root[:, None]
    matrix = np.concatenate([B.real, B.imag], axis=0)
    b = rhs.ravel()*root
    target = np.concatenate([b.real, b.imag])
    scales = np.linalg.norm(matrix, axis=0)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise RuntimeError('Invalid full-space residual basis')
    coefficients, _, rank, singular = np.linalg.lstsq(matrix/scales, target, rcond=1e-11)
    coefficients /= scales
    return ((P@coefficients).reshape(shape), (KP@coefficients).reshape(shape),
            {'rank': int(rank), 'columns': len(directions),
             'column_scaled_condition': float(singular[0]/singular[-1])})


def orthogonal_direction(candidate, directions, weights):
    """Remove old directions in the weighted real receiver inner product."""
    shape = candidate.shape
    root = np.sqrt(np.broadcast_to(weights[None], shape).ravel())
    P = np.stack([p.ravel() for p in directions], 1)
    C = P*root[:, None]
    C = np.concatenate([C.real, C.imag], 0)
    scales = np.linalg.norm(C, axis=0)
    c = candidate.ravel()*root
    c = np.concatenate([c.real, c.imag])
    before = float(np.linalg.norm(c))
    beta = np.linalg.lstsq(C/scales, c, rcond=1e-11)[0]/scales
    p = candidate.ravel()-P@beta
    # A second pass controls cancellation in inherited short-recurrence bases.
    c2 = p*root
    c2 = np.concatenate([c2.real, c2.imag])
    beta2 = np.linalg.lstsq(C/scales, c2, rcond=1e-11)[0]/scales
    p -= P@beta2
    norm = float(np.sqrt(np.sum(abs(p*root)**2)))
    if not math.isfinite(norm) or norm <= max(1e-12*before, 1e-300):
        return None
    return (p/norm).reshape(shape)


@torch.no_grad()
def solve_dual(rhs, weights, lambda_squared, action, metric, max_actions=20,
               relative_gap=.02, state=None, checkpoint=None, saved_actions=None,
               warm_start=None):
    if (rhs.ndim != 2 or not rhs.is_complex() or weights.shape != (rhs.shape[1],) or
        not 1 <= max_actions <= 32 or lambda_squared <= 0 or not 0 < relative_gap < 1):
        raise ValueError('Bounded positive full-dual system required')
    active = weights > 0
    rhs = rhs.detach()*active[None]
    if state is not None:
        state = dict(state)
        for key in ['rhs', 'weights', 'spectrum', 'y', 'Ky', 'residual', 'p']:
            if key in state:
                state[key] = state[key].to(rhs.device)
        if (state['lambda_squared'] != lambda_squared or not torch.equal(state['rhs'], rhs) or
            not torch.equal(state['weights'], weights) or not torch.equal(state['spectrum'], metric.spectrum)):
            raise RuntimeError('Continuation changed the saved physical system')
    else:
        state = {'initialized': False, 'action_count': 0, 'history': [], 'rhs': rhs.clone(),
                 'lambda_squared': float(lambda_squared), 'weights': weights.clone(),
                 'spectrum': metric.spectrum.clone()}
    if state['action_count'] and saved_actions is None:
        raise RuntimeError('Native product archive required for continuation')
    directions, products = load_products(saved_actions, state['action_count'], rhs)
    rhs_np, weights_np = rhs.cpu().numpy(), weights.cpu().numpy()

    def save():
        if checkpoint is not None:
            checkpoint(state)

    def fit():
        y, Ky, details = residual_minimum(rhs_np, weights_np, lambda_squared, directions, products)
        state.update(initialized=True, y=torch.from_numpy(y).to(rhs.device),
                     Ky=torch.from_numpy(Ky).to(rhs.device))
        state['residual'] = rhs-state['Ky']-lambda_squared*state['y']
        bounds = dual_bounds(rhs, state['y'], state['Ky'], lambda_squared, weights)
        state['history'].append({'action': state['action_count'],
            'full_residual_minimum': True, 'basis': details, **bounds})
        save()
        return bounds

    if not directions:
        first = rhs
        scale = None
        if warm_start is not None:
            old_y = warm_start['y'].to(rhs.device)*active[None]
            if inner(old_y, old_y, weights) > 1e-40:
                first, scale = old_y, warm_start['metric_scale']
        save()
        product = action(first, 0)*active[None]
        if scale is None:
            scale = inner(first, product, weights)/max(inner(first, metric.action(first), weights), 1e-300)
        if not math.isfinite(scale) or scale <= 0:
            raise RuntimeError('Nonpositive required native dual action')
        state.update(action_count=1, metric_scale=scale)
        directions.append(first.cpu().numpy().copy())
        products.append(product.cpu().numpy().copy())
    bounds = fit()
    while not (bounds['dual'] > 0 and bounds['relative_quality_gap'] <= relative_gap):
        if state['action_count'] >= max_actions:
            break
        candidate = metric.inverse(state['residual'], state['metric_scale'], lambda_squared)*active[None]
        p = orthogonal_direction(candidate.cpu().numpy(), directions, weights_np)
        if p is None:
            p = orthogonal_direction(state['residual'].cpu().numpy(), directions, weights_np)
        if p is None:
            raise RuntimeError('Saved directions exhausted the resolvable receiver space')
        direction = torch.from_numpy(p).to(rhs.device)
        product = action(direction, state['action_count'])*active[None]
        directions.append(p.copy())
        products.append(product.cpu().numpy().copy())
        state['action_count'] += 1
        bounds = fit()
    return {'y': state['y'], 'Ky': state['Ky'], 'bounds': bounds, 'state': state,
        'action_count': state['action_count'],
        'converged': bounds['dual'] > 0 and bounds['relative_quality_gap'] <= relative_gap}
