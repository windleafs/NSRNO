"""Real-Hilbert recycled Galerkin/CG for complex-represented receiver data.

K=H H* is real linear, not necessarily complex linear. Only real coefficients
may combine cached images. This module never creates an acoustic operator.
"""
import math
import torch


def inner(a, b):
    return (a.flatten().conj()*b.flatten()).real.sum()


class ReceiverCoarseSpace:
    def __init__(self, operator_key, maximum=16, dependence_tolerance=1e-8):
        if not operator_key or maximum < 0:
            raise ValueError('Explicit fixed-operator identity and bounded space required')
        self.key, self.maximum, self.tolerance = operator_key, maximum, dependence_tolerance
        self.vectors, self.images, self.shape = [], [], None

    def check(self, operator_key):
        if operator_key != self.key:
            raise RuntimeError('Never reuse receiver images at another material/acquisition operator')

    @torch.no_grad()
    def add(self, direction, Kdirection, operator_key):
        self.check(operator_key)
        if direction.shape != Kdirection.shape or not direction.is_complex():
            raise ValueError('Matching complex receiver vectors required')
        if self.shape is None:
            self.shape = direction.shape
        if direction.shape != self.shape or len(self.vectors) >= self.maximum:
            return False
        v, kv = direction.detach().flatten().clone(), Kdirection.detach().flatten().clone()
        original = float(torch.sqrt(inner(v, v)))
        if not math.isfinite(original) or original <= 1e-30:
            return False
        for _ in range(2):
            for q, kq in zip(self.vectors, self.images):
                coefficient = inner(q, v)
                v -= coefficient*q
                kv -= coefficient*kq
        norm = float(torch.sqrt(inner(v, v)))
        if norm <= self.tolerance*original:
            return False
        self.vectors.append(v/norm)
        self.images.append(kv/norm)
        return True

    @torch.no_grad()
    def factor(self, shift, operator_key):
        self.check(operator_key)
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError('A strictly positive fixed normal-system shift is required')
        if not self.vectors:
            return None
        V, KV = torch.stack(self.vectors, 1), torch.stack(self.images, 1)
        AV = KV+shift*V
        gram = (V.conj().T@AV).real
        gram = (gram+gram.T)/2
        values, rotation = torch.linalg.eigh(gram)
        threshold = max(float(values.max())*1e-12, 1e-30)
        keep = values > threshold
        if not bool(keep.any()):
            return None
        U = rotation[:, keep].to(V.dtype)
        return {'W': V@U, 'AW': AV@U, 'values': values[keep],
            'candidate_dimension': len(self.vectors), 'retained_dimension': int(keep.sum())}


@torch.no_grad()
def coarse_correction(residual, factor):
    if factor is None:
        return torch.zeros_like(residual), torch.zeros_like(residual)
    W, AW, values = [factor[k] for k in ['W', 'AW', 'values']]
    coefficients = ((W.conj().T@residual.flatten()).real/values).to(W.dtype)
    return (W@coefficients).reshape_as(residual), (AW@coefficients).reshape_as(residual)


@torch.no_grad()
def complement_direction(residual, factor):
    if factor is None:
        return residual.clone()
    W, AW, values = [factor[k] for k in ['W', 'AW', 'values']]
    coefficients = ((AW.conj().T@residual.flatten()).real/values).to(W.dtype)
    return (residual.flatten()-W@coefficients).reshape_as(residual)


@torch.no_grad()
def solve_shifted(K, rhs, shift, space, operator_key, *, initial=None, initial_K=None,
        maximum_actions=100, relative_tolerance=1e-8, checkpoint=None):
    """All K calls are actual solve actions; callers own propagation accounting.

    Known initial_K must refer to the exact same operator key. Returned Ky is
    accumulated from used actions; an actual acoustic final field remains the
    authoritative residual check in the reconstruction caller.
    """
    space.check(operator_key)
    factor = space.factor(shift, operator_key)
    rhs_norm = float(torch.sqrt(inner(rhs, rhs)))
    x = torch.zeros_like(rhs) if initial is None else initial.detach().clone()
    actions, pending = 0, []
    def action(v):
        nonlocal actions
        if actions >= maximum_actions:
            raise RuntimeError('The explicit actual receiver action budget is exhausted')
        actions += 1
        result = K(v, actions)
        if result.shape != rhs.shape or not bool(torch.isfinite(result).all()):
            raise RuntimeError('Invalid actual receiver operator result')
        if len(space.vectors)+len(pending) < space.maximum:
            pending.append((v.detach().clone(), result.detach().clone()))
        return result
    if initial_K is not None:
        Kx = initial_K.detach().clone()
    elif initial is None or not bool(x.any()):
        Kx = torch.zeros_like(rhs)
    else:
        Kx = action(x)
    residual = rhs-Kx-shift*x
    correction, A_correction = coarse_correction(residual, factor)
    x += correction
    Kx += A_correction-shift*correction
    residual -= A_correction
    z = complement_direction(residual, factor)
    rho = float(inner(residual, z))
    direction = z.clone()
    history = []
    relative = float(torch.sqrt(inner(residual, residual)))/max(rhs_norm, 1e-30)
    while relative > relative_tolerance and actions < maximum_actions:
        if not math.isfinite(rho) or rho <= 0:
            raise RuntimeError('Recycled complement lost positive residual curvature')
        Kp = action(direction)
        Ap = Kp+shift*direction
        denominator = float(inner(direction, Ap))
        if not math.isfinite(denominator) or denominator <= 0:
            raise RuntimeError('The shifted real-Hilbert system is not positive')
        fraction = rho/denominator
        x += fraction*direction
        Kx += fraction*Kp
        residual -= fraction*Ap
        relative = float(torch.sqrt(inner(residual, residual)))/max(rhs_norm, 1e-30)
        history.append({'action': actions, 'relative_recursive_residual': relative})
        if checkpoint is not None:
            checkpoint({'x': x, 'Kx': Kx, 'residual': residual, 'actions': actions})
        if relative <= relative_tolerance:
            break
        z = complement_direction(residual, factor)
        next_rho = float(inner(residual, z))
        direction = z+(next_rho/rho)*direction
        rho = next_rho
    for vector, image in pending:
        space.add(vector, image, operator_key)
    accumulated_residual = rhs-Kx-shift*x
    return {'x': x, 'accumulated_Kx': Kx, 'actions': actions,
        'relative_recursive_residual': relative,
        'relative_accumulated_residual': float(torch.sqrt(inner(accumulated_residual, accumulated_residual)))/max(rhs_norm, 1e-30),
        'converged': relative <= relative_tolerance,
        'coarse_dimension_used': factor['retained_dimension'] if factor else 0,
        'next_coarse_dimension': len(space.vectors), 'history': history,
        'real_linear_complex_representation': True, 'original_operator_not_modified': True}
