"""Weighted real-Hilbert PCG for the full receiver-space source dual problem."""
import math
import torch


def inner(a, b, weights):
    return float((weights[None]*(a.conj()*b).real).sum())


def dual_bounds(rhs, y, Ky, lambda_squared, weights, source_energy=None):
    """Exact for a linear adjoint pair; native evaluations are numerical estimates.

    source_energy, if supplied, is ||F* y||^2 from the actual native adjoint.
    Its difference from <y, K y> is retained, not hidden by the recurrence.
    """
    yr = inner(y, rhs, weights)
    yy = inner(y, y, weights)
    yKy = inner(y, Ky, weights)
    energy = yKy if source_energy is None else float(source_energy)
    residual = rhs-Ky-lambda_squared*y
    residual_squared = max(0., inner(residual, residual, weights))
    gap_from_residual = residual_squared/(2*lambda_squared)
    dual = yr-.5*energy-.5*lambda_squared*yy
    primal = .5*energy+.5*inner(rhs-Ky, rhs-Ky, weights)/lambda_squared
    identity_defect = abs(energy-yKy)
    # This is a numerical stopping safeguard, not a rigorous floating-error bound.
    quality_gap = max(primal-dual, gap_from_residual, 0.)+4*identity_defect
    return {'dual': dual, 'primal': primal, 'gap_from_dual_residual': gap_from_residual,
        'primal_minus_dual': primal-dual, 'numerical_identity_defect': identity_defect,
        'numerical_quality_gap': quality_gap,
        'relative_quality_gap': quality_gap/max(abs(dual), 1e-30),
        'dual_residual_squared': residual_squared,
        'numerical_lower': dual-2*identity_defect,
        'numerical_upper': primal+2*identity_defect,
        'native_float_evaluations_are_not_rigorous_global_error_bounds': source_energy is not None}


class FourierReceiverMetric:
    """A fixed positive metric on receiver samples; no material/image filtering."""
    def __init__(self, spectrum):
        if spectrum.ndim != 2 or not torch.isfinite(spectrum).all() or not torch.all(spectrum > 0):
            raise ValueError('Positive finite receiver-frequency metric required')
        self.spectrum = spectrum.double()

    def action(self, value):
        return torch.fft.ifft(torch.fft.fft(value, dim=0, norm='ortho')*self.spectrum,
                              dim=0, norm='ortho')

    def inverse(self, value, scale, lambda_squared):
        return torch.fft.ifft(torch.fft.fft(value, dim=0, norm='ortho')/
                              (scale*self.spectrum+lambda_squared), dim=0, norm='ortho')


@torch.no_grad()
def solve_dual(rhs, weights, lambda_squared, action, metric, max_actions=8,
               relative_gap=.02, state=None, checkpoint=None):
    """Return the full-space iterate and saved recurrence, never fake convergence.

    action(p, index) supplies Kp from a required inverse matvec and may persist
    its result. Resuming retains the total action count; old actions are not run
    again. The first Rayleigh action calibrates the numerical metric and forms
    a valid initial iterate; PCG then restarts in the full receiver space.
    """
    if (rhs.ndim != 2 or not rhs.is_complex() or weights.shape != (rhs.shape[1],) or
        not math.isfinite(lambda_squared) or lambda_squared <= 0 or
        not 1 <= max_actions <= 64 or not 0 < relative_gap < 1):
        raise ValueError('Finite positive dual system and bounded solve required')
    active = weights > 0
    rhs = rhs.detach()*active[None]
    if state is None:
        state = {'initialized': False, 'action_count': 0, 'history': [],
                 'rhs': rhs.clone(), 'lambda_squared': float(lambda_squared),
                 'weights': weights.clone(), 'spectrum': metric.spectrum.clone()}
    else:
        state = dict(state)
        for key in ['rhs', 'weights', 'spectrum', 'y', 'Ky', 'residual', 'p']:
            if key in state:
                state[key] = state[key].to(rhs.device)
        if (state['lambda_squared'] != lambda_squared or
            not torch.equal(state['rhs'], rhs) or not torch.equal(state['weights'], weights) or
            not torch.equal(state['spectrum'], metric.spectrum)):
            raise RuntimeError('Saved full-dual recurrence belongs to another fixed system')

    def save():
        if checkpoint is not None:
            checkpoint(state)

    rr = inner(rhs, rhs, weights)
    if rr <= 1e-40:
        zero = torch.zeros_like(rhs)
        return {'y': zero, 'Ky': zero, 'converged': True, 'action_count': 0,
                'bounds': dual_bounds(rhs, zero, zero, lambda_squared, weights), 'state': state}
    if not state['initialized']:
        save()
        Kr = action(rhs, 0)*active[None]
        rKr = inner(rhs, Kr, weights)
        denominator = rKr+lambda_squared*rr
        if not math.isfinite(denominator) or denominator <= 0 or rKr <= 0:
            raise RuntimeError('Nonpositive required native dual action')
        alpha = rr/denominator
        y, Ky = alpha*rhs, alpha*Kr
        residual = rhs-Ky-lambda_squared*y
        scale = rKr/max(inner(rhs, metric.action(rhs), weights), 1e-300)
        z = metric.inverse(residual, scale, lambda_squared)*active[None]
        rho = inner(residual, z, weights)
        state.update(initialized=True, action_count=1, y=y, Ky=Ky, residual=residual,
                     p=z, rho=rho, metric_scale=scale)
        bounds = dual_bounds(rhs, y, Ky, lambda_squared, weights)
        state['history'].append({'action': 1, 'first_rayleigh_restart': True, **bounds})
        save()
    bounds = dual_bounds(rhs, state['y'], state['Ky'], lambda_squared, weights)
    while bounds['relative_quality_gap'] > relative_gap and state['action_count'] < max_actions:
        p, rho = state['p'], state['rho']
        if not math.isfinite(rho) or rho <= 0:
            raise RuntimeError('Nonpositive full-dual preconditioned residual')
        Kp = action(p, state['action_count'])*active[None]
        denominator = inner(p, Kp, weights)+lambda_squared*inner(p, p, weights)
        if not math.isfinite(denominator) or denominator <= 0:
            raise RuntimeError('Full-dual action lost positive curvature')
        alpha = rho/denominator
        state['y'] = state['y']+alpha*p
        state['Ky'] = state['Ky']+alpha*Kp
        residual = rhs-state['Ky']-lambda_squared*state['y']
        z = metric.inverse(residual, state['metric_scale'], lambda_squared)*active[None]
        next_rho = inner(residual, z, weights)
        state['p'] = z+(next_rho/rho)*p
        state['rho'], state['residual'] = next_rho, residual
        state['action_count'] += 1
        bounds = dual_bounds(rhs, state['y'], state['Ky'], lambda_squared, weights)
        state['history'].append({'action': state['action_count'], 'first_rayleigh_restart': False, **bounds})
        save()
    return {'y': state['y'], 'Ky': state['Ky'],
            'converged': bounds['relative_quality_gap'] <= relative_gap,
            'action_count': state['action_count'], 'bounds': bounds, 'state': state}
