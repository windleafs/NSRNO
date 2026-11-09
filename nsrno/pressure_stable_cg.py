"""Independent PCG in a GPU batch, with true positive energy curvature.

All transmissions remain present. There is no ill-conditioned transmission
Gram inversion. Every accepted field decreases its own actual augmented
objective. Roundoff stagnation is reported as NOT converged, never as a pass.
"""
import torch


def dot_real(a, b):
    x, y = a.flatten(1), b.flatten(1)
    result = torch.zeros(len(x), dtype=torch.float64, device=x.device)
    for j in range(0, x.shape[1], 131072):
        u, v = x[:, j:j+131072].to(torch.complex128), y[:, j:j+131072].to(torch.complex128)
        result.add_((u.conj()*v).real.sum(1))
    return result


def scale(values, coefficients):
    return values*coefficients.to(values.real.dtype).reshape(-1, *([1]*(values.ndim-1)))


def select(mask, a, b):
    return torch.where(mask.reshape(-1, *([1]*(a.ndim-1))), a, b)


@torch.no_grad()
def update_independent(operator, rhs, target, response, current, penalty,
                       forward, adjoint, precondition, maximum=64, tolerance=.003):
    if maximum < 1 or penalty <= 0:
        raise ValueError('Positive field iteration budget and penalty required')
    def gradient(eq, rx):
        return operator.readout_adjoint(rx, response)+penalty*operator.adjoint(eq)
    full = gradient(rhs, target)
    den = dot_real(full, full).clamp_min(1e-30)
    pre_full = adjoint(full)
    pre_den = dot_real(pre_full, pre_full).clamp_min(1e-30)
    del full, pre_full
    base = current.clone()
    eq = rhs-operator(base)
    rx = target-operator.readout(base, response)
    cost = dot_real(rx, rx)+penalty*dot_real(eq, eq)
    before = cost.clone()
    grad = gradient(eq, rx)
    right = adjoint(grad)
    physical = (dot_real(grad, grad)/den).sqrt()
    pre = (dot_real(right, right)/pre_den).sqrt()
    if not torch.isfinite(cost).all() or not torch.isfinite(physical).all():
        raise RuntimeError('Nonfinite initial field/objective in stable CG')
    converged = (physical <= tolerance)&(pre <= tolerance)
    active = ~converged
    reasons = ['converged' if bool(v) else 'iteration_limit' for v in converged]
    best_normal = torch.maximum(physical, pre)
    stalls = torch.zeros(len(base), dtype=torch.long, device=base.device)
    correction = torch.zeros_like(base)
    residual = right.clone()
    z = precondition(residual)
    direction = select(active, z, torch.zeros_like(z))
    rho = dot_real(residual, z)
    history = []
    for iteration in range(1, maximum+1):
        if not bool(active.any()):
            break
        field = forward(direction)
        bf = operator(field)
        cf = operator.readout(field, response)
        image = adjoint(gradient(bf, cf))
        # Algebraically d*P*HPd; evaluating by energies avoids a noisy,
        # almost rank-deficient 11x11 curvature Gram at the FP32 floor.
        curvature = dot_real(cf, cf)+penalty*dot_real(bf, bf)
        valid = active&torch.isfinite(rho)&torch.isfinite(curvature)&(rho > 0)&(curvature > 0)
        broken = active&~valid
        for j in broken.nonzero().flatten().tolist():
            reasons[j] = 'nonpositive_or_nonfinite_search_curvature'
        active &= valid
        alpha = torch.where(active, rho/curvature.clamp_min(1e-300), 0.)
        correction.add_(scale(direction, alpha))
        residual.sub_(scale(image, alpha))
        recursive = (dot_real(residual, residual)/pre_den).sqrt()
        row = {'iteration': iteration,
            'recursive_preconditioned_global_normal_residual': recursive.tolist()}
        check = (iteration % 4 == 0 or iteration == maximum or bool(broken.any())
                 or bool((recursive[active] <= tolerance).all()))
        del field, bf, cf, image
        if check:
            candidate = base+forward(correction)
            er = rhs-operator(candidate)
            dr = target-operator.readout(candidate, response)
            proposed_cost = dot_real(dr, dr)+penalty*dot_real(er, er)
            finite = torch.isfinite(proposed_cost)
            improve = active&finite&(proposed_cost <= cost)
            old_cost = cost.clone()
            base = select(improve, candidate, base)
            eq = select(improve, er, eq)
            rx = select(improve, dr, rx)
            cost = torch.where(improve, proposed_cost, cost)
            grad = gradient(eq, rx)
            true_residual = adjoint(grad)
            physical = (dot_real(grad, grad)/den).sqrt()
            pre = (dot_real(true_residual, true_residual)/pre_den).sqrt()
            normal = torch.maximum(physical, pre)
            drift = (dot_real(true_residual-residual, true_residual-residual)/
                     dot_real(true_residual, true_residual).clamp_min(1e-30)).sqrt()
            if not torch.isfinite(normal).all():
                raise RuntimeError('Nonfinite true field normal in stable CG')
            newly_converged = active&(physical <= tolerance)&(pre <= tolerance)
            for j in newly_converged.nonzero().flatten().tolist():
                reasons[j] = 'converged'
            active &= ~newly_converged
            small_change = (old_cost-cost).abs() <= old_cost.abs().clamp_min(1e-30)*1e-6
            stagnant = active&small_change&(normal >= best_normal*.995)&(drift > .1)
            stalls = torch.where(stagnant, stalls+1, torch.zeros_like(stalls))
            stopped = active&(stalls >= 3)&(iteration >= 16)
            for j in stopped.nonzero().flatten().tolist():
                reasons[j] = 'true_residual_roundoff_stagnation_NOT_converged'
            active &= ~stopped
            best_normal = torch.minimum(best_normal, normal)
            row.update(physical_global_normal_residual=physical.tolist(),
                preconditioned_global_normal_residual=pre.tolist(), augmented_cost=cost.tolist(),
                residual_drift=drift.tolist(), active_shots=int(active.sum()),
                independent_true_residual_restart=True)
            history.append(row)
            # Restart all active, independent directions from the accepted
            # true residual. No correction from a rejected checkpoint leaks.
            correction.zero_()
            residual = true_residual
            z = precondition(residual)
            direction = select(active, z, torch.zeros_like(z))
            rho = dot_real(residual, z)
            del candidate, er, dr, proposed_cost, grad
            continue
        history.append(row)
        z = precondition(residual)
        next_rho = dot_real(residual, z)
        beta = torch.where(active, next_rho/rho.clamp_min(1e-300), 0.)
        direction = select(active, z+scale(direction, beta), torch.zeros_like(z))
        rho = next_rho
    # These residuals correspond to the best accepted field, not recurrence.
    eq = rhs-operator(base)
    rx = target-operator.readout(base, response)
    grad = gradient(eq, rx)
    true_residual = adjoint(grad)
    physical = (dot_real(grad, grad)/den).sqrt()
    pre = (dot_real(true_residual, true_residual)/pre_den).sqrt()
    after = dot_real(rx, rx)+penalty*dot_real(eq, eq)
    if not torch.isfinite(after).all() or bool((after > before*(1+1e-6)+1e-20).any()):
        raise RuntimeError('Stable field solve increased an actual shot objective')
    return base, {'solver': 'batched_independent_CG_true_energy', 'iterations': len(history),
        'history': history, 'physical_global_normal_residual': physical.tolist(),
        'preconditioned_global_normal_residual': pre.tolist(),
        'augmented_cost_before': before.tolist(), 'augmented_cost_after': after.tolist(),
        'converged': bool(((physical <= tolerance)&(pre <= tolerance)).all()),
        'shot_stop_reasons': reasons, 'all_transmissions_retained': len(base),
        'requested_tolerance_unchanged': tolerance}, eq, rx
