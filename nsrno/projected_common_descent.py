"""Weighted physical projection and a full-field common descent direction."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .independent_physical_coordinates import SCALES


CAPS = np.array([.02, .04, .12])[:, None, None]/SCALES


def weighted_project(coordinates, control, diagonal, active):
    """Exact B^-1 projection onto the existing c/Z/alpha constraints."""
    control = np.asarray(control, np.float64)
    active = np.asarray(active, bool)&coordinates.mask
    B = np.broadcast_to(np.asarray(diagonal, np.float64), control.shape)
    if (control.shape != coordinates.absolute.shape or active.shape != control.shape[1:] or
        not np.isfinite(control).all() or not np.isfinite(B).all() or
        np.any(B[:, active] <= 0)):
        raise ValueError('Finite controls and positive active diagonal metric required')
    bc = np.where(active, B[0], 1.)
    br = np.where(active, B[1], 1.)
    absolute = coordinates.absolute+control
    c, rho = absolute[:2]
    Z = .05*c+.1*rho
    valid = ((c >= coordinates.c_lower)&(c <= coordinates.c_upper)&
        (Z >= coordinates.logZ_lower)&(Z <= coordinates.logZ_upper))
    best_c, best_rho = c.copy(), rho.copy()
    distance = np.where(valid, 0., np.inf)
    def consider(candidate_c, candidate_rho):
        nonlocal best_c, best_rho, distance
        trial = (candidate_c-c)**2/bc+(candidate_rho-rho)**2/br
        choose = trial < distance
        best_c = np.where(choose, candidate_c, best_c)
        best_rho = np.where(choose, candidate_rho, best_rho)
        distance = np.minimum(distance, trial)
    for boundary in [coordinates.c_lower, coordinates.c_upper]:
        candidate_rho = np.clip(rho, (coordinates.logZ_lower-.05*boundary)/.1,
                                (coordinates.logZ_upper-.05*boundary)/.1)
        consider(boundary, candidate_rho)
    normal_squared = .05**2*bc+.1**2*br
    for boundary in [coordinates.logZ_lower, coordinates.logZ_upper]:
        candidate_c = np.clip(c-bc*.05*(Z-boundary)/normal_squared,
                             coordinates.c_lower, coordinates.c_upper)
        candidate_rho = (boundary-.05*candidate_c)/.1
        consider(candidate_c, candidate_rho)
    projected = np.stack([best_c, best_rho,
        np.clip(absolute[2], coordinates.alpha_lower, coordinates.alpha_upper)])
    return (projected-coordinates.absolute)*active[None]


def common_direction(coordinates, control, data_gradient, prior_gradient, active, lateral):
    """Solve the projected min-max local direction using a scalar dual.

    Both gradients are already required by the actual inverse. No propagation,
    finite difference, Hessian action, or patient array is loaded here.
    """
    active = np.asarray(active, bool)&coordinates.mask
    gD = np.asarray(data_gradient, np.float64)*active[None]
    gR = np.asarray(prior_gradient, np.float64)*active[None]
    if (gD.shape != coordinates.absolute.shape or gR.shape != gD.shape or
        not np.isfinite(gD).all() or not np.isfinite(gR).all()):
        raise ValueError('Finite complete native independent gradients required')
    nD, nR = float(np.linalg.norm(gD)), float(np.linalg.norm(gR))
    if min(nD, nR) <= 1e-30:
        return np.zeros_like(gD), {'status': 'zero_objective_gradient_at_current_material',
            'data_gradient_norm': nD, 'prior_gradient_norm': nR, 'actual_common_step_available': False}
    reference = active&np.asarray(lateral, bool)[None]
    counts = reference.sum(1)
    rows = active.any(1)
    profiles = np.sqrt(np.sum(((gD/nD)**2+(gR/nR)**2)*reference[None], axis=-1)
                       /np.maximum(counts[None], 1))
    smooth = gaussian_filter1d(profiles, 40., axis=-1, mode='nearest')
    floors = np.array([max(.05*float(np.median(v[rows & (v > 0)])), 1e-12)
                       if np.any(rows & (v > 0)) else 1e-12 for v in smooth])
    B = CAPS**2/np.maximum(smooth[:, :, None], floors[:, None, None])
    a = float(np.sum(B*gR*gR))
    b = float(np.sum(B*gD*gD))
    if min(a, b) <= 0 or not np.isfinite(a+b):
        raise RuntimeError('Positive weighted gradient norms required')
    uR, uD = gR/np.sqrt(a), gD/np.sqrt(b)
    limits = []
    for field in range(3):
        for gradient in [uR, uD]:
            maximum = float(np.max(abs(B[field]*gradient[field])[active]))
            if maximum > 0:
                limits.append(float(CAPS[field, 0, 0])/maximum)
    eta = .95*min(limits)
    difference = uD-uR
    def evaluate(weight):
        mixture = (1-weight)*uR+weight*uD
        projected = weighted_project(coordinates, control-eta*B*mixture, B, active)
        direction = projected-control
        derivative = float(np.sum(difference*direction))
        quadratic = float(np.sum(direction*direction/B))/(2*eta)
        dual = float(np.sum(mixture*direction))+quadratic
        return direction, derivative, dual
    d0, derivative0, dual0 = evaluate(0.)
    d1, derivative1, dual1 = evaluate(1.)
    iterations = 0
    if derivative0 <= 0:
        weight, direction, dual = 0., d0, dual0
    elif derivative1 >= 0:
        weight, direction, dual = 1., d1, dual1
    else:
        lo, hi = 0., 1.
        for iteration in range(30):
            weight = .5*(lo+hi)
            direction, derivative, dual = evaluate(weight)
            iterations = iteration+1
            if derivative == 0:
                lo = hi = weight
                break
            if derivative > 0:
                lo = weight
            else:
                hi = weight
        weight = .5*(lo+hi)
        direction, derivative, dual = evaluate(weight)
    pre_scale_slopes = [float(np.sum(uR*direction)), float(np.sum(uD*direction))]
    projected_norm_squared = float(np.sum(direction*direction/B))
    ratio = float(np.max(abs(direction)/CAPS))
    cap_scale = min(1., 1/max(ratio, 1e-300))
    direction = direction*cap_scale
    normalized_slopes = [float(np.sum(uR*direction)), float(np.sum(uD*direction))]
    original_slopes = [float(np.sum(gR*direction)), float(np.sum(gD*direction))]
    available = bool(max(original_slopes) < 0 and max(normalized_slopes) < -1e-14 and
                     float(np.linalg.norm(direction)) > 1e-12)
    result = {'status': 'projected_full_field_common_descent' if available
              else 'no_resolved_projected_common_descent_at_current_material',
        'actual_common_step_available': available, 'dual_data_weight': float(weight),
        'dual_bisections': iterations, 'eta': eta, 'cap_global_scale': cap_scale,
        'normalized_prior_data_slopes': normalized_slopes,
        'prior_slope': original_slopes[0], 'RF_slope': original_slopes[1],
        'pre_cap_projection_descent_bound': -projected_norm_squared/eta,
        'pre_cap_normalized_prior_data_slopes': pre_scale_slopes,
        'dual_value_before_global_cap': dual,
        'weighted_gradient_cosine': float(np.sum(gD*B*gR)/np.sqrt(a*b)),
        'physical_log_caps': [.02, .04, .12],
        'actual_maximum_physical_log_increment': np.max(abs(SCALES*direction), axis=(1, 2)).tolist(),
        'metric_floors': floors.tolist(),
        'minimum_active_metric': B[:, rows].min(axis=(1, 2)).tolist(),
        'maximum_active_metric': B[:, rows].max(axis=(1, 2)).tolist(),
        'sigma_rows_for_metric_only': 40., 'material_filtered': False,
        'new_acoustic_derivatives_for_direction_selection': 0}
    return direction, result
