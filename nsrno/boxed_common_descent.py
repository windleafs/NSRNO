"""Weighted common descent in the physical domain intersected with a step box."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .independent_physical_coordinates import SCALES

CAPS = np.array([.02, .04, .12])[:, None, None]/SCALES


def project_box(coordinates, target, metric, active, center):
    target, center = np.asarray(target, np.float64), np.asarray(center, np.float64)
    active = np.asarray(active, bool)&coordinates.mask
    B = np.broadcast_to(np.asarray(metric, np.float64), target.shape)
    if (target.shape != center.shape or target.shape != coordinates.absolute.shape or
        not np.isfinite(target).all() or not np.isfinite(center).all() or
        not np.isfinite(B).all() or np.any(B[:, active] <= 0)):
        raise ValueError('Finite independent fields and positive active metric required')
    origin = coordinates.absolute+center
    absolute = coordinates.absolute+target
    cx, cy = absolute[:2]
    bx, by = np.where(active, B[0], 1.), np.where(active, B[1], 1.)
    xl = np.maximum(coordinates.c_lower, origin[0]-CAPS[0])
    xh = np.minimum(coordinates.c_upper, origin[0]+CAPS[0])
    yl, yh = origin[1]-CAPS[1], origin[1]+CAPS[1]
    zl, zh = coordinates.logZ_lower, coordinates.logZ_upper
    z = .05*cx+.1*cy
    valid = (cx >= xl)&(cx <= xh)&(cy >= yl)&(cy <= yh)&(z >= zl)&(z <= zh)
    bestx, besty = cx.copy(), cy.copy()
    distance = np.where(valid, 0., np.inf)
    def consider(x, y, feasible):
        nonlocal bestx, besty, distance
        cost = (x-cx)**2/bx+(y-cy)**2/by
        choose = feasible&(cost < distance)
        bestx, besty = np.where(choose, x, bestx), np.where(choose, y, besty)
        distance = np.where(choose, cost, distance)
    for x in [xl, xh]:
        lo, hi = np.maximum(yl, (zl-.05*x)/.1), np.minimum(yh, (zh-.05*x)/.1)
        consider(x, np.clip(cy, lo, hi), lo <= hi)
    for y in [yl, yh]:
        lo, hi = np.maximum(xl, (zl-.1*y)/.05), np.minimum(xh, (zh-.1*y)/.05)
        consider(np.clip(cx, lo, hi), y, lo <= hi)
    normal = .05**2*bx+.1**2*by
    for boundary in [zl, zh]:
        lo, hi = np.maximum(xl, (boundary-.1*yh)/.05), np.minimum(xh, (boundary-.1*yl)/.05)
        x = np.clip(cx-bx*.05*(z-boundary)/normal, lo, hi)
        consider(x, (boundary-.05*x)/.1, lo <= hi)
    if not np.isfinite(distance[active]).all():
        raise RuntimeError('Trust box and physical domain have no resolved intersection')
    al = np.maximum(coordinates.alpha_lower, origin[2]-CAPS[2])
    ah = np.minimum(coordinates.alpha_upper, origin[2]+CAPS[2])
    result = np.stack([bestx, besty, np.clip(absolute[2], al, ah)])-coordinates.absolute
    return np.where(active[None], result, center)


def common_box_direction(coordinates, control, data_gradient, prior_gradient, active, lateral):
    active = np.asarray(active, bool)&coordinates.mask
    gD, gR = [np.asarray(g, np.float64)*active[None] for g in [data_gradient, prior_gradient]]
    if not np.isfinite(gD).all() or not np.isfinite(gR).all():
        raise ValueError('Finite required inverse gradients only')
    nD, nR = float(np.linalg.norm(gD)), float(np.linalg.norm(gR))
    if min(nD, nR) <= 1e-30:
        return np.zeros_like(gD), {'actual_common_step_available':False,'status':'zero_objective_gradient'}
    reference = active&np.asarray(lateral, bool)[None]
    counts, rows = reference.sum(1), active.any(1)
    profiles = np.sqrt(np.sum(((gD/nD)**2+(gR/nR)**2)*reference[None], axis=-1)/np.maximum(counts[None], 1))
    smooth = gaussian_filter1d(profiles, 40., axis=-1, mode='nearest')
    floors = np.array([max(.05*float(np.median(v[rows&(v > 0)])), 1e-12)
        if np.any(rows&(v > 0)) else 1e-12 for v in smooth])
    B = CAPS**2/np.maximum(smooth[:, :, None], floors[:, None, None])
    a, b = float(np.sum(B*gR*gR)), float(np.sum(B*gD*gD))
    uR, uD = gR/np.sqrt(a), gD/np.sqrt(b)
    limits = []
    for field in range(3):
        for gradient in [uR, uD]:
            rms = float(np.sqrt(np.mean((B[field]*gradient[field])[reference]**2)))
            if rms > 0:
                limits.append(float(CAPS[field, 0, 0])/rms)
    eta = .95*min(limits)
    difference = uD-uR
    def evaluate(weight):
        mixture = (1-weight)*uR+weight*uD
        direction = project_box(coordinates, control-eta*B*mixture, B, active, control)-control
        derivative = float(np.sum(difference*direction))
        return direction, derivative
    d0, s0 = evaluate(0.)
    d1, s1 = evaluate(1.)
    iterations = 0
    if s0 <= 0:
        weight, direction = 0., d0
    elif s1 >= 0:
        weight, direction = 1., d1
    else:
        lo, hi = 0., 1.
        for _ in range(30):
            weight = (lo+hi)/2
            direction, slope = evaluate(weight)
            iterations += 1
            if slope > 0:
                lo = weight
            else:
                hi = weight
        weight = (lo+hi)/2
        direction, _ = evaluate(weight)
    normalized = [float(np.sum(g*direction)) for g in [uR, uD]]
    original = [float(np.sum(g*direction)) for g in [gR, gD]]
    maxima = np.max(abs(SCALES*direction), axis=(1, 2))
    if np.any(maxima > np.array([.02, .04, .12])+1e-10):
        raise RuntimeError('Exact projection left the declared per-pixel trust box')
    available = max(normalized) < -1e-14 and max(original) < 0
    return direction, {'status':'boxed_full_field_common_descent' if available else 'no_common_descent_resolved',
        'actual_common_step_available':bool(available),'RF_slope':original[1],
        'prior_slope':original[0],'normalized_prior_data_slopes':normalized,
        'dual_data_weight':weight,'dual_bisections':iterations,'eta':eta,
        'weighted_gradient_cosine':float(np.sum(gD*B*gR)/np.sqrt(a*b)),
        'metric_norm_profile_sigma_rows':40.,'physical_log_caps':[.02,.04,.12],
        'actual_maximum_physical_log_increment':maxima.tolist(),
        'physical_log_increment_RMS_in_FOV':[float(np.sqrt(np.mean((SCALES[f]*direction[f])[reference]**2))) for f in range(3)],
        'global_outlier_cap_scale':False,'material_filtered':False,
        'new_patient_derivative_probes':0}
