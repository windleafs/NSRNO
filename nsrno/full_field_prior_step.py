"""Fresh full-field prior directions and a data-constrained local material step."""
import time
import numpy as np
from .depth_rms_preconditioner import depth_rms_direction
from .independent_physical_coordinates import SCALES
from .constrained_prior_saved_model_CPU import project_data_box


def regularization(coordinates, prior, control, roi):
    material = coordinates.m*np.exp(SCALES*control)
    material[2] = coordinates.edge+coordinates.tissue*np.exp(.70*control[2])
    value, gradient = prior.energy_and_scaled_log_gradient(material, coordinates.edge, roi)
    tv, gt = coordinates.regularization(control, .001)
    return value+tv, gradient+gt, {'marginal_prior': value, 'TV': tv}


def build_directions(coordinates, control, data_gradient, prior_gradient, active, lateral):
    caps = np.array([.02, .04, .12])[:, None, None]/SCALES
    prior_direction, prior_metric = depth_rms_direction(prior_gradient, active, lateral, caps)
    data_direction, data_metric = depth_rms_direction(data_gradient, active, lateral, caps)
    directions = np.zeros((4, *control.shape), np.float64)
    for field in range(3):
        directions[field, field] = .5*prior_direction[field]
    directions[3] = .5*data_direction
    absolute = coordinates.absolute+control
    margin_c = np.maximum(0., np.minimum(absolute[0]-coordinates.c_lower,
                                       coordinates.c_upper-absolute[0]))
    margin_alpha = np.maximum(0., np.minimum(absolute[2]-coordinates.alpha_lower,
                                           coordinates.alpha_upper-absolute[2]))
    logZ = .05*absolute[0]+.10*absolute[1]
    margin_Z = np.maximum(0., np.minimum(logZ-coordinates.logZ_lower, coordinates.logZ_upper-logZ))
    sums = [np.sum(abs(directions[:, 0]), 0),
            np.sum(abs(directions[:, 2]), 0),
            np.sum(abs(.05*directions[:, 0]+.10*directions[:, 1]), 0)]
    local = np.ones(active.shape)
    for margin, size in zip([margin_c, margin_alpha, margin_Z], sums):
        local = np.minimum(local, .99*margin/np.maximum(size, 1e-300))
    local = np.clip(local, 0., 1.)*active
    directions *= local[None, None]
    if np.max(np.sum(abs(directions), 0)/caps) > 1.00001:
        raise RuntimeError('The coefficient box exceeds the declared physical caps')
    material = coordinates.material(control)
    physical = np.empty(directions.shape, np.float32)
    for j, direction in enumerate(directions):
        physical[j] = np.stack([.05*material[0]*direction[0], .1*material[1]*direction[1],
                               .7*(material[2]-coordinates.edge)*direction[2]])
    return directions, physical, {
        'physical_log_caps': [.02, .04, .12], 'prior_metric': prior_metric, 'data_metric': data_metric,
        'fresh_full_field_gradients_not_old_retained_shapes': True,
        'prior_gradient_directional_products': np.einsum('aijk,ijk->a', directions, prior_gradient).tolist(),
        'data_gradient_directional_products': np.einsum('aijk,ijk->a', directions, data_gradient).tolist(),
        'minimum_active_feasibility_scale': float(local[active].min()),
        'fraction_active_scaled_for_physical_bounds': float(np.mean(local[active] < 1.))}


def solve_prior_step(H, linear, RF0, coordinates, prior, control, roi, directions):
    started = time.time()
    H = .5*(np.asarray(H)+np.asarray(H).T)
    eigen = np.linalg.eigvalsh(H)
    if eigen[0] < -1e-10*max(float(eigen[-1]), 1e-30):
        raise RuntimeError('The required all44 Gram matrix is not positive semidefinite')
    correction = max(0., -float(eigen[0]))
    H = H+correction*np.eye(4)
    linear = np.asarray(linear)
    q = lambda theta: float(RF0+linear@theta+.5*theta@H@theta)
    def evaluate(theta):
        value, gradient, parts = regularization(coordinates, prior,
            control+np.einsum('a,aijk->ijk', theta, directions), roi)
        return value, np.einsum('aijk,ijk->a', directions, gradient), parts
    theta = np.zeros(4)
    value, gradient, parts = evaluate(theta)
    baseline = value
    eta = 1/max(float(np.max(abs(gradient))), 1e-20)
    history = [{'iteration': 0, 'regularization': value, 'RF_model': RF0,
                'coefficients': theta.tolist(), **parts}]
    reason = 'maximum_30_CPU_prior_steps'
    for iteration in range(1, 31):
        if time.time()-started >= 300:
            reason = '300_second_CPU_budget'
            break
        accepted = False
        for line in range(12):
            if time.time()-started >= 300:
                reason = '300_second_CPU_budget'
                break
            candidate = project_data_box(theta-eta*gradient, H, linear, q, RF0)
            change = candidate-theta
            if np.max(abs(change)) < 1e-5:
                reason = 'small_prior_step_in_current_local_model'
                break
            trial, trial_gradient, parts = evaluate(candidate)
            if trial <= value+1e-4*min(float(gradient@change), 0.):
                theta, value, gradient = candidate, trial, trial_gradient
                history.append({'iteration': iteration, 'regularization': value, 'RF_model': q(theta),
                                'coefficients': theta.tolist(), 'accepted_CPU_line': line, **parts})
                eta *= 1.5
                accepted = True
                break
            eta *= .5
        if not accepted:
            if not reason.startswith(('small_', '300_')):
                reason = 'no_further_monotone_CPU_prior_step'
            break
    response = .5*float(theta@H@theta)
    trust = min(1., np.sqrt(.25*RF0/max(response, 1e-30)))
    theta *= trust
    return theta, {'stop_reason': reason, 'seconds': time.time()-started,
        'initial_regularization': baseline, 'optimized_regularization_before_RF_trust': value,
        'RF_linear_response_trust_fraction': trust, 'RF_model_after_trust': q(theta),
        'coefficients_after_trust': theta.tolist(), 'history': history,
        'roundoff_PSD_correction': correction, 'native_RF_still_requires_actual_line_search': True}
