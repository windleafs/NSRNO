"""Actual conditional spatial-prior step with the old data direction unchanged."""
import time
import numpy as np
from .depth_rms_preconditioner import depth_rms_direction
from .independent_physical_coordinates import SCALES
from .constrained_prior_saved_model_CPU import project_data_box


def regularization(coordinates, prior, control, roi):
    material = coordinates.m*np.exp(SCALES*control)
    material[2] = coordinates.edge+coordinates.tissue*np.exp(.7*control[2])
    value, gradient = prior.energy_and_scaled_log_gradient(material, coordinates.edge, roi)
    tv, gt = coordinates.regularization(control, .001)
    return value+tv, gradient+gt, {**prior.last_parts, 'TV': tv}


def build_directions(coordinates, control, prior_gradient, active, lateral,
                     old_control_direction, old_physical_direction):
    caps = np.array([.02, .04, .12])[:, None, None]/SCALES
    proposal, metric = depth_rms_direction(prior_gradient, active, lateral, caps)
    directions = np.zeros((4, *control.shape), np.float64)
    for field in range(3):
        directions[field, field] = .5*proposal[field]
    data = np.asarray(old_control_direction, np.float64)
    directions[3] = data
    absolute = coordinates.absolute+control
    margins = [np.maximum(0., np.minimum(absolute[0]-coordinates.c_lower,
                                        coordinates.c_upper-absolute[0])),
               np.maximum(0., np.minimum(absolute[2]-coordinates.alpha_lower,
                                        coordinates.alpha_upper-absolute[2]))]
    logZ = .05*absolute[0]+.1*absolute[1]
    margins.append(np.maximum(0., np.minimum(logZ-coordinates.logZ_lower,
                                            coordinates.logZ_upper-logZ)))
    occupied = [abs(data[0]), abs(data[2]), abs(.05*data[0]+.1*data[1])]
    sizes = [np.sum(abs(directions[:3, 0]), 0), np.sum(abs(directions[:3, 2]), 0),
             np.sum(abs(.05*directions[:3, 0]+.1*directions[:3, 1]), 0)]
    scale = np.ones(active.shape)
    for margin, used, size in zip(margins, occupied, sizes):
        remaining = .99*margin-used
        if np.min(remaining[active]) < -1e-10:
            raise RuntimeError('The unchanged data direction is not feasible in the original box')
        scale = np.minimum(scale, np.maximum(0., remaining)/np.maximum(size, 1e-300))
    scale = np.clip(scale, 0., 1.)*active
    directions[:3] *= scale[None, None]
    if not np.array_equal(directions[3], data):
        raise RuntimeError('Data direction changed; its cached JVPs cannot be reused')
    if np.max(np.sum(abs(directions), 0)/caps) > 1.00001:
        raise RuntimeError('Original physical logarithmic step caps exceeded')
    material = coordinates.material(control)
    physical = np.empty(directions.shape, np.float32)
    for j in range(3):
        direction = directions[j]
        physical[j] = np.stack([.05*material[0]*direction[0], .1*material[1]*direction[1],
                               .7*(material[2]-coordinates.edge)*direction[2]])
    physical[3] = old_physical_direction
    if not np.array_equal(physical[3], old_physical_direction):
        raise RuntimeError('Physical data direction changed')
    return directions, physical, {'physical_log_caps': [.02, .04, .12],
        'prior_metric': metric, 'unchanged_data_direction_and_JVPs_reused': True,
        'only_new_prior_directions_scaled_for_feasibility': True,
        'fraction_active_prior_scaled': float(np.mean(scale[active] < 1.)),
        'prior_gradient_directional_products': np.einsum('aijk,ijk->a', directions, prior_gradient).tolist()}


def solve_prior_step(H, linear, RF0, coordinates, prior, control, roi, directions):
    started = time.time()
    H = .5*(np.asarray(H)+np.asarray(H).T)
    eigen = np.linalg.eigvalsh(H)
    if eigen[0] < -1e-10*max(float(eigen[-1]), 1e-30):
        raise RuntimeError('Actual all44 Gram matrix is not positive semidefinite')
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
        'roundoff_PSD_correction': correction, 'actual_native_RF_line_still_required': True}
