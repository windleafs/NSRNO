"""A prior-aware inverse proposal using the completed exact all44 local model.

No new native response, patient derivative, truth or frequency is requested.
The result remains an unverified physical candidate, not a new RF incumbent.
"""
from pathlib import Path
import argparse
import itertools
import json
import time
import numpy as np
from .factorized_material_prior import FactorizedMaterialPrior
from .independent_physical_coordinates import IndependentPhysicalCoordinates, SCALES
from .low_frequency_0p5 import atomic, digest


def box_spd(A, b):
    n = len(b)
    best = (0., np.zeros(n))
    for face in itertools.product([-1, 0, 1], repeat=n):
        face = np.asarray(face)
        free, fixed = face == 0, face != 0
        x = face.astype(np.float64)
        if free.any():
            x[free] = np.linalg.solve(A[np.ix_(free, free)],
                -b[free]-A[np.ix_(free, fixed)]@x[fixed])
        if np.max(abs(x)) > 1+1e-10:
            continue
        x = np.clip(x, -1, 1)
        value = float(.5*x@A@x+b@x)
        if value < best[0]:
            best = value, x.copy()
    x = best[1]
    residual = x-np.clip(x-(A@x+b)/np.diag(A), -1, 1)
    if np.max(abs(residual)) > 1e-6:
        raise RuntimeError('Exact small box projection failed its KKT condition')
    return x


def project_data_box(v, H, b, q, cap):
    x = np.clip(v, -1, 1)
    if q(x) <= cap:
        return x
    identity = np.eye(len(v))
    lo, hi = 0., 1/max(float(np.max(np.diag(H))), 1e-12)
    for _ in range(60):
        x = box_spd(identity+hi*H, hi*b-v)
        if q(x) <= cap:
            break
        hi *= 2
    else:
        raise RuntimeError('Data-constrained prior projection has no verified feasible bracket')
    for _ in range(40):
        middle = .5*(lo+hi)
        candidate = box_spd(identity+middle*H, middle*b-v)
        if q(candidate) <= cap:
            hi, x = middle, candidate
        else:
            lo = middle
    if q(x) > cap+1e-12:
        raise RuntimeError('Projected prior step violated the fixed data-model constraint')
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root, out = args.reference_root.resolve(), args.out.resolve()
    if out.exists():
        raise RuntimeError('Inspect this same saved CPU candidate; do not silently rerun it')
    if not out.is_relative_to(Path.home()/'nsrno_artifacts'):
        raise ValueError('Dedicated same-account artifact volume only')
    run = json.loads((root/'run.json').read_text())
    if run['status'] != 'all44_joint_tangent_stage_finished' or Path(f'/proc/{run["pid"]}').exists():
        raise RuntimeError('Use the completed, fixed all44 experiment only')
    qualified = json.loads(args.qualification.read_text())
    if (qualified['status'] != 'factorized_TRAIN_prior_physical_log_gradient_CPU_passed' or
        qualified['prior_SHA256'] != digest(args.prior) or qualified['source_SHA256'] !=
        digest(Path(__file__).with_name('factorized_material_prior.py'))):
        raise RuntimeError('New prior must retain its necessary CPU chain-rule qualification')
    # Read one explicit computational local model, never traverse task JSON.
    model = json.loads((root/'jobs/0001_joint_tangents.json').read_text())
    H = np.asarray(model['H'], np.float64)
    H = .5*(H+H.T)
    eigen = np.linalg.eigvalsh(H)
    if eigen[0] < -1e-10*max(eigen[-1], 1e-30):
        raise RuntimeError('Saved data model lost positive semidefiniteness')
    correction = max(0., -float(eigen[0]))
    H += correction*np.eye(len(H))
    b, initial_RF = np.asarray(model['linear']), model['loss']
    theta0 = np.asarray(run['coefficients'], np.float64)
    q = lambda x: float(initial_RF+b@x+.5*x@H@x)
    epsilon = max(abs(run['candidate_RF']-q(theta0)), 1e-8*initial_RF)
    cap = q(theta0)+epsilon
    plan = json.loads((root/'native_plan.json').read_text())
    parent = Path(run['parent_root'])
    anchor = Path(plan['native_anchor_material'])
    edge_file = Path('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    if digest(anchor) != run['native_anchor_material_SHA256'] or digest(edge_file) != plan['known_edge_SHA256']:
        raise RuntimeError('Original coordinate map changed')
    edge = np.broadcast_to(np.load(edge_file), (961, 1121)).astype(np.float64)
    coordinates = IndependentPhysicalCoordinates(np.load(anchor), edge)
    control0 = np.load(parent/'best_control.npy').astype(np.float64)
    directions = np.load(root/'arrays/control_directions.npy').astype(np.float64)
    if directions.shape != (4, 3, 961, 1121):
        raise ValueError('Exactly the retained independent material directions required')
    material0 = coordinates.m
    tissue0 = material0[2]-edge
    z = (np.arange(961)-60)*.05
    x = (np.arange(1121)-560)*.05
    roi = (z[:, None] >= 1)&(z[:, None] <= 43)&(abs(x)[None] <= 19.1)
    zz, xx = np.ogrid[:961, :1121]
    active = (zz >= 80)&(zz < 937)&(xx >= 24)&(xx < 1097)
    prior = FactorizedMaterialPrior(args.prior, tail=.1, temperature=0.)
    out.mkdir()
    started = time.time()
    def evaluate(theta):
        control = control0+np.einsum('a,aijk->ijk', theta, directions)
        material = material0*np.exp(SCALES*control)
        material[2] = edge+tissue0*np.exp(.7*control[2])
        R, gradient = prior.energy_and_scaled_log_gradient(material, edge, roi)
        TV, gt = coordinates.regularization(control, .001)
        coefficient_gradient = np.einsum('aijk,ijk->a', directions, gradient+gt)
        return R+TV, coefficient_gradient, R, TV
    theta = theta0.copy()
    value, gradient, R0, TV0 = evaluate(theta)
    initial_value = value
    eta = 1/max(np.max(abs(gradient)), 1e-20)
    history = [{'iteration': 0, 'regularization': value, 'prior': R0, 'TV': TV0,
                'RF_model': q(theta), 'coefficients': theta.tolist()}]
    stop_reason = 'maximum_30_CPU_inverse_steps'
    for iteration in range(1, 31):
        if time.time()-started >= 300:
            stop_reason = '300_second_CPU_budget'
            break
        accepted = False
        for line in range(12):
            if time.time()-started >= 300:
                stop_reason = '300_second_CPU_budget'
                break
            candidate = project_data_box(theta-eta*gradient, H, b, q, cap)
            change = candidate-theta
            if np.max(abs(change)) < 1e-5:
                stop_reason = 'small_data_constrained_prior_step_in_retained_space'
                break
            trial, trial_gradient, R, TV = evaluate(candidate)
            slope = float(gradient@change)
            if trial <= value+1e-4*min(slope, 0.):
                theta, value, gradient = candidate, trial, trial_gradient
                history.append({'iteration': iteration, 'regularization': value, 'prior': R,
                    'TV': TV, 'RF_model': q(theta), 'coefficients': theta.tolist(),
                    'data_constraint_slack': cap-q(theta), 'accepted_CPU_line': line})
                accepted = True
                eta *= 1.5
                atomic(out/'progress.json', {'iteration': iteration, 'history_last': history[-1],
                    'seconds': time.time()-started, 'native_RF_verified': False,
                    'new_wave_propagations': 0, 'patient_truth_read': False})
                break
            eta *= .5
        if not accepted:
            if not stop_reason.startswith(('small_', '300_')):
                stop_reason = 'no_further_monotone_CPU_prior_step'
            break
    control = control0+np.einsum('a,aijk->ijk', theta, directions)
    if np.max(abs(coordinates.project(control, active)-control)) > 1e-9:
        raise RuntimeError('Saved prior proposal is not physically feasible')
    proposal = coordinates.material(control)
    baseline = np.load(root/'final/material.npy').astype(np.float64)
    change = proposal.astype(np.float64)-baseline
    np.save(out/'candidate_control.npy', control)
    np.save(out/'candidate_material.npy', proposal)
    result = {'status': 'data_constrained_independent_prior_CPU_inverse_candidate',
        'seconds': time.time()-started, 'stop_reason': stop_reason,
        'initial_regularization': initial_value, 'final_regularization': value,
        'coefficients_before': theta0.tolist(), 'coefficients_after': theta.tolist(),
        'RF_model_before': q(theta0), 'RF_model_after': q(theta), 'RF_model_cap': cap,
        'cap_slack_from_previous_actual_prediction_error': epsilon,
        'prior_reference_SHA256': digest(args.prior),
        'initial_material_SHA256': digest(root/'final/material.npy'),
        'candidate_material_SHA256': digest(out/'candidate_material.npy'),
        'physical_increment_RMS_in_FOV': [float(np.sqrt(np.mean(v[roi]**2))) for v in change],
        'regularization_steps': len(history)-1, 'history': history,
        'roundoff_PSD_correction': correction, 'native_RF_verified': False,
        'native_incumbent_not_replaced': True, 'patient_truth_read': False,
        'new_native_forward_or_derivative_passes': 0, 'new_data_or_frequency': False,
        'original134_unit_2_RF_campaign_unchanged': True,
        'retained_space_only_not_full_field_or_imaging_completion': True,
        'input_direction_SHA256': digest(root/'arrays/control_directions.npy'),
        'source_SHA256': digest(__file__)}
    atomic(out/'result.json', result)
    compact = {k: v for k, v in result.items() if k != 'history'}
    print(json.dumps(compact), flush=True)


if __name__ == '__main__':
    main()
