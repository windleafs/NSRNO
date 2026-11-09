"""Actual convex GN subproblem with full-grid control bounds and dual certificate."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.optimize import LinearConstraint, minimize, nnls
from .independent_physical_coordinates import SCALES


def solve_bounded_subspace(matrix, gradient, images, caps, damping=.01,
                           tolerance=2e-6, maximum_cuts=80):
    matrix = np.asarray(matrix, np.float64)
    gradient = np.asarray(gradient, np.float64)
    images = np.asarray(images, np.float64)
    caps = np.asarray(caps, np.float64)
    dimension = len(gradient)
    if (matrix.shape != (dimension, dimension) or images.ndim != 4 or
        images.shape[:2] != (dimension, 3) or caps.shape != (3, 1, 1) or
        np.any(caps <= 0) or not np.isfinite(matrix).all() or not np.isfinite(images).all()
            or not np.isfinite(gradient).all() or not 0 < damping):
        raise ValueError('Finite convex subspace model and independent control caps required')
    symmetric = (matrix+matrix.T)/2
    diagonal = np.maximum(np.diag(symmetric), max(float(np.diag(symmetric).max()), 1e-30)*1e-12)
    scaling = np.sqrt(diagonal)
    quadratic = symmetric/scaling[:, None]/scaling[None]+damping*np.eye(dimension)
    np.linalg.cholesky(quadratic)
    linear = gradient/scaling
    value = -np.linalg.solve(quadratic, linear)
    rows, seen, history = [], set(), []
    optimizer_success = True
    for iteration in range(maximum_cuts+1):
        if rows:
            constraint = np.stack(rows)
            solved = minimize(lambda y: float(linear@y+.5*y@quadratic@y), value,
                jac=lambda y: linear+quadratic@y, method='SLSQP',
                constraints=[LinearConstraint(constraint, -1.+1e-7, 1.-1e-7)],
                options={'ftol': 1e-12, 'maxiter': 500})
            value = solved.x
            optimizer_success = bool(solved.success)
        coefficient = value/scaling
        field = np.einsum('k,kizx->izx', coefficient, images, optimize=True)
        ratios = abs(field)/caps
        maxima = ratios.max(axis=(1, 2))
        history.append({'iteration': iteration, 'constraints': len(rows),
            'maximum_field_ratios': maxima.tolist(), 'solver_success': optimizer_success,
            'damped_model_value': float(linear@value+.5*value@quadratic@value)})
        if maxima.max() <= 1+tolerance:
            break
        additions = 0
        for material in range(3):
            if maxima[material] <= 1+tolerance:
                continue
            z, x = np.unravel_index(int(ratios[material].argmax()), ratios[material].shape)
            key = (material, int(z), int(x))
            if key not in seen:
                rows.append(images[:, material, z, x]/scaling/caps[material, 0, 0])
                seen.add(key)
                additions += 1
        if additions == 0:
            raise RuntimeError('Cut QP is infeasible beyond tolerance at an existing constraint')
    else:
        raise RuntimeError('Actual subspace cut budget ended before full-grid feasibility')
    objective = float(linear@value+.5*value@quadratic@value)
    if rows:
        constraint = np.stack(rows)
        projection = constraint@value
        active = abs(projection) >= 1-2e-5
        normals = np.sign(projection[active])[:, None]*constraint[active]
    else:
        normals = np.empty((0, dimension))
    if len(normals):
        multiplier, stationarity = nnls(normals.T, -(linear+quadratic@value),
                                       maxiter=max(1000, 10*len(normals)))
        dual_linear = linear+normals.T@multiplier
    else:
        multiplier = np.empty(0)
        stationarity = float(np.linalg.norm(linear+quadratic@value))
        dual_linear = linear
    # These nonnegative multipliers provide a lower bound for the ORIGINAL
    # full-grid QP, since every retained row is one of its actual constraints.
    dual = float(-.5*dual_linear@np.linalg.solve(quadratic, dual_linear)-multiplier.sum())
    gap = objective-dual
    relative_gap = gap/max(abs(objective), abs(dual), 1e-20)
    if (not np.isfinite(coefficient).all() or maxima.max() > 1+tolerance or
            relative_gap > 2e-5 or relative_gap < -2e-5 or objective >= 0):
        raise RuntimeError(f'Actual bounded subspace certificate failed: {maxima}, {relative_gap}')
    return coefficient, {'dimension': dimension, 'damping': damping,
        'constraint_generation_history': history, 'retained_voxel_constraints': len(rows),
        'maximum_linear_field_cap_ratios': maxima.tolist(),
        'full_grid_feasibility_tolerance': tolerance, 'primal_damped_model_value': objective,
        'dual_lower_bound': dual, 'relative_primal_dual_gap': relative_gap,
        'dual_stationarity_L2': float(stationarity), 'SLSQP_success': optimizer_success,
        'undamped_model_change': float(gradient@coefficient+.5*coefficient@symmetric@coefficient),
        'scope': 'this current at-most-six-dimensional convex update model, not full material convergence'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    project = Path.cwd().resolve()
    source, out = args.source_root.resolve(), args.out.resolve()
    if source != project/'out/v5/reconstruction_neural_subspace_native_v1' or not out.is_relative_to(project/'out/v5'):
        raise ValueError('The saved current inverse model only')
    if out.exists():
        raise RuntimeError('Preserve the actual CPU model solve; never reset it')
    folder = source/'candidates/step01'
    step = json.loads((source/'steps/step01.json').read_text())
    images = np.load(folder/'control_basis.npy')
    gradient = np.load(folder/'actual_objective_gradient.npy')
    slopes = np.einsum('kizx,izx->k', images, gradient, optimize=True)
    matrix = np.asarray(step['coupled_matrix'])
    parent = json.loads((Path(json.loads((source/'run.json').read_text())['parent_root'])/'run.json').read_text())
    cap_factor = min(1., float(parent.get('next_physical_cap_factor', 1.)))
    caps = np.array([.004, .006, .05])[:, None, None]/SCALES*cap_factor
    coefficient, certificate = solve_bounded_subspace(matrix, slopes, images, caps,
                                                    damping=step['damping'])
    previous_scale = step['lines'][-1]['common_neural_parameter_scale']
    previous = np.asarray(step['coefficients'])*previous_scale
    symmetric = (matrix+matrix.T)/2
    diagonal = np.maximum(np.diag(symmetric), max(float(np.diag(symmetric).max()), 1e-30)*1e-12)
    previous_value = float(slopes@previous+.5*previous@(symmetric+step['damping']*np.diag(diagonal))@previous)
    result = {'status': 'saved_actual_GN_model_bounded_CPU_solution_not_reconstruction',
        'source_root': str(source), 'source_step': 1, 'original_common_scale': previous_scale,
        'previous_damped_model_change': previous_value,
        'previous_undamped_model_change': float(slopes@previous+.5*previous@symmetric@previous),
        'bounded_coefficients': coefficient.tolist(), 'certificate': certificate,
        'model_decrease_gain_factor': certificate['primal_damped_model_value']/previous_value,
        'same_six_actual_native_products_reused': True, 'new_acoustic_solves': 0,
        'new_patient_material_derivative_probes': 0, 'patient_truth_read': False,
        'source_model_SHA256': {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [source/'steps/step01.json', folder/'control_basis.npy',
                         folder/'actual_objective_gradient.npy', folder/'base_neural_parameters.npy']},
        'code_SHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.mkdir()
    np.save(out/'bounded_coefficients.npy', coefficient)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['status', 'previous_damped_model_change',
        'previous_undamped_model_change', 'bounded_coefficients', 'model_decrease_gain_factor',
        'same_six_actual_native_products_reused', 'new_acoustic_solves', 'patient_truth_read',
        'code_SHA256']} | {'certificate': {k: certificate[k] for k in ['dimension',
        'retained_voxel_constraints', 'maximum_linear_field_cap_ratios',
        'primal_damped_model_value', 'dual_lower_bound', 'relative_primal_dual_gap',
        'dual_stationarity_L2', 'SLSQP_success', 'undamped_model_change']},
        'complete_CPU_result': str(out/'result.json')}), flush=True)


if __name__ == '__main__':
    main()
