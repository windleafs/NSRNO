"""One actual whitened receiver-space four-channel wavefield subproblem."""
from pathlib import Path
import math
import numpy as np
import torch
from .native_multi_equation import MultiEquationWorkspace
from .native_full_source_dual import receiver_adjoint
from .low_frequency_0p5 import atomic


def inner(a, b):
    return float((a.conj()*b).real.sum())


def evaluate_source(material, acquisition, shot, sqrt_weight, prediction, node,
        folder, work, guard, progress, max_actions=12, gamma=.1, tolerance=.01):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = dict(work.state)
    a = acquisition
    wave = a.waveform(shot)
    zero = torch.zeros_like(wave)
    workspace = MultiEquationWorkspace(material, a.config, node)
    if bool(workspace.node.flatten()[a.si[a.sw != 0].long()].any()):
        raise RuntimeError('Physical source nodes must remain outside the material/residual node support')
    rhs = ((a.observed[shot]-prediction)*sqrt_weight[None]).detach()
    rr0 = inner(rhs, rhs)
    if not rr0 > 1e-24:
        raise RuntimeError('Inspect a zero original source residual before choosing its fixed penalty')
    def action(value, index):
        guard()
        progress(phase='actual_inner_operator', action=index)
        raw_dual = receiver_adjoint(a, value, sqrt_weight, 1.)
        norm, cp, _ = work.run(f'{folder.name}_K{index:02d}_state_adjoint',
            lambda: workspace.adjoint(raw_dual, a.ri, a.rw), adjoint=1)
        raw, _ = work.run(f'{folder.name}_K{index:02d}_adjoint_replay_and_auxiliary_forward',
            lambda: workspace.from_dual(zero, a.si, a.sw, a.ri, a.rw, raw_dual, cp),
            adjoint=1, auxiliary=1)
        result = (a.coefficients(raw)*sqrt_weight[None]).detach()
        quadratic = inner(value, result)
        error = abs(quadratic-norm)/max(abs(quadratic), abs(norm), 1e-30)
        atomic(folder/f'K{index:02d}_required_action.json', {'receiver_quadratic': quadratic,
            'source_norm': norm, 'relative_identity_error': error,
            'not_a_separate_probe': True})
        if not np.isfinite([quadratic, norm, error]).all() or quadratic <= 0 or error > .01:
            np.savez(folder/f'K{index:02d}_failed_actual_action.npz', direction=value.cpu().numpy(), result=result.cpu().numpy())
            raise RuntimeError(f'Required four-channel action inconsistent:dot={quadratic},norm={norm},relative={error};do not replay')
        del cp, raw, raw_dual
        return result, quadratic, error
    y = torch.zeros_like(rhs)
    residual, direction = rhs.clone(), rhs.clone()
    rho, penalty = rr0, None
    history = []
    for index in range(1, max_actions+1):
        Kp, curvature, error = action(direction, index)
        if penalty is None:
            penalty = gamma*curvature/rr0
            if not np.isfinite(penalty) or penalty <= 0:
                raise RuntimeError('A positive fixed per-source penalty is required')
        Ap = Kp+penalty*direction
        denominator = inner(direction, Ap)
        if not denominator > 0:
            raise RuntimeError('Actual shifted source normal system lost positive curvature')
        fraction = rho/denominator
        y.add_(direction, alpha=fraction)
        residual.add_(Ap, alpha=-fraction)
        new_rho = inner(residual, residual)
        relative = math.sqrt(max(new_rho, 0)/rr0)
        history.append({'action': index, 'lambda_squared': penalty,
            'relative_recurrence_residual': relative, 'required_operator_identity_error': error})
        atomic(folder/'inner_history.json', {'history': history, 'patient_truth_read': False})
        path = folder/'dual_state.pt'
        temporary = path.with_suffix('.partial.pt')
        torch.save({'y': y.cpu(), 'residual': residual.cpu(), 'direction': direction.cpu(),
            'lambda_squared': penalty, 'completed_actions': index, 'no_automatic_replay': True}, temporary)
        temporary.replace(path)
        progress(phase='inner_action_completed', action=index, relative_residual=relative,
            lambda_squared=penalty)
        if relative <= tolerance:
            break
        direction = residual+(new_rho/rho)*direction
        rho = new_rho
        del Kp, Ap
    guard()
    progress(phase='required_final_state_adjoint', completed_actions=len(history))
    raw_dual = receiver_adjoint(a, y, sqrt_weight, 1.)
    source_norm, cp, _ = work.run(f'{folder.name}_final_y_state_adjoint',
        lambda: workspace.adjoint(raw_dual, a.ri, a.rw), adjoint=1)
    augmented_raw, statistics = work.run(f'{folder.name}_final_y_replay_augmented_forward_statistics',
        lambda: workspace.from_dual(wave, a.si, a.sw, a.ri, a.rw, raw_dual, cp, True),
        adjoint=1, auxiliary=1)
    augmented = a.scattered_coefficients(augmented_raw, shot).detach()
    actual_Ky = (augmented-prediction)*sqrt_weight[None]
    actual_residual = (a.observed[shot]-augmented)*sqrt_weight[None]
    normal_residual = rhs-actual_Ky-penalty*y
    quality = math.sqrt(inner(normal_residual, normal_residual)/rr0)
    quadratic = inner(y, actual_Ky)
    identity_error = abs(quadratic-source_norm)/max(abs(quadratic), abs(source_norm), 1e-30)
    atomic(folder/'required_final_field_diagnostics.json', {'actual_relative_normal_residual': quality,
        'receiver_quadratic': quadratic, 'source_norm': source_norm,
        'relative_identity_error': identity_error})
    if not np.isfinite([quality, source_norm, identity_error]).all() or identity_error > .01:
        np.savez(folder/'failed_actual_final_field.npz', y=y.cpu().numpy(), rhs=rhs.cpu().numpy(), actual_Ky=actual_Ky.cpu().numpy())
        raise RuntimeError(f'Required augmented field disagrees:dot={quadratic},norm={source_norm},relative={identity_error}')
    RF_augmented = inner(actual_residual, actual_residual)
    primal = .5*(RF_augmented+penalty*source_norm)
    dual = penalty*(inner(rhs, y)-.5*source_norm-.5*penalty*inner(y, y))
    relative_gap = max(0., primal-dual)/max(abs(primal), abs(dual), 1e-30)
    weighted = statistics.cpu().numpy()*penalty
    if not np.isfinite(weighted).all() or np.any(weighted[[0, 2, 4, 6]] < 0):
        raise RuntimeError('Nonfinite actual all-equation conditional statistics')
    details = {'completed_inner_actions': len(history), 'maximum_inner_actions': max_actions,
        'lambda_squared': penalty, 'penalty_fixed_after_first_required_action': True,
        'physical_RF_before_from_saved_same_material': rr0, 'augmented_RF': RF_augmented,
        'weighted_source_penalty': .5*penalty*source_norm,
        'wavefield_primal': primal, 'wavefield_dual': dual, 'relative_primal_dual_gap': relative_gap,
        'actual_relative_normal_residual': quality, 'actual_source_norm_identity_error': identity_error,
        'actual_inner_converged': quality <= tolerance and relative_gap <= .02,
        'maximum_recurrence_operator_identity_error': max(h['required_operator_identity_error'] for h in history),
        'native_work_delta': {k: value-before.get(k, 0) for k, value in work.state.items()
            if k.startswith(('issued_', 'completed_'))},
        'patient_truth_read': False, 'new_material_Jacobian_or_JVPs': 0,
        'all_four_residual_channels': True, 'centered_statistics_include_lambda_squared': True,
        'statistics_archived_only_in_worker_aggregate': True}
    atomic(folder/'source_result.json', details)
    return weighted, details
