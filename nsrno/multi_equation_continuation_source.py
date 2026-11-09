"""One fixed-penalty warm/recycled actual four-channel source subproblem.

New module only; the first running full44 source implementation is unchanged.
"""
from pathlib import Path
import math
import numpy as np
import torch
from .native_multi_equation import MultiEquationWorkspace
from .native_full_source_dual import receiver_adjoint
from .multi_equation_source_solve import inner
from .warm_recycled_receiver_solver import solve_warm_recycled
from .low_frequency_0p5 import atomic, digest


def evaluate_source(material, acquisition, shot, sqrt_weight, prediction, node,
        folder, work, guard, progress, warm_record, space, operator_key,
        max_actions=12, tolerance=.01):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = dict(work.state)
    if digest(warm_record['dual_state_path']) != warm_record['dual_state_SHA256']:
        raise RuntimeError('Saved completed parent dual changed;do not replay')
    saved = torch.load(warm_record['dual_state_path'], map_location='cpu', weights_only=False)
    penalty = float(warm_record['lambda_squared'])
    if (float(saved['lambda_squared']) != penalty or penalty <= 0 or
            not np.isfinite(penalty)):
        raise RuntimeError('Preserve the parent source penalty exactly')
    a = acquisition
    wave = a.waveform(shot)
    zero = torch.zeros_like(wave)
    workspace = MultiEquationWorkspace(material, a.config, node)
    if bool(workspace.node.flatten()[a.si[a.sw != 0].long()].any()):
        raise RuntimeError('Physical source nodes must remain outside residual node support')
    rhs = ((a.observed[shot]-prediction)*sqrt_weight[None]).detach()
    rr0 = inner(rhs, rhs)
    if not rr0 > 1e-24:
        raise RuntimeError('Inspect zero/nonfinite original residual before a source solve')
    warm = saved['y'].to(device=rhs.device)
    if warm.shape != rhs.shape or warm.dtype != rhs.dtype:
        raise RuntimeError('Saved dual belongs to another receiver space')
    del saved
    errors, history = [], []

    def action(value, index):
        guard()
        progress(phase='actual_warm_recycled_operator', action=index)
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
            'necessary_current_K_on_old_y': index == 1, 'not_a_separate_probe': True})
        # K is positive semidefinite; the unchanged positive shift makes A SPD.
        if not np.isfinite([quadratic, norm, error]).all() or quadratic < 0 or error > .01:
            np.savez(folder/f'K{index:02d}_failed_actual_action.npz',
                direction=value.cpu().numpy(), result=result.cpu().numpy())
            raise RuntimeError(f'Required continuation action inconsistent:q={quadratic},norm={norm},relative={error};do not replay')
        errors.append(error)
        del cp, raw, raw_dual
        return result

    def checkpoint(packet):
        relative = math.sqrt(inner(packet['residual'], packet['residual'])/rr0)
        row = {'action': packet['completed_actions'], 'lambda_squared': penalty,
            'relative_recurrence_residual': relative,
            'required_operator_identity_error': errors[-1]}
        if history and history[-1]['action'] == row['action']:
            history[-1] = row
        else:
            history.append(row)
        atomic(folder/'inner_history.json', {'history': history, 'patient_truth_read': False,
            'first_action_recomputes_current_K_on_saved_y': True,
            'lambda_squared_unchanged_across_material_steps': True})
        path = folder/'dual_state.pt'
        temporary = path.with_suffix('.partial.pt')
        torch.save({k: v.detach().cpu() if torch.is_tensor(v) else v
            for k, v in packet.items()}, temporary)
        temporary.replace(path)
        progress(phase='warm_recycled_inner_action_completed', action=packet['completed_actions'],
            relative_residual=relative, lambda_squared=penalty)

    try:
        solved = solve_warm_recycled(action, rhs, penalty, space, operator_key, warm,
            maximum_actions=max_actions, relative_tolerance=tolerance, checkpoint=checkpoint)
    except BaseException:
        torch.save({'operator_key': space.key, 'shape': tuple(space.shape or ()),
            'vectors': [v.cpu() for v in space.vectors],
            'current_operator_images': [v.cpu() for v in space.images],
            'lambda_squared': penalty, 'no_automatic_replay': True},
            folder/'failed_current_coarse_space.pt')
        raise
    y = solved['x']
    guard()
    progress(phase='required_final_state_adjoint', completed_actions=solved['actual_K_actions'])
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
        np.savez(folder/'failed_actual_final_field.npz', y=y.cpu().numpy(),
            rhs=rhs.cpu().numpy(), actual_Ky=actual_Ky.cpu().numpy())
        raise RuntimeError('Required continuation final field disagrees;inspect without replay')
    RF_augmented = inner(actual_residual, actual_residual)
    primal = .5*(RF_augmented+penalty*source_norm)
    dual = penalty*(inner(rhs, y)-.5*source_norm-.5*penalty*inner(y, y))
    relative_gap = max(0., primal-dual)/max(abs(primal), abs(dual), 1e-30)
    weighted = statistics.cpu().numpy()*penalty
    if not np.isfinite(weighted).all() or np.any(weighted[[0, 2, 4, 6]] < 0):
        raise RuntimeError('Nonfinite actual continuation conditional statistics')
    final_state = {'y': y.cpu(), 'actual_native_Ky': actual_Ky.cpu(),
        'actual_normal_residual': normal_residual.cpu(), 'lambda_squared': penalty,
        'completed_actions': solved['actual_K_actions'], 'operator_key': operator_key,
        'native_Ky_belongs_only_to_this_operator_key': True, 'no_automatic_replay': True}
    temporary = folder/'dual_state.partial.pt'
    torch.save(final_state, temporary)
    temporary.replace(folder/'dual_state.pt')
    details = {'completed_inner_actions': solved['actual_K_actions'],
        'completed_CG_updates_after_necessary_warm_action': solved['inner_CG_actions'],
        'maximum_inner_actions': max_actions, 'lambda_squared': penalty,
        'penalty_fixed_after_first_required_action': False,
        'penalty_fixed_from_completed_parent_source': True,
        'warm_start_requires_current_operator_action': True,
        'old_material_K_images_reused': False, 'operator_key': operator_key,
        'coarse_dimension_used': solved['coarse_dimension_used'],
        'next_coarse_dimension': solved['next_coarse_dimension'],
        'physical_RF_before_from_saved_same_material': rr0, 'augmented_RF': RF_augmented,
        'weighted_source_penalty': .5*penalty*source_norm,
        'wavefield_primal': primal, 'wavefield_dual': dual, 'relative_primal_dual_gap': relative_gap,
        'actual_relative_normal_residual': quality,
        'actual_source_norm_identity_error': identity_error,
        'actual_inner_converged': quality <= tolerance and relative_gap <= .02,
        'maximum_recurrence_operator_identity_error': max(errors),
        'native_work_delta': {k: value-before.get(k, 0) for k, value in work.state.items()
            if k.startswith(('issued_', 'completed_'))},
        'patient_truth_read': False, 'new_material_Jacobian_or_JVPs': 0,
        'all_four_residual_channels': True, 'centered_statistics_include_lambda_squared': True,
        'statistics_archived_only_in_worker_aggregate': True,
        'actual_native_Ky_saved_for_inspection_not_cross_material_reuse': True}
    atomic(folder/'source_result.json', details)
    return weighted, details

