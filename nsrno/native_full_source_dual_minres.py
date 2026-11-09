"""Actual full receiver-space source dual, using the frozen native propagators."""
from pathlib import Path
import ctypes
import hashlib
import json
import numpy as np
import torch
from .native_source_extension import SourceExtensionWorkspace, library
from .native_time import ptr
from .source_extension_ray import spatial_weight
from .source_dual_receiver_metric import receiver_metric
from .full_source_dual_cg import dual_bounds, inner
from .full_source_dual_minres import solve_dual
from .low_frequency_0p5 import atomic, digest


class NativeWork:
    def __init__(self, folder):
        self.path = Path(folder)/'native_work.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'issued_adjoint_direction_passes': 0, 'completed_adjoint_direction_passes': 0,
            'issued_physical_forward_passes': 0, 'completed_physical_forward_passes': 0,
            'issued_auxiliary_forward_passes': 0, 'completed_auxiliary_forward_passes': 0,
            'completed_native_phases': 0, 'inflight_phase': None}
        if self.state['inflight_phase'] is not None:
            raise RuntimeError('Inspect partially executed native phase; do not automatically repeat it')

    def run(self, name, function, adjoint=0, physical=0, auxiliary=0):
        amounts = {'adjoint_direction_passes': adjoint, 'physical_forward_passes': physical,
                   'auxiliary_forward_passes': auxiliary}
        for key, number in amounts.items():
            self.state['issued_'+key] += number
        self.state['inflight_phase'] = name
        atomic(self.path, self.state)
        result = function()
        for key, number in amounts.items():
            self.state['completed_'+key] += number
        self.state['completed_native_phases'] += 1
        self.state['inflight_phase'] = None
        atomic(self.path, self.state)
        return result


class DualWorkspace(SourceExtensionWorkspace):
    @torch.no_grad()
    def auxiliary_forward(self, zero_wave, si, sw, ri, rw, raw_dual, weight, cp_dual):
        """K action needs no augmented-forward checkpoint bank."""
        raw = torch.empty((self.c.nt, self.c.nr), device=self.device)
        lib = library()
        before = [self.m, zero_wave, si, sw, ri, rw, raw_dual, weight, cp_dual]
        after = [raw, None, self.state, self.other, self.vp, self.k,
                 self.adj, self.prev, self.work, self.forcing_tape()]
        self.check(lib, lib.extension_forward(ctypes.byref(self.c), *[ptr(t) for t in before],
            ctypes.c_float(1.), *[ptr(t) for t in after], self.stream))
        return raw


def receiver_adjoint(a, value, weights, coefficient):
    with torch.enable_grad():
        raw = torch.zeros((a.config.nt, a.config.nr), device=weights.device, requires_grad=True)
        coefficients = a.coefficients(raw)
        scalar = coefficient*(weights[None]*(coefficients.conj()*value.detach()).real).sum()
        result, = torch.autograd.grad(scalar, raw)
    return result.detach().contiguous()


def save_state(path, state):
    snapshot = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in state.items()}
    temporary = path.with_suffix('.partial.pt')
    torch.save(snapshot, temporary)
    temporary.replace(path)
    last = state['history'][-1] if state['history'] else None
    atomic(path.with_suffix('.progress.json'), {'initialized': state['initialized'],
        'completed_K_actions': state['action_count'], 'last_inner_estimate': last,
        'patient_truth_read': False, 'actual_inverse_subproblem_not_a_validation_probe': True})


def evaluate_full_shot(material, a, shot, weights, coefficient, centre_hz, lambda_squared,
                       folder, calibration=None, need_gradient=True, cached_prediction=None,
                       max_actions=20, relative_gap=.02, warm_start_path=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    work = NativeWork(folder)
    wave = a.waveform(shot)
    zero_wave = torch.zeros_like(wave)
    source_weight = spatial_weight(a, wave, centre_hz, material.device)
    workspace = DualWorkspace(material, a.config)
    if cached_prediction is None:
        raw, _ = work.run('physical_forward', lambda: workspace.physical_forward(
            wave, a.si, a.sw, a.ri, a.rw, False), physical=1)
        prediction = a.scattered_coefficients(raw, shot).detach()
        del raw
    else:
        prediction = torch.from_numpy(np.asarray(cached_prediction)).to(material.device)
        if prediction.shape != a.observed[shot].shape:
            raise ValueError('Wrong same-material cached original RF shape')
    rhs = (a.observed[shot]-prediction).detach()*(weights > 0)[None]
    rr = coefficient*inner(rhs, rhs, weights)
    effective_weights = coefficient*weights
    metric, metric_details = receiver_metric(a, source_weight, weights, coefficient)
    manifest = {'material_array_SHA256': hashlib.sha256(material.detach().cpu().numpy().tobytes()).hexdigest(),
        'source_shot': int(shot), 'lambda_squared': lambda_squared, 'centre_hz': centre_hz,
        'data_rhs_SHA256': hashlib.sha256(rhs.cpu().numpy().tobytes()).hexdigest(),
        'metric': metric_details, 'patient_truth_read': False}
    manifest_file = folder/'operator_manifest.json'
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        raise RuntimeError('Same saved full-dual task changed its fixed physical system')
    atomic(manifest_file, manifest)
    action_folder = folder/'operator_actions'
    action_folder.mkdir(exist_ok=True)
    identity_errors = []

    def action(value, index):
        p = value.detach().cpu().numpy()
        signature = hashlib.sha256(p.tobytes()).hexdigest()
        output = action_folder/f'action{index:03d}.npz'
        receipt = output.with_suffix('.json')
        if output.exists() or receipt.exists():
            if not output.exists() or not receipt.exists():
                raise RuntimeError('Inspect partial saved K action; do not blindly repeat native work')
            old = json.loads(receipt.read_text())
            if old['direction_SHA256'] != signature or old['array_SHA256'] != digest(output):
                raise RuntimeError('Saved K action belongs to another direction')
            identity_errors.append(old['same_required_action_relative_identity_error'])
            with np.load(output) as saved:
                return torch.from_numpy(saved['Kp'].copy()).to(material.device)
        raw_dual = receiver_adjoint(a, value, weights, coefficient)
        aa, cp = work.run(f'K{index:03d}_adjoint', lambda: workspace.adjoint(
            raw_dual, a.ri, a.rw, source_weight, True), adjoint=1)
        raw_aux = work.run(f'K{index:03d}_checkpoint_and_auxiliary_forward',
            lambda: workspace.auxiliary_forward(zero_wave, a.si, a.sw, a.ri, a.rw,
                raw_dual, source_weight, cp), adjoint=1, auxiliary=1)
        result = a.coefficients(raw_aux).detach()*(weights > 0)[None]
        dot = inner(value, result, effective_weights)
        error = abs(dot-aa)/max(abs(dot), abs(aa), 1e-30)
        if not np.isfinite([dot, aa, error]).all() or dot <= 0 or error > .01:
            raise RuntimeError('A required full-dual K action is numerically inconsistent')
        temporary = output.with_name(output.stem+'.partial.npz')
        np.savez(temporary, p=p, Kp=result.cpu().numpy())
        temporary.replace(output)
        atomic(receipt, {'action': index, 'direction_SHA256': signature,
            'array_SHA256': digest(output), 'native_source_energy': aa, 'receiver_quadratic': dot,
            'same_required_action_relative_identity_error': error,
            'used_in_actual_full_dual_residual_minimum': True, 'new_material_derivative_probe': False})
        identity_errors.append(error)
        return result

    state_path = folder/'dual_state.pt'
    state = torch.load(state_path, map_location=material.device, weights_only=False) if state_path.exists() else None
    solved = solve_dual(rhs, effective_weights, lambda_squared, action, metric,
        max_actions=max_actions, relative_gap=relative_gap, state=state,
        checkpoint=lambda value: save_state(state_path, value), saved_actions=action_folder,
        warm_start=(torch.load(warm_start_path, map_location=material.device, weights_only=False)
                    if state is None and warm_start_path else None))
    details = {'shot': int(shot), 'physical_RF_contribution': rr,
        'inner_K_actions': solved['action_count'], 'recurrence_inner_converged': solved['converged'],
        'recurrence_bounds': solved['bounds'], 'metric': metric_details,
        'maximum_same_required_action_identity_error': max(identity_errors, default=0.),
        'native_work': dict(work.state), 'patient_truth_read': False,
        'material_field_lowpass': False, 'full_data_dual_not_residual_ray': True,
        'actual_native_inner_converged': False, 'gradient_ready': False}
    if not solved['converged']:
        atomic(folder/'result.json', details)
        return None, None, calibration, details
    y = solved['y']
    raw_y = receiver_adjoint(a, y, weights, coefficient)
    aa, cp_y = work.run('final_y_actual_adjoint', lambda: workspace.adjoint(
        raw_y, a.ri, a.rw, source_weight, True), adjoint=1)
    cp_augmented = None
    if need_gradient:
        raw_augmented, cp_augmented = work.run('final_y_augmented_forward',
            lambda: workspace.augmented_forward(wave, a.si, a.sw, a.ri, a.rw,
                raw_y, source_weight, cp_y, 1.), adjoint=1, auxiliary=1)
        actual_Ky = a.scattered_coefficients(raw_augmented, shot).detach()-prediction
    else:
        raw_aux = work.run('final_y_actual_auxiliary_forward',
            lambda: workspace.auxiliary_forward(zero_wave, a.si, a.sw, a.ri, a.rw,
                raw_y, source_weight, cp_y), adjoint=1, auxiliary=1)
        actual_Ky = a.coefficients(raw_aux).detach()
    actual_Ky = actual_Ky*(weights > 0)[None]
    bounds = dual_bounds(rhs, y, actual_Ky, lambda_squared, effective_weights, aa)
    details['actual_native_bounds'] = bounds
    details['actual_native_inner_converged'] = bounds['relative_quality_gap'] <= relative_gap
    if not details['actual_native_inner_converged']:
        # Retain the actual reliable residual for a future same-system continuation.
        state = solved['state']
        residual = rhs-actual_Ky-lambda_squared*y
        z = metric.inverse(residual, state['metric_scale'], lambda_squared)*(weights > 0)[None]
        state.update(Ky=actual_Ky, residual=residual, p=z, rho=inner(residual, z, effective_weights))
        state['history'].append({'native_reliable_residual_update': True, **bounds})
        save_state(state_path, state)
        details['native_work'] = dict(work.state)
        atomic(folder/'result.json', details)
        return None, None, calibration, details
    if calibration is None or calibration.get('normalization') is None:
        if not need_gradient or bounds['dual'] <= 0:
            raise RuntimeError('First full-dual normalization requires an actual converged inverse gradient')
        calibration = {'lambda_squared': lambda_squared,
            'normalization': rr/(2*bounds['dual']), 'anchor_physical_RF': rr,
            'anchor_actual_inner_bounds': bounds}
    if calibration['lambda_squared'] != lambda_squared:
        raise RuntimeError('Frozen full-dual regularization changed')
    factor = 2*calibration['normalization']
    objective = factor*bounds['dual']
    gradient = None
    if need_gradient:
        gradient = -factor*work.run('actual_augmented_material_pullback',
            lambda: workspace.augmented_pullback(wave, a.si, a.sw, a.ri, a.rw,
                raw_y, source_weight, cp_y, cp_augmented, 1.), adjoint=2, auxiliary=1)
        details['gradient_ready'] = True
    details.update(objective=objective, objective_numerical_lower=factor*bounds['numerical_lower'],
        objective_numerical_upper=factor*bounds['numerical_upper'], calibration=calibration,
        native_work=dict(work.state), full_dual_gradient_uses_numerically_solved_auxiliary_variable=True)
    atomic(folder/'result.json', details)
    return objective, gradient, calibration, details
