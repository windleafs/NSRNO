"""Corrected residual-ray dual objective. Uses real sources, never source coding."""
import math
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from .native_source_extension import SourceExtensionWorkspace
from .native_tangent import pullback


def spatial_weight(acquisition, wave, centre_hz, device):
    c = acquisition.config
    active = wave.detach().square().sum(0).cpu().numpy() > 0
    si = acquisition.si.detach().cpu().numpy()
    sw = acquisition.sw.detach().cpu().numpy()
    support = np.zeros((c.nz, c.nx), bool)
    support.ravel()[si[active][sw[active] != 0]] = True
    if not support.any():
        raise ValueError('Empty unchanged physical source support')
    distance = distance_transform_edt(~support, sampling=c.dx)
    h = 1540./centre_hz
    z, x = np.ogrid[:c.nz, :c.nx]
    mask = (z >= 80) & (z < 937) & (x >= 24) & (x < 1097)
    weight = (h/np.sqrt(h*h+distance*distance))*mask
    return torch.from_numpy(np.ascontiguousarray(weight, np.float32)).to(device)


def ray_scalars(rr, aa, calibration=None):
    if not math.isfinite(rr+aa) or rr <= 0 or aa <= 0:
        raise ValueError('A finite nonzero actual residual and extension norm are required')
    if calibration is None:
        # Fixed after the first actual inverse gradient in this source family.
        # This is a numerical metric from observed RF, not a truth-dependent prior.
        lam2 = .1*aa/rr
        calibration = {'lambda_squared': lam2, 'normalization': (aa+lam2*rr)/rr,
                       'anchor_actual_residual_squared': rr, 'anchor_extension_norm_squared': aa}
    lam2, normalization = calibration['lambda_squared'], calibration['normalization']
    denominator = aa+lam2*rr
    alpha = rr/denominator
    return normalization*rr*rr/denominator, alpha, calibration


def evaluate_shot(material, acquisition, shot, weights, coefficient, centre_hz,
                  calibration=None, need_gradient=True):
    """One true source contribution; no patient truth or material-Jacobian probes."""
    if not need_gradient and calibration is None:
        raise ValueError('Line evaluation must use the same frozen family calibration')
    a = acquisition
    wave = a.waveform(shot)
    weight = spatial_weight(a, wave, centre_hz, material.device)
    workspace = SourceExtensionWorkspace(material, a.config)
    raw, cp_physical = workspace.physical_forward(wave, a.si, a.sw, a.ri, a.rw, need_gradient)
    # Only the small linear receiver readout is differentiated here. No retained
    # time-propagation graph or true internal wavefield is used.
    with torch.enable_grad():
        raw.requires_grad_(True)
        prediction = a.scattered_coefficients(raw, shot)
        residual = a.observed[shot]-prediction
        rr_tensor = coefficient*(weights[None]*residual.abs().square()).sum()
        raw_r, = torch.autograd.grad(-.5*rr_tensor, raw)
    prediction = prediction.detach()
    residual = residual.detach()
    raw_r = raw_r.detach().contiguous()
    rr = float(rr_tensor.detach())
    aa, cp_dual = workspace.adjoint(raw_r, a.ri, a.rw, weight, need_gradient)
    loss, alpha, calibration = ray_scalars(rr, aa, calibration)
    gradient = None
    correction_norm = None
    extended_change_squared = None
    if need_gradient:
        raw_augmented, cp_augmented = workspace.augmented_forward(
            wave, a.si, a.sw, a.ri, a.rw, raw_r, weight, cp_dual, alpha)
        with torch.enable_grad():
            raw_augmented.requires_grad_(True)
            augmented = a.scattered_coefficients(raw_augmented, shot)
            dual_gradient = (residual-(augmented.detach()-prediction)
                             -calibration['lambda_squared']*alpha*residual)
            scalar = coefficient*(weights[None]*(augmented.conj()*dual_gradient).real).sum()
            raw_gy, = torch.autograd.grad(scalar, raw_augmented)
        correction_norm = float(coefficient*(weights[None]*dual_gradient.abs().square()).sum())
        extended_change_squared = float(coefficient*(weights[None]*
            (augmented.detach()-prediction).abs().square()).sum())
        physical_correction = pullback(material, wave, a.si, a.sw, a.ri, a.rw,
                                       a.config, cp_physical, raw_gy.detach())
        del cp_physical
        augmented_gradient = workspace.augmented_pullback(wave, a.si, a.sw, a.ri, a.rw,
            raw_r, weight, cp_dual, cp_augmented, alpha)
        gradient = -2*alpha*calibration['normalization']*(augmented_gradient+physical_correction)
    details = {'shot': int(shot), 'actual_physical_RF_contribution': rr,
        'ray_objective_contribution': loss, 'raw_extension_norm_squared': aa,
        'chosen_extension_norm_squared': alpha*alpha*aa, 'ray_coefficient': alpha,
        'dual_stationarity_residual_squared': correction_norm,
        'augmented_minus_physical_RF_squared': extended_change_squared,
        'calibration': calibration, 'exact_reduced_ray_gradient_correction_included': need_gradient,
        'true_source_not_encoded': True, 'patient_truth_read': False,
        'not_a_full_dual_or_material_convergence_certificate': True,
        'adjoint_direction_passes_including_checkpoint_replays': 5 if need_gradient else 1,
        'physical_forward_passes': 1,
        'augmented_forward_passes': 1 if need_gradient else 0,
        'ordinary_physical_checkpoint_replay_passes': 1 if need_gradient else 0,
        'augmented_checkpoint_replay_passes': 1 if need_gradient else 0}
    return loss, gradient, calibration, details
