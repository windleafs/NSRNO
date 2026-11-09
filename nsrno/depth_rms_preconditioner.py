"""Prepared, not yet run in inversion: positive depth RMS scaling, no field smoothing."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def depth_rms_direction(gradient, active_mask, lateral_ROI,
                        normalized_step_caps, sigma_rows=40., relative_floor=.02):
    """Use an already-required inverse gradient; launches no propagation/probe.

    The smoothed object is a positive step metric, never a material field.
    A fixed-code line search and full original-source acceptance remain the
    caller's responsibility. This is not a measured Hessian approximation.
    """
    value = np.asarray(gradient, np.float64)
    mask = np.asarray(active_mask, bool)
    lateral = np.asarray(lateral_ROI, bool)
    caps = np.asarray(normalized_step_caps, np.float64).reshape(3, 1, 1)
    if value.ndim != 3 or value.shape[0] != 3 or mask.shape != value.shape[1:]:
        raise ValueError('Three independent fields and the full active grid are required')
    if lateral.shape != (value.shape[-1],) or not np.all(caps > 0):
        raise ValueError('Explicit full-ROI column mask and positive native control caps')
    if not 0 < relative_floor <= 1 or not 1 <= sigma_rows <= 80:
        raise ValueError('Prespecified positive preconditioner range exceeded')
    if not np.all(np.isfinite(value)):
        raise ValueError('The ordinary inverse gradient must be finite')
    reference_mask = mask & lateral[None, :]
    count = reference_mask.sum(-1).astype(np.float64)
    if not np.any(count > 0):
        raise ValueError('An unknown tissue ROI is required')
    squared_sum = np.sum(value**2 * reference_mask[None], axis=-1)
    smooth_sum = gaussian_filter1d(squared_sum, sigma_rows, axis=-1, mode='nearest')
    smooth_count = gaussian_filter1d(count, sigma_rows, mode='nearest')
    row_rms = np.sqrt(smooth_sum / np.maximum(smooth_count[None], 1e-30))
    global_rms = np.sqrt(np.sum(squared_sum, axis=-1) / count.sum())
    diagonal = np.maximum(row_rms, relative_floor * global_rms[:, None])
    diagonal = np.maximum(diagonal, 1e-24) / .1
    direction = np.clip(-value / diagonal[:, :, None], -caps, caps) * mask[None]
    active_rows = np.any(mask, axis=1)
    return direction, {
        'metric': 'per-field positive RMS over depth-smoothed squared ordinary gradient',
        'global_RMS_in_scaled_native_control_units': global_rms.tolist(),
        'minimum_active_diagonal': diagonal[:, active_rows].min(-1).tolist(),
        'maximum_active_diagonal': diagonal[:, active_rows].max(-1).tolist(),
        'smoothed_metric_sigma_rows': float(sigma_rows),
        'relative_positive_floor': float(relative_floor),
        'material_field_smoothed': False,
        'independent_controls_retained': True,
        'extra_acoustic_derivatives_or_HVPs': 0,
        'descent_for_input_gradient_in_exact_arithmetic': True,
        'full_original_RF_acceptance_still_required': True,
    }
