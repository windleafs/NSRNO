"""Odd density direction with an even, separately learned amplitude.

The raw difference of nearly equal material outputs can suppress both the
update and its gradient. Normalize the odd *direction* and retain the parent's
even log gain. This keeps density odd at a fixed baseline without prescribing
global linearity of the three-parameter inverse.
"""
import torch
from .relative_phase_secant_uno import RelativePhaseSecantUNO
from .conditional_secant_uno import secant_loss


class BalancedOddDensitySecantUNO(RelativePhaseSecantUNO):
    def __init__(self, query_chunk=1024):
        super().__init__(query_chunk)
        self.config.update(
            architecture='conditional_relative_phase_UNO_normalized_odd_density_v1',
            density_correction='r * exp(even_log_gain) * normalized_odd_shape',
            density_direction='half difference of shared parent shapes at opposite RF residuals',
            density_gain='half sum of shared parent log gains at opposite RF residuals',
            density_update_exact_odd_at_fixed_baseline=True,
            density_raw_output_cancellation_does_not_set_amplitude=True,
            c_and_alpha_updates_unconstrained_nonlinear=True,
            global_three_parameter_inverse_linearity_assumed=False,
            output_requires_two_shared_weight_network_evaluations=True,
            auxiliary_shape_and_gain_losses='same original independent-field loss for all three fields')

    def forward(self, signal, baseline_roi, context, residual_rms):
        positive = super().forward(signal, baseline_roi, context, residual_rms)
        flipped = torch.cat((signal[:, :, :2], -signal[:, :, 2:4], signal[:, :, 4:]), 2)
        negative = super().forward(flipped, baseline_roi, context, residual_rms)
        raw_odd = .5 * (positive['shape'][:, 1:2] - negative['shape'][:, 1:2])
        raw_odd_rms = raw_odd[:, :, 5:].square().mean((-2, -1), keepdim=True).add(1e-12).sqrt()
        density_shape = raw_odd / raw_odd_rms
        density_gain = .5 * (positive['log_gain'][:, 1:2] + negative['log_gain'][:, 1:2])
        r = residual_rms.reshape(-1, 1)
        density_amplitude = r * density_gain.exp()
        density = density_amplitude[:, :, None, None] * density_shape
        delta = torch.cat((positive['delta_scaled'][:, :1], density,
                           positive['delta_scaled'][:, 2:]), 1)
        shape = torch.cat((positive['shape'][:, :1], density_shape, positive['shape'][:, 2:]), 1)
        log_gain = torch.cat((positive['log_gain'][:, :1], density_gain, positive['log_gain'][:, 2:]), 1)
        amplitude = torch.cat((positive['amplitude_scaled'][:, :1], density_amplitude,
                               positive['amplitude_scaled'][:, 2:]), 1)
        prediction = baseline_roi + self.scales * delta
        prediction = torch.where(self.gel, baseline_roi, prediction)
        return {**positive, 'prediction': prediction, 'delta_scaled': delta,
                'shape': shape, 'log_gain': log_gain, 'amplitude_scaled': amplitude,
                'density_raw_odd_shape_RMS': raw_odd_rms.flatten(1)}
