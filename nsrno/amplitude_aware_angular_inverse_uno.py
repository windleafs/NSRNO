"""Angular-QR inverse with RF amplitude visible to finite nonlinear updates.

The leading density path remains exactly residual-linear at fixed baseline.
Only the finite c/alpha/remainder head receives log1p(residual RMS).
No existing frozen training source is modified.
"""
import torch
from torch import nn
from .angular_qr_inverse_uno import AngularQRInverseUNO


class AmplitudeAwareAngularInverseUNO(AngularQRInverseUNO):
    def __init__(self, query_chunk=256):
        super().__init__(query_chunk)
        old = self.nonlinear_stem
        replacement = nn.Conv2d(old.in_channels + 1, old.out_channels, 1)
        with torch.no_grad():
            replacement.weight.zero_()
            replacement.weight[:, :old.in_channels].copy_(old.weight)
            replacement.bias.copy_(old.bias)
        self.nonlinear_stem = replacement
        self.config.update(
            architecture='amplitude_aware_angular_QR_inverse_UNO_v1',
            residual_amplitude_visible_to_finite_nonlinear_heads=True,
            residual_amplitude_feature='log1p(global RF residual RMS in training units)',
            additional_amplitude_input_weights_initialized_to_zero=True,
            additional_parameters_beyond_angular_QR=old.out_channels,
            added_parameter_count=self.config['added_parameter_count'] + old.out_channels,
            initial_parent_function_preserved=True,
            amplitude_does_not_condition_density_linear_leading_path=True,
            forced_positive_homogeneity_of_c_and_alpha_removed=True)
        # Correct an inherited descriptive flag, without changing frozen parents.
        self.readout.config.update(
            features_linear_in_both_RF_inputs_at_fixed_current_material=False,
            features_linear_in_residual_at_fixed_baseline_RF_and_material=True,
            baseline_magnitude_and_phase_views_not_linear_in_baseline=True)

    def load_parent_angular_state(self, state):
        """Lift a closed parent checkpoint with its original function preserved."""
        value = dict(state)
        old = value['nonlinear_stem.weight']
        if old.shape[1] + 1 != self.nonlinear_stem.weight.shape[1]:
            raise ValueError('Expected the original angular-QR finite-head input width')
        lifted = old.new_zeros(self.nonlinear_stem.weight.shape)
        lifted[:, :old.shape[1]] = old
        value['nonlinear_stem.weight'] = lifted
        self.load_state_dict(value, strict=True)

    def forward(self, signal, baseline_roi, context, residual_rms):
        baseline_features, difference, coverage = self.readout(signal, context)
        base = (baseline_roi - self.centres) / self.scales
        ctx = (context - self.centres) / self.scales
        bulk = torch.cat((ctx.mean((-2, -1)), ctx.std((-2, -1), unbiased=False)), 1)
        bulk = bulk[:, :, None, None].expand(-1, -1, 216, 192)
        state = self.state(torch.cat(
            (baseline_features.asinh(), coverage, base, self.coordinates, bulk), 1))
        b = baseline_features.reshape(1, -1, 2, 216, 192)
        d = difference.reshape_as(b)
        norm = torch.hypot(b[:, :, 0], b[:, :, 1])
        denominator = torch.where(norm > 0, norm, torch.ones_like(norm))
        pr, pi = b[:, :, 0] / denominator, b[:, :, 1] / denominator
        rotated = torch.stack(
            (d[:, :, 0] * pr + d[:, :, 1] * pi,
             d[:, :, 1] * pr - d[:, :, 0] * pi), 2).flatten(1, 2)
        signed = torch.cat((difference, rotated), 1)
        r = residual_rms.reshape(-1, 1, 1, 1)
        amplitude = r.log1p().expand(-1, -1, 216, 192)
        nonlinear = self.nonlinear_head(self.nonlinear(self.nonlinear_stem(
            torch.cat((signed, state, base, self.coordinates, amplitude), 1))))
        density = self.density_head(self.density(self.density_stem(signed), state))
        density = density * (12 * torch.tanh(self.density_log_gain(state) / 12)).exp()
        delta = torch.cat(
            (r * nonlinear[:, :1],
             r.clamp_min(1e-12) * density + r.square() * torch.tanh(nonlinear[:, 1:2]),
             r * nonlinear[:, 2:]), 1)
        delta = delta.masked_fill(self.gel, 0.)
        prediction = baseline_roi + self.scales * delta
        update_amplitude = delta[:, :, 5:].detach().square().mean((-2, -1)).sqrt()
        shape = delta / update_amplitude.clamp_min(1e-12)[:, :, None, None]
        gain = (update_amplitude.clamp_min(1e-12) /
                residual_rms.reshape(-1, 1).clamp_min(1e-12)).log()
        return {'prediction': prediction, 'delta_scaled': delta, 'shape': shape,
                'amplitude_scaled': update_amplitude, 'log_gain': gain,
                'residual_RMS': residual_rms.reshape(-1, 1)}
