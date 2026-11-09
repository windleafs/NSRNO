"""Untrained joint output charts: integrated density and differentiated attenuation primitive."""
import torch
from torch import nn
from torch.nn import functional as F
from .gradient_density_angular_inverse_uno import GradientDensityAngularInverseUNO
from .depth_primitive import coefficient_from_mean_primitive


class CoefficientChartAngularInverseUNO(GradientDensityAngularInverseUNO):
    def __init__(self, query_chunk=256):
        super().__init__(query_chunk)
        self.alpha_primitive_head = nn.Conv2d(16, 1, 1, bias=False)
        nn.init.zeros_(self.alpha_primitive_head.weight)
        self.config.update(
            architecture='density_gradient_alpha_primitive_angular_inverse_UNO_v1',
            alpha_primitive='normalized depth antiderivative on unknown tissue rows',
            alpha_primitive_is_not_an_exact_native_ray_integral=True,
            alpha_new_component_residual_linear_at_fixed_baseline=True,
            total_alpha_not_forced_linear_or_odd=True,
            original_alpha_finite_nonlinear_path_retained=True,
            alpha_primitive_initially_zero=True,
            added_parameters_beyond_gradient_density_parent=16,
            added_parameters_beyond_amplitude_parent=48,
            added_parameter_count=self.config['added_parameter_count'] + 16,
            independent_three_field_output_codomain_retained=True)

    def load_parent_amplitude_state(self, state):
        value = self.state_dict()
        allowed = ('density_head.gradient_head.', 'density_head.integration.',
                   'alpha_primitive_head.')
        expected = set()
        for name, tensor in state.items():
            target = 'density_head.direct.weight' if name == 'density_head.weight' else name
            if target not in value:
                raise ValueError('Unexpected parent tensor: ' + name)
            value[target] = tensor
            expected.add(target)
        if any(not name.startswith(allowed) for name in set(value) - expected):
            raise ValueError('Parent checkpoint does not cover the retained model')
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
        linear_features = self.density(self.density_stem(signed), state)
        density = self.density_head(linear_features)
        primitive = self.alpha_primitive_head(linear_features)[:, :, 5:]
        alpha_linear = F.pad(coefficient_from_mean_primitive(primitive), (0, 0, 5, 0))
        density = density * (12 * torch.tanh(self.density_log_gain(state) / 12)).exp()
        delta = torch.cat(
            (r * nonlinear[:, :1],
             r.clamp_min(1e-12) * density + r.square() * torch.tanh(nonlinear[:, 1:2]),
             r * nonlinear[:, 2:] + r.clamp_min(1e-12) * alpha_linear), 1)
        delta = delta.masked_fill(self.gel, 0.)
        prediction = baseline_roi + self.scales * delta
        update_amplitude = delta[:, :, 5:].detach().square().mean((-2, -1)).sqrt()
        shape = delta / update_amplitude.clamp_min(1e-12)[:, :, None, None]
        gain = (update_amplitude.clamp_min(1e-12) /
                residual_rms.reshape(-1, 1).clamp_min(1e-12)).log()
        return {'prediction': prediction, 'delta_scaled': delta, 'shape': shape,
                'amplitude_scaled': update_amplitude, 'log_gain': gain,
                'residual_RMS': residual_rms.reshape(-1, 1)}
