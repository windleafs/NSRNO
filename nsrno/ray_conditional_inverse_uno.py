"""Two-dimensional UNO following current-state phase-preserving RF readout.

c/alpha corrections are nonlinear. Density has a residual-linear leading path
conditioned on the current state, plus a bounded quadratic-amplitude correction
for finite nonlinear changes.
No native propagator or Geo-Flow model is used here.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .current_ray_rf_readout import CurrentRayRFReadout
from .uno_rf import UShape, SpectralMap


class LinearOperatorBlock(nn.Module):
    def __init__(self, cin, cout, modes):
        super().__init__()
        self.spectral = SpectralMap(cin, cout, modes)
        self.local = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False),
                                   nn.Conv2d(cout, cout, 3, padding=1, groups=cout, bias=False))

    def forward(self, value, size):
        resized = F.interpolate(value, size=size, mode='bilinear', align_corners=False)
        return .5 * (self.spectral(value, size) + self.local(resized))


class GatedLinearUShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [16, 24, 32, 24, 16]
        modes = [(16, 16), (12, 12), (8, 8), (12, 12), (16, 16)]
        self.blocks = nn.ModuleList([LinearOperatorBlock(self.channels[max(0, i-1)], c, modes[i])
                                     for i, c in enumerate(self.channels)])
        self.skips = nn.ModuleList([nn.Conv2d(24, 24, 1, bias=False),
                                    nn.Conv2d(16, 16, 1, bias=False)])
        self.gates = nn.ModuleList([nn.Conv2d(16, c, 1) for c in self.channels])
        for gate in self.gates:
            nn.init.zeros_(gate.weight); nn.init.zeros_(gate.bias)

    def forward(self, value, state):
        h, w = value.shape[-2:]
        sizes = [(h, w), (h//2, w//2), (h//4, w//4), (h//2, w//2), (h, w)]
        def run(i, x):
            def block(v, condition):
                gate = 1 + .25 * torch.tanh(F.interpolate(self.gates[i](condition), sizes[i],
                                                        mode='bilinear', align_corners=False))
                return self.blocks[i](v, sizes[i]) * gate
            return checkpoint(block, x, state, use_reentrant=False) if torch.is_grad_enabled() else block(x, state)
        a = run(0, value); b = run(1, a); c = run(2, b)
        d = run(3, c) + self.skips[0](b)
        return run(4, d) + self.skips[1](a)


class RayConditionalInverseUNO(nn.Module):
    def __init__(self, query_chunk=256):
        super().__init__()
        self.readout = CurrentRayRFReadout(query_chunk=query_chunk)
        channels = self.readout.feature_channels_per_RF_type
        coverage = 11 * len(self.readout.lags)
        self.register_buffer('scales', torch.tensor([80., 100., .2])[None, :, None, None])
        self.register_buffer('centres', torch.tensor([1540., 1000., .5])[None, :, None, None])
        z, x = torch.meshgrid(torch.linspace(0, 1, 216), torch.linspace(-1, 1, 192), indexing='ij')
        self.register_buffer('coordinates', torch.stack((x, z))[None])
        self.register_buffer('gel', torch.arange(216)[None, None, :, None] < 5)
        self.state = nn.Sequential(
            nn.Conv2d(channels + coverage + 11, 16, 1), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
        self.nonlinear_stem = nn.Conv2d(2 * channels + 21, 24, 1)
        self.nonlinear = UShape([24, 32, 48, 32, 24],
                               [(16, 16), (12, 12), (8, 8), (12, 12), (16, 16)], True)
        self.nonlinear_head = nn.Sequential(nn.Conv2d(24, 24, 1), nn.GELU(), nn.Conv2d(24, 3, 1))
        self.density_stem = nn.Conv2d(2 * channels, 16, 1, bias=False)
        self.density = GatedLinearUShape()
        self.density_head = nn.Conv2d(16, 1, 1, bias=False)
        self.density_log_gain = nn.Conv2d(16, 1, 1)
        nn.init.normal_(self.nonlinear_head[-1].weight, std=.01)
        nn.init.zeros_(self.nonlinear_head[-1].bias)
        nn.init.normal_(self.density_head.weight, std=.03)
        nn.init.zeros_(self.density_log_gain.weight); nn.init.zeros_(self.density_log_gain.bias)
        self.config = {
            'architecture': 'current_ray_readout_spatial_UNO_signed_density_v1',
            'readout': self.readout.config,
            'outputs': 'independent sound speed, density, tissue attenuation corrections',
            'density_value_path': 'bias-free linear UNO; state-only gates and baseline-phase rotation',
            'density_leading_path_exact_linear_in_RF_residual_at_fixed_baseline': True,
            'finite_density_nonlinearity': 'r_RMS**2 * tanh(neural_correction)',
            'finite_density_nonlinear_remainder_bound_in_scaled_units': 'absolute value <= r_RMS**2',
            'density_total_update_is_not_globally_linear_or_odd': True,
            'c_and_alpha_finite_nonlinearity_preserved': True,
            'original_RF_amplitude_retained_through_residual_RMS': True,
            'input_RF_normalization_floor_compensated_in_linear_density_path': 1e-12,
            'no_development_truth_as_input': True, 'forced_smooth_c': False,
            'shared_material_edges': False, 'output_ROI_shape': [3, 216, 192],
            'neural_proposal_is_not_completed_physical_joint_reconstruction': True,
        }

    def forward(self, signal, baseline_roi, context, residual_rms):
        baseline_features, difference, coverage = self.readout(signal, context)
        base = (baseline_roi - self.centres) / self.scales
        ctx = (context - self.centres) / self.scales
        bulk = torch.cat((ctx.mean((-2, -1)), ctx.std((-2, -1), unbiased=False)), 1)[:, :, None, None].expand(-1, -1, 216, 192)
        state = self.state(torch.cat((baseline_features.asinh(), coverage, base, self.coordinates, bulk), 1))
        # Rotate each complex focused residual by a fixed baseline phase.
        b = baseline_features.reshape(1, -1, 2, 216, 192)
        d = difference.reshape_as(b)
        norm = torch.hypot(b[:, :, 0], b[:, :, 1])
        denominator = torch.where(norm > 0, norm, torch.ones_like(norm))
        pr, pi = b[:, :, 0] / denominator, b[:, :, 1] / denominator
        rotated = torch.stack((d[:, :, 0] * pr + d[:, :, 1] * pi,
                               d[:, :, 1] * pr - d[:, :, 0] * pi), 2).flatten(1, 2)
        signed = torch.cat((difference, rotated), 1)
        r = residual_rms.reshape(-1, 1, 1, 1)
        nonlinear = self.nonlinear_head(self.nonlinear(self.nonlinear_stem(
            torch.cat((signed, state, base, self.coordinates), 1))))
        density = self.density_head(self.density(self.density_stem(signed), state))
        density = density * (12 * torch.tanh(self.density_log_gain(state) / 12)).exp()
        delta = torch.cat((r * nonlinear[:, :1],
                           r.clamp_min(1e-12) * density + r.square() * torch.tanh(nonlinear[:, 1:2]),
                           r * nonlinear[:, 2:]), 1)
        delta = delta.masked_fill(self.gel, 0.)
        prediction = baseline_roi + self.scales * delta
        amplitude = delta[:, :, 5:].detach().square().mean((-2, -1)).sqrt()
        shape = delta / amplitude.clamp_min(1e-12)[:, :, None, None]
        gain = (amplitude.clamp_min(1e-12) / residual_rms.reshape(-1, 1).clamp_min(1e-12)).log()
        return {'prediction': prediction, 'delta_scaled': delta, 'shape': shape,
                'amplitude_scaled': amplitude, 'log_gain': gain,
                'residual_RMS': residual_rms.reshape(-1, 1)}


def secant_loss(result, baseline, target):
    scale = target.new_tensor([80., 100., .2])[None, :, None, None]
    exact = ((target - baseline) / scale)[:, :, 5:]
    amplitude = exact.square().mean((-2, -1)).sqrt()
    floor = target.new_tensor([5 / 80, 5 / 100, .02 / .2])[None]
    relative = ((result['delta_scaled'][:, :, 5:] - exact).square().mean((-2, -1)) /
                amplitude.square().clamp_min(floor.square()))
    return relative.mean(), {
        'relative_c_rho_alpha': relative.detach().mean(0).cpu().tolist(),
        'shape_loss': 0., 'log_gain_loss': 0.,
        'material_correction_loss_only': True,
        'target_amplitude_scaled': amplitude.detach().cpu().tolist(),
        'predicted_amplitude_scaled': result['amplitude_scaled'].cpu().tolist(),
    }
