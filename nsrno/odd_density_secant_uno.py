"""Parameter-specific signed density updates; c and alpha remain nonlinear.

This is a learned local correction proposal, not a proof that the global
three-field inverse is odd. Existing finite training RF for density is nearly
antisymmetric; finite sound-speed RF is not.
"""
import torch
from .relative_phase_secant_uno import RelativePhaseSecantUNO


class OddDensitySecantUNO(RelativePhaseSecantUNO):
    def __init__(self, query_chunk=1024):
        super().__init__(query_chunk)
        self.config.update(
            architecture='conditional_relative_phase_UNO_signed_density_v1',
            density_correction='half difference of shared network at opposite RF residuals',
            density_update_exact_odd_at_fixed_baseline=True,
            c_and_alpha_updates_unconstrained_nonlinear=True,
            global_three_parameter_inverse_linearity_assumed=False,
            output_requires_two_shared_weight_network_evaluations=True,
            auxiliary_shape_and_gain_losses='c and alpha only; density uses material correction error')

    def forward(self, signal, baseline_roi, context, residual_rms):
        positive = super().forward(signal, baseline_roi, context, residual_rms)
        flipped = torch.cat((signal[:, :, :2], -signal[:, :, 2:4], signal[:, :, 4:]), 2)
        negative = super().forward(flipped, baseline_roi, context, residual_rms)
        density = .5 * (positive['delta_scaled'][:, 1:2] - negative['delta_scaled'][:, 1:2])
        delta = torch.cat((positive['delta_scaled'][:, :1], density,
                           positive['delta_scaled'][:, 2:]), 1)
        safe_amplitude = density[:, :, 5:].square().mean((-2, -1)).clamp_min(1e-24).sqrt()
        shape_density = density / safe_amplitude[:, :, None, None]
        r = residual_rms.reshape(-1, 1)
        gain_density = (safe_amplitude / r.clamp_min(1e-12)).log()
        amplitude_density = torch.where(r == 0, torch.zeros_like(safe_amplitude), safe_amplitude)
        shape = torch.cat((positive['shape'][:, :1], shape_density, positive['shape'][:, 2:]), 1)
        log_gain = torch.cat((positive['log_gain'][:, :1], gain_density, positive['log_gain'][:, 2:]), 1)
        amplitude = torch.cat((positive['amplitude_scaled'][:, :1], amplitude_density,
                               positive['amplitude_scaled'][:, 2:]), 1)
        prediction = baseline_roi + self.scales * delta
        prediction = torch.where(self.gel, baseline_roi, prediction)
        return {**positive, 'prediction': prediction, 'delta_scaled': delta,
                'shape': shape, 'log_gain': log_gain, 'amplitude_scaled': amplitude}


def secant_loss(result, baseline, target):
    scale = target.new_tensor([80., 100., .2])[None, :, None, None]
    exact = ((target - baseline) / scale)[:, :, 5:]
    amplitude = exact.square().mean((-2, -1)).sqrt()
    floor = target.new_tensor([5 / 80, 5 / 100, .02 / .2])[None]
    relative = ((result['delta_scaled'][:, :, 5:] - exact).square().mean((-2, -1)) /
                amplitude.square().clamp_min(floor.square()))
    active = amplitude > 1e-5
    nonlinear_active = active.clone()
    nonlinear_active[:, 1] = False
    exact_shape = exact / amplitude.clamp_min(1e-12)[:, :, None, None]
    shape_error = (result['shape'][:, :, 5:] - exact_shape).square().mean((-2, -1))
    shape_loss = (shape_error * nonlinear_active).sum() / nonlinear_active.sum().clamp_min(1)
    wanted_gain = (amplitude.clamp_min(1e-12) / result['residual_RMS'].clamp_min(1e-12)).log()
    gain_error = (result['log_gain'] - wanted_gain).square()
    gain_loss = (gain_error * nonlinear_active).sum() / nonlinear_active.sum().clamp_min(1)
    loss = relative.mean() + .5 * shape_loss + .1 * gain_loss
    return loss, {
        'relative_c_rho_alpha': relative.detach().mean(0).cpu().tolist(),
        'shape_loss': float(shape_loss.detach()), 'log_gain_loss': float(gain_loss.detach()),
        'density_has_amplitude_shape_auxiliary_loss': False,
        'target_amplitude_scaled': amplitude.detach().cpu().tolist(),
        'predicted_amplitude_scaled': result['amplitude_scaled'].detach().cpu().tolist(),
    }
