"""Same current-ray inverse UNO with additional preconditioned angular inputs."""
import torch
from torch import nn
from .ray_conditional_inverse_uno import RayConditionalInverseUNO
from .angular_qr_ray_rf_readout import AngularQRRayRFReadout
from .angular_receiver_basis import raw_receiver_feature_indices


class AngularQRInverseUNO(RayConditionalInverseUNO):
    def __init__(self, query_chunk=256):
        super().__init__(query_chunk)
        parent_parameters = sum(p.numel() for p in self.parameters())
        old_channels = self.readout.feature_channels_per_RF_type
        old_auxiliary = 11 * len(self.readout.lags)
        readout = AngularQRRayRFReadout(query_chunk=query_chunk)
        channels = readout.feature_channels_per_RF_type
        auxiliary = readout.auxiliary_channels
        raw_index = raw_receiver_feature_indices(11, len(readout.lags))
        signed_index = torch.cat((raw_index, channels + raw_index))
        old_state = self.state[0]
        new_state = nn.Conv2d(channels + auxiliary + 11, 16, 1)
        old_nonlinear = self.nonlinear_stem
        new_nonlinear = nn.Conv2d(2 * channels + 21, 24, 1)
        old_density = self.density_stem
        new_density = nn.Conv2d(2 * channels, 16, 1, bias=False)
        with torch.no_grad():
            new_state.weight.zero_()
            new_state.weight.index_copy_(1, raw_index, old_state.weight[:, :old_channels])
            new_state.weight[:, channels:channels + old_auxiliary].copy_(
                old_state.weight[:, old_channels:old_channels + old_auxiliary])
            new_state.weight[:, channels + auxiliary:].copy_(
                old_state.weight[:, old_channels + old_auxiliary:])
            new_state.bias.copy_(old_state.bias)
            new_nonlinear.weight.zero_()
            new_nonlinear.weight.index_copy_(1, signed_index, old_nonlinear.weight[:, :2 * old_channels])
            new_nonlinear.weight[:, 2 * channels:].copy_(old_nonlinear.weight[:, 2 * old_channels:])
            new_nonlinear.bias.copy_(old_nonlinear.bias)
            new_density.weight.zero_()
            new_density.weight.index_copy_(1, signed_index, old_density.weight)
        self.readout = readout
        self.state[0] = new_state
        self.nonlinear_stem = new_nonlinear
        self.density_stem = new_density
        self.config.update(
            architecture='current_ray_angular_QR_inverse_UNO_v1',
            readout=readout.config,
            initial_parent_function_preserved=True,
            additional_QR_input_weights_initialized_to_zero=True,
            added_parameter_count=sum(p.numel() for p in self.parameters()) - parent_parameters,
            new_RF_measurements_added=False,
            pre_aggregation_baseline_phase_echo_features=True,
            geometric_Gram_is_not_material_Jacobian=True,
            alpha_objective='equal local field error and normalized cumulative-depth error',
            alpha_total_weight_equal_to_other_material_fields=True,
            alpha_cumulative_objective_does_not_remove_local_field_loss=True)


def secant_loss(result, baseline, target):
    scale = target.new_tensor([80., 100., .2])[None, :, None, None]
    exact = ((target - baseline) / scale)[:, :, 5:]
    update = result['delta_scaled'][:, :, 5:]
    amplitude = exact.square().mean((-2, -1)).sqrt()
    floor = target.new_tensor([5 / 80, 5 / 100, .02 / .2])[None]
    relative = ((update - exact).square().mean((-2, -1)) /
                amplitude.square().clamp_min(floor.square()))
    height = exact.shape[-2]
    cumulative_target = exact[:, 2].cumsum(-2) / height
    cumulative_error = (update[:, 2] - exact[:, 2]).cumsum(-2) / height
    ramp = torch.arange(1, height + 1, device=target.device, dtype=target.dtype) / height
    cumulative_floor_squared = floor[0, 2].square() * ramp.square().mean()
    cumulative_relative = cumulative_error.square().mean((-2, -1)) / cumulative_target.square().mean((-2, -1)).clamp_min(cumulative_floor_squared)
    # Each material keeps one third of the total weight. Alpha's share is
    # divided between local and cumulative coefficients; no field is removed.
    alpha_objective = .5 * (relative[:, 2] + cumulative_relative)
    loss = (relative[:, 0] + relative[:, 1] + alpha_objective).mean() / 3
    return loss, {
        'relative_c_rho_alpha': relative.detach().mean(0).cpu().tolist(),
        'shape_loss': 0., 'log_gain_loss': 0.,
        'alpha_cumulative_relative_MSE': float(cumulative_relative.detach().mean()),
        'alpha_optimized_local_cumulative_mean': float(alpha_objective.detach().mean()),
        'local_alpha_loss_retained': True,
        'target_amplitude_scaled': amplitude.detach().cpu().tolist(),
        'predicted_amplitude_scaled': result['amplitude_scaled'].cpu().tolist(),
    }
