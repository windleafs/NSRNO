"""Untrained alternative: direct density path plus zero-initialized gradient integration."""
import torch
from torch import nn
from torch.nn import functional as F
from .amplitude_aware_angular_inverse_uno import AmplitudeAwareAngularInverseUNO
from .anchored_gradient_integration import AnchoredGradientIntegration


class IntegratedDensityHead(nn.Module):
    def __init__(self, direct, gel_rows=5, height=216, width=192):
        super().__init__()
        if direct.bias is not None:
            raise ValueError('The residual-leading density head must remain bias-free')
        self.direct = direct
        self.gel_rows, self.height, self.width = gel_rows, height, width
        self.gradient_head = nn.Conv2d(direct.in_channels, 2, 1, bias=False)
        nn.init.zeros_(self.gradient_head.weight)
        self.integration = AnchoredGradientIntegration(height - gel_rows, width)

    def forward(self, features):
        if features.shape[-2:] != (self.height, self.width):
            raise ValueError('Complete density ROI features are required')
        gradient = self.gradient_head(features)[:, :, self.gel_rows:]
        integrated = self.integration(gradient[:, :1, :, :-1], gradient[:, 1:2])
        return self.direct(features) + F.pad(integrated, (0, 0, self.gel_rows, 0))


class GradientDensityAngularInverseUNO(AmplitudeAwareAngularInverseUNO):
    def __init__(self, query_chunk=256):
        super().__init__(query_chunk)
        self.density_head = IntegratedDensityHead(self.density_head)
        self.config.update(
            architecture='gradient_integrated_density_angular_QR_inverse_UNO_v1',
            density_gradient_integration=self.density_head.integration.config,
            direct_parent_density_path_retained=True,
            density_gradient_path_initially_zero=True,
            added_parameters_beyond_amplitude_parent=32,
            added_parameter_count=self.config['added_parameter_count'] + 32,
            density_leading_path_exact_linear_in_RF_residual_at_fixed_baseline=True,
            field_loss_and_finite_nonlinear_heads_not_changed_by_this_module=True,
            no_new_measurement_information_created=True)

    def load_parent_amplitude_state(self, state):
        """Lift a closed amplitude-aware model; keep the new integral path zero."""
        value = dict(state)
        value['density_head.direct.weight'] = value.pop('density_head.weight')
        current = self.state_dict()
        for key in current:
            if key.startswith('density_head.gradient_head.') or key.startswith('density_head.integration.'):
                value[key] = current[key]
        self.load_state_dict(value, strict=True)
