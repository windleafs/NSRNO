"""Conditional inverse UNO with explicit baseline-phase RF interactions.

At nonzero baseline b the extra complex feature is
    u * conj(b) / abs(b),  u = (target - baseline) / residual_RMS.
The raw complex baseline and residual remain available. Residual magnitude is
not amplified by a weak baseline. No material focusing or new wave solves occur.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .conditional_secant_uno import ConditionalSecantUNO, SecantSourceEncoder
from .raw_rf_inverse_uno import fourier_positions


def relative_phase_channels(signal):
    if signal.shape[-3] != 6:
        raise ValueError('Expected baseline(2), residual(2), valid, time')
    br, bi, ur, ui = signal.unbind(-3)[:4]
    norm = torch.hypot(br, bi)
    denominator = torch.where(norm > 0, norm, torch.ones_like(norm))
    pr, pi = br / denominator, bi / denominator
    same = ur * pr + ui * pi
    quadrature = ui * pr - ur * pi
    return torch.cat((signal[..., :4, :, :],
                      torch.stack((same, quadrature), dim=-3),
                      signal[..., 4:, :, :]), dim=-3)


class RelativePhaseSourceEncoder(SecantSourceEncoder):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(133, 24, 1)

    def forward(self, signal, geometry):
        b = signal.shape[0]
        packed = F.pixel_unshuffle(relative_phase_channels(signal), 4)
        h, w = packed.shape[-2:]
        rx = torch.linspace(-1, 1, 192, device=signal.device).reshape(1, 1, 192, 1).expand(b, 1, 192, 544)
        rx = F.avg_pool2d(rx, 4)
        geom = geometry[:, :, None, None].expand(-1, -1, h, w)
        encoded = self.uno(self.stem(torch.cat((packed, rx, geom), 1))).flatten(2).transpose(1, 2)
        valid = F.avg_pool2d(signal[:, 4:5], 4).flatten(2).transpose(1, 2)
        time = F.avg_pool2d(signal[:, 5:6], 4).flatten(2).transpose(1, 2) / valid.clamp_min(1 / 16)
        positions = torch.cat((rx.flatten(2).transpose(1, 2), time,
                               geometry[:, :3, None].transpose(1, 2).expand(-1, h * w, -1)), 2)
        query_positions = torch.cat((self.query_coordinates[None].expand(b, -1, -1),
                                     geometry[:, :3, None].transpose(1, 2).expand(-1, 512, -1)), 2)
        queries = self.queries + self.position(fourier_positions(query_positions))
        keys = self.key(encoded) + self.position(fourier_positions(positions))
        distance = query_positions[:, :, :2, None] - positions[:, :, :2].transpose(1, 2)[:, None]
        bias = -.5 * ((distance[:, :, 0] / .16) ** 2 + (distance[:, :, 1] / .055) ** 2)
        bias = bias.masked_fill(valid[..., 0][:, None, :] == 0, float('-inf'))[:, None]
        tokens = self.readout(queries, keys, self.value(encoded), bias)
        return tokens + self.refine(tokens), self.position(fourier_positions(query_positions))


class RelativePhaseSecantUNO(ConditionalSecantUNO):
    def __init__(self, query_chunk=1024):
        super().__init__(query_chunk)
        original = self.encoder
        encoder = RelativePhaseSourceEncoder()
        encoder.load_state_dict({k: v for k, v in original.state_dict().items()
                                 if not k.startswith('stem.')}, strict=False)
        # Keep the exact same initial parent function; learn only the extra
        # phase interaction from zero input weights, with all other weights
        # copied from the same seeded fresh parent.
        with torch.no_grad():
            encoder.stem.weight.zero_()
            encoder.stem.weight[:, :64].copy_(original.stem.weight[:, :64])
            encoder.stem.weight[:, 96:128].copy_(original.stem.weight[:, 64:96])
            encoder.stem.weight[:, 128:133].copy_(original.stem.weight[:, 96:101])
            encoder.stem.bias.copy_(original.stem.bias)
        self.encoder = encoder
        self.config.update(
            architecture='conditional_relative_phase_RF_UNO_v1',
            RF_inputs=['current_RF/global_scale', 'unit_RMS_complex_RF_difference',
                       'residual_rotated_by_conjugate_baseline_phase', 'valid', 'time'],
            encoder_signal_channels=8,
            extra_parameter_count=768,
            raw_RF_preserved=True,
            RF_residual_divided_by_baseline_amplitude=False,
            relative_phase_at_zero_baseline='zero; raw residual still available',
            initial_parent_function_preserved_by_zero_extra_weights=True,
            signed_odd_finite_inverse_constraint=False,
            per_record_RMS_includes_only_original_RF_difference=True)
