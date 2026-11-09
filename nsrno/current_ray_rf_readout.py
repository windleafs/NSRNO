"""Linear, phase-preserving RF readout on current-material straight-ray times.

This is an approximate feature operator, not the native wave solver and not a
proof of identifiability. Its geometry is computed only from the supplied
current material, with no development label access.
"""
import hashlib
import math
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .ray_propagation import RayPropagation


def upsample_baseband(signal, samples, dt, factor=4):
    # Input is demodulated at exactly 3 MHz; its finite DFT carrier is rounded.
    # Undo the small rounding rotation before periodic band-limited resampling.
    period = samples * dt
    centre = round(3e6 * period)
    carrier = centre / period
    t = torch.arange(samples, device=signal.device, dtype=torch.float32) * dt
    rotation = torch.polar(torch.ones_like(t), -2 * math.pi * (carrier - 3e6) * t)
    value = signal[..., :samples] * rotation
    spectrum = torch.fft.fftshift(torch.fft.fft(value, dim=-1, norm='forward'), dim=-1)
    length = samples * factor
    left = length // 2 - samples // 2
    spectrum = F.pad(spectrum, (left, length - samples - left))
    return torch.fft.ifft(torch.fft.ifftshift(spectrum, dim=-1), dim=-1, norm='forward'), carrier, period


class CurrentRayRFReadout(nn.Module):
    def __init__(self, query_chunk=256, host_cache_entries=8, output_shape=(216, 192)):
        super().__init__()
        self.query_chunk = query_chunk
        self.output_shape = tuple(output_shape)
        if len(self.output_shape) != 2 or min(self.output_shape) < 2:
            raise ValueError('A two-dimensional output grid is required')
        self.host_cache_entries = host_cache_entries
        self.host_times = OrderedDict()
        angles = np.linspace(-8, 8, 11)
        tref = 4 / 3e6 + .0191 * np.abs(np.sin(angles * math.pi / 180)) / 1540
        spec = {'dx_m': .0002, 'medium_shape': [2, 241, 281], 'x0_m': -.028,
                'z0_m': -.003, 'receivers': 192, 'pitch_m': .0002,
                'angles_deg': angles.tolist(), 'source_tref_s': tref.tolist(),
                'ray_fan_count': 513, 'ray_receiver_chunk': 8}
        self.rays = RayPropagation(spec)
        z, x = torch.meshgrid(torch.linspace(0, .043, self.output_shape[0]),
                              torch.linspace(-.0191, .0191, self.output_shape[1]), indexing='ij')
        self.register_buffer('x', x.flatten())
        self.register_buffer('z', z.flatten())
        self.register_buffer('ray_to_ROI_grid', torch.stack((x / .028, 2 * z / .045 - 1), -1)[None])
        self.register_buffer('theta', torch.tensor(angles * math.pi / 180, dtype=torch.float32))
        self.register_buffer('rx', torch.linspace(-.0191, .0191, 192))
        self.register_buffer('lags', torch.tensor(
            [-3.2, -1.6, -.8, -.4, -.2, 0., .2, .4, .8, 1.6, 3.2]) * 1e-6)
        self.feature_channels_per_RF_type = 11 * len(self.lags) * 3 * 2
        self.config = {
            'geometry': 'current full-domain slowness integrated on straight rays',
            'geometry_resolution_mm': .2,
            'native_wave_solver_equivalence_claimed': False,
            'ray_bending_and_relaxation_dispersion_not_exactly_modelled': True,
            'angles_kept_separate': 11, 'receivers': 192,
            'RF_types': ['baseline/global_scale', 'unit_RMS_signed_residual'],
            'complex_carrier_restoration_hz': 3e6,
            'bandlimited_time_upsampling': 4,
            'lags_us': [float(v * 1e6) for v in self.lags],
            'receiver_moments': ['monopole', 'straight-ray dipole', 'signed lateral direction'],
            'normalization': 'common known geometric l2 norm; no RF-dependent normalization',
            'features_linear_in_both_RF_inputs_at_fixed_current_material': True,
            'query_chunk': query_chunk, 'host_geometry_cache_entries': host_cache_entries,
            'output_shape': list(self.output_shape),
            'disk_feature_cache': False,
        }

    def _times(self, context):
        if context.shape != (1, 3, 241, 281):
            raise ValueError('Current readout uses one full material state per update')
        key = hashlib.sha256(context[:, :1].detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
        if key in self.host_times:
            self.host_times.move_to_end(key)
            rx, tx = self.host_times[key]
            return rx.to(context.device), tx.to(context.device)
        material = context[:, :2].float()
        receive = []
        grid = self.ray_to_ROI_grid
        for start in range(0, 192, 8):
            indices = torch.arange(start, min(192, start + 8), device=context.device)
            raw = self.rays.receive_time(material, indices, material=True)
            receive.append(F.grid_sample(raw, grid, align_corners=True, padding_mode='border').flatten(2))
        receive = torch.cat(receive, 1).transpose(1, 2).contiguous()  # [1,Q,R]
        indices = torch.arange(11, device=context.device)
        raw = self.rays.transmit_time(material, indices, material=True)
        transmit = F.grid_sample(raw, grid, align_corners=True, padding_mode='border').flatten(2)  # [1,A,Q]
        if self.host_cache_entries:
            self.host_times[key] = (receive.cpu(), transmit.cpu())
            while len(self.host_times) > self.host_cache_entries:
                self.host_times.popitem(last=False)
        return receive, transmit

    @torch.no_grad()
    def forward(self, signal, context):
        if signal.shape != (1, 11, 6, 192, 544):
            raise ValueError('Complete same-clock 11-shot conditional RF is required')
        valid_counts = signal[:, :, 4].sum(-1)
        samples = int(valid_counts[0, 0, 0].item())
        if samples not in (361, 541) or not torch.all(valid_counts == samples):
            raise ValueError('Each update must have one actual complete acquisition clock')
        dt = float(signal[0, 0, 5, 0, 1].item()) * 90e-6
        if not dt > 0:
            raise ValueError('Invalid physical sample time')
        # B,A,RF-type,R,T complex data. Padding is never treated as observation.
        raw = torch.stack((torch.complex(signal[:, :, 0], signal[:, :, 1]),
                           torch.complex(signal[:, :, 2], signal[:, :, 3])), 2)
        upsampled, carrier, period = upsample_baseband(raw, samples, dt)
        receive, transmit = self._times(context)
        count = len(self.x); lags = len(self.lags)
        output = signal.new_empty((1, 2, 11, count, lags, 3, 2))
        coverage = signal.new_empty((1, 11, count, lags))
        sine = self.theta.sin()[:, None, None]
        cosine = self.theta.cos()[:, None, None]
        for start in range(0, count, self.query_chunk):
            end = min(count, start + self.query_chunk); n = end - start
            time = (receive[:, None, start:end, :, None] +
                    transmit[:, :, start:end, None, None] +
                    self.lags[None, None, None, None, :])
            observable = (time >= 0) & (time <= period - dt)
            coordinate = time * (upsampled.shape[-1] / period)
            lower = coordinate.floor().long().clamp(0, upsampled.shape[-1] - 2)
            fraction = (coordinate - lower).clamp(0, 1)
            # Gather every receiver trace at all source-dependent query times.
            index = lower.permute(0, 1, 3, 2, 4).reshape(1, 11, 1, 192, n * lags).expand(-1, -1, 2, -1, -1)
            first = torch.gather(upsampled, -1, index)
            second = torch.gather(upsampled, -1, index + 1)
            frac = fraction.permute(0, 1, 3, 2, 4).reshape(1, 11, 1, 192, n * lags)
            sampled = first + (second - first) * frac
            sampled = sampled.reshape(1, 11, 2, 192, n, lags).permute(0, 1, 2, 4, 3, 5)
            rotation = torch.polar(torch.ones_like(time), 2 * math.pi * carrier * time)
            sampled = sampled * (rotation * observable)[:, :, None]
            lateral = self.rx[None] - self.x[start:end, None]
            z = self.z[start:end, None]
            distance = torch.hypot(lateral, z).clamp_min(.0001)
            side = lateral / distance
            dipole = (lateral[None] * sine - z[None] * cosine) / distance[None]
            footprint = (self.x[start:end][None] - self.z[start:end][None] *
                         self.theta.tan()[:, None]).abs() <= .0191
            main = (.02 / distance).sqrt()[None] * footprint[:, :, None]
            common_norm = main.square().sum(-1).sqrt().clamp_min(1e-12)
            weights = torch.stack((main, main * dipole, main * side[None]), -1) / common_norm[:, :, None, None]
            # Complex phase is preserved before receiver aggregation; no abs/log.
            projected = torch.einsum('batqrl,aqrm->batqlm', sampled, weights.to(sampled.dtype))
            output[:, :, :, start:end] = torch.view_as_real(projected).permute(0, 2, 1, 3, 4, 5, 6)
            coverage[:, :, start:end] = (
                observable.float() * main.square()[None, :, :, :, None]).sum(-2) / common_norm.square()[None, :, :, None]
        features = output.permute(0, 1, 2, 4, 5, 6, 3).reshape(1, 2, -1, *self.output_shape)
        coverage = coverage.permute(0, 1, 3, 2).reshape(1, 11 * lags, *self.output_shape)
        return features[:, 0], features[:, 1], coverage
