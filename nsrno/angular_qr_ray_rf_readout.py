"""Add observed-aperture orthogonal angular modes to current-state RF readout."""
import math
import torch
from torch.nn import functional as F
from .current_ray_rf_readout import CurrentRayRFReadout, upsample_baseband
from .angular_receiver_basis import angular_receiver_qr


def upsample_periodic(signal, samples, factor=4):
    spectrum = torch.fft.fftshift(torch.fft.fft(signal[..., :samples], dim=-1, norm='forward'), dim=-1)
    length = samples * factor
    left = length // 2 - samples // 2
    spectrum = F.pad(spectrum, (left, length - samples - left))
    return torch.fft.ifft(torch.fft.ifftshift(spectrum, dim=-1), dim=-1, norm='forward')


class AngularQRRayRFReadout(CurrentRayRFReadout):
    def __init__(self, query_chunk=256, host_cache_entries=8, output_shape=(216, 192)):
        super().__init__(query_chunk, host_cache_entries, output_shape)
        self.feature_channels_per_RF_type *= 4
        self.auxiliary_channels = 3 * 11 * len(self.lags)
        self.config.update(
            receiver_moments=['original monopole', 'original dipole', 'original lateral',
                              'QR monopole', 'QR independent dipole', 'QR independent lateral'],
            angular_QR_uses_actual_observed_receiver_mask_per_lag=True,
            angular_QR_geometry_dtype='float64; returned weights float32',
            angular_QR_rank_deficient_modes='zero',
            angular_QR_noise_vector_norm_amplification='at most 1 for the QR block',
            conditioning_fields='coverage plus two relative residual column norms',
            original_three_angular_moments_retained=True,
            additional_state_view='receiver-time baseline envelope magnitude',
            additional_residual_view='receiver-time residual referenced to baseline phase, before aggregation',
            relative_echo_views_carrier_restoration=False,
            residual_is_not_divided_by_baseline_amplitude=True)

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
        upsampled_main, carrier, period = upsample_baseband(raw, samples, dt)
        magnitude = raw[:, :, 0].abs()
        phase_reference = raw[:, :, 0].conj() / torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
        relative_views = torch.stack((torch.complex(magnitude, torch.zeros_like(magnitude)),
                                      raw[:, :, 1] * phase_reference), 2)
        upsampled_relative = upsample_periodic(relative_views, samples)
        upsampled = torch.cat((upsampled_main, upsampled_relative), 2)
        receive, transmit = self._times(context)
        count = len(self.x); lags = len(self.lags)
        output = signal.new_empty((1, 4, 11, count, lags, 6, 2))
        coverage = signal.new_empty((1, 11, count, lags))
        relative_norms = signal.new_empty((1, 11, count, lags, 2))
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
            index = lower.permute(0, 1, 3, 2, 4).reshape(1, 11, 1, 192, n * lags).expand(-1, -1, 4, -1, -1)
            first = torch.gather(upsampled, -1, index)
            second = torch.gather(upsampled, -1, index + 1)
            frac = fraction.permute(0, 1, 3, 2, 4).reshape(1, 11, 1, 192, n * lags)
            sampled = first + (second - first) * frac
            sampled = sampled.reshape(1, 11, 4, 192, n, lags).permute(0, 1, 2, 4, 3, 5)
            rotation = torch.polar(torch.ones_like(time), 2 * math.pi * carrier * time)
            carrier_views = torch.stack((rotation, rotation, torch.ones_like(rotation), torch.ones_like(rotation)), 2)
            sampled = sampled * carrier_views * observable[:, :, None]
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
            original_projected = torch.einsum('batqrl,aqrm->batqlm', sampled, weights.to(sampled.dtype))
            qr, spreads = angular_receiver_qr(main, dipole, side, observable)
            orthogonal_projected = torch.einsum('batqrl,aqrlm->batqlm', sampled, qr.to(sampled.dtype))
            projected = torch.cat((original_projected, orthogonal_projected), -1)
            relative_norms[:, :, start:end] = spreads[None]
            output[:, :, :, start:end] = torch.view_as_real(projected).permute(0, 2, 1, 3, 4, 5, 6)
            coverage[:, :, start:end] = (
                observable.float() * main.square()[None, :, :, :, None]).sum(-2) / common_norm.square()[None, :, :, None]
        features = output.permute(0, 1, 2, 4, 5, 6, 3).reshape(1, 4, -1, *self.output_shape)
        coverage = coverage.permute(0, 1, 3, 2).reshape(1, 11 * lags, *self.output_shape)
        relative_norms = relative_norms.permute(0, 1, 3, 4, 2).reshape(1, 11 * lags * 2, *self.output_shape)
        baseline = torch.cat((features[:, 0], features[:, 2]), 1)
        difference = torch.cat((features[:, 1], features[:, 3]), 1)
        return baseline, difference, torch.cat((coverage, relative_norms), 1)
