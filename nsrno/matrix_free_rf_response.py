"""Exact linear RF processing using FFT convolutions instead of a dense response matrix.

The SOS coefficients, odd endpoint extension, steady initial states,
polyphase FIR, and retained DFT mask match the existing SciPy definition.
This module has no patient/material reader and never runs a wave solver.
"""
import numpy as np
from scipy import signal
import torch
from torch import nn


def power_two(length):
    return 1 << (int(length) - 1).bit_length()


class MatrixFreeRFResponse(nn.Module):
    """Map [native time, ...] to [RF time, ...], with automatic transpose."""
    def __init__(self, native_samples, native_dt_s, nominal_band_hz,
                 retained_frequency_mask, down=10, device='cpu'):
        super().__init__()
        self.native_samples = int(native_samples)
        self.down = int(down)
        self.native_dt_s = float(native_dt_s)
        self.nominal_band_hz = tuple(float(v) for v in nominal_band_hz)
        if self.native_samples < 2 or self.down != 10:
            raise ValueError('Current acquisition requires native samples and downsampling by ten')
        self.rf_samples = (self.native_samples + self.down - 1) // self.down
        sos = signal.butter(4, self.nominal_band_hz, fs=1 / self.native_dt_s,
                            btype='bandpass', output='sos')
        ntaps = 2 * len(sos) + 1 - min(int((sos[:, 2] == 0).sum()),
                                     int((sos[:, 5] == 0).sum()))
        self.pad = 3 * ntaps
        if self.native_samples <= self.pad:
            raise ValueError('Record shorter than the required odd extension')
        self.extended_samples = self.native_samples + 2 * self.pad
        impulse = np.zeros(self.extended_samples, np.float64); impulse[0] = 1.
        response = signal.sosfilt(sos, impulse)
        self.filter_fft_samples = power_two(2 * self.extended_samples - 1)
        self.dc_gain = float(np.prod(np.sum(sos[:, :3], axis=1) /
                                     np.sum(sos[:, 3:], axis=1)))
        fir_half = 10 * self.down
        fir = signal.firwin(2 * fir_half + 1, 1 / self.down, window=('kaiser', 5.))
        self.resample_fft_samples = power_two(self.native_samples + len(fir) - 1)
        mask = np.asarray(retained_frequency_mask, dtype=bool)
        if mask.shape != (self.rf_samples // 2 + 1,):
            raise ValueError('Supply the explicit retained DFT mask for this record')
        self.register_buffer('filter_spectrum', torch.fft.rfft(
            torch.from_numpy(response).to(device), n=self.filter_fft_samples))
        self.register_buffer('resample_spectrum', torch.fft.rfft(
            torch.from_numpy(fir).to(device), n=self.resample_fft_samples))
        self.register_buffer('retained_mask', torch.from_numpy(mask.copy()).to(device))
        self.register_buffer('resample_indices',
                             (fir_half + self.down * torch.arange(self.rf_samples, device=device)))

    @staticmethod
    def shaped(spectrum, dimensions):
        return spectrum.reshape((-1,) + (1,) * (dimensions - 1))

    def causal_with_steady_initial_state(self, value):
        # A constant prehistory x[0] is equivalent to zero-state filtering
        # x-x[0], followed by adding the DC gain times x[0].
        shifted = value - value[:1]
        filtered = torch.fft.irfft(
            torch.fft.rfft(shifted, n=self.filter_fft_samples, dim=0) *
            self.shaped(self.filter_spectrum, value.ndim),
            n=self.filter_fft_samples, dim=0)[:self.extended_samples]
        return filtered + self.dc_gain * value[:1]

    def forward(self, raw):
        if raw.shape[0] != self.native_samples:
            raise ValueError('Native RF record length changed')
        if raw.device != self.filter_spectrum.device or raw.dtype not in [torch.float32, torch.float64]:
            raise ValueError('Matching device and real floating RF are required')
        output_dtype = raw.dtype
        value = raw.double()
        extended = torch.cat((2 * value[:1] - value[1:self.pad + 1].flip(0),
                              value,
                              2 * value[-1:] - value[-self.pad - 1:-1].flip(0)), dim=0)
        first = self.causal_with_steady_initial_state(extended)
        second = self.causal_with_steady_initial_state(first.flip(0)).flip(0)
        filtered = second[self.pad:-self.pad]
        convolved = torch.fft.irfft(
            torch.fft.rfft(filtered, n=self.resample_fft_samples, dim=0) *
            self.shaped(self.resample_spectrum, raw.ndim),
            n=self.resample_fft_samples, dim=0)
        sampled = convolved[self.resample_indices]
        spectrum = torch.fft.rfft(sampled, dim=0)
        projected = torch.fft.irfft(
            spectrum * self.shaped(self.retained_mask, raw.ndim),
            n=self.rf_samples, dim=0)
        return projected.to(output_dtype)

    def storage_info(self):
        return {
            'native_samples': self.native_samples, 'RF_samples': self.rf_samples,
            'filter_fft_samples': self.filter_fft_samples,
            'resample_fft_samples': self.resample_fft_samples,
            'coefficient_buffer_bytes': sum(b.numel() * b.element_size() for b in self.buffers()),
            'old_dense_float32_matrix_bytes': self.native_samples * self.rf_samples * 4,
            'arithmetic': 'float64 FFT; result cast to input dtype',
            'buffers_are_not_total_training_or_GPU_peak_memory': True,
            'patient_or_material_data_read': False,
        }
