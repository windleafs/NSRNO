"""Invertible signed RF/complex baseband representation on the V5 clock."""
import math

import numpy as np
from scipy import fft

RF_SAMPLES = 2401
RF_FS = 40e6
IQ_SAMPLES = 640
CENTRE_BIN = 360
FIRST_BIN = CENTRE_BIN - IQ_SAMPLES // 2
LAST_BIN = FIRST_BIN + IQ_SAMPLES
WINDOW_EDGES_US = (0., 8., 18., 36., 61.)


def time_windows(length, duration_us):
    time = np.arange(length) * duration_us / length
    edges = np.searchsorted(time, WINDOW_EDGES_US)
    return [slice(start, end) for start, end in zip(edges[:-1], edges[1:])]


def encode(rf):
    if rf.shape[-1] != RF_SAMPLES:
        raise ValueError(f"Expected {RF_SAMPLES} RF samples")
    spectrum = fft.rfft(rf, axis=-1, norm="forward")
    selected = spectrum[..., FIRST_BIN:LAST_BIN]
    iq = math.sqrt(2.) * fft.ifft(fft.ifftshift(selected, axes=-1), axis=-1, norm="forward")
    discarded_energy = 2. * (np.sum(np.abs(spectrum[..., :FIRST_BIN])**2, dtype=np.float64)
                            + np.sum(np.abs(spectrum[..., LAST_BIN:])**2, dtype=np.float64))
    discarded_energy -= np.sum(np.abs(spectrum[..., 0])**2, dtype=np.float64)
    total_energy = np.sum(np.square(rf, dtype=np.float64)) / RF_SAMPLES
    error = math.sqrt(max(0., discarded_energy) / total_energy)
    return np.stack((iq.real, iq.imag), axis=-3).astype(np.float32, copy=False), error


def decode(iq):
    import torch
    import torch.nn.functional as F

    iq = iq.float()
    if iq.shape[-3] != 2 or iq.shape[-1] != IQ_SAMPLES:
        raise ValueError("IQ axes must end in [2, receiver, 640]")
    signal = torch.complex(iq[..., 0, :, :], iq[..., 1, :, :])
    spectrum = torch.fft.fftshift(torch.fft.fft(signal, dim=-1, norm="forward"), dim=-1)
    spectrum = F.pad(spectrum / math.sqrt(2.), (FIRST_BIN, RF_SAMPLES // 2 + 1 - LAST_BIN))
    return torch.fft.irfft(spectrum, n=RF_SAMPLES, dim=-1, norm="forward")
