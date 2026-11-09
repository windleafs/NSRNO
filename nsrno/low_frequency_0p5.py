"""Fixed, compact 0.5 MHz acquisition for one existing V5 development material.

Only the source and receiving band change. The 3 MHz relaxation parameters
are retained. This module never imports torch or reads a patient array.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy import fft, signal

F0 = 0.5e6
NATIVE_NT = 36001
NATIVE_DT = 2.5e-9
DOWN = 10
RF_NT = 3601
RF_DT = NATIVE_DT * DOWN
BAND = (0.25e6, 0.75e6)
RETAIN = (0.10e6, 1.25e6)
AMPLITUDE = 1e5 * np.sqrt(F0 / 3e6)
SAMPLE = 'train_200'
CASE = 'Neg_35_Left'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def frequency_mask():
    frequency = fft.rfftfreq(RF_NT, RF_DT)
    return (frequency >= RETAIN[0]) & (frequency <= RETAIN[1])


def processed_coefficients(raw):
    """SciPy definition, double arithmetic; [time,receiver] -> [receiver,freq]."""
    raw = np.asarray(raw)
    if raw.shape != (NATIVE_NT, 192) or not np.isfinite(raw).all():
        raise ValueError('Require the complete 90 us, 192 receiver record')
    sos = signal.butter(4, BAND, fs=1 / NATIVE_DT,
                        btype='bandpass', output='sos')
    filtered = signal.sosfiltfilt(sos, raw.astype(np.float64), axis=0)
    sampled = signal.resample_poly(filtered, 1, DOWN, axis=0)
    if sampled.shape != (RF_NT, 192):
        raise ValueError('Processed RF clock changed')
    return np.ascontiguousarray(
        fft.rfft(sampled, axis=0, norm='ortho')[frequency_mask()].T)


def save_coefficients(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    value = np.asarray(value, dtype=np.complex128)
    if value.shape != (192, int(frequency_mask().sum())) or not np.isfinite(value).all():
        raise ValueError('Invalid retained channel RF coefficients')
    temporary = path.with_name(path.stem + '.partial.npy')
    with temporary.open('wb') as stream:
        np.save(stream, value)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)
    return digest(path)


def decode_coefficients(value):
    value = np.asarray(value)
    if value.shape[-2:] != (192, int(frequency_mask().sum())):
        raise ValueError('Unexpected RF coefficient layout')
    spectrum = np.zeros((*value.shape[:-1], RF_NT // 2 + 1), np.complex128)
    spectrum[..., frequency_mask()] = value
    return fft.irfft(spectrum, n=RF_NT, axis=-1, norm='ortho')


def acquisition_metadata():
    frequency = fft.rfftfreq(RF_NT, RF_DT)
    return {
        'source_f0_hz': F0, 'source_sigma_s': 1 / F0,
        'source_peak_pressure_Pa': float(AMPLITUDE),
        'amplitude_rule': '1e5*sqrt(0.5/3), fixed Gaussian source time-integrated energy',
        'native_nt': NATIVE_NT, 'native_dt_s': NATIVE_DT,
        'record_us': 90, 'RF_nt': RF_NT, 'RF_dt_s': RF_DT,
        'receivers': 192, 'nominal_band_hz': list(BAND),
        'retained_frequency_hz': frequency[frequency_mask()].tolist(),
        'retained_DFT_bins': np.flatnonzero(frequency_mask()).tolist(),
        'RF_format': 'complex128 orthonormal positive-frequency DFT, receiver/frequency',
        'time_RF_exactly_reconstructable_for_declared_frequency_mask': True,
        'nominal_frequency_count': int(((frequency >= BAND[0]) & (frequency <= BAND[1])).sum()),
        'retained_frequency_count': int(frequency_mask().sum()),
        'processing': '4th order Butterworth sosfiltfilt, resample_poly(1,10), explicit DFT mask',
        'three_independent_material_fields': True, 'FOV_depth_mm': [1, 43],
        'case': CASE, 'sample': SAMPLE,
        'constitutive_model': 'exact original 3 MHz two-pole relaxation fit, not refitted',
        'low_frequency_is_extrapolation_of_original_causal_model': True,
    }
