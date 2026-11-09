"""Positive f-k receiver covariance metric, solely for the inner numerical solve."""
import hashlib
import numpy as np
import torch
from .full_source_dual_cg import FourierReceiverMetric

_CACHE = {}


def receiver_metric(acquisition, spatial_weight, weights, coefficient):
    a, c = acquisition, acquisition.config
    ri = a.ri.detach().cpu().numpy()
    rw = a.rw.detach().cpu().numpy()
    rx = ((ri % c.nx)*rw).sum(1)*c.dx
    rz = ((ri // c.nx)*rw).sum(1)*c.dx
    pitch = (rx[-1]-rx[0])/(len(rx)-1)
    w = spatial_weight.detach().cpu().numpy()
    frequency = np.asarray(a.frequencies)
    weight_frequency = coefficient*weights.detach().cpu().numpy()
    key = hashlib.sha256(w.tobytes()+frequency.tobytes()+weight_frequency.tobytes()+
                         np.array([pitch, float(rz.mean()), c.dx, c.dt]).tobytes()).hexdigest()
    if key not in _CACHE:
        if pitch <= 0 or np.max(abs(rx-(rx[0]+np.arange(len(rx))*pitch))) > 1e-6:
            spectrum = np.ones((len(rx), len(frequency)), np.float64)
            kind = 'identity_for_nonuniform_receivers'
        else:
            # Known nominal medium is a numerical covariance approximation only.
            # No estimated material is filtered or replaced by these constants.
            kx = 2*np.pi*np.fft.fftfreq(len(rx), d=pitch)
            z = np.arange(c.nz)*c.dx-float(rz.mean())
            valid = z > 0
            depth_power = (w*w).mean(1)[valid]
            depth = z[valid]
            spectrum = np.zeros((len(rx), len(frequency)), np.float64)
            for lo in range(0, len(frequency), 16):
                f = frequency[lo:lo+16]
                attenuation = np.log(10)/20*.5*100*(f/1e6)**1.1
                k = 2*np.pi*f/1540.+1j*attenuation
                temporal = (2*np.sin(np.pi*f*c.dt)/c.dt)**2
                value = np.zeros((len(rx), len(f)), np.float64)
                for alias in [-1, 0, 1]:
                    kz = np.sqrt(k[None]**2-(kx[:, None]+alias*2*np.pi/pitch)**2)
                    integral = (np.exp(-2*kz.imag[:, :, None]*depth[None, None])*depth_power).sum(2)*c.dx
                    value += temporal[None]*integral/(4*np.maximum(abs(kz)**2, 1e-20))
                spectrum[:, lo:lo+len(f)] = value*weight_frequency[None, lo:lo+len(f)]
            keep = weight_frequency > 0
            median = float(np.median(spectrum[:, keep]))
            if not np.isfinite(median) or median <= 0:
                raise RuntimeError('Invalid known-geometry numerical covariance')
            spectrum = np.clip(spectrum/median, 1e-6, 1e6)
            kind = 'known_geometry_f_k_covariance_with_fixed_positive_eigenvalues'
        _CACHE[key] = spectrum, kind
    spectrum, kind = _CACHE[key]
    metric = FourierReceiverMetric(torch.from_numpy(spectrum.copy()).to(weights.device))
    return metric, {'kind': kind, 'receiver_pitch_m': float(pitch),
        'reference_speed_m_s': 1540., 'reference_tissue_alpha': .5,
        'spatial_aliases': [-1, 0, 1], 'eigenvalue_relative_bounds': [1e-6, 1e6],
        'changes_only_solver_metric': True, 'material_field_smoothed': False,
        'new_acoustic_propagations': 0, 'source_metric_SHA256': key}
