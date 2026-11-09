"""Candidate data-space phase-increment loss; no acoustic model or material access.

For a pure delay, neighbouring frequency phase cancels the carrier and depends
on delta_f * delay. General multi-echo inversion is not guaranteed. An RF
amplitude fidelity term remains necessary for independent rho/alpha imaging.
"""
import numpy as np
import torch


class FrequencyPhaseIncrement:
    def __init__(self, observed, dft_mask, band_hz, local=True, samples=3601,
                 dt=2.5e-8, lag_scale_s=1e-6, sigma_s=6e-6,
                 centre_step_s=12e-6, phase_floor_fraction=.01,
                 weak_window_energy_floor=.001):
        if observed.ndim!=3 or not observed.is_complex():
            raise ValueError('Complex source x receiver x stored frequency observations required')
        mask = np.asarray(dft_mask, bool)
        if mask.shape!=(samples//2+1,) or mask.sum()!=observed.shape[-1]:
            raise ValueError('Exact supplied positive-DFT mask required')
        frequency = np.fft.rfftfreq(samples, dt)
        bins = np.flatnonzero(mask)
        selected = bins[(frequency[bins]>=band_hz[0])&(frequency[bins]<=band_hz[1])]
        if len(selected)<2 or not .001<=phase_floor_fraction<=.1:
            raise ValueError('Dense valid frequencies and a fixed positive phase floor required')
        if not 0<weak_window_energy_floor<1 or not lag_scale_s>0:
            raise ValueError('Positive fixed data normalization parameters required')
        self.samples, self.dt = samples, dt
        self.sources, self.receivers = observed.shape[:2]
        self.device = observed.device
        self.input_bins = torch.tensor(bins, device=self.device, dtype=torch.long)
        self.phase_bins = torch.tensor(selected, device=self.device, dtype=torch.long)
        self.input_band = torch.tensor((frequency[bins]>=band_hz[0])&
            (frequency[bins]<=band_hz[1]), device=self.device)
        self.stored_frequencies = observed.shape[-1]
        self.dtype = torch.complex128
        t = torch.arange(samples, device=self.device, dtype=torch.float64)*dt
        centres = np.arange(0, float(t[-1].cpu())+1e-12, centre_step_s) if local else np.array([0.])
        if local:
            centre = torch.tensor(centres, device=self.device)[:, None]
            windows = torch.exp(-.5*((t[None]-centre)/sigma_s)**2)
            windows /= windows.square().sum(0, keepdim=True).sqrt().clamp_min(1e-30)
        else:
            windows = torch.ones((1, samples), device=self.device, dtype=torch.float64)
        self.windows = windows
        delta_f = np.diff(frequency[selected])
        self.contiguous = torch.tensor(np.diff(selected)==1, device=self.device)
        self.phase_scale = torch.tensor(1-np.cos(2*np.pi*delta_f*lag_scale_s),
                                       device=self.device).clamp_min(1e-15)
        with torch.no_grad():
            data = self._windowed(observed.to(self.dtype))
            amplitude = data.abs()
            local_rms = amplitude.square().mean(-1, keepdim=True).sqrt()
            global_floor = max(float(amplitude.square().mean().sqrt())*1e-12, 1e-30)
            self.prediction_floor = (phase_floor_fraction*local_rms).clamp_min(global_floor)
            self.unit_data = data/amplitude.clamp_min(self.prediction_floor)
            valid = ((amplitude[..., 1:]>3*self.prediction_floor)&
                     (amplitude[..., :-1]>3*self.prediction_floor)&self.contiguous)
            self.pair_weight = amplitude[..., 1:]*amplitude[..., :-1]*valid
            energy = self.pair_weight.sum((0, 2, 3))
            self.window_denominator = energy.clamp_min(
                max(float(energy.max())*weak_window_energy_floor, 1e-30))
        self.config = {'data_loss':'adjacent-frequency unit-phase increment chord',
            'local_time_windows':bool(local), 'window_centres_us':(centres*1e6).tolist(),
            'Gaussian_sigma_us':sigma_s*1e6 if local else None,
            'sum_squared_windows_equals_one':True,
            'phase_floor_fraction_of_data_RMS':phase_floor_fraction,
            'weak_window_energy_floor_fraction':weak_window_energy_floor,
            'reference_delay_us':lag_scale_s*1e6,
            'data_frequency_pairs':int(np.sum(np.diff(selected)==1)),
            'window_input_uses_only_declared_phase_band':True,
            'source_count':self.sources, 'receivers':self.receivers,
            'physical_material_arrays_read':False,
            'RF_amplitude_loss_still_required':True,
            'nonlinear_source_encoding_unbiasedness_not_claimed':True,
            'single_delay_formula':'1-cos(2*pi*delta_f*tau); no general multi-arrival guarantee'}

    def _windowed(self, coefficients):
        spectrum = torch.zeros((*coefficients.shape[:-1], self.samples//2+1),
                               device=coefficients.device, dtype=self.dtype)
        spectrum[..., self.input_bins] = coefficients*self.input_band
        signal = torch.fft.irfft(spectrum, n=self.samples, dim=-1, norm='ortho')
        windowed = signal.unsqueeze(-3)*self.windows[:, None, :]
        return torch.fft.rfft(windowed, dim=-1, norm='ortho').index_select(-1, self.phase_bins)

    def loss(self, prediction, source):
        if prediction.shape!=(self.receivers, self.stored_frequencies) or not 0<=source<self.sources:
            raise ValueError('One actual acquired source at the exact stored frequencies')
        if prediction.device!=self.device:
            raise ValueError('Known data and actual prediction must use the same device')
        value = self._windowed(prediction.to(self.dtype))
        unit = value/value.abs().clamp_min(self.prediction_floor[source])
        transfer = unit*self.unit_data[source].conj()
        product = transfer[..., 1:]*transfer[..., :-1].conj()
        # Unlike |z[k+1]-z[k]|², the value is positive when both model phases
        # vanish. Stabilization still needs the retained physical RF term.
        discrepancy = (1-product.real).clamp_min(0)/self.phase_scale
        per_window = (self.pair_weight[source]*discrepancy).sum((1, 2))
        return (per_window/self.window_denominator).mean()
