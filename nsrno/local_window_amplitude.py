"""Windowed spectral magnitude fidelity for a kinematic stage, not final signed RF."""
import torch


class LocalWindowAmplitude:
    def __init__(self, phase_backend, observed, weak_window_energy_floor=.001):
        if observed.shape!=(phase_backend.sources, phase_backend.receivers,
                           phase_backend.stored_frequencies):
            raise ValueError('Exact same observations as the local phase backend')
        if not 0<weak_window_energy_floor<1:
            raise ValueError('Fixed positive observed-energy floor required')
        self.backend = phase_backend
        with torch.no_grad():
            self.amplitude = phase_backend._windowed(observed).abs()
            energy = self.amplitude.square().sum((0, 2, 3))
            self.denominator = energy.clamp_min(max(float(energy.max())*
                weak_window_energy_floor, 1e-30))
        self.config = {'loss':'local-window spectral magnitude squared error',
            'normalization':'all original sources and receivers, independently per time window',
            'observed_energy_floor':weak_window_energy_floor,
            'retains_physical_amplitude':True, 'source_gain_fitted':False,
            'invariant_to_overall_signal_sign':True,
            'final_signed_RF_stage_still_required':True}

    def loss(self, prediction, source):
        if prediction.shape!=(self.backend.receivers, self.backend.stored_frequencies):
            raise ValueError('One original source with exact stored frequencies')
        if not 0<=source<self.backend.sources:
            raise ValueError('Original source index required')
        value = self.backend._windowed(prediction).abs()
        squared = (value-self.amplitude[source]).square().sum((1, 2))
        return (squared/self.denominator).mean()
