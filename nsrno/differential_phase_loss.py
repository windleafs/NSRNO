"""Self-consistent spectral phase differences, using the exact modeled RF.

No synthetic low-frequency wave equation is assumed. The feature of the exact
prediction is compared with the same feature of the observation. It is a
kinematic initializer, not a density/attenuation amplitude likelihood.
"""
import math
import torch


class DifferentialPhaseLoss:
    def __init__(self, acquisition, window_provider, *, lag_bins=16,
                 spectral_floor=1e-3, cache_cpu=False):
        if lag_bins < 1 or spectral_floor <= 0:
            raise ValueError('Positive frequency lag and power floor are required')
        self.observed = acquisition.observed
        self.provider = window_provider
        self.nfft = window_provider.nfft
        self.device = self.observed.device
        self.lag_bins = lag_bins
        self.spectral_floor = spectral_floor
        self.cache_cpu = cache_cpu
        if cache_cpu and self.device.type != 'cpu':
            raise ValueError('All-shot target caching is permitted only on CPU')
        low, high = (4e6, 7.5e6) if acquisition.source == 'hf' else (1.5e6, 4.5e6)
        df = 40e6/self.nfft
        self.indices = torch.arange(math.ceil(low/df), math.floor(high/df)-lag_bins+1,
                                    device=self.device)
        self.frequency_difference_hz = lag_bins*df
        self._targets = {}

    def _feature(self, ft, floor):
        a = ft[..., self.indices]
        b = ft[..., self.indices+self.lag_bins]
        return b*a.conj()/((a.abs().square()+floor)*(b.abs().square()+floor)).clamp_min(1e-30).sqrt()

    def target(self, shot):
        if shot in self._targets:
            return self._targets[shot]
        windows = self.provider.windows(shot)
        ft = torch.fft.rfft(self.observed[shot][None]*windows, n=self.nfft, dim=-1)
        power = ft.abs().square()
        floor = self.spectral_floor*power.amax(-1, keepdim=True)
        feature = self._feature(ft, floor)
        weight = (power[..., self.indices]*power[..., self.indices+self.lag_bins]).sqrt()
        weight = weight/weight.sum(-1, keepdim=True).clamp_min(1e-30)
        if not self.cache_cpu:
            self._targets.clear()
        self._targets[shot] = windows, feature, weight, floor
        return self._targets[shot]

    def terms(self, predicted, shot):
        windows, target, weight, floor = self.target(shot)
        ft = torch.fft.rfft(predicted[None]*windows, n=self.nfft, dim=-1)
        difference = self._feature(ft, floor)-target
        # Each complex feature has modulus <= 1; terms are bounded by one.
        trace_penalty = .25*(weight*difference.abs().square()).sum(-1)
        return (trace_penalty*self.provider.trace_weights[shot]).sum(-1).unbind(0)
