"""Fixed-observation reverse matching filters, followed by full-RF refinement.

This is an initialization objective. Finite bandwidth/stabilization give a
nonzero self-filter penalty; it is not a substitute for the final RF likelihood.
No material truth is read here.
"""
import math
import torch
from torch.nn import functional as F
from .rf_depth_weights import arrival_bins


class AdaptiveMatchingLoss:
    def __init__(self, acquisition, *, bins=None, sigma_us=1.5,
                 stabilization=.1, taper_samples=17, filter_energy_floor=1e-3,
                 cache_cpu=False):
        if sigma_us <= 0 or stabilization <= 0:
            raise ValueError('Positive lag width and stabilization are required')
        if taper_samples < 1 or taper_samples % 2 != 1:
            raise ValueError('An odd positive smoothing length is required')
        self.observed = acquisition.observed
        self.device = self.observed.device
        self.bins = (torch.from_numpy(arrival_bins(acquisition)).to(self.device)
                     if bins is None else bins)
        self.sigma_us = float(sigma_us)
        self.stabilization = float(stabilization)
        self.filter_energy_floor = float(filter_energy_floor)
        if self.filter_energy_floor < 0:
            raise ValueError('Nonnegative fixed-observation filter floor is required')
        if cache_cpu and self.device.type != 'cpu':
            raise ValueError('All-shot target caching is permitted only on CPU')
        self.cache_cpu = cache_cpu
        self._targets = {}
        self.taper_samples = taper_samples
        self.nt = self.observed.shape[-1]
        self.nfft = 1 << math.ceil(math.log2(2*self.nt))
        lag = torch.fft.fftfreq(self.nfft, device=self.device,
                               dtype=self.observed.dtype)*self.nfft/40.
        self.gaussian = torch.exp(-.5*(lag/self.sigma_us).square())
        self.kernel = torch.hann_window(taper_samples, periodic=False,
                                       device=self.device, dtype=self.observed.dtype)
        self.kernel = (self.kernel/self.kernel.sum()).reshape(1, 1, -1)
        with torch.no_grad():
            energy = []
            for shot in range(self.observed.shape[0]):
                target = self.observed[shot][None]*self.windows(shot)
                energy.append(target.double().square().sum(-1))
            energy = torch.stack(energy)
            self.trace_weights = (energy/energy.sum((0, 2), keepdim=True).clamp_min(1e-30)).to(self.observed.dtype)

    def windows(self, shot):
        masks = (self.bins[shot][None] == torch.arange(5, device=self.device)[:, None, None]).to(self.observed.dtype)
        if self.taper_samples > 1:
            radius = self.taper_samples//2
            packed = masks.reshape(-1, 1, self.nt)
            packed = F.pad(packed, (radius, radius), mode='replicate')
            masks = F.conv1d(packed, self.kernel).reshape_as(masks)
        return masks.clamp_min(0).sqrt()

    def target_filter(self, shot):
        if shot in self._targets:
            return self._targets[shot]
        windows = self.windows(shot)
        target = self.observed[shot][None]*windows
        observed_ft = torch.fft.rfft(target, n=self.nfft, dim=-1)
        observed_power = observed_ft.abs().square()
        floor = self.stabilization*observed_power.amax(-1, keepdim=True)
        denominator = (observed_power+floor).clamp_min(1e-30)
        transfer = observed_ft.conj()/denominator
        self_filter = torch.fft.irfft(observed_power/denominator, n=self.nfft, dim=-1)
        energy_floor = self.filter_energy_floor*self_filter.square().sum(-1)
        if not self.cache_cpu:
            self._targets.clear()
        self._targets[shot] = (windows, transfer, energy_floor)
        return self._targets[shot]

    def terms(self, predicted, shot):
        windows, transfer, fixed_floor = self.target_filter(shot)
        predicted_ft = torch.fft.rfft(predicted[None]*windows, n=self.nfft, dim=-1)
        filt_ft = predicted_ft*transfer
        filt = torch.fft.irfft(filt_ft, n=self.nfft, dim=-1)
        energy = filt.square()
        concentration = (energy*self.gaussian).sum(-1)/(energy.sum(-1)+fixed_floor).clamp_min(1e-20)
        # A zero prediction has concentration zero and maximum penalty one.
        penalties = (1-concentration)*self.trace_weights[shot]
        return penalties.sum(-1).unbind(0)
