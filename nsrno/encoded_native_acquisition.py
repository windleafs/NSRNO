"""Not yet executed: within-family Rademacher sources for actual native inversion.

This reduces propagation count, not measurement requirements or material fields.
The executable float32 rounding convention is not exactly source-linear; use
unchanged full-source native RF checks to accept any resulting material.
"""
import numpy as np
import torch
from .native_time import native_wave


class EncodedNativeAcquisition:
    """Wrap an existing authorized eleven-shot acquisition without new data.

    Construction performs receiving operations on known-water references and
    builds an exact-source CPU cache. It launches no acoustic propagation.
    Instantiate only after the previous GPU experiment has exited.
    """
    def __init__(self, acquisition, max_host_waveform_bytes=2*2**30):
        self.acquisition = acquisition
        if acquisition.observed.ndim != 3 or acquisition.observed.shape[:2] != (11, 192):
            raise ValueError('All eleven shots and the fixed 192 receiver array are required')
        if acquisition.config.nt != 36001 or acquisition.meta['dt'] != 2.5e-9:
            raise ValueError('Retain original native 90 us clock')
        self.frequencies = acquisition.frequencies
        self.device = acquisition.observed.device
        self._waveforms = []
        self.source_bytes = 0
        self.native_wave_shape = None
        self.signs = None
        self.observed = None
        self.reference = None
        with torch.no_grad():
            for shot in range(11):
                value = acquisition.waveform(shot).detach().cpu().numpy()
                if value.dtype != np.float32 or value.shape[0] != 36001:
                    raise ValueError('Original waveform bytes/clock are required')
                if self.native_wave_shape is None:
                    self.native_wave_shape = value.shape
                elif value.shape != self.native_wave_shape:
                    raise ValueError('All encoded sources must use the same injection geometry')
                nonzero = np.flatnonzero(np.any(value != 0, axis=1))
                if not len(nonzero):
                    raise ValueError('An acquired source cannot be identically zero')
                # Exactly zero float32 samples only. Copy releases the full-array base.
                prefix = value[:int(nonzero[-1])+1].copy()
                self.source_bytes += prefix.nbytes
                if self.source_bytes > max_host_waveform_bytes:
                    raise RuntimeError('Prespecified host waveform cache budget exceeded')
                self._waveforms.append(prefix)
                del value
            if hasattr(acquisition, 'raw_reference'):
                if acquisition.raw_reference.shape != (11, 36001, 192):
                    raise ValueError('Matched original native known-water reference required')
                references = []
                for shot in range(11):
                    known_raw = torch.from_numpy(
                        np.array(acquisition.raw_reference[shot], copy=True)).to(self.device)
                    references.append(acquisition.coefficients(known_raw).detach())
                    del known_raw
                self._reference_coefficients = torch.stack(references)
                self.reference_convention = 'double receiving coefficients minus encoded known-water coefficients'
                self.exact_original_float32_subtraction_parity = False
            else:
                if acquisition.reference.shape != acquisition.observed.shape:
                    raise ValueError('Original low-frequency reference coefficients required')
                self._reference_coefficients = acquisition.reference
                self.reference_convention = 'original low-frequency known-water coefficient subtraction'
                self.exact_original_float32_subtraction_parity = True
        self.config = {
            'encoding': 'independent Rademacher +-1 within each eleven-shot family',
            'source_clock_shifted': False,
            'waveform_scaling': 'unnormalized +-1; original family objective has no extra factor11',
            'source_linear_identity_is_exact_arithmetic_only': True,
            'full_original_44_source_RF_acceptance_required': True,
            'raw_source_count': 11, 'encoded_propagations_per_family': 1,
            'native_wave_shape': list(self.native_wave_shape),
            'CPU_source_cache_bytes': self.source_bytes,
            'only_exact_float32_zero_source_tails_omitted_from_cache': True,
            'reference_convention': self.reference_convention,
            'patient_truth_read': False,
            'new_acquisition_data': False,
            'new_identifiability_or_material_derivative_probe': False,
        }

    def set_encoding(self, signs):
        signs = np.asarray(signs)
        if signs.shape != (11,) or not np.all((signs == -1) | (signs == 1)):
            raise ValueError('Use all eleven independent +-1 symbols without selection')
        self.signs = signs.astype(np.int8).copy()
        coefficients = self.acquisition.observed.new_tensor(self.signs)
        self.observed = torch.tensordot(coefficients, self.acquisition.observed, dims=([0], [0]))
        self.reference = torch.tensordot(coefficients, self._reference_coefficients, dims=([0], [0]))

    def draw(self, generator):
        self.set_encoding(generator.choice(np.array([-1, 1], np.int8), size=11))
        return self.signs.copy()

    def waveform(self):
        if self.signs is None:
            raise RuntimeError('Set and record the encoding before actual inversion')
        value = np.zeros(self.native_wave_shape, np.float32)
        for sign, prefix in zip(self.signs, self._waveforms):
            if sign == 1:
                value[:len(prefix)] += prefix
            else:
                value[:len(prefix)] -= prefix
        return torch.from_numpy(value).to(self.device)

    def prediction(self, material):
        if self.signs is None:
            raise RuntimeError('Actual source and measured RF must share the same encoding')
        acquisition = self.acquisition
        wave = self.waveform()
        raw = native_wave(material, wave, acquisition.si, acquisition.sw,
                          acquisition.ri, acquisition.rw, acquisition.config)
        return acquisition.coefficients(raw) - self.reference

    def objective(self, prediction, frequency_weights, coefficient):
        """Use unchanged weights computed from ALL original family observations."""
        if self.observed is None or prediction.shape != self.observed.shape:
            raise ValueError('Actual encoded data layout mismatch')
        if frequency_weights.shape != (prediction.shape[-1],):
            raise ValueError('Original full-band energy normalization is required')
        residual = prediction - self.observed
        return coefficient * (residual.abs().square() * frequency_weights[None]).sum()
