"""Coarse independent corrections of a fixed full-native material estimate.

Only the control interpolation and its transpose are exact algebraic maps.
The coarse PDE remains an approximation and needs actual fine-grid RF scoring.
No patient truth reader is present.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve1d, maximum_filter, minimum_filter
import torch
from torch.nn import functional as F
from .native_gn_coordinates import Coordinates, SCALES


class FrequencyMultilevelCoordinates:
    def __init__(self, native_anchor, known_edge, factor=4, pml=24):
        if factor not in (2,4) or pml != 24:
            raise ValueError('This candidate uses twofold or fourfold spacing and 24 PML cells')
        self.factor = factor
        self.pml = pml
        self.native_anchor = np.asarray(native_anchor, np.float32)
        if self.native_anchor.shape != (3, 961, 1121):
            raise ValueError('Full native estimate required')
        self.edge_native = np.broadcast_to(known_edge, self.native_anchor.shape[1:]).copy()
        self.fine = Coordinates(self.native_anchor, self.edge_native)
        self.offset = pml - pml // factor
        self.edge = self.restrict(self.edge_native)
        tissue = self.native_anchor[2] - self.edge_native
        # Positive arithmetic averaging is a chosen coarse model, not an
        # exact homogenization of variable-density fine scattering.
        self.anchor = self.restrict(np.stack(
            (self.native_anchor[0], self.native_anchor[1], tissue)))
        self.anchor[2] += self.edge
        self.shape = self.anchor.shape
        self.gel_rows = pml + (80-pml)//factor
        z = np.arange(self.shape[1])[:, None]
        x = np.arange(self.shape[2])[None, :]
        self.mask = ((z >= self.gel_rows) & (z < self.shape[1]-pml) &
                     (x >= pml) & (x < self.shape[2]-pml))
        self.anchor[0, :self.gel_rows] = 1540.
        self.anchor[1, :self.gel_rows] = 1000.
        self.anchor[2, :self.gel_rows] = .002 + self.edge[:self.gel_rows]
        self.tissue = self.anchor[2] - self.edge
        if np.any(self.tissue <= 0):
            raise ValueError('Positive coarse tissue attenuation required')
        self._fine_mask = torch.from_numpy(self.fine.mask.copy()).to(torch.float64)
        self._coarse_mask = torch.from_numpy(self.mask.copy()).to(torch.float64)
        self.low, self.high = self._bounds()

    def restrict(self, value):
        # Full weighting centred at the same physical grid nodes.
        f = self.factor
        kernel = (f-np.abs(np.arange(-f+1,f,dtype=np.float64))) / f**2
        value = convolve1d(np.asarray(value, np.float64), kernel, axis=-1, mode='nearest')
        value = convolve1d(value, kernel, axis=-2, mode='nearest')
        p, f = self.pml, self.factor
        value = value[..., p:-p:f, p:-p:f]
        pads = [(0, 0)] * (value.ndim-2) + [(p, p), (p, p)]
        return np.pad(value, pads, mode='edge')

    def _prolong_tensor(self, value):
        value = value * self._coarse_mask
        o = self.offset
        cropped = value[:, o:-o, o:-o]
        result = F.interpolate(cropped[None], size=self.native_anchor.shape[1:],
                               mode='bilinear', align_corners=True)[0]
        return result * self._fine_mask

    def prolong(self, value):
        return self._prolong_tensor(torch.as_tensor(
            np.asarray(value, np.float64))).detach().numpy()

    def transpose(self, value):
        control = torch.zeros(self.shape, dtype=torch.float64, requires_grad=True)
        result = self._prolong_tensor(control)
        dual = torch.as_tensor(np.asarray(value, np.float64))
        derivative, = torch.autograd.grad((result * dual).sum(), control)
        return derivative.detach().numpy()

    def _bounds(self):
        # Bilinear interpolation has nonnegative weights. Bound each control
        # by every native point in its support, ensuring the prolonged log
        # material remains in the original physical box without clipping.
        low, high = self.fine.bounds(trust=(1., 1., 10.))
        low = np.where(self.fine.mask[None], low, -np.inf)
        high = np.where(self.fine.mask[None], high, np.inf)
        radius = 2*self.factor + 1
        low = maximum_filter(low, size=(1, radius, radius), mode='nearest')
        high = minimum_filter(high, size=(1, radius, radius), mode='nearest')
        p, f = self.pml, self.factor
        low = low[:, p:-p:f, p:-p:f]
        high = high[:, p:-p:f, p:-p:f]
        low = np.pad(low, ((0,0),(p,p),(p,p)), mode='constant')
        high = np.pad(high, ((0,0),(p,p),(p,p)), mode='constant')
        low[:, ~self.mask] = 0.
        high[:, ~self.mask] = 0.
        if np.any(low > 0) or np.any(high < 0) or not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError('The native anchor is not a feasible control origin')
        return low, high

    def coarse_material(self, value):
        value = np.asarray(value, np.float64) * self.mask
        dc, dz, da = SCALES * value
        result = np.stack((self.anchor[0]*np.exp(dc),
                           self.anchor[1]*np.exp(dz-dc),
                           self.edge + self.tissue*np.exp(da)))
        return result.astype(np.float32)

    def native_material(self, value):
        return self.fine.material(self.prolong(value))

    def regularization(self, value, weight=.001):
        native = self.prolong(value)
        cost, gradient, _, _ = self.fine.regularization(native, weight)
        return cost, self.transpose(gradient)

    def info(self):
        return {
            'factor': self.factor, 'native_shape': list(self.native_anchor.shape),
            'coarse_shape': list(self.shape),
            'native_spacing_m': 50e-6, 'coarse_spacing_m': self.factor*50e-6,
            'same_native_time_step_s': 2.5e-9,
            'pml_cells_retained': self.pml,
            'additional_PML_domain_extension_each_side_m': self.offset*self.factor*50e-6,
            'coarse_coupling_rows': self.gel_rows,
            'coarse_material_averaging_is_approximate': True,
            'independent_c_logZ_log_tissue_alpha_controls': True,
            'prolongation': 'bilinear independent log corrections of actual native anchor',
            'no_clipping_of_prolonged_native_material': True,
            'bounds_use_all_native_points_in_each_control_support': True,
            'original_native_regularizer_and_its_exact_control_pullback': True,
            'final_full_native_RF_acceptance_required': True,
            'patient_truth_read': False,
        }
