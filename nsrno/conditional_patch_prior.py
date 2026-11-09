"""Independent conditional TRAIN patch energies on the native material grid.

This module has no data-loader imports. Its only file input is a fitted prior.
It neither shares labels between materials nor filters the material fields.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.special import logsumexp
from .factorized_material_prior import FactorizedMaterialPrior


DILATIONS = (1, 4, 16)
OFFSETS = np.array([(z, x) for z in range(-2, 3) for x in range(-2, 3)], np.int64)
CENTER = 12
LOWER = np.array([(1400.-1540.)/80.,
    (1540e3*np.exp(-.25)/1680.-1000.)/100., -8.])
UPPER = np.array([(1680.-1540.)/80.,
    (1540e3*np.exp(.25)/1400.-1000.)/100., 2.])


class PatchModel:
    def __init__(self, weight, mean, variance, pca_center, basis, field, tail=.1):
        self.weight = np.asarray(weight, np.float64)
        self.mean = np.asarray(mean, np.float64)
        self.variance = np.asarray(variance, np.float64)
        self.pca_center = np.asarray(pca_center, np.float64)
        self.basis = np.asarray(basis, np.float64)
        self.field, self.tail = int(field), float(tail)
        if (self.mean.shape != self.variance.shape or self.mean.shape != (len(self.weight), 25) or
            self.pca_center.shape != (25,) or self.basis.shape != (25, 25) or
            self.field not in (0, 1, 2) or not 0 < self.tail < 1 or
            any(not np.isfinite(x).all() for x in
                [self.weight, self.mean, self.variance, self.pca_center, self.basis]) or
            np.any(self.weight <= 0) or np.any(self.variance <= 0) or
            abs(self.weight.sum()-1) > 1e-10 or
            not np.allclose(self.basis.T@self.basis, np.eye(25), rtol=0, atol=1e-10)):
            raise ValueError('Positive normalized full-dimensional patch mixture required')
        self.precision = 1/self.variance
        self.mean_precision = self.mean*self.precision
        self.core_normalizer = (np.log((1-self.tail)*self.weight)
            -.5*(25*np.log(2*np.pi)+np.log(self.variance).sum(1)))
        self.mean_squared_precision = (self.mean*self.mean_precision).sum(1)
        self.center_mean = self.pca_center[CENTER]+self.mean@self.basis[CENTER]
        self.center_variance = self.variance@(self.basis[CENTER]**2)
        self.center_normalizer = (np.log((1-self.tail)*self.weight)
            -.5*np.log(2*np.pi*self.center_variance))

    def value_gradient(self, patches):
        """Conditional negative log density per neighbor, and dE/dpatch.

        The uniform joint tail has exactly the uniform center marginal used
        below. The center term is subtracted as an NLL, not added twice.
        """
        patches = np.asarray(patches, np.float64)
        if patches.ndim != 2 or patches.shape[1] != 25 or not np.isfinite(patches).all():
            raise ValueError('Finite batches of 25-dimensional patches required')
        width = UPPER[self.field]-LOWER[self.field]
        inside = (patches >= LOWER[self.field])&(patches <= UPPER[self.field])
        log_tail = np.where(inside.all(1), np.log(self.tail)-25*np.log(width), -np.inf)
        transformed = (patches-self.pca_center)@self.basis
        quadratic = ((transformed*transformed)@self.precision.T
            -2*transformed@self.mean_precision.T+self.mean_squared_precision)
        core = self.core_normalizer-.5*quadratic
        log_joint = np.logaddexp(logsumexp(core, axis=1), log_tail)
        response = np.exp(core-log_joint[:, None])
        g_transformed = transformed*(response@self.precision)-response@self.mean_precision
        gradient = g_transformed@self.basis.T
        center = patches[:, CENTER]
        difference = center[:, None]-self.center_mean
        center_core = self.center_normalizer-.5*difference*difference/self.center_variance
        center_tail = np.where(inside[:, CENTER], np.log(self.tail)-np.log(width), -np.inf)
        log_center = np.logaddexp(logsumexp(center_core, axis=1), center_tail)
        response_center = np.exp(center_core-log_center[:, None])
        gradient[:, CENTER] -= (response_center*difference/self.center_variance).sum(1)
        return (-log_joint+log_center)/24., gradient/24.


class ConditionalPatchPrior:
    def __init__(self, path, dilations=DILATIONS):
        self.dilations = tuple(int(d) for d in dilations)
        if not self.dilations or any(d not in DILATIONS for d in self.dilations):
            raise ValueError('Declared native patch scales required')
        self.models = {}
        with np.load(Path(path), allow_pickle=False) as state:
            for field in range(3):
                for dilation in self.dilations:
                    prefix = f'f{field}_d{dilation}_'
                    self.models[field, dilation] = PatchModel(
                        *[state[prefix+k] for k in ['weight', 'mean', 'variance', 'pca_center', 'basis']],
                        field=field, tail=.1)
        self._roi_key, self._centers = None, {}
        self.last_parts = None

    def centers(self, roi):
        key = (roi.shape, hashlib.sha256(roi.tobytes()).hexdigest())
        if key != self._roi_key:
            centers = {}
            for dilation in self.dilations:
                valid = minimum_filter(roi, size=4*dilation+1, mode='constant', cval=0)
                centers[dilation] = np.nonzero(valid)
                if not len(centers[dilation][0]):
                    raise ValueError('ROI has no complete native patch at a declared scale')
            self._centers, self._roi_key = centers, key
        return self._centers

    def energy_and_scaled_log_gradient(self, material, known_edge, roi, weight=1., chunk=8192):
        """Energy and derivative in log c/.05, log rho/.1, log tissue alpha/.7."""
        material = np.asarray(material, np.float64)
        roi = np.asarray(roi, bool)
        if (material.shape != (3, *roi.shape) or not roi.any() or weight < 0 or chunk < 1 or
            not np.isfinite(material).all()):
            raise ValueError('Finite independent physical fields and nonempty ROI required')
        tissue = material[2]-np.broadcast_to(known_edge, roi.shape)
        if np.any(material[:2, roi] <= 0) or np.any(tissue[roi] <= 0):
            raise ValueError('Positive c, rho, and tissue absorption required')
        features = np.zeros_like(material)
        features[0] = (material[0]-1540.)/80.
        features[1] = (material[1]-1000.)/100.
        features[2, roi] = np.log(tissue[roi]/.5)/.7
        gradients = np.zeros_like(material)
        value, parts = 0., {}
        centers = self.centers(roi)
        for field in range(3):
            for dilation in self.dilations:
                z, x = centers[dilation]
                offsets = OFFSETS*dilation
                factor = weight/(len(self.dilations)*len(z))
                score = 0.
                for start in range(0, len(z), chunk):
                    zz = z[start:start+chunk, None]+offsets[None, :, 0]
                    xx = x[start:start+chunk, None]+offsets[None, :, 1]
                    energy, gradient = self.models[field, dilation].value_gradient(features[field, zz, xx])
                    score += factor*float(energy.sum())
                    np.add.at(gradients[field], (zz, xx), factor*gradient)
                value += score
                parts[f'f{field}_d{dilation}'] = score
        gradients[0] *= .05*material[0]/80.
        gradients[1] *= .1*material[1]/100.
        # The standardized alpha feature equals its scaled-log control plus
        # a fixed anchor, so its physical pullback factor is exactly one.
        self.last_parts = {'conditional_spatial': value, 'per_field_scale': parts,
            'centers_per_scale': {str(d): len(zx[0]) for d, zx in centers.items()}}
        return value, gradients


class CombinedIndependentPrior:
    def __init__(self, marginal_path, patch_path):
        self.marginal = FactorizedMaterialPrior(marginal_path, tail=.1, temperature=0.)
        self.spatial = ConditionalPatchPrior(patch_path)
        self.last_parts = None

    def energy_and_scaled_log_gradient(self, material, known_edge, roi):
        v0, g0 = self.marginal.energy_and_scaled_log_gradient(material, known_edge, roi)
        v1, g1 = self.spatial.energy_and_scaled_log_gradient(material, known_edge, roi)
        self.last_parts = {'marginal': v0, 'conditional_spatial': v1,
                           'spatial_details': self.spatial.last_parts}
        return v0+v1, g0+g1
