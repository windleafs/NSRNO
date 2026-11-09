"""Frozen TRAIN denoiser score pullbacks, not a squared-residual energy.

Only physical-feature and view transforms are differentiated. No denoiser
Jacobian is formed. An approximate CNN score need not be conservative.
"""
import numpy as np
import torch
from torch.nn import functional as F
from .learned_spatial_energy import LearnedSpatialEnergy
from .global_cdf_prior import GlobalCDFPrior


class FrozenScoreGuidance:
    def __init__(self, neural_path, cdf_path):
        fixed = LearnedSpatialEnergy(neural_path)
        self.model, self.sigmas = fixed.model, fixed.sigmas
        self.cdf = GlobalCDFPrior(cdf_path)

    def neural_pullback(self, coordinates, control):
        control = np.asarray(control, np.float64)
        if control.shape != (3, 961, 1121) or not np.isfinite(control).all():
            raise ValueError('Independent finite full native controls required')
        absolute = coordinates.absolute+control
        result = np.zeros_like(absolute)
        diagnostics = {}
        for field in range(3):
            value = torch.from_numpy(absolute[field, 80:921, 178:943].copy())[None, None]
            value.requires_grad_(True)
            if field == 0:
                feature = (1540.*torch.exp(.05*value)-1540.)/80.
            elif field == 1:
                feature = (1000.*torch.exp(.1*value)-1000.)/100.
            else:
                feature = value
            views = [feature, F.adaptive_avg_pool2d(feature, (211, 192)),
                F.interpolate(feature, size=(128, 128), mode='bilinear', align_corners=False)]
            linear_pullback_surrogate = 0.
            for scale, view in enumerate(views):
                with torch.no_grad():
                    denoised = self.model.fields[field](view.detach(), self.sigmas[scale], scale)
                    negative_score = (view.detach()-denoised)/self.sigmas[scale]**2
                # This scalar only implements the chain-rule pullback of a fixed
                # covector. Its numerical value is not a prior energy.
                linear_pullback_surrogate += (view*negative_score).mean()/3
                diagnostics[f'f{field}_s{scale}'] = float(negative_score.square().mean().sqrt())
            derivative, = torch.autograd.grad(linear_pullback_surrogate, value)
            result[field, 80:921, 178:943] = derivative.detach().numpy()[0, 0]
        if not np.isfinite(result).all():
            raise RuntimeError('Nonfinite required frozen-score pullback')
        return result, diagnostics

    def spatial_terms(self, coordinates, control, gradient):
        cdf, gc, details = self.cdf.evaluate(coordinates, control, gradient=gradient)
        tv, gt = coordinates.regularization(control, .001)
        return {'CDF': cdf, 'TV': tv, 'CDF_details': details}, gc+gt if gradient else None

    def guide(self, coordinates, control):
        gn, per_view = self.neural_pullback(coordinates, control)
        parts, gs = self.spatial_terms(coordinates, control, True)
        vector = gn+gs
        if not np.isfinite(vector).all():
            raise RuntimeError('Nonfinite composite actual guide')
        return vector, {**parts, 'negative_feature_score_RMS': per_view,
            'sigma_by_scale': self.sigmas, 'network_Jacobian_used': False,
            'exact_feature_and_view_pullback': True, 'known_global_scalar_energy': False,
            'same_frozen_weights_and_CDF': True}

    def signed_line_work(self, coordinates, control, direction, fraction, initial_parts):
        if not 0 < fraction <= 1:
            raise ValueError('A positive bounded actual candidate fraction is required')
        nodes, weights = np.polynomial.legendre.leggauss(3)
        samples, neural_work = [], 0.
        for node, weight in zip(nodes, weights):
            t = .5*fraction*(node+1)
            gn, _ = self.neural_pullback(coordinates, control+t*direction)
            slope = float(np.sum(gn*direction))
            neural_work += .5*fraction*weight*slope
            samples.append({'fraction': float(t), 'negative_score_slope': slope})
        after, _ = self.spatial_terms(coordinates, control+fraction*direction, False)
        cdf_delta, tv_delta = after['CDF']-initial_parts['CDF'], after['TV']-initial_parts['TV']
        total = float(neural_work+cdf_delta+tv_delta)
        if not np.isfinite(total):
            raise RuntimeError('Nonfinite signed guide work')
        return total, {'neural_signed_line_work_GL3': float(neural_work),
            'exact_CDF_difference': cdf_delta, 'exact_TV_difference': tv_delta,
            'composite_signed_line_work': total, 'parts_after': after, 'nodes': samples,
            'approximate_quadrature_not_error_certificate': True,
            'not_a_path_independent_energy': True, 'old_residual_energy_not_an_acceptance_rule': True}
