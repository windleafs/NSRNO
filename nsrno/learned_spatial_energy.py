"""CPU energy and full gradient of three independently frozen TRAIN denoisers."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .independent_denoising_prior import IndependentMaterialDenoisers, denoising_energy


class LearnedSpatialEnergy:
    def __init__(self, checkpoint):
        torch.set_num_threads(1)
        state = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
        if (state['training_case'] != 'Neg_07_Left' or state['development_arrays_read'] or
            state['reserved_arrays_read'] or state['completed_steps'] != 3000):
            raise RuntimeError('Only the fixed TRAIN-only final model is allowed')
        self.model = IndependentMaterialDenoisers().double().eval().requires_grad_(False)
        self.model.load_state_dict(state['state'])
        self.sigmas = [.05, .2, .6]

    def evaluate(self, coordinates, control, gradient=True):
        if np.asarray(control).shape != (3, 961, 1121):
            raise ValueError('Independent full native controls required')
        absolute = coordinates.absolute+np.asarray(control, np.float64)
        total, per_scale = 0., {}
        derivative = np.zeros_like(absolute) if gradient else None
        for field in range(3):
            value = torch.from_numpy(absolute[field, 80:921, 178:943].copy())[None, None]
            value.requires_grad_(gradient)
            with torch.set_grad_enabled(gradient):
                if field == 0:
                    feature = (1540.*torch.exp(.05*value)-1540.)/80.
                elif field == 1:
                    feature = (1000.*torch.exp(.1*value)-1000.)/100.
                else:
                    feature = value
                medium = F.adaptive_avg_pool2d(feature, (211, 192))
                global_view = F.interpolate(feature, size=(128, 128), mode='bilinear', align_corners=False)
                energy = 0.
                for scale, view in enumerate([feature, medium, global_view]):
                    e = denoising_energy(self.model.fields[field], view, self.sigmas[scale], scale)
                    energy = energy+e/3
                    per_scale[f'f{field}_s{scale}'] = float(e.detach())
                total += float(energy.detach())
                if gradient:
                    g, = torch.autograd.grad(energy, value)
                    derivative[field, 80:921, 178:943] = g.detach().numpy()[0, 0]
        TV, TV_gradient = coordinates.regularization(control, .001)
        if gradient:
            derivative += TV_gradient
            if not np.isfinite(derivative).all():
                raise RuntimeError('Nonfinite full neural-energy control gradient')
        if not np.isfinite(total+TV):
            raise RuntimeError('Nonfinite frozen neural energy')
        return total+TV, derivative, {'neural_energy':total, 'per_field_scale':per_scale,
            'TV':TV, 'sigma_by_scale':self.sigmas, 'full_denoiser_chain_derivative':gradient,
            'old_marginal_and_conditional_patch_terms_removed':True}
