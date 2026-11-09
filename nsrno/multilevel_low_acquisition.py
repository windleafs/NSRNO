"""Coarse low-frequency PDE on an extended PML; actual target remains fine RF.

A new caller must lock an actual fine-material/RF anchor and use:
  prediction(u) = coarse(u) - coarse(0) + fine(anchor).
The additive correction is fixed, not refitted to observed data per material.
"""
import numpy as np
import torch
from .acquisition_single_case_0p5 import SingleCaseLowAcquisition
from .native_time import Config
from .time_acquisition import sparse


class MultilevelLowAcquisition(SingleCaseLowAcquisition):
    def __init__(self, root, family, coarse_shape, device='cuda:0', chunk=128, factor=4):
        super().__init__(root, family, device=device, chunk=chunk)
        if factor != 4 or tuple(coarse_shape) != (277, 317):
            raise ValueError('Require the specified extended-PML coarse grid')
        g = self.geometry
        p = self.meta['pml']
        dx = self.meta['dx'] * factor
        offset = p - p//factor
        origin = (g['origin'].astype(np.float64) - offset*dx).astype(np.float32)
        spacing = (g['spacing'] * np.float32(factor)).astype(np.float32)
        si, sw = sparse(g['source_coordinates'], origin, spacing, coarse_shape[1])
        ri, rw = sparse(g['receiver_coordinates'], origin, spacing, coarse_shape[1], rec=True)
        # native_time.cu injects src*dt/(1540*dx). With the ORIGINAL source
        # quadrature points retained, integral of injection over cell area is
        # proportional to dx*sum(sw*src). Dividing sw by factor preserves it.
        sw = sw / np.float32(factor)
        self.si = torch.from_numpy(si).to(device)
        self.sw = torch.from_numpy(sw).to(device)
        self.ri = torch.from_numpy(ri).to(device)
        self.rw = torch.from_numpy(rw).to(device)
        fit = self.index['absorption']
        self.config = Config(
            coarse_shape[0], coarse_shape[1], self.config.nt,
            len(g['source_coordinates']), len(g['xe']), p, chunk,
            dx, self.meta['dt'], *fit['taus_s'], *fit['eta_per_unit_alpha'])
        if min(si[:, 0] // coarse_shape[1]) < p+4:
            raise ValueError('Known sources must remain beyond the PML and stencil margin')
        self.coarse_info = {
            'coarse_shape': list(coarse_shape), 'coarse_origin': origin.tolist(),
            'coarse_spacing': spacing.tolist(), 'coarse_PML_cells': p,
            'original_source_points_retained': len(g['source_coordinates']),
            'original_waveform_hash_checks_retained': True,
            'source_interpolation_weights_rescaling': 1/factor,
            'source_rescaling_from_native_injection_formula': 'src*dt/(1540*dx)',
            'new_relaxation_fit': False,
            'new_time_step': False,
            'coarse_PDE_not_claimed_identical_to_fine_PDE': True,
            'actual_fine_RF_anchor_correction_required': True,
        }
