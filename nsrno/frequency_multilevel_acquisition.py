"""Independent low/high source objects with reusable coarse physical geometry."""
import copy
import numpy as np
import torch
from .native_time import Config
from .time_acquisition import sparse
from .acquisition_single_case_0p5 import SingleCaseLowAcquisition
from .full90_3mhz_acquisition import Full90ThreeMHzAcquisition


class LowHalfMHzAcquisition(SingleCaseLowAcquisition):
    def scattered_coefficients(self,raw,shot):
        return self.coefficients(raw)-self.reference[shot]


def coarse_acquisition(fine,coarse_shape,factor):
    if factor not in (2,4):
        raise ValueError('Specified physical spacing factors only')
    result=copy.copy(fine)
    g=fine.geometry
    p=fine.meta['pml']
    dx=fine.meta['dx']*factor
    offset=p-p//factor
    origin=(g['origin'].astype(np.float64)-offset*dx).astype(np.float32)
    spacing=(g['spacing']*np.float32(factor)).astype(np.float32)
    si,sw=sparse(g['source_coordinates'],origin,spacing,coarse_shape[1])
    ri,rw=sparse(g['receiver_coordinates'],origin,spacing,coarse_shape[1],rec=True)
    result.si=torch.from_numpy(si).to(fine.device)
    result.sw=torch.from_numpy(sw/np.float32(factor)).to(fine.device)
    result.ri=torch.from_numpy(ri).to(fine.device)
    result.rw=torch.from_numpy(rw).to(fine.device)
    fit=fine.index['absorption']
    result.config=Config(coarse_shape[0],coarse_shape[1],fine.config.nt,
        fine.config.ns,fine.config.nr,p,fine.config.chunk,dx,fine.meta['dt'],
        *fit['taus_s'],*fit['eta_per_unit_alpha'])
    if min(si[:,0]//coarse_shape[1])<p+4:
        raise ValueError('Source must remain beyond the PML stencil margin')
    result.coarse_info={
        'factor':factor,'shape':list(coarse_shape),'native_clock_retained':True,
        'source_spatial_integral_rescaling':1/factor,
        'original_source_points_and_waveforms_retained':True,
        'response_buffers_and_observation_shared_with_fine_acquisition':True,
        'actual_fine_RF_acceptance_required':True}
    return result


def matched_acquisitions(low_root,high_index,device,chunk=128):
    return {
        'low_lf':LowHalfMHzAcquisition(low_root,'lf',device=device,chunk=chunk),
        'low_point3':LowHalfMHzAcquisition(low_root,'point3',device=device,chunk=chunk),
        'high_lf':Full90ThreeMHzAcquisition(high_index,'lf',device=device,chunk=chunk),
        'high_point3':Full90ThreeMHzAcquisition(high_index,'point3',device=device,chunk=chunk),
    }
