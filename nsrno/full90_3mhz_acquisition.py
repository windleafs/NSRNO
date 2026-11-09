"""Matched existing 90 us 3 MHz observations, full 11-shot source families.

No truth arrays are read. The response uses the original receiving definition
and dense nominal-band DFT samples without a large cached matrix.
"""
from pathlib import Path
import json
import numpy as np
import torch
from .native_time import Config
from .time_acquisition import sparse
from .matrix_free_rf_response import MatrixFreeRFResponse
from .low_frequency_0p5 import digest


class Full90ThreeMHzAcquisition:
    def __init__(self,index_path,family,device='cuda:0',chunk=128):
        index_path=Path(index_path)
        self.index=json.loads(index_path.read_text())
        if self.index['status']!='matched_existing_3MHz_full90_ready':
            raise ValueError('Prepare the fixed existing-data adapter first')
        if family not in ['lf','point3']:
            raise ValueError(family)
        self.family=family
        self.device=device
        self.f0=3e6
        self.meta=json.loads(Path('out/v5/time_triparam_v1/geometry.json').read_text())
        with np.load('out/v5/time_triparam_v1/geometry.npz') as a:
            self.geometry={k:a[k] for k in a.files}
        self.frequencies=np.asarray(self.index['frequency_hz'])
        self.mask=torch.from_numpy(np.asarray(self.index['DFT_mask'],bool)).to(device)
        record=self.index['families'][family]
        if digest(record['observed_coefficients'])!=record['observed_coefficients_SHA256']:
            raise ValueError('Fixed observed RF changed')
        self.observed=torch.from_numpy(np.load(record['observed_coefficients'])).to(device)
        if digest(record['raw_reference'])!=record['raw_reference_SHA256']:
            raise ValueError('Original known-water reference changed')
        self.raw_reference=np.load(record['raw_reference'],mmap_mode='r')
        if self.raw_reference.shape!=(11,36001,192) or self.raw_reference.dtype!=np.float32:
            raise ValueError('Full original native water reference required')
        self.angles=self.index['angles_deg']
        with np.load(self.index['source_layout']) as a:
            self.point_weights=a['source_weights']
            self.point_positions=a['transmit_x_m']
        if self.observed.shape!=(11,192,len(self.frequencies)):
            raise ValueError('All angles and receivers are required')
        fit=self.index['absorption']
        g=self.geometry
        nz,nx=self.meta['shape']
        self.config=Config(nz,nx,36001,len(g['source_coordinates']),len(g['xe']),
            self.meta['pml'],chunk,self.meta['dx'],self.meta['dt'],
            *fit['taus_s'],*fit['eta_per_unit_alpha'])
        si,sw=sparse(g['source_coordinates'],g['origin'],g['spacing'],nx)
        ri,rw=sparse(g['receiver_coordinates'],g['origin'],g['spacing'],nx,rec=True)
        self.si=torch.from_numpy(si).to(device)
        self.sw=torch.from_numpy(sw).to(device)
        self.ri=torch.from_numpy(ri).to(device)
        self.rw=torch.from_numpy(rw).to(device)
        self.response=MatrixFreeRFResponse(36001,self.meta['dt'],
            [1.5e6,4.5e6],np.asarray(self.index['DFT_mask'],bool),device=device)

    def waveform(self,shot):
        g=self.geometry
        x,xe=g['x'],g['xe']
        columns=np.flatnonzero((x>=xe[0])&(x<=xe[-1]))
        if self.family=='point3':
            delays=np.zeros(len(columns))
            weights=self.point_weights[shot]
        else:
            raw=x[columns]*np.sin(np.deg2rad(self.angles[shot]))/1540.
            delays=raw-raw.min()
            weights=g['weights']
        # Exact original benchmark.source_arrays evaluation order.
        sigma=1/self.f0
        time=np.arange(36001)*self.meta['dt']
        tau=time[None,:]-(4*sigma+delays[:,None])
        value=(1e5*weights[:,None]*np.exp(-.5*(tau/sigma)**2)*
               np.sin(2*np.pi*self.f0*tau)).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(value.T)).to(self.device)

    def coefficients(self,raw):
        # Coarse differences use total pressure; their fixed anchor offset
        # cancels the water reference. No actual fine RF is claimed here.
        response=self.response(raw.double())
        return torch.fft.rfft(response,dim=0,norm='ortho')[self.mask].T

    def scattered_coefficients(self,raw,shot):
        # The old generator subtracts the native float32 water pressure
        # BEFORE receiving operations. Preserve this rounding convention.
        reference=torch.from_numpy(np.array(self.raw_reference[shot],copy=True)).to(self.device)
        value=raw-reference
        response=self.response(value.double())
        return torch.fft.rfft(response,dim=0,norm='ortho')[self.mask].T
