"""Native 0.5 MHz observed RF, fixed-source, matrix-free receiver processing."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .low_frequency_0p5 import (F0, NATIVE_NT, NATIVE_DT, AMPLITUDE, SAMPLE,
    frequency_mask, BAND, digest)
from .matrix_free_rf_response import MatrixFreeRFResponse
from .native_time import Config, native_wave
from .time_acquisition import sparse


class SingleCaseLowAcquisition:
    def __init__(self, root, family, device='cuda:0', chunk=128):
        root = Path(root)
        self.index = json.loads((root / 'index.json').read_text())
        if (self.index['status'] != 'paired_0p5MHz_generation_finished' or
                self.index['completed_wave_solves'] != 44 or self.index['sample'] != SAMPLE):
            raise ValueError('Require the actual completed, authorized single-case acquisition')
        if family not in ['lf', 'point3']:
            raise ValueError(family)
        self.family = family
        self.device = device
        self.meta = json.loads(Path('out/v5/time_triparam_v1/geometry.json').read_text())
        with np.load('out/v5/time_triparam_v1/geometry.npz') as a:
            self.geometry = {k: a[k] for k in a.files}
        layout_path = root / 'source_layout.npz'
        if digest(layout_path) != self.index['source_layout_SHA256']:
            raise ValueError('Frozen source layout changed')
        with np.load(layout_path) as a:
            self.layout = {k: a[k] for k in a.files}
        nz, nx = self.meta['shape']
        fit = self.index['absorption']
        self.config = Config(nz, nx, NATIVE_NT, len(self.geometry['source_coordinates']),
            len(self.geometry['xe']), self.meta['pml'], chunk, self.meta['dx'],
            NATIVE_DT, *fit['taus_s'], *fit['eta_per_unit_alpha'])
        if self.meta['dt'] != NATIVE_DT or (nz, nx) != (961, 1121):
            raise ValueError('Retain the original native grid and clock')
        g = self.geometry
        si, sw = sparse(g['source_coordinates'], g['origin'], g['spacing'], nx)
        ri, rw = sparse(g['receiver_coordinates'], g['origin'], g['spacing'], nx, rec=True)
        self.si = torch.from_numpy(si).to(device)
        self.sw = torch.from_numpy(sw).to(device)
        self.ri = torch.from_numpy(ri).to(device)
        self.rw = torch.from_numpy(rw).to(device)
        mask = frequency_mask()
        self.mask = torch.from_numpy(mask).to(device)
        self.frequencies = np.fft.rfftfreq(3601, 2.5e-8)[mask]
        self.response = MatrixFreeRFResponse(
            NATIVE_NT, NATIVE_DT, BAND, mask, device=device)
        self.records = {}
        arrays = {}
        for kind in ['reference', SAMPLE]:
            records = sorted([r for r in self.index['shots']
                              if r['family'] == family and r['kind'] == kind],
                             key=lambda r: r['shot'])
            if [r['shot'] for r in records] != list(range(11)):
                raise ValueError('All 11 shots required')
            values = []
            for row in records:
                if digest(row['path']) != row['SHA256']:
                    raise ValueError('Frozen RF changed')
                values.append(np.load(row['path']))
            self.records[kind] = records
            arrays[kind] = torch.from_numpy(np.stack(values)).to(device)
        self.reference = arrays['reference']
        self.observed = arrays[SAMPLE]
        if self.observed.shape != (11, 192, len(self.frequencies)):
            raise ValueError('Complete channel RF required')
        self.angles = self.index['angles_deg']

    def waveform(self, shot):
        # Reproduce benchmark_calibration.source_arrays exactly, including
        # float32 conversion before equal-energy source scaling.
        g = self.geometry
        columns = self.layout['source_columns']
        if self.family == 'lf':
            raw_delay = g['x'][columns] * np.sin(np.deg2rad(self.angles[shot])) / 1540.
            delays = raw_delay - raw_delay.min()
            weights = g['weights']
        else:
            delays = np.zeros(len(columns))
            weights = self.layout['point_weights'][shot]
        sigma = 1 / F0
        time = np.arange(NATIVE_NT) * NATIVE_DT
        tau = time[None, :] - (4 * sigma + delays[:, None])
        wave = (1e5 * weights[:, None] * np.exp(-.5 * (tau / sigma)**2) *
                np.sin(2 * np.pi * F0 * tau)).astype(np.float32)
        wave *= np.float32(AMPLITUDE / 1e5)
        wave = np.ascontiguousarray(wave.T)
        expected = self.records['reference'][shot]['source_SHA256']
        if hashlib.sha256(wave.tobytes()).hexdigest() != expected:
            raise ValueError('Source bytes differ from the actual original-solver acquisition')
        return torch.from_numpy(wave).to(self.device)

    def coefficients(self, raw):
        # Double receiving arithmetic avoids cancelling the large known
        # incident field at a float32 FFT before reference subtraction.
        rf = self.response(raw.double())
        return torch.fft.rfft(rf, dim=0, norm='ortho')[self.mask].T
