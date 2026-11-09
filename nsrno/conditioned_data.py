"""Shared, target-independent assembly for conditioned frequency training."""
from pathlib import Path
import json
from functools import lru_cache
import numpy as np
import torch
from nsrno.conditioned_wave import FrequencyGeometry,ConditionedFrequencySystem

ROOT=Path('out/v5/conditioned_wave_v1')
@lru_cache(maxsize=32)
def receiver_response(frequency,dt=2.5e-9):
    """Known V5 zero-phase Butterworth and polyphase-resampling response.

    This is the infinite-signal transfer; finite record edge transients remain
    part of the measured RF discrepancy. No target-fitted gain is used.
    """
    from scipy.signal import butter,sosfreqz,firwin,freqz
    sos=butter(4,[4e6,7.5e6],fs=1/dt,btype='bandpass',output='sos')
    _,h=sosfreqz(sos,worN=[frequency],fs=1/dt)
    down=10; half_len=10*down
    taps=firwin(2*half_len+1,1/down,window=('kaiser',5.))
    omega=2*np.pi*frequency*dt
    _,fir=freqz(taps,worN=[omega])
    return float(abs(h[0])**2*(fir[0]*np.exp(1j*omega*half_len)).real)

def key(bin_index,angle): return f'b{bin_index}_a{angle}'
def field_path(row,bin_index,angle): return ROOT/'teachers'/f'{row["id"]}_{key(bin_index,angle)}.npy'

class ConditionedData:
    def __init__(self,device='cuda'):
        self.index=json.load(open(ROOT/'index.json')); self.spec=self.index['spec']
        self.rows=self.index['records']; self.cache=Path(self.index['source_cache']); self.device=device
        self.geometry=FrequencyGeometry(self.spec).to(device)
        self.sources={k:torch.from_numpy(v).to(device) for k,v in np.load(ROOT/'sources.npz').items()}
        self.references={}; self.media={}

    def material(self,row):
        # CPU cache keeps device memory available to unrolled differentiation.
        if row['id'] not in self.media:
            self.media[row['id']]=(torch.from_numpy(np.load(self.cache/row['medium'])),
                                   torch.from_numpy(np.load(row['alpha'])))
        return tuple(v.to(self.device) for v in self.media[row['id']])

    def reference(self,bin_index,angle):
        name=key(bin_index,angle)
        if name not in self.references:
            archive=np.load(ROOT/'reference'/(name+'.npz'))
            self.references[name]={k:self.geometry.pad_field(torch.from_numpy(archive[k]).to(self.device))
                for k in ('pressure','operator_pressure','source')}
        return self.references[name]

    def assemble(self,row,bin_index,angle,requires_grad=False):
        medium,alpha=self.material(row)
        if requires_grad: medium=medium.detach().requires_grad_(True)
        s=ConditionedFrequencySystem(self.geometry,medium,alpha,bin_index*40e6/2401)
        ref=self.reference(bin_index,angle)
        # Scattered equation. Reference discretization error is canceled.
        rhs=s.source(self.sources[key(bin_index,angle)])-ref['source']-s(ref['pressure'])+ref['operator_pressure']
        return s,rhs,medium

    def teacher(self,row,bin_index,angle):
        return self.geometry.pad_field(torch.from_numpy(np.load(field_path(row,bin_index,angle))).to(self.device))

    def target(self,row,bin_index,angle):
        iq=np.load(self.cache/row['iq'],mmap_mode='r')[angle]
        spectrum=np.fft.fftshift(np.fft.fft(iq[0]+1j*iq[1],axis=-1,norm='forward'),axes=-1)/np.sqrt(2)
        return torch.from_numpy(spectrum[:,bin_index-40].astype(np.complex64)).to(self.device)
