"""Observation-only spectra for the existing plane and point 3-MHz groups."""
from pathlib import Path
import json
import numpy as np
import torch
from .conditioned_wave import FrequencyGeometry
from .v5_observations import plane_observation,canonical_id
from .lf_frequency_data import receiver_response

ROOTS={'lf':'out/v5/conditioned_lf_full_v1','point3':'out/v5/conditioned_point_full_v1'}


class FrequencyGroupData:
    def __init__(self,source,sample,device='cpu',geometry=None):
        if source not in ROOTS:raise ValueError(source)
        self.source=source;self.sample=sample;self.device=device;self.root=Path(ROOTS[source])
        if not (self.root/'index.json').exists():raise RuntimeError(f'Full {source} reference is not ready')
        self.spec=json.loads((self.root/'index.json').read_text())['spec']
        self.bins=self.spec['bins'];self.first=self.bins[0]
        if self.bins!=list(range(self.first,self.bins[-1]+1)):raise ValueError('Contiguous physical frequencies required')
        self.N=self.spec['rf_samples'];self.references={}
        self.geometry=geometry or FrequencyGeometry(self.spec).to(device)
        if source=='lf':
            metadata=plane_observation('lf',sample)
            iq=np.load(metadata['iq'],mmap_mode='r')
            transformed=np.fft.fftshift(np.fft.fft(iq[:,0]+1j*iq[:,1],axis=-1,norm='forward'),axes=-1)/np.sqrt(2)
            self.energy=float(np.sum(abs(transformed)**2,dtype=np.float64))
            self.spectra=transformed[:,:,self.first-40:self.bins[-1]-39].astype(np.complex64)
        else:
            point=json.loads(Path('out/v5/point_tx_3mhz_v1/index.json').read_text())
            row=next(r for r in point['records'] if r['id']==canonical_id(sample))
            rf=np.load(row['rf'],mmap_mode='r')
            transformed=np.fft.rfft(rf,axis=-1,norm='forward')
            self.energy=float(np.sum(abs(transformed)**2,dtype=np.float64))
            self.spectra=transformed[:,:,self.first:self.bins[-1]+1].astype(np.complex64)
        if self.spectra.shape!=(11,192,len(self.bins)):raise ValueError('All 11 transmissions and 192 receivers required')

    def frequency(self,position):return self.bins[position]*40e6/self.N
    def target(self,position,angle):return torch.from_numpy(self.spectra[angle,:,position]).to(self.device)
    def reference(self,position,angle):
        if angle not in self.references:
            self.references[angle]=np.load(self.root/'reference'/f'a{angle:02d}.npy',mmap_mode='r')
        return self.geometry.pad_field(torch.from_numpy(np.array(self.references[angle][position])).to(self.device))
