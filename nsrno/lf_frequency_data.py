"""Acquisition-only access to 3-MHz dense spectra and uniform reference fields."""
from pathlib import Path
from functools import lru_cache
import json,math
import numpy as np
import torch
from .conditioned_wave import FrequencyGeometry
from .v5_observations import plane_observation


@lru_cache(maxsize=512)
def receiver_response(frequency):
    from scipy.signal import butter,sosfreqz,firwin,freqz
    dt=2.5e-9
    sos=butter(4,[1.5e6,4.5e6],fs=1/dt,btype='bandpass',output='sos')
    _,h=sosfreqz(sos,worN=[frequency],fs=1/dt)
    taps=firwin(201,.1,window=('kaiser',5.))
    omega=2*np.pi*frequency*dt
    _,fir=freqz(taps,worN=[omega])
    return float(abs(h[0])**2*(fir[0]*np.exp(1j*omega*100)).real)


class LowFrequencyData:
    def __init__(self,sample='train_200',device='cpu',root='out/v5/conditioned_lf_full_v1',allow_partial=False):
        self.root=Path(root);self.device=device;self.sample=sample
        if (self.root/'index.json').exists():
            self.spec=json.loads((self.root/'index.json').read_text())['spec']
        elif allow_partial:
            index=json.loads(Path('out/v5/forward_cache_3mhz_v1/index.json').read_text())
            low=json.loads(Path('out/v5/low_frequency_3mhz_v1/index.json').read_text())
            meta=json.loads(Path('out/v5/time_triparam_v1/geometry.json').read_text())
            self.spec={**index['grid'],**index['acquisition'],**low['absorption'],
                       'native_dt_s':meta['dt'],'pml':meta['pml'],'bins':list(range(91,271))}
        else:raise RuntimeError('The full eleven-angle reference is not complete')
        self.geometry=FrequencyGeometry(self.spec).to(device)
        observation=plane_observation('lf',sample)
        iq=np.load(observation['iq'],mmap_mode='r')
        transform=np.fft.fftshift(np.fft.fft(iq[:,0]+1j*iq[:,1],axis=-1,norm='forward'),axes=-1)/math.sqrt(2)
        self.energy=float(np.sum(abs(transform)**2,dtype=np.float64))
        self.spectra=transform[:,:,91-40:271-40].astype(np.complex64)
        self.reference_maps={}
        self.observation_metadata=observation

    def target(self,bin_index,angle):
        if bin_index not in self.spec['bins']:raise ValueError('Frequency outside the full LF band')
        return torch.from_numpy(self.spectra[angle,:,bin_index-91]).to(self.device)

    def reference(self,bin_index,angle):
        if angle not in self.reference_maps:
            self.reference_maps[angle]=np.load(self.root/'reference'/f'a{angle:02d}.npy',mmap_mode='r')
        value=np.array(self.reference_maps[angle][bin_index-91])
        return self.geometry.pad_field(torch.from_numpy(value).to(self.device))
