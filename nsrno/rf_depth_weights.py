"""Fixed geometry-based arrival weights; not a claim of true RF localization."""
import numpy as np
import torch

NAMES=['earlier_than_midpoint_5mm','midpoint_5_to_12mm','midpoint_12_to_25mm','midpoint_25_to_43mm','later_or_off_midpoint']
WEIGHTS=[.10,.20,.25,.30,.15]


def arrival_bins(acquisition):
    a=acquisition;g=a.geometry;t=np.arange(a.observed.shape[-1])/40e6
    bins=np.zeros(tuple(a.observed.shape),np.uint8);receiver=g['xe'];reference_speed=1540.
    for z in [.005,.012,.025,.043]:
        if a.source=='point3':
            offset=a.source_positions[:,None]-receiver[None,:]
            boundary=4/a.f0+np.sqrt(offset**2+(2*z)**2)/reference_speed
        else:
            theta=np.deg2rad(a.angles);cols=np.flatnonzero((g['x']>=receiver[0])&(g['x']<=receiver[-1]))
            raw=np.sin(theta[:,None])*g['x'][cols][None,:]/reference_speed
            tref=4/a.f0-raw.min(axis=1)
            boundary=tref[:,None]+(np.sin(theta[:,None])*receiver[None,:]+2*z*np.cos(theta[:,None]))/reference_speed
        bins+=(t[None,None,:]>=boundary[:,:,None])
    return bins


class DepthWeights:
    def __init__(self,acquisition):
        self.bins=torch.from_numpy(arrival_bins(acquisition)).to(acquisition.device)
        energy=acquisition.observed.double().square()
        self.den=[max(float(energy[self.bins==k].sum()),1e-20) for k in range(5)]

    def terms(self,difference,shot):
        bins=self.bins[shot]
        return [difference[bins==k].square().sum()/den for k,den in enumerate(self.den)]
