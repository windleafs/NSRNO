"""Same inverse UNO backbone with complex angular/spectral inputs."""
import numpy as np
import torch
from torch import nn
from .inverse_uno import InverseUNO,map_loss,pair_loss,HEIGHT,WIDTH
from .angular_spectral_features import CHANNELS

INPUT_CHANNELS=2*CHANNELS+2


def load_input(row):
    value=np.zeros((INPUT_CHANNELS,HEIGHT,WIDTH),np.float32)
    for source,offset,present in [('hf',0,2*CHANNELS),('lf',CHANNELS,2*CHANNELS+1)]:
        if source in row['features']:
            feature=np.load(row['features'][source],mmap_mode='r')
            if feature.shape!=(CHANNELS,HEIGHT,WIDTH):raise ValueError('Wrong angular/spectral feature shape')
            value[offset:offset+CHANNELS]=feature
            value[present]=1
    return value


def load_batch(rows,device):
    x=torch.from_numpy(np.stack([load_input(r) for r in rows])).to(device)
    y=torch.from_numpy(np.stack([np.load(r['label']) for r in rows])).to(device)
    return x,y


class AngularSpectralInverseUNO(InverseUNO):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv2d(INPUT_CHANNELS+2,24,1)
        self.config={**self.config,'architecture':'angular_spectral_inverse_UNO_v1',
            'input_channels':INPUT_CHANNELS,'feature_channels_per_source':CHANNELS,
            'backbone_unchanged':True,'point_acquisitions_used':False,
            'input_statistics':'complex subbands plus orthogonal bistatic angular mode'}
