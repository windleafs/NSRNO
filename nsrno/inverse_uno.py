"""Independent data-conditioned spatial inverse UNO; outputs are initial maps."""
import math
import numpy as np
import torch
from torch import nn
from .uno_rf import UShape

HEIGHT,WIDTH=216,192
SCALE=(80.,100.,.2)

def load_input(row):
    value=np.zeros((62,HEIGHT,WIDTH),np.float32)
    for key,offset,mask in [('hf',0,60),('lf',30,61)]:
        if key in row['features']:
            value[offset:offset+30]=np.load(row['features'][key],mmap_mode='r')
            value[mask]=1
    return value

def load_batch(rows,device):
    x=torch.from_numpy(np.stack([load_input(r) for r in rows])).to(device)
    y=torch.from_numpy(np.stack([np.load(r['label']) for r in rows])).to(device)
    return x,y

class InverseUNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.config={'architecture':'all_angle_focus_spatial_inverse_UNO_v1','input_channels':62,'image_shape':[HEIGHT,WIDTH],
            'outputs':['c','rho','alpha_tissue'],'role':'coarse initialization for native physical joint inversion','focus_speeds_m_s':[1455.,1540.,1600.]}
        z,x=torch.meshgrid(torch.linspace(0,1,HEIGHT),torch.linspace(-1,1,WIDTH),indexing='ij')
        self.register_buffer('coordinates',torch.stack([x,z])[None])
        self.register_buffer('gel',(torch.arange(HEIGHT)[None,None,:,None]<5))
        self.stem=nn.Conv2d(64,24,1)
        self.operator=UShape([24,32,48,32,24],[(16,16),(12,12),(8,8),(12,12),(16,16)],checkpointing=False)
        self.head=nn.Sequential(nn.Conv2d(24,24,1),nn.GELU(),nn.Conv2d(24,3,1))
        nn.init.normal_(self.head[-1].weight,std=.001);nn.init.zeros_(self.head[-1].bias)

    def forward(self,features):
        features=features.float()
        raw=self.head(self.operator(self.stem(torch.cat([features,self.coordinates.expand(features.shape[0],-1,-1,-1)],1))))
        c=1540+140*torch.tanh(raw[:,0]);rho=1000+180*torch.tanh(raw[:,1])
        alpha=.002+1.998*torch.sigmoid(raw[:,2]+math.log(.498/1.5))
        result=torch.stack([c,rho,alpha],1)
        fixed=torch.tensor([1540.,1000.,.002],device=result.device)[None,:,None,None]
        return torch.where(self.gel,fixed,result)

def map_loss(prediction,target):
    scale=prediction.new_tensor(SCALE)[None,:,None,None]
    residual=(prediction-target)/scale
    return residual[:,:,5:].square().mean(),residual[:,:,5:].square().mean((0,2,3))

def pair_loss(prediction,target):
    delta=prediction[1,:,5:]-prediction[0,:,5:]
    exact=target[1,:,5:]-target[0,:,5:]
    floor=prediction.new_tensor([5.,5.,.02])
    variance=exact.square().mean((-2,-1)).clamp_min(floor.square())
    return ((delta-exact).square().mean((-2,-1))/variance).mean()
