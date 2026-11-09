"""Unrestricted fine fields with explicit global and coarse search directions.

The physical model and material bounds are unchanged. Coarse directions are a
preconditioner, not a smooth-speed constraint: the finest field remains free.
"""
import math
import torch
from torch.nn import functional as F
from .reconstruct_time import Parameters

class MultiScaleParameters(Parameters):
    def __init__(self,shape,step):
        super().__init__(shape,step)
        self.levels=[(1,1),(8,10),(32,36),self.small]
        self.counts=[3*h*w for h,w in self.levels]

    @staticmethod
    def constrain(raw):
        return torch.stack([1.75*torch.tanh(raw[0]/1.75),2.5*torch.tanh(raw[1]/2.5),
            -3+5*torch.tanh(raw[2]/3.2+math.atanh(.6))])

    @staticmethod
    def unconstrain(value):
        return torch.stack([1.75*torch.atanh((value[0]/1.75).clamp(-.999999,.999999)),
            2.5*torch.atanh((value[1]/2.5).clamp(-.999999,.999999)),
            3.2*(torch.atanh(((value[2]+3)/5).clamp(-.999999,.999999))-math.atanh(.6))])

    def raw_fields(self,packed):
        raw=0.;begin=0
        for shape,count in zip(self.levels,self.counts):
            level=packed[begin:begin+count].reshape(1,3,*shape)
            raw=raw+F.interpolate(level,size=self.small,mode='bilinear',align_corners=True)[0]
            begin+=count
        return raw

    def fields(self,packed):return self.constrain(self.raw_fields(packed))

    def initialize(self,previous=None):
        if previous is None:value=torch.zeros(3,*self.small)
        else:value=F.interpolate(previous[None],size=self.small,mode='bilinear',align_corners=True)[0]
        residual=self.unconstrain(value);packed=[]
        for shape in self.levels:
            if shape==(1,1):level=residual.mean((-2,-1),keepdim=True)
            else:level=F.interpolate(residual[None],size=shape,mode='bilinear',align_corners=True)[0]
            packed.append(level.flatten())
            residual=residual-F.interpolate(level[None],size=self.small,mode='bilinear',align_corners=True)[0]
        return torch.cat(packed)

    def material(self,packed):return super().material(self.fields(packed))
    def regularization(self,packed,weight):return super().regularization(self.fields(packed),weight)
    def bounds(self):return [(None,None)]*sum(self.counts)

    def coordinate_norms(self,gradient):
        norms=torch.zeros(3);begin=0
        for count,shape in zip(self.counts,self.levels):
            norms+=gradient[begin:begin+count].reshape(3,*shape).square().sum((-2,-1))
            begin+=count
        return norms.sqrt().tolist()
