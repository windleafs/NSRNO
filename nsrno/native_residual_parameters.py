"""Coarse updates preserve the RF-derived native fine initialization."""
import torch
from torch.nn import functional as F
from .multiscale_parameters import MultiScaleParameters


class NativeResidualParameters(MultiScaleParameters):
    def initialize(self,previous=None):
        value=torch.zeros(3,*self.native) if previous is None else F.interpolate(previous[None],size=self.native,mode='bilinear',align_corners=True)[0]
        raw_native=self.unconstrain(value)
        coarse_value=F.interpolate(value[None],size=self.small,mode='bilinear',align_corners=True)[0]
        coarse_raw=self.unconstrain(coarse_value)
        self.reference_raw=raw_native-F.interpolate(coarse_raw[None],size=self.native,mode='bilinear',align_corners=True)[0]
        residual=coarse_raw;packed=[]
        for shape in self.levels:
            level=residual.mean((-2,-1),keepdim=True) if shape==(1,1) else F.interpolate(residual[None],size=shape,mode='bilinear',align_corners=True)[0]
            packed.append(level.flatten())
            residual=residual-F.interpolate(level[None],size=self.small,mode='bilinear',align_corners=True)[0]
        return torch.cat(packed)

    def fields(self,packed):
        raw=F.interpolate(self.raw_fields(packed)[None],size=self.native,mode='bilinear',align_corners=True)[0]+self.reference_raw
        return self.constrain(raw)
