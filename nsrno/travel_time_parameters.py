"""Vertical travel-time c/Z/attenuation coordinates for exact 2-D FWI."""
import numpy as np
import torch
from torch.nn import functional as F
from .multiscale_parameters import MultiScaleParameters
from .reconstruct_time import Parameters

def interpolate_monotone(coordinates,values,query):
    # coordinates [z,x], values [channel,z,x], common query [output_z].
    positions=coordinates.T.contiguous()
    targets=query[None].expand(positions.shape[0],-1).contiguous()
    index=(torch.searchsorted(positions.detach(),targets,right=True)-1).clamp(0,positions.shape[1]-2)
    lower=torch.gather(positions,1,index);upper=torch.gather(positions,1,index+1)
    fraction=((targets-lower)/(upper-lower)).clamp(0,1).to(values.dtype)
    by_column=values.permute(0,2,1)
    idx=index[None].expand(values.shape[0],-1,-1)
    left=torch.gather(by_column,2,idx);right=torch.gather(by_column,2,idx+1)
    return (left+(right-left)*fraction[None]).permute(0,2,1).contiguous()

class TravelTimeParameters(MultiScaleParameters):
    def __init__(self,shape,step):
        super().__init__((1326,shape[1]),step)
        self.physical_shape=shape;self.physical_dx=50e-6;self.travel_dx=40e-6
        self.travel_start=100;self.physical_start=80
        self.travel_z=torch.arange(1326,dtype=torch.float64)*self.travel_dx-.003
        self.physical_z=torch.arange(shape[0],dtype=torch.float64)*self.physical_dx-.003
        edge=torch.from_numpy(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')).float()
        self.physical_edge=(edge[0] if edge.ndim==2 else edge)[None].expand(shape)
        self.reference_raw=torch.zeros(3,*self.native)

    def native_travel_fields(self,packed):
        raw=F.interpolate(self.raw_fields(packed)[None],size=self.native,mode='bilinear',align_corners=True)[0]+self.reference_raw
        q=self.constrain(raw)
        c=1540+80*q[0];impedance=1540e3*torch.exp(.1*q[1]);beta=.5*torch.exp(.7*q[2])
        mask=(torch.arange(self.native[0])[:,None]>=self.travel_start)
        return torch.stack([torch.where(mask,c,1540.),torch.where(mask,impedance,1540e3),torch.where(mask,beta,.002)])

    def material(self,packed):
        c,impedance,beta=self.native_travel_fields(packed)
        increments=.5*(c[self.travel_start+1:].double()+c[self.travel_start:-1].double())/1540*self.travel_dx
        start=torch.full((1,c.shape[1]),.001,dtype=torch.float64)
        body=torch.cat([start,start+torch.cumsum(increments,0)],0)
        top=self.travel_z[:self.travel_start,None].expand(-1,c.shape[1])
        physical_coordinate=torch.cat([top,body],0)
        cp,zp,bp=interpolate_monotone(physical_coordinate,torch.stack([c,impedance,beta]),self.physical_z)
        rho=zp/cp;alpha=bp*1540/cp+self.physical_edge
        mask=torch.arange(self.physical_shape[0])[:,None]>=self.physical_start
        return torch.stack([torch.where(mask,cp,1540.),torch.where(mask,rho,1000.),torch.where(mask,alpha,.002+self.physical_edge)])

    @torch.no_grad()
    def initialize(self,previous=None):
        if previous is None:
            self.reference_raw.zero_();return super().initialize(None)
        physical=Parameters(self.physical_shape,1)
        q=F.interpolate(previous[None],size=self.physical_shape,mode='bilinear',align_corners=True)[0]
        material=physical.material(q);c,rho,total_alpha=material
        alpha=(total_alpha-self.physical_edge).clamp_min(.001)
        increments=.5*(1/c[self.physical_start+1:].double()+1/c[self.physical_start:-1].double())*1540*self.physical_dx
        start=torch.full((1,c.shape[1]),.001,dtype=torch.float64)
        body=torch.cat([start,start+torch.cumsum(increments,0)],0)
        top=self.physical_z[:self.physical_start,None].expand(-1,c.shape[1])
        travel_coordinate=torch.cat([top,body],0)
        ct,zt,bt=interpolate_monotone(travel_coordinate,torch.stack([c,c*rho,alpha*c/1540]),self.travel_z)
        target=torch.stack([(ct-1540)/80,torch.log(zt/1540e3)/.1,torch.log(bt.clamp_min(.001)/.5)/.7])
        target[:,:self.travel_start]=0
        raw=self.unconstrain(target)
        coarse=F.interpolate(raw[None],size=self.small,mode='bilinear',align_corners=True)[0]
        packed=super().initialize(self.constrain(coarse))
        self.reference_raw=raw-F.interpolate(self.raw_fields(packed)[None],size=self.native,mode='bilinear',align_corners=True)[0]
        return packed

    def checkpoint_fields(self,packed):
        c,rho,total_alpha=self.material(packed)
        tissue=(total_alpha-self.physical_edge).clamp_min(.001)
        result=torch.stack([(c-1540)/80,torch.log(c*rho/1540e3)/.1,torch.log(tissue/.5)/.7])
        result=torch.where((torch.arange(self.physical_shape[0])[:,None]<self.physical_start)[None],0.,result)
        return result
