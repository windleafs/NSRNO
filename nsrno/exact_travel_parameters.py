"""Travel-time initialization preserving an RF-derived physical material.

The correction is only initialization. The existing differentiable material
map and exact native propagator are unchanged.
"""
import numpy as np
import torch
from torch.nn import functional as F
from .travel_time_parameters import TravelTimeParameters
from .multiscale_parameters import MultiScaleParameters
from .reconstruct_time import Parameters


def coordinates_from_speed(c,start,dx):
    body=np.concatenate([np.full((1,c.shape[1]),.001),.001+np.cumsum(.5*(c[start+1:]+c[start:-1])/1540*dx,axis=0)],axis=0)
    top=np.broadcast_to((np.arange(start)*dx-.003)[:,None],(start,c.shape[1]))
    return np.concatenate([top,body],axis=0)


def interpolation_rows(coordinates,query):
    positions=torch.from_numpy(coordinates.T.copy())
    targets=torch.from_numpy(np.broadcast_to(query,(coordinates.shape[1],len(query))).copy())
    index=(torch.searchsorted(positions,targets,right=True)-1).clamp(0,coordinates.shape[0]-2).T.numpy()
    column=np.arange(coordinates.shape[1])[None,:]
    lower=coordinates[index,column];upper=coordinates[index+1,column]
    fraction=np.clip((query[:,None]-lower)/(upper-lower),0,1)
    if np.any(np.diff(index,axis=0)<1):raise ValueError('Physical samples must have disjoint or neighboring interpolation supports')
    return index,fraction


def sample(values,index,fraction):
    column=np.arange(values.shape[-1])[None,:]
    left=values[:,index,column];right=values[:,index+1,column]
    return left+(right-left)*fraction[None]


def project(values,target,index,fraction):
    """Minimum-norm correction through a batched tridiagonal BB^T solve."""
    rhs=target-sample(values,index,fraction);diagonal=(1-fraction)**2+fraction**2
    off=np.zeros_like(diagonal)
    off[1:]=np.where(index[1:]==index[:-1]+1,(1-fraction[1:])*fraction[:-1],0)
    pivot=diagonal.copy();solution=rhs.copy()
    for row in range(1,len(pivot)):
        multiplier=off[row]/pivot[row-1]
        pivot[row]-=multiplier*off[row];solution[:,row]-=multiplier[None]*solution[:,row-1]
    if np.any(pivot<=0):raise ValueError('Interpolation normal matrix is not positive definite')
    solution[:,-1]/=pivot[-1]
    for row in range(len(pivot)-2,-1,-1):solution[:,row]=(solution[:,row]-off[row+1][None]*solution[:,row+1])/pivot[row][None]
    correction=np.zeros_like(values);column=np.arange(values.shape[-1])[None,:]
    for channel in range(values.shape[0]):
        np.add.at(correction[channel],(index,column),(1-fraction)*solution[channel])
        np.add.at(correction[channel],(index+1,column),fraction*solution[channel])
    return values+correction


class ExactTravelTimeParameters(TravelTimeParameters):
    @torch.no_grad()
    def initialize(self,previous=None):
        packed=super().initialize(previous)
        if previous is None:return packed
        physical=Parameters(self.physical_shape,1)
        q=F.interpolate(previous[None],size=self.physical_shape,mode='bilinear',align_corners=True)[0]
        target=physical.material(q);desired_c=target[0].numpy().astype(np.float64)
        c,z,b=self.native_travel_fields(packed).numpy().astype(np.float64)
        query=self.physical_z.numpy();iterations=0
        for iterations in range(24):
            coordinate=coordinates_from_speed(c,self.travel_start,self.travel_dx)
            index,fraction=interpolation_rows(coordinate,query)
            error=float(np.max(abs(sample(c[None],index,fraction)[0]-desired_c)))
            if error<2e-6:break
            c=project(c[None],desired_c[None],index,fraction)[0];c[:self.travel_start]=1540.
        # Use the same float32 constrained speed representation as the live map.
        cq=torch.from_numpy(((c-1540)/80).astype(np.float32))
        dummy=torch.stack([cq,torch.zeros_like(cq),torch.zeros_like(cq)])
        c=(1540+80*self.constrain(self.unconstrain(dummy))[0]).numpy().astype(np.float64)
        coordinate=coordinates_from_speed(c,self.travel_start,self.travel_dx);index,fraction=interpolation_rows(coordinate,query)
        cp=sample(c[None],index,fraction)[0]
        rho=target[1].numpy().astype(np.float64)
        alpha=(target[2]-self.physical_edge).clamp_min(.001).numpy().astype(np.float64)
        zb=project(np.stack([z,b]),np.stack([cp*rho,alpha*cp/1540]),index,fraction)
        if np.any(zb<=0):raise ValueError('Exact initialization generated nonpositive impedance or attenuation')
        travel_target=torch.from_numpy(np.stack([(c-1540)/80,np.log(zb[0]/1540e3)/.1,np.log(zb[1]/.5)/.7]).astype(np.float32))
        travel_target[:,:self.travel_start]=0
        if torch.any(travel_target[0].abs()>=1.75) or torch.any(travel_target[1].abs()>=2.5) or torch.any((travel_target[2]<=-8)|(travel_target[2]>=2)):
            raise ValueError('Exact initialization exceeds the existing material bounds')
        raw=self.unconstrain(travel_target);coarse=F.interpolate(raw[None],size=self.small,mode='bilinear',align_corners=True)[0]
        packed=MultiScaleParameters.initialize(self,self.constrain(coarse))
        self.reference_raw=raw-F.interpolate(self.raw_fields(packed)[None],size=self.native,mode='bilinear',align_corners=True)[0]
        rendered=self.material(packed)
        errors=[float((a-b).abs().max()) for a,b in zip(rendered,target)]
        self.initialization_report={'speed_projection_iterations':iterations+1,'maximum_material_error':errors,'source':'RF-derived physical initialization; no truth read'}
        if errors[0]>.02 or errors[1]>.02 or errors[2]>2e-4:raise ValueError(f'Cannot preserve physical initialization: {errors}')
        return packed
