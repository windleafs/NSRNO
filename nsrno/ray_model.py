"""Learn complex pulse responses after material-dependent coherent aggregation."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .forward_model import C_REF, RHO_REF
from .scatter_model import FineScatterForwardOperator
from .ray_propagation import RayPropagation
from .rf_representation import IQ_SAMPLES, RF_FS, RF_SAMPLES, decode


def make_background(medium, spec):
    c, rho = medium.float().unbind(1)
    h, w = c.shape[-2:]
    z = torch.arange(h, device=c.device) * spec["dx_m"] + spec["z0_m"]
    delay = torch.cumsum((c.reciprocal()-1/C_REF)*(z>=-1e-8)[None,:,None],1)*(2e6*spec["dx_m"])
    fields = torch.stack(((c/C_REF).log()/.05,(rho/RHO_REF).log()/.1,delay),1)
    return F.avg_pool2d(F.pad(fields,(0,-w%8,0,-h%8),mode="replicate"),8)


class RayArrivalForwardOperator(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.base = FineScatterForwardOperator(spec["base_spec"])
        self.base.requires_grad_(False)
        self.propagation = RayPropagation(spec)
        self.modes = 8
        self.kernel_size = (3,33)
        self.pulse_real = nn.Parameter(torch.zeros(self.modes,2,*self.kernel_size))
        self.pulse_imag = nn.Parameter(torch.zeros(self.modes,2,*self.kernel_size))
        self.context = nn.Sequential(nn.Conv2d(8,24,(3,9),padding=(1,4)),nn.SiLU(),
                                     nn.Conv2d(24,24,(3,9),padding=(1,4)),nn.SiLU(),
                                     nn.Conv2d(24,self.modes,1))
        self.register_buffer("feature_scales",torch.tensor(spec.get("ray_feature_scales",[1.,1.])).float())
        time = torch.arange(IQ_SAMPLES) * (RF_SAMPLES/RF_FS/IQ_SAMPLES)
        gate = .5-.5*torch.cos(math.pi*((time*1e6-16)/2).clamp(0,1))
        self.register_buffer("late_gate",gate)

    @property
    def template(self):
        return self.base.template

    @property
    def rf_scale(self):
        return self.base.rf_scale

    def _context(self, background, indices):
        b,a=indices.shape
        p=self.propagation
        time=torch.arange(IQ_SAMPLES,device=background.device)*(RF_SAMPLES/RF_FS/IQ_SAMPLES)
        theta=p.theta[indices].flatten()[:,None,None]
        tref=p.tref[indices].flatten()[:,None,None]
        x=p.rx[None,:,None]
        z=(C_REF*(time[None,None,:]-tref)-x*theta.sin())/(1+theta.cos())
        x0=self.spec["x0_m"]+3.5*self.spec["dx_m"]
        z0=self.spec["z0_m"]+3.5*self.spec["dx_m"]
        gx=(2*(x-x0)/(8*self.spec["dx_m"]*(background.shape[-1]-1))-1).expand_as(z)
        gz=2*(z-z0)/(8*self.spec["dx_m"]*(background.shape[-2]-1))-1
        sampled=F.grid_sample(background.repeat_interleave(a,0).float(),torch.stack((gx,gz),-1),
                              padding_mode="border",align_corners=True)
        ones=torch.ones_like(z)
        geometry=torch.stack((ones*(time[None,None,:]*1e6/30-1),
                              ones*x/.02,ones*theta.sin(),ones*theta.cos(),ones*tref*1e6),1)
        return torch.cat((sampled,geometry),1)

    def predict_cached(self, background, indices, features, baseline):
        b,a=indices.shape
        if features.shape[:4] != (b,a,2,2):
            raise ValueError("Ray features must be [B,A,source,complex,R,T]")
        normalized=features.float()/self.feature_scales[None,None,:,None,None,None]
        re,im=normalized.flatten(0,1).unbind(2)
        padding=(self.kernel_size[0]//2,self.kernel_size[1]//2)
        u=F.conv2d(re,self.pulse_real,padding=padding)-F.conv2d(im,self.pulse_imag,padding=padding)
        v=F.conv2d(im,self.pulse_real,padding=padding)+F.conv2d(re,self.pulse_imag,padding=padding)
        context=self.context(F.avg_pool2d(self._context(background,indices),4))
        context=F.interpolate(context,size=re.shape[-2:],mode="bilinear",align_corners=False)
        gates=(1+.5*torch.tanh(context))/math.sqrt(self.modes)
        correction=torch.stack(((u*gates).sum(1),(v*gates).sum(1)),1).float()
        correction=correction.reshape(b,a,2,self.spec["receivers"],IQ_SAMPLES)
        return baseline + self.rf_scale*self.late_gate*correction

    def predict_iq(self, medium, angle_indices=None):
        indices=torch.arange(len(self.spec["angles_deg"]),device=medium.device) if angle_indices is None else torch.as_tensor(angle_indices,device=medium.device)
        if indices.ndim==1:
            indices=indices[None].expand(len(medium),-1)
        if indices.ndim!=2 or len(indices)!=len(medium):
            raise ValueError("Expected shared or per-record angle indices")
        unique,inverse=torch.unique(indices,return_inverse=True)
        with torch.autocast(medium.device.type,enabled=False):
            features=self.propagation(medium,unique,material=self.spec["ray_delay"]=="material")
            # Canonical frozen baseline: float32, one record and all angles.
            # Its bf16 implementation is sensitive to angle-batch shape.
            all_baselines=torch.cat([self.base.predict_iq(record[None].float()) for record in medium],0)
        features=features[torch.arange(len(medium),device=medium.device)[:,None],inverse]
        baseline=all_baselines[torch.arange(len(medium),device=medium.device)[:,None],indices]
        return self.predict_cached(make_background(medium,self.spec),indices,features,baseline)

    def forward(self, medium, angle_indices=None):
        return decode(self.predict_iq(medium,angle_indices))
