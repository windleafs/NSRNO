"""Complex frequency-band and orthogonal angular RF focusing statistics.

All eleven transmissions and all valid receivers contribute. Angular
components are statistics, not a claim of Born-exact parameter separation.
No material truth or Geo-Flow model enters this transform.
"""
import math
import numpy as np
import torch
from .focus_features import FocusFeatures,FOCUS_SPEEDS,HEIGHT,WIDTH
from .rf_representation import decode

CHANNELS=72
BANDS={3e6:((1.5e6,2.4e6),(2.4e6,3.3e6),(3.3e6,4.5e6)),
       7.5e6:((4e6,5.2e6),(5.2e6,6.4e6),(6.4e6,7.5e6))}


def angular_projection(sum0,sum_mu,energy,w0,w1,w2,expected,angle_floor=1e-6):
    """Project complex samples on 1 and a centered, unit-RMS angular mode."""
    den=w0.clamp_min(1e-12)
    center=w1/den
    variance=(w2/den-center.square()).clamp_min(0)
    mean=sum0/den
    contrast=(sum_mu/den-center*mean)/(variance+angle_floor).sqrt()
    rms=(energy/den).clamp_min(0).sqrt()
    coherence=(sum0.abs().square()/(den*energy).clamp_min(1e-12)).clamp(0,1)
    coverage=(w0/expected.clamp_min(1e-12)).clamp(0,1)
    return mean,contrast,rms,coherence,variance.sqrt(),coverage


class AngularSpectralFocus(FocusFeatures):
    def __init__(self,device='cuda:0'):
        super().__init__(device)
        delta=(self.xx[:,None]-self.xe[None]).abs()
        sign=(self.xe[None]-self.xx[:,None]).sign()
        zgel=self.zz[:,None].clamp(max=.001)
        zbody=(self.zz[:,None]-.001).clamp_min(0)
        self.outward_sin=[]
        self.outward_cos=[]
        self.incident_sin=[]
        self.incident_cos=[]
        theta=torch.linspace(-8,8,11,device=device)*math.pi/180
        for c in FOCUS_SPEEDS:
            lower=torch.zeros_like(delta)
            upper=torch.full_like(delta,1/max(c,1540.))*(1-1e-7)
            for _ in range(26):
                p=.5*(lower+upper)
                cg=(1-(1540*p).square()).clamp_min(1e-12).sqrt()
                cb=(1-(c*p).square()).clamp_min(1e-12).sqrt()
                horizontal=zgel*p*1540/cg+zbody*p*c/cb
                smaller=horizontal<delta
                lower=torch.where(smaller,p,lower);upper=torch.where(smaller,upper,p)
            p=.5*(lower+upper)
            local_c=torch.where(zbody>0,c,1540.)
            sine=sign*p*local_c
            sine=torch.where(self.zz[:,None]<1e-6,sign,sine).clamp(-1,1)
            self.outward_sin.append(sine)
            self.outward_cos.append(-(1-sine.square()).clamp_min(0).sqrt())
            incoming=torch.where(self.zz[None]>.001,c/1540.,1.)*theta.sin()[:,None]
            self.incident_sin.append(incoming)
            self.incident_cos.append((1-incoming.square()).clamp_min(0).sqrt())
        self.apodization=self.weight.sum(0)
        self.expected=11*self.apodization.sum(-1)

    @torch.no_grad()
    def __call__(self,iq,f0):
        if float(f0) not in BANDS:raise ValueError('Only existing 3/7.5 MHz acquisitions are supported')
        iq=torch.as_tensor(iq,dtype=torch.float32,device=self.device)
        rf=decode(iq);n=rf.shape[-1]
        frequency=torch.fft.fftfreq(n,1/40e6,device=self.device)
        masks=torch.stack([(frequency>=lo)&(frequency<hi) for lo,hi in BANDS[float(f0)]])
        spectrum=torch.fft.fft(rf,dim=-1)
        analytic=torch.fft.ifft(spectrum[None]*(2*masks[:,None,None,:]),dim=-1)
        del spectrum,rf,iq
        scale=(221.6426 if f0==3e6 else 117.0216)/20
        features=[];pcount=HEIGHT*WIDTH
        for k,_ in enumerate(FOCUS_SPEEDS):
            sum0=torch.zeros((3,pcount),device=self.device,dtype=torch.complex64)
            sum_mu=torch.zeros_like(sum0)
            energy=torch.zeros((3,pcount),device=self.device)
            w0=torch.zeros(pcount,device=self.device)
            w1=torch.zeros_like(w0);w2=torch.zeros_like(w0)
            for angle in range(11):
                sample=(self.receiver[k]+self.transmit[k,angle,:,None]+self.tref[f0][angle])*40e6
                valid=(sample>=0)&(sample<=n-1)
                index=sample.floor().clamp(0,n-2).long()
                fraction=(sample-index).clamp(0,1)
                first=torch.gather(analytic[:,angle],2,index.T[None].expand(3,-1,-1)).transpose(1,2)
                second=torch.gather(analytic[:,angle],2,(index+1).T[None].expand(3,-1,-1)).transpose(1,2)
                value=(first+(second-first)*fraction[None])*valid[None]
                mu=(self.incident_sin[k][angle,:,None]*self.outward_sin[k]+
                    self.incident_cos[k][angle,:,None]*self.outward_cos[k])
                w=self.apodization*valid
                sum0+=(value*w[None]).sum(-1)
                sum_mu+=(value*(w*mu)[None]).sum(-1)
                energy+=(value.abs().square()*w[None]).sum(-1)
                w0+=w.sum(-1);w1+=(w*mu).sum(-1);w2+=(w*mu.square()).sum(-1)
                del sample,valid,index,fraction,first,second,value,mu,w
            mean,contrast,rms,coherence,spread,coverage=angular_projection(
                sum0,sum_mu,energy,w0,w1,w2,self.expected)
            for band in range(3):
                features.extend([torch.asinh(mean[band].real/scale),torch.asinh(mean[band].imag/scale),
                    torch.asinh(contrast[band].real/scale),torch.asinh(contrast[band].imag/scale),
                    torch.log1p(rms[band]/scale),coherence[band],spread,coverage])
        result=torch.stack(features).reshape(CHANNELS,HEIGHT,WIDTH)
        if not torch.isfinite(result).all():raise RuntimeError('Nonfinite angular/spectral features')
        return result.cpu().numpy().astype(np.float16)
