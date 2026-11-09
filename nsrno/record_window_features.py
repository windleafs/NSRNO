"""Identical angular/spectral statistics on 60 or 90 us 3 MHz RF."""
import numpy as np
import torch
from .angular_spectral_features import AngularSpectralFocus,angular_projection,BANDS,CHANNELS
from .focus_features import FOCUS_SPEEDS,HEIGHT,WIDTH
from .rf_representation import decode


class RecordWindowFocus(AngularSpectralFocus):
    @torch.no_grad()
    def from_iq60(self,iq):
        iq=torch.as_tensor(iq,dtype=torch.float32,device=self.device)
        return self.from_rf(decode(iq))

    @torch.no_grad()
    def from_rf(self,rf):
        rf=torch.as_tensor(rf,dtype=torch.float32,device=self.device)
        if rf.shape[:2]!=(11,192) or rf.shape[-1] not in [2401,3601]:
            raise ValueError('Require 11 x 192 channels at 40 MHz for 60 or 90 us')
        n=rf.shape[-1]
        frequency=torch.fft.fftfreq(n,1/40e6,device=self.device)
        masks=torch.stack([(frequency>=lo)&(frequency<hi) for lo,hi in BANDS[3e6]])
        spectrum=torch.fft.fft(rf,dim=-1)
        analytic=torch.fft.ifft(spectrum[None]*(2*masks[:,None,None,:]),dim=-1)
        del spectrum,rf
        scale=221.6426/20
        features=[];pcount=HEIGHT*WIDTH
        for k,_ in enumerate(FOCUS_SPEEDS):
            sum0=torch.zeros((3,pcount),device=self.device,dtype=torch.complex64)
            sum_mu=torch.zeros_like(sum0)
            energy=torch.zeros((3,pcount),device=self.device)
            w0=torch.zeros(pcount,device=self.device)
            w1=torch.zeros_like(w0);w2=torch.zeros_like(w0)
            for angle in range(11):
                sample=(self.receiver[k]+self.transmit[k,angle,:,None]+self.tref[3e6][angle])*40e6
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
        if not torch.isfinite(result).all():raise RuntimeError('Nonfinite record-window features')
        return result.cpu().numpy().astype(np.float16)
