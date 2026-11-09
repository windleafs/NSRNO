"""Known-geometry full-angle complex RF focusing, independent of Geo-Flow."""
from pathlib import Path
import math
import numpy as np
import torch
from .rf_representation import decode

FOCUS_SPEEDS=(1455.,1540.,1600.)
HEIGHT,WIDTH=216,192

class FocusFeatures:
    def __init__(self,device='cuda:0'):
        self.device=device
        g=np.load('out/v5/time_triparam_v1/geometry.npz')
        self.x=torch.linspace(-.0191,.0191,WIDTH,device=device)
        self.z=torch.arange(HEIGHT,device=device)*.0002
        zz,xx=torch.meshgrid(self.z,self.x,indexing='ij')
        self.xx=xx.flatten();self.zz=zz.flatten();self.xe=torch.from_numpy(g['xe'].astype(np.float32)).to(device)
        delta=(self.xx[:,None]-self.xe[None]).abs();zgel=self.zz[:,None].clamp(max=.001);zbody=(self.zz[:,None]-.001).clamp_min(0)
        sine=delta/torch.sqrt(delta.square()+self.zz[:,None].square()+1e-10)
        near=torch.exp(-.5*(sine/.35)**4)
        far=torch.sigmoid(12*(sine-.5))*torch.exp(-.5*(sine/.95)**8)
        self.weight=torch.stack([near,far]);self.weight_sum=self.weight.sum(-1)
        self.receiver=[];self.transmit=[]
        angles=torch.linspace(-8,8,11,device=device)*math.pi/180
        x=g['x'];xe=g['xe'];cols=(x>=xe[0])&(x<=xe[-1])
        self.tref={}
        for f0 in [3e6,7.5e6]:
            delay=x[cols][None,:]*np.sin(np.linspace(-8,8,11)*np.pi/180)[:,None]/1540
            self.tref[f0]=torch.from_numpy((4/f0-delay.min(1)).astype(np.float32)).to(device)
        for c in FOCUS_SPEEDS:
            # Horizontal ray slowness is monotonic in horizontal displacement.
            lower=torch.zeros_like(delta)
            upper=torch.where(zbody>0,torch.full_like(delta,1/max(c,1540.)),torch.full_like(delta,1/1540.))*(1-1e-7)
            for _ in range(26):
                p=(lower+upper)*.5
                cg=torch.sqrt((1-(p*1540)**2).clamp_min(1e-12));cb=torch.sqrt((1-(p*c)**2).clamp_min(1e-12))
                horizontal=zgel*p*1540/cg+zbody*p*c/cb
                condition=horizontal<delta;lower=torch.where(condition,p,lower);upper=torch.where(condition,upper,p)
            p=(lower+upper)*.5
            tt=zgel/(1540*torch.sqrt((1-(p*1540)**2).clamp_min(1e-12)))+zbody/(c*torch.sqrt((1-(p*c)**2).clamp_min(1e-12)))
            tt=torch.where(self.zz[:,None]<1e-6,delta/1540,tt)
            self.receiver.append(tt)
            tx=self.xx[None]*angles.sin()[:,None]/1540+self.zz.clamp(max=.001)[None]*angles.cos()[:,None]/1540
            tx=tx+(self.zz-.001).clamp_min(0)[None]*torch.sqrt(1/c**2-(angles.sin()/1540)**2)[:,None]
            self.transmit.append(tx)
        self.receiver=torch.stack(self.receiver);self.transmit=torch.stack(self.transmit)

    @torch.no_grad()
    def __call__(self,iq,f0):
        iq=torch.as_tensor(iq,dtype=torch.float32,device=self.device)
        rf=decode(iq);n=rf.shape[-1]
        spectrum=torch.fft.fft(rf,dim=-1);hilbert=torch.zeros(n,device=self.device);hilbert[0]=1;hilbert[1:(n+1)//2]=2
        analytic=torch.fft.ifft(spectrum*hilbert,dim=-1)
        scale=(221.6426 if f0==3e6 else 117.0216)/20
        features=[];pcount=HEIGHT*WIDTH
        for k,c in enumerate(FOCUS_SPEEDS):
            sums=torch.zeros(2,pcount,device=self.device,dtype=torch.complex64)
            energy=torch.zeros(2,pcount,device=self.device);weight=torch.zeros_like(energy)
            for angle in range(11):
                sample=(self.receiver[k]+self.transmit[k,angle,:,None]+self.tref[f0][angle])*40e6
                valid=(sample>=0)&(sample<=n-1);index=sample.floor().clamp(0,n-2).long()
                fraction=(sample-index).clamp(0,1)
                first=torch.gather(analytic[angle],1,index.T).T
                second=torch.gather(analytic[angle],1,(index+1).T).T
                value=(first+(second-first)*fraction)*valid
                for group in range(2):
                    w=self.weight[group]*valid
                    sums[group]+=(value*w).sum(-1)
                    energy[group]+=(value.abs().square()*w).sum(-1)
                    weight[group]+=w.sum(-1)
            for group in range(2):
                denominator=weight[group].clamp_min(1e-12);mean=sums[group]/denominator
                rms=(energy[group]/denominator).sqrt()
                coherence=(sums[group].abs().square()/(denominator*energy[group]).clamp_min(1e-12)).clamp(0,1)
                coverage=(weight[group]/(11*self.weight_sum[group]).clamp_min(1e-12)).clamp(0,1)
                features.extend([torch.asinh(mean.real/scale),torch.asinh(mean.imag/scale),torch.log1p(rms/scale),coherence,coverage])
        result=torch.stack(features).reshape(30,HEIGHT,WIDTH)
        if not torch.isfinite(result).all():raise ValueError('Nonfinite RF features')
        return result.cpu().numpy().astype(np.float16)
