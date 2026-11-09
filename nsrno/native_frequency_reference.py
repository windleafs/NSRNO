"""Diagnostic pressure operator matching V5 split-field PML and relaxation."""
import cmath,math
import numpy as np
import torch
from torch.nn import functional as F
from nsrno.wave_operator import C0,RHO0

class NativeFrequencyReference:
    def __init__(self,model,record,metadata,frequency):
        self.model=model; self.dx=model.dx; self.dt=metadata['dt_s']; self.f=frequency
        self.D=cmath.exp(2j*math.pi*frequency*self.dt); self.q=1/self.D
        self.k0=2*math.sin(math.pi*frequency*self.dt)/(self.dt*C0)
        self.nt=metadata['nt']; self.taus=metadata['fit']['taus_s']
        self.record=record; self.meta=metadata
        self.mask=self.pad(np.ones_like(record['c'],dtype=np.float32))
        def coefficient(key): return F.pad(torch.from_numpy(record[key]).cuda()[None,None],model.pad,mode='replicate')[0,0]
        self.c,self.rho,self.eta1,self.eta2=(coefficient(k) for k in ('c','rho','eta1','eta2'))
        self.a=RHO0/self.rho
        self.ax=2*self.a*torch.roll(self.a,-1,-1)/(self.a+torch.roll(self.a,-1,-1))*self.mask
        self.az=2*self.a*torch.roll(self.a,-1,-2)/(self.a+torch.roll(self.a,-1,-2))*self.mask
        xx=torch.arange(model.W,device='cuda')-model.px; zz=torch.arange(model.H,device='cuda')-model.pz
        def damping(position,size,stagger):
            left=torch.where(position<metadata['pml'],(1-(position+stagger)/metadata['pml']).pow(4),0.)
            right=torch.where(position>=size-metadata['pml'],(1-(size-1-position-stagger)/metadata['pml']).pow(4),0.)
            return (left+right)*2*C0/self.dx
        self.arx=1-self.dt*damping(xx,model.w,0.)[None,:]
        self.arz=1-self.dt*damping(zz,model.h,0.)[:,None]
        self.avx=1-self.dt*damping(xx,model.w,.5)[None,:]
        self.avz=1-self.dt*damping(zz,model.h,.5)[:,None]
        self.srx=(self.D-1)/(self.D-self.arx); self.srz=(self.D-1)/(self.D-self.arz)
        self.svx=(self.D-1)/(self.D-self.avx); self.svz=(self.D-1)/(self.D-self.avz)
        self.mx=(self.arx!=1).float(); self.mz=(self.arz!=1).float()
        self.gvx=1+(1-self.avx)*self.q; self.gvz=1+(1-self.avz)*self.q
        self.amplitude=(1+(self.eta1+self.eta2)/2).square()
        self.H=sum(e/(self.amplitude*(1+t*(self.D-1)/self.dt)) for e,t in zip((self.eta1,self.eta2),self.taus))
        modulus=self.amplitude*(1-self.H)/(1-(1-self.q)*self.H)
        self.mass=self.a*(C0/self.c).square()/modulus

    def pad(self,array): return F.pad(torch.from_numpy(np.array(array)).cuda(),self.model.pad)
    def div(self,x,z):
        return self.srx*torch.fft.ifft2(-self.model.Dx.conj()*torch.fft.fft2(x))+self.srz*torch.fft.ifft2(-self.model.Dz.conj()*torch.fft.fft2(z))
    def spatial(self,p):
        spectrum=torch.fft.fft2(p)
        x=self.ax*self.svx*torch.fft.ifft2(self.model.Dx*spectrum)
        z=self.az*self.svz*torch.fft.ifft2(self.model.Dz*spectrum)
        # The generated kernel forms pressure from the provisional undamped
        # density/velocity, then overwrites PML states. Retain that exact order.
        def part(flux,gain,mask,ar,symbol):
            regular=torch.fft.ifft2(-symbol.conj()*torch.fft.fft2(flux))
            provisional=torch.fft.ifft2(-symbol.conj()*torch.fft.fft2(gain*flux))
            return mask/(self.D-ar)*regular+(1+(1-mask)/(self.D-ar))*provisional
        return (self.D-1)/self.D/self.k0**2*(part(x,self.gvx,self.mx,self.arx,self.model.Dx)+part(z,self.gvz,self.mz,self.arz,self.model.Dz))
    def __call__(self,p):
        return self.mask*(self.spatial(p)+self.mass*p)+(1-self.mask)*p
    def endpoint(self):
        factor=self.q**self.nt*self.dt/(2401/40e6)
        ep,evx,evz,erx,erz,es1,es2=[self.pad(self.record[k+'_final'])*factor for k in ('p','vx','vy','rhox','rhoy','S1','S2')]
        es=es1+es2
        memory=-self.H/self.q*(ep+es)-sum((1-self.dt/t)*s/(1-(1-self.dt/t)*self.q) for s,t in zip((es1,es2),self.taus))
        density=(self.q*memory/(1-self.H)-es)/(self.c.square()*self.amplitude)
        spectrum=torch.fft.fft2(ep)
        def part(ev,er,ax,av,ar,gain,mask,symbol):
            gradient=torch.fft.ifft2(symbol*spectrum)
            velocity=(self.dt*self.D/(self.D-av)*(ax/RHO0)*gradient-av*self.D/(self.D-av)*ev)*self.mask
            provisional=gain*velocity-(1-av)*ev
            div_v=torch.fft.ifft2(-symbol.conj()*torch.fft.fft2(velocity))
            div_provisional=torch.fft.ifft2(-symbol.conj()*torch.fft.fft2(provisional))
            stored=-self.dt*self.D/(self.D-ar)*self.rho*(mask*div_v+(1-mask)*div_provisional)-ar*self.D/(self.D-ar)*er
            return self.q*stored-er-self.dt*self.rho*div_provisional
        density_endpoint=part(evx,erx,self.ax,self.avx,self.arx,self.gvx,self.mx,self.model.Dx)+part(evz,erz,self.az,self.avz,self.arz,self.gvz,self.mz,self.model.Dz)
        return self.mask*self.a*C0*C0*(density_endpoint-density)
