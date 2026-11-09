"""Variable-density Helmholtz residual network on the native material grid.

FFT in time uses exp(-i omega t); outgoing spatial waves use exp(-i k r).
The learned component is a preconditioner, not a replacement for the PDE.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

C0=1540.; RHO0=1000.

class WaveResidualOperator(nn.Module):
    def __init__(self,spec,iterations=8,preconditioner='shifted'):
        super().__init__(); self.spec=spec; self.iterations=iterations; self.preconditioner=preconditioner
        h,w=spec['medium_shape'][-2:]; self.h,self.w=h,w
        self.H=math.ceil((h+64)/32)*32; self.W=math.ceil((w+64)/32)*32
        self.pz=(self.H-h)//2; self.px=(self.W-w)//2
        self.pad=(self.px,self.W-w-self.px,self.pz,self.H-h-self.pz)
        dx=spec['dx_m']; self.dx=dx
        qx=2*math.pi*torch.fft.fftfreq(self.W); qz=2*math.pi*torch.fft.fftfreq(self.H)
        # Eighth-order staggered gradient; divergence is its negative adjoint.
        coefficients=(1225/1024,-245/3072,49/5120,-5/7168)
        def symbol(q):
            return 2j*sum(a*torch.sin((i+.5)*q) for i,a in enumerate(coefficients))*torch.exp(.5j*q)/dx
        self.register_buffer('Dx',symbol(qx)[None,:]); self.register_buffer('Dz',symbol(qz)[:,None])
        self.register_buffer('lap_symbol',-(self.Dx.abs().square()+self.Dz.abs().square()))
        self.register_buffer('kx',qx/dx)
        self.register_buffer('x',torch.arange(self.W)*dx+spec['x0_m']-self.px*dx)
        self.register_buffer('z',torch.arange(self.H)*dx+spec['z0_m']-self.pz*dx)
        iz=torch.minimum(torch.arange(self.H),torch.arange(self.H).flip(0))[:,None]
        ix=torch.minimum(torch.arange(self.W),torch.arange(self.W).flip(0))[None,:]
        self.register_buffer('sponge',2*((48-torch.minimum(iz,ix)).clamp_min(0)/48).square())
        self.register_buffer('rx_cols',torch.round((torch.arange(192)*.0002-.0191-spec['x0_m'])/dx).long()+self.px)
        self.face=int(round(-spec['z0_m']/dx))+self.pz
        self.register_buffer('angles',torch.tensor(spec['angles_deg'])*math.pi/180)
        self.register_buffer('trefs',torch.tensor(spec['source_tref_s']))
        self.context=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.Tanh(),nn.Conv2d(12,2,3,padding=1))
        nn.init.zeros_(self.context[-1].weight); nn.init.zeros_(self.context[-1].bias)
        self.spectral=nn.Parameter(torch.zeros(2,4))
        self.log_shift=nn.Parameter(torch.tensor(math.log(.15)))
        self.response=nn.Parameter(torch.tensor([[1.,0.,0.],[0.,0.,0.]]))
        self.alpha_log=nn.Parameter(torch.tensor(math.log(.5)))

    def setup(self,medium,frequency,learned=True):
        if medium.shape[0]!=1: raise ValueError('One native medium per frequency solve')
        c,rho=medium.float().unbind(1)
        # Full-resolution coefficients; no material smoothing or density freezing.
        a=RHO0/rho; b=a*(C0/c).square()
        a=F.pad(a[None],self.pad,mode='replicate')[0,0]
        b=F.pad(b[None],self.pad,mode='replicate')[0,0]
        c=F.pad(c[None],self.pad,mode='replicate')[0,0]
        omega=2*math.pi*frequency
        k0=(2*math.sin(omega*2.5e-9/2)/2.5e-9 if self.spec.get('attenuation')=='relaxation' else omega)/C0
        alpha=self.alpha_log.exp().clamp(.01,3.)*100*math.log(10)/20*(frequency/1e6)**1.1
        tissue=(self.z[:,None]>=0).float()*(1-torch.exp(-((c/C0).log()/.03).square()-(a.log()/.05).square()))
        # Effective attenuation inferred by a shared scalar, not a hidden alpha label.
        if self.spec.get('attenuation')=='relaxation':
            eta=[self.alpha_log.exp().clamp(.01,3.)*tissue*c/C0*v for v in self.spec['eta_per_unit_alpha']]
            amplitude=(1+.5*(eta[0]+eta[1])).square()
            step=torch.exp(torch.as_tensor(1j*omega*2.5e-9,device=c.device))
            h=sum(e/(amplitude*(1+t*(step-1)/2.5e-9)) for e,t in zip(eta,self.spec['taus_s']))
            modulus=amplitude*(1-h)/(1-(1-1/step)*h)
            kappa_complex=b/modulus-1j*self.sponge
        else:
            kappa_complex=b*(1-1j*alpha*tissue*c/(2*math.pi*frequency)).square()-1j*self.sponge
        ax=2*a*torch.roll(a,-1,-1)/(a+torch.roll(a,-1,-1))
        az=2*a*torch.roll(a,-1,-2)/(a+torch.roll(a,-1,-2))
        data=(ax,az,kappa_complex,k0)
        context=F.avg_pool2d(torch.stack(((c/C0).log(),-a.log(),torch.ones_like(a)*(frequency/6e6-1)))[None],4)
        if learned:
            gain=F.interpolate(self.context(context),size=(self.H,self.W),mode='bilinear',align_corners=False)[0]
            gain=(1+.5*torch.tanh(gain[0]))*torch.exp(.4j*torch.tanh(gain[1]))
            shift=self.log_shift.exp().clamp(.03,1.)
        else: gain=1.; shift=.15
        if self.preconditioner in ('free_space','free_space_tapered'):
            # Open-space Green function: every residual point can reach every
            # receiver immediately; 2x padding prevents circular convolution.
            zz=torch.minimum(torch.arange(2*self.H,device=c.device),torch.arange(2*self.H,device=c.device).flip(0)+1)*self.dx
            xx=torch.minimum(torch.arange(2*self.W,device=c.device),torch.arange(2*self.W,device=c.device).flip(0)+1)*self.dx
            kr=k0*torch.sqrt(zz[:,None].square()+xx[None,:].square()).clamp_min(self.dx/math.sqrt(math.pi))
            green=.25*torch.complex(torch.special.bessel_y0(kr),torch.special.bessel_j0(kr))*(k0*self.dx)**2
            green_spectrum=torch.fft.fft2(green)
            if self.preconditioner=='free_space_tapered':
                # Match the outgoing correction to this operator's quadratic
                # absorbing layer (WKB amplitude), before residual line search.
                gain=gain*torch.exp(-k0*48*self.dx/3*(self.sponge/2).pow(1.5))
            qx=2*math.pi*torch.fft.fftfreq(2*self.W,d=self.dx,device=c.device)/k0
            qz=2*math.pi*torch.fft.fftfreq(2*self.H,d=self.dx,device=c.device)/k0
            q2=qx[None,:].square()+qz[:,None].square()
            angular=qx[None,:].square()/(q2+1e-8)
        elif self.preconditioner in ('shifted','shifted_tapered'):
            q2=-self.lap_symbol/k0**2
            angular=self.Dx.abs().square()/(self.Dx.abs().square()+self.Dz.abs().square()+1)
            green_spectrum=1/(1-q2-1j*shift)
            if self.preconditioner=='shifted_tapered':
                gain=gain*torch.exp(-k0*48*self.dx/3*(self.sponge/2).pow(1.5))
        else: raise ValueError(self.preconditioner)
        basis=torch.stack((torch.ones_like(q2),q2/(1+q2),angular,angular*q2/(1+q2)))
        mod=torch.einsum('ck,khw->chw',self.spectral,basis) if learned else torch.zeros((2,*q2.shape),device=c.device)
        multiplier=torch.exp(.5*torch.tanh(mod[0])+.5j*torch.tanh(mod[1]))*green_spectrum
        return data,gain,multiplier

    def apply_operator(self,u,data):
        ax,az,b,k0=data; spectrum=torch.fft.fft2(u)
        gx=torch.fft.ifft2(self.Dx*spectrum); gz=torch.fft.ifft2(self.Dz*spectrum)
        div=torch.fft.ifft2(-self.Dx.conj()*torch.fft.fft2(ax*gx)-self.Dz.conj()*torch.fft.fft2(az*gz))
        return div/k0**2+b*u

    def incident(self,frequency,angle):
        theta=self.angles[angle]; omega=2*math.pi*frequency
        k0=(2*math.sin(omega*2.5e-9/2)/2.5e-9 if self.spec.get('attenuation')=='relaxation' else omega)/C0
        aperture=torch.where(self.x.abs()<=.0191,.5*(1+torch.cos(math.pi*self.x/.0191)),0.)
        boundary=aperture*torch.exp(-1j*k0*self.x*theta.sin())
        # Discrete dispersion relation for the same staggered stencil, avoiding
        # cumulative continuum/stencil phase error in the incident field.
        sx=self.Dx[0].abs().square(); desired=(k0**2-sx).clamp_min(0).sqrt()*self.dx
        q=torch.as_tensor(desired).clamp(0,math.pi)
        coeff=(1225/1024,-245/3072,49/5120,-5/7168)
        for _ in range(6):
            value=2*sum(a*torch.sin((j+.5)*q) for j,a in enumerate(coeff))
            derivative=2*sum(a*(j+.5)*torch.cos((j+.5)*q) for j,a in enumerate(coeff))
            q=(q-(value-desired)/derivative.clamp_min(.05)).clamp(0,math.pi)
        kz=torch.complex(q/self.dx,-(sx-k0**2).clamp_min(0).sqrt())
        propagation=torch.exp(-1j*self.z.abs()[:,None]*kz[None,:])
        return torch.fft.ifft(torch.fft.fft(boundary)[None,:]*propagation,dim=-1)

    def _iteration(self,u,rhs,ax,az,b,k0,gain,multiplier):
        data=(ax,az,b,k0); residual=rhs-self.apply_operator(u,data)
        direction=gain*torch.fft.ifft2(multiplier*torch.fft.fft2(residual,s=multiplier.shape))[:self.H,:self.W]
        image=self.apply_operator(direction,data)
        # Exact complex least-squares line search: ||r-alpha Ld|| <= ||r||.
        alpha=(image.conj()*residual).sum()/image.abs().square().sum().clamp_min(1e-24)
        return u+alpha*direction

    def forward(self,medium,frequency,angle,learned=True,return_trace=False):
        data,gain,multiplier=self.setup(medium,frequency,learned)
        incident=self.incident(frequency,angle)
        ax,az,b,k0=data
        # Scattered equation, including density gradients and compressibility.
        contrast=(ax-1,az-1,b-(1-1j*self.sponge),k0)
        rhs=-self.apply_operator(incident,contrast)
        u=torch.zeros_like(incident); trace=[]
        denominator=rhs.abs().square().mean().clamp_min(1e-20)
        for iteration in range(self.iterations):
            if self.training and torch.is_grad_enabled():
                u=checkpoint(self._iteration,u,rhs,*data,gain,multiplier,use_reentrant=False)
            else: u=self._iteration(u,rhs,*data,gain,multiplier)
            if return_trace:
                trace.append(float(((rhs-self.apply_operator(u,data)).abs().square().mean()/denominator).sqrt().detach()))
        residual_loss=(rhs-self.apply_operator(u,data)).abs().square().mean()/denominator
        prediction=u[self.face,self.rx_cols]*self.source_factor(frequency,angle)
        return prediction,residual_loss,trace

    def source_factor(self,frequency,angle):
        # Gaussian-sine source Fourier coefficient on the unchanged V5 clock.
        source_f0=self.spec.get('source_f0_hz',7.5e6)
        sigma=1/source_f0; duration=2401/40e6; omega=2*math.pi*frequency
        source=-.5j*1e5*sigma*math.sqrt(2*math.pi)/duration*(
            torch.exp(torch.as_tensor(-.5*(sigma*(omega-2*math.pi*source_f0))**2,device=self.x.device))-
            torch.exp(torch.as_tensor(-.5*(sigma*(omega+2*math.pi*source_f0))**2,device=self.x.device)))
        source=source*torch.exp(-1j*omega*self.trefs[angle])
        f=frequency/6e6-1
        coefficients=self.response@torch.tensor([1.,f,f*f],device=self.x.device)
        response=torch.complex(coefficients[0],coefficients[1])
        return source*response

    @torch.no_grad()
    def predict_band_rf(self,medium,angle,bin_indices):
        """Inference-only RF projected onto specified positive-frequency bins.

        This is not the complete V5 RF unless the full relevant spectrum is
        supplied. Use forward() for differentiable single-frequency inversion.
        """
        spectrum=torch.zeros((192,1201),device=medium.device,dtype=torch.complex64)
        for index in bin_indices:
            if not 0<index<1201: raise ValueError('Positive RF bin out of range')
            spectrum[:,index]=self(medium,index*40e6/2401,angle)[0]
        return torch.fft.irfft(spectrum,n=2401,dim=-1,norm='forward')

def load_wave_operator(path,device='cpu',iterations=None):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=WaveResidualOperator(saved['spec'],saved['iterations'] if iterations is None else iterations,
                               saved.get('preconditioner','shifted'))
    model.load_state_dict(saved['model']); return model.to(device).eval()
