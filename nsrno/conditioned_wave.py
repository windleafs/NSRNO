"""V5 discrete wave equation with explicit attenuation and acquisition source.

No target field, learned material coefficient, or fitted source response is used.
Alpha is an explicit input; its prescribed eta dependence on c is differentiable.
"""
import cmath, math
import torch
from torch import nn
from torch.nn import functional as F
from nsrno.wave_operator import C0,RHO0

class FrequencyGeometry(nn.Module):
    def __init__(self,spec):
        super().__init__(); self.spec=spec
        self.h,self.w=spec['medium_shape'][-2:]; self.dx=spec['dx_m']
        self.H=math.ceil((self.h+64)/32)*32; self.W=math.ceil((self.w+64)/32)*32
        if 'fft_shape' in spec:
            self.H,self.W=spec['fft_shape']
            if self.H-self.h<16 or self.W-self.w<16:
                raise ValueError('FFT padding must separate the8th-order finite-stencil support')
        self.pz=(self.H-self.h)//2; self.px=(self.W-self.w)//2
        self.pad=(self.px,self.W-self.w-self.px,self.pz,self.H-self.h-self.pz)
        qx=2*math.pi*torch.fft.fftfreq(self.W); qz=2*math.pi*torch.fft.fftfreq(self.H)
        def symbol(q):
            coeff=(1225/1024,-245/3072,49/5120,-5/7168)
            return 2j*sum(a*torch.sin((j+.5)*q) for j,a in enumerate(coeff))*torch.exp(.5j*q)/self.dx
        self.register_buffer('Dx',symbol(qx)[None,:]); self.register_buffer('Dz',symbol(qz)[:,None])
        self.register_buffer('lap',-(self.Dx.abs().square()+self.Dz.abs().square()))
        self.register_buffer('mask',F.pad(torch.ones(self.h,self.w),self.pad))
        self.register_buffer('rx_cols',torch.round((torch.arange(192)*.0002-.0191-spec['x0_m'])/self.dx).long()+self.px)
        self.face=round(-spec['z0_m']/self.dx)+self.pz
        self.dt=spec['native_dt_s']; self.pml=spec['pml']
        xx=torch.arange(self.W)-self.px; zz=torch.arange(self.H)-self.pz
        def damping(pos,size,stagger):
            left=torch.where(pos<self.pml,(1-(pos+stagger)/self.pml).pow(4),0.)
            right=torch.where(pos>=size-self.pml,(1-(size-1-pos-stagger)/self.pml).pow(4),0.)
            return (left+right)*2*C0/self.dx
        for name,values in [('arx',1-self.dt*damping(xx,self.w,0.)[None,:]),
                ('arz',1-self.dt*damping(zz,self.h,0.)[:,None]),
                ('avx',1-self.dt*damping(xx,self.w,.5)[None,:]),
                ('avz',1-self.dt*damping(zz,self.h,.5)[:,None])]: self.register_buffer(name,values)

    def pad_field(self,p): return F.pad(p,self.pad)
    def native(self,p): return p[...,self.pz:self.pz+self.h,self.px:self.px+self.w]
    def coefficients(self,m):
        # UltraSolver crops the original 24-cell rim, then edge pads the crop.
        p=self.pml
        m=F.pad(m[...,p:-p,p:-p],(p,p,p,p),mode='replicate')
        return F.pad(m,self.pad,mode='replicate')
    def channels(self,p): return p[self.face,self.rx_cols]

class ConditionedFrequencySystem:
    def __init__(self,geometry,medium,alpha,frequency):
        if medium.shape!=(2,geometry.h,geometry.w): raise ValueError('Expected native [2,z,x] medium')
        self.g=geometry; self.frequency=float(frequency); g=geometry
        self.c,self.rho=g.coefficients(medium).unbind(0)
        self.alpha=g.coefficients(alpha[None])[0]
        self.a=RHO0/self.rho; self.D=cmath.exp(2j*math.pi*frequency*g.dt); self.q=1/self.D
        self.k0=2*math.sin(math.pi*frequency*g.dt)/(g.dt*C0)
        self.ax=2*self.a*torch.roll(self.a,-1,-1)/(self.a+torch.roll(self.a,-1,-1))*g.mask
        self.az=2*self.a*torch.roll(self.a,-1,-2)/(self.a+torch.roll(self.a,-1,-2))*g.mask
        eta=[self.alpha*self.c/C0*v for v in g.spec['eta_per_unit_alpha']]
        amplitude=(1+(eta[0]+eta[1])/2).square()
        h=sum(e/(amplitude*(1+t*(self.D-1)/g.dt)) for e,t in zip(eta,g.spec['taus_s']))
        modulus=amplitude*(1-h)/(1-(1-self.q)*h)
        self.mass=self.a*(C0/self.c).square()/modulus
        self.svx=(self.D-1)/(self.D-g.avx); self.svz=(self.D-1)/(self.D-g.avz)
        self.gvx=1+(1-g.avx)*self.q; self.gvz=1+(1-g.avz)*self.q
        self.mx=(g.arx!=1).float(); self.mz=(g.arz!=1).float()

    def __call__(self,p):
        g=self.g; spectrum=torch.fft.fft2(p)
        x=self.ax*self.svx*torch.fft.ifft2(g.Dx*spectrum)
        z=self.az*self.svz*torch.fft.ifft2(g.Dz*spectrum)
        def part(flux,gain,mask,ar,symbol):
            regular=torch.fft.ifft2(-symbol.conj()*torch.fft.fft2(flux))
            provisional=torch.fft.ifft2(-symbol.conj()*torch.fft.fft2(gain*flux))
            return mask/(self.D-ar)*regular+(1+(1-mask)/(self.D-ar))*provisional
        spatial=(self.D-1)/self.D/self.k0**2*(part(x,self.gvx,self.mx,g.arx,g.Dx)+part(z,self.gvz,self.mz,g.arz,g.Dz))
        return g.mask*(spatial+self.mass*p)+(1-g.mask)*p

    def source(self,line):
        g=self.g
        zi,xi=g.spec['source_c_lookup_zx']
        source_c=self.c[zi+g.pz,xi+g.px]
        # Pressure is formed BEFORE the source increments both density states.
        profile=F.pad(line[None],(g.px,g.pad[1],g.face,g.H-g.face-1))
        return profile*self.a*C0**2*2*g.dt/(source_c*g.dx*(self.D-1))

    def fixed_preconditioner(self,shift=.15):
        multiplier=1/(1+self.g.lap/self.k0**2-1j*shift)
        return lambda r:self.g.mask*torch.fft.ifft2(multiplier*torch.fft.fft2(r))

class LearnedPreconditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.context=nn.Sequential(nn.Conv2d(5,16,3,padding=1),nn.Tanh(),nn.Conv2d(16,16,3,padding=1),
                                   nn.Tanh(),nn.Conv2d(16,4,3,padding=1))
        nn.init.zeros_(self.context[-1].weight); nn.init.zeros_(self.context[-1].bias)
        self.spectral=nn.Parameter(torch.zeros(2,4))
        self.log_shift=nn.Parameter(torch.tensor(math.log(.15)))

    def setup(self,system,spatial=True,spectral=True):
        s=system; g=s.g
        context=torch.stack(((s.c/C0).log()/.06,(s.rho/RHO0).log()/.08,
            torch.log1p(s.alpha),torch.ones_like(s.c)*(s.frequency/6e6-1),g.mask))[None]
        context=F.avg_pool2d(context,8)
        maps=F.interpolate(self.context(context),size=(g.H,g.W),mode='bilinear',align_corners=False)[0]
        inner=(1+.5*torch.tanh(maps[0]))*torch.exp(.5j*torch.tanh(maps[1]))
        outer=(1+.5*torch.tanh(maps[2]))*torch.exp(.5j*torch.tanh(maps[3]))
        if not spatial: inner=outer=1.
        q2=-g.lap/s.k0**2; angular=g.Dx.abs().square()/(-g.lap+1)
        basis=torch.stack((torch.ones_like(q2),q2/(1+q2),angular,angular*q2/(1+q2)))
        mod=torch.einsum('ck,khw->chw',self.spectral,basis) if spectral else torch.zeros((2,g.H,g.W),device=q2.device)
        multiplier=torch.exp(.5*torch.tanh(mod[0])+.5j*torch.tanh(mod[1]))/(1-q2-1j*self.log_shift.exp().clamp(.01,1.))
        def apply(r): return g.mask*outer*torch.fft.ifft2(multiplier*torch.fft.fft2(inner*r))
        return apply

def differentiable_gmres(operator,precondition,rhs,iterations=8):
    """One right-preconditioned GMRES cycle with twice MGS and autograd.

    Tiny least squares is complex128 on CPU; all field operations stay on GPU.
    The returned solution uses exactly `iterations` physical operator calls.
    """
    beta=rhs.norm().clamp_min(1e-30); basis=[rhs/beta]; directions=[]; columns=[]
    for j in range(iterations):
        z=precondition(basis[j]); directions.append(z)
        w=operator(z); qs=torch.stack(basis)
        coeff=(qs.conj()*w).flatten(1).sum(1)
        w=w-(qs*coeff[:,None,None]).sum(0)
        correction=(qs.conj()*w).flatten(1).sum(1)
        w=w-(qs*correction[:,None,None]).sum(0)
        norm=w.norm().clamp_min(1e-30)
        columns.append(F.pad(torch.cat((coeff+correction,norm[None].to(rhs.dtype))),(0,iterations-j-1)))
        basis.append(w/norm)
    h=torch.stack(columns,dim=1)
    target=F.pad(beta[None].to(rhs.dtype),(0,iterations))
    y=torch.linalg.lstsq(h.to('cpu',torch.complex128),target.to('cpu',torch.complex128)).solution.to(rhs.device,rhs.dtype)
    return (torch.stack(directions)*y[:,None,None]).sum(0)
