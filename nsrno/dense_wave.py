"""Dense V5 acquisition, controlled physical solves, and implicit adjoints."""
from pathlib import Path
from collections import OrderedDict
import json,math,time,copy
from types import SimpleNamespace
import numpy as np
import torch
from nsrno.conditioned_wave import FrequencyGeometry,ConditionedFrequencySystem,differentiable_gmres,LearnedPreconditioner
from nsrno.conditioned_data import receiver_response
from nsrno.wave_solver import gmres

ROOT=Path('out/v5/conditioned_dense_v1')

class DenseData:
    def __init__(self,device='cuda',reference_limit=4,root=ROOT,fast_fft=False):
        self.root=Path(root); self.index=json.load(open(self.root/'index.json')); self.spec=self.index['spec']; self.rows=self.index['records']
        self.first=self.spec['bins'][0]; self.count=len(self.spec['bins']); self.last=self.spec['bins'][-1]
        self.cache=Path(self.index['source_cache']); self.device=device
        geometry_spec={**self.spec,'fft_shape':[1024,1280]} if fast_fft else self.spec
        self.geometry=FrequencyGeometry(geometry_spec).to(device)
        self.sources=torch.from_numpy(np.load(self.root/'sources.npy')).to(device)
        self.references={}; self.fields=OrderedDict(); self.limit=reference_limit; self.media={}; self.spectra={}; self.full_energies={}
    def material(self,row):
        if row['id'] not in self.media:
            self.media[row['id']]=(torch.from_numpy(np.load(self.cache/row['medium'])),torch.from_numpy(np.load(row['alpha'])))
        return tuple(v.to(self.device) for v in self.media[row['id']])
    def target(self,row,bin_index,angles):
        if row['id'] not in self.spectra:
            iq=np.load(self.cache/row['iq'],mmap_mode='r')
            transform=np.fft.fftshift(np.fft.fft(iq[:,0]+1j*iq[:,1],axis=-1,norm='forward'),axes=-1)/np.sqrt(2)
            self.full_energies[row['id']]=np.sum(abs(transform)**2,axis=(1,2),dtype=np.float64)/(1-row['rf_representation_relative_rms']**2)
            self.spectra[row['id']]=transform[:,:,self.first-40:self.last-39].astype(np.complex64)
        return torch.from_numpy(self.spectra[row['id']][angles,:,bin_index-self.first]).to(self.device)
    def reference(self,bin_index,angles):
        fields=[]
        for angle in angles:
            name=(bin_index,angle)
            if name not in self.fields:
                if angle not in self.references: self.references[angle]=np.load(self.root/'reference'/f'a{angle:02d}.npy',mmap_mode='r')
                array=np.array(self.references[angle][bin_index-self.first])
                self.fields[name]=self.geometry.pad_field(torch.from_numpy(array).to(self.device))
                while len(self.fields)>self.limit: self.fields.popitem(last=False)
            else: self.fields.move_to_end(name)
            fields.append(self.fields[name])
        return torch.stack(fields)
    def assemble(self,medium,alpha,bin_index,angles,dtype=None):
        g=self.geometry
        if dtype is not None: medium=medium.to(dtype); alpha=alpha.to(dtype)
        system=ConditionedFrequencySystem(g,medium,alpha,bin_index*40e6/2401)
        pref=self.reference(bin_index,angles)
        if dtype==torch.float64: pref=pref.to(torch.complex128)
        source=self.sources[bin_index-self.first,angles]
        if dtype==torch.float64: source=source.to(torch.complex128)
        uniform=torch.empty_like(medium); uniform[0]=1540.; uniform[1]=1000.
        reference_system=ConditionedFrequencySystem(g,uniform,torch.full_like(alpha,.002),system.frequency)
        rhs=source_rhs(system,source)-source_rhs(reference_system,source)-system(pref)+reference_system(pref)
        return system,rhs,pref,source

def source_rhs(system,line):
    g=system.g; zi,xi=g.spec['source_c_lookup_zx']; source_c=system.c[zi+g.pz,xi+g.px]
    profile=torch.nn.functional.pad(line[:,None,:],(g.px,g.pad[1],g.face,g.H-g.face-1))
    return profile*system.a*1540.**2*2*g.dt/(source_c*g.dx*(system.D-1))

def channels(g,p): return p[...,g.face,g.rx_cols]

def projected_operator(system,p):
    mask=system.g.mask
    return mask*system(mask*p)+(1-mask)*p

def adjoint_operator(system,p):
    """Analytic adjoint of the projected native discrete pressure equation."""
    s=system; g=s.g; y=g.mask*p
    scalar=(s.D-1)/s.D/s.k0**2
    weighted=scalar.conjugate()*y
    def axis(ax,sv,gain,mask,ar,symbol):
        left=mask/(s.D-ar); right=1+(1-mask)/(s.D-ar)
        first=torch.fft.ifft2(-symbol*torch.fft.fft2(left.conj()*weighted))
        second=torch.fft.ifft2(-symbol*torch.fft.fft2(right.conj()*weighted))
        flux=(ax*sv).conj()*(first+gain.conj()*second)
        return torch.fft.ifft2(symbol.conj()*torch.fft.fft2(flux))
    value=axis(s.ax,s.svx,s.gvx,s.mx,g.arx,g.Dx)+axis(s.az,s.svz,s.gvz,s.mz,g.arz,g.Dz)+s.mass.conj()*y
    return g.mask*value+(1-g.mask)*p

def batched_gmres_cycle(operator,precondition,rhs,iterations=8):
    if not torch.is_grad_enabled():
        # Inference can reuse preallocated basis matrices. Avoid materializing
        # a [K,angles,z,x] weighted tensor at each orthogonalization step.
        shape=rhs.shape; n=rhs.numel(); beta=rhs.norm().clamp_min(1e-30)
        basis=torch.empty((iterations+1,n),dtype=rhs.dtype,device=rhs.device)
        directions=torch.empty((iterations,n),dtype=rhs.dtype,device=rhs.device)
        h=torch.zeros((iterations+1,iterations),dtype=rhs.dtype,device=rhs.device)
        basis[0]=rhs.flatten()/beta; used=iterations
        for j in range(iterations):
            z=precondition(basis[j].reshape(shape)); directions[j]=z.flatten()
            w=operator(z).flatten()
            for _ in range(2):
                coefficient=basis[:j+1].conj()@w
                w=w-basis[:j+1].T@coefficient; h[:j+1,j]+=coefficient
            norm=w.norm(); h[j+1,j]=norm
            if float(norm)<1e-20: used=j+1; break
            basis[j+1]=w/norm
        target=torch.zeros(used+1,dtype=rhs.dtype,device=rhs.device); target[0]=beta
        y=torch.linalg.lstsq(h[:used+1,:used].cpu().to(torch.complex128),target.cpu().to(torch.complex128)).solution.to(rhs.device,rhs.dtype)
        return (directions[:used].T@y).reshape(shape)
    beta=rhs.norm().clamp_min(1e-30); basis=[rhs/beta]; directions=[]; columns=[]
    expand=(-1,)+((1,)*rhs.ndim)
    for j in range(iterations):
        z=precondition(basis[j]); directions.append(z); w=operator(z); q=torch.stack(basis)
        c=(q.conj()*w).flatten(1).sum(1); w=w-(q*c.reshape(expand)).sum(0)
        correction=(q.conj()*w).flatten(1).sum(1); w=w-(q*correction.reshape(expand)).sum(0)
        norm=w.norm().clamp_min(1e-30)
        columns.append(torch.nn.functional.pad(torch.cat((c+correction,norm[None].to(rhs.dtype))),(0,iterations-j-1)))
        basis.append(w/norm)
    h=torch.stack(columns,dim=1); target=torch.nn.functional.pad(beta[None].to(rhs.dtype),(0,iterations))
    solution=torch.linalg.lstsq(h.cpu().to(torch.complex128),target.cpu().to(torch.complex128)).solution.to(rhs.device,rhs.dtype)
    return (torch.stack(directions)*solution.reshape(expand)).sum(0)

def bicgstab(operator,precondition,rhs,x,tolerance=.001,max_iterations=800):
    """Short-recurrence right-preconditioned solve for shared-angle RHS.

    Actual FP64 residuals are checked by the outer refinement. This method
    avoids storing/reorthogonalizing hundreds of full multi-angle fields.
    """
    scale=rhs.norm().clamp_min(1e-30); r=rhs-operator(x); shadow=r.clone()
    p=torch.zeros_like(r); v=torch.zeros_like(r)
    rho_old=torch.ones((),dtype=rhs.dtype,device=rhs.device); alpha=rho_old; omega=rho_old
    calls=1; reason='iteration_budget'; used=0
    for iteration in range(max_iterations):
        if float(r.norm()/scale)<tolerance: reason='converged'; break
        rho=(shadow.conj()*r).sum()
        if not bool(torch.isfinite(rho)) or float(rho.abs())<1e-30 or float(omega.abs())<1e-30:
            reason='breakdown'; break
        beta=(rho/rho_old)*(alpha/omega); p=r+beta*(p-omega*v)
        phat=precondition(p); v=operator(phat); calls+=1
        denominator=(shadow.conj()*v).sum()
        if float(denominator.abs())<1e-30: reason='breakdown'; break
        alpha=rho/denominator; s=r-alpha*v
        if float(s.norm()/scale)<tolerance:
            x=x+alpha*phat; r=s; used=iteration+1; reason='converged'; break
        shat=precondition(s); t=operator(shat); calls+=1
        omega=(t.conj()*s).sum()/t.abs().square().sum().clamp_min(1e-30)
        x=x+alpha*phat+omega*shat; r=s-omega*t; rho_old=rho; used=iteration+1
        if iteration%40==39:
            r=rhs-operator(x); calls+=1
            if not bool(torch.isfinite(r).all()) or float(r.norm()/scale)>1e5: reason='diverged'; break
    actual=float((operator(x)-rhs).norm()/scale); calls+=1
    if actual<tolerance: reason='converged'
    return x,{'reason':reason,'iterations':used,'steps':calls,'residual':actual,'method':'BiCGSTAB'}

def learned_apply(network,system,adjoint=False):
    """Apply the learned linear inverse or its exact complex adjoint."""
    s=system; g=s.g; c=s.c.float(); rho=s.rho.float(); alpha=s.alpha.float()
    context=torch.stack(((c/1540.).log()/.06,(rho/1000.).log()/.08,torch.log1p(alpha),
        torch.ones_like(c)*(s.frequency/6e6-1),g.mask))[None]
    maps=torch.nn.functional.interpolate(network.context(torch.nn.functional.avg_pool2d(context,8)),
        size=(g.H,g.W),mode='bilinear',align_corners=False)[0]
    inner=(1+.5*torch.tanh(maps[0]))*torch.exp(.5j*torch.tanh(maps[1]))
    outer=(1+.5*torch.tanh(maps[2]))*torch.exp(.5j*torch.tanh(maps[3]))
    q2=-g.lap/s.k0**2; angular=g.Dx.abs().square()/(-g.lap+1)
    basis=torch.stack((torch.ones_like(q2),q2/(1+q2),angular,angular*q2/(1+q2)))
    mod=torch.einsum('ck,khw->chw',network.spectral,basis)
    multiplier=torch.exp(.5*torch.tanh(mod[0])+.5j*torch.tanh(mod[1]))/(1-q2-1j*network.log_shift.exp().clamp(.01,1.))
    if adjoint:
        return lambda r:g.mask*inner.conj()*torch.fft.ifft2(multiplier.conj()*torch.fft.fft2(outer.conj()*g.mask*r))
    return lambda r:g.mask*outer*torch.fft.ifft2(multiplier*torch.fft.fft2(inner*r))

@torch.no_grad()
def solve_system(system,rhs,preconditioner=None,adjoint=False,tolerance=1e-5,max_cycles=100,warm_steps=8,mixed=False,method='gmres',initial=None):
    if mixed and rhs.dtype==torch.complex128:
        # FP32 inverse applications with true FP64 residual correction retain
        # the same physical equation and stopping criterion.
        start=time.perf_counter(); approximate=copy.copy(system)
        for name,value in vars(system).items():
            if isinstance(value,torch.Tensor):
                setattr(approximate,name,value.to(torch.complex64 if value.is_complex() else torch.float32))
        operator=(lambda p:adjoint_operator(system,p)) if adjoint else (lambda p:projected_operator(system,p))
        x=torch.zeros_like(rhs) if initial is None else initial.to(rhs.dtype)
        scale=rhs.norm().clamp_min(1e-30); total_steps=0; inner=[]
        for outer in range(5):
            residual=rhs-operator(x); relative=float(residual.norm()/scale)
            if relative<tolerance: break
            inner_tolerance=min(.03,max(.001,tolerance/(max(relative,1e-30)*3)))
            inner_method='gmres_learned' if method=='bicgstab' and outer>0 else method
            update,info=solve_system(approximate,residual.to(torch.complex64),preconditioner,
                adjoint=adjoint,tolerance=inner_tolerance,max_cycles=max_cycles,
                warm_steps=warm_steps if outer==0 and initial is None else 0,mixed=False,method=inner_method)
            x=x+update.to(torch.complex128); total_steps+=info['steps']; inner.append(info)
        residual=operator(x)-rhs; relative=float(residual.norm()/scale)
        individual=residual.flatten(1).norm(dim=1)/rhs.flatten(1).norm(dim=1).clamp_min(1e-30)
        return x,{'reason':'converged' if relative<tolerance else 'iteration_budget','steps':total_steps,
            'residual':relative,'max_angle_residual':float(individual.max()),'seconds':time.perf_counter()-start,
            'dtype':'mixed FP32 inverse / FP64 residual','refinements':len(inner),'inner':inner}
    g=system.g; start=time.perf_counter(); traces=[]; operator=(lambda p:adjoint_operator(system,p)) if adjoint else (lambda p:projected_operator(system,p))
    multiplier=1/(1+g.lap/system.k0**2-.15j)
    if adjoint: multiplier=multiplier.conj()
    fixed=lambda r:g.mask*torch.fft.ifft2(multiplier*torch.fft.fft2(r))
    initial=torch.zeros_like(rhs) if initial is None else initial.to(rhs.dtype)
    if preconditioner is not None and warm_steps and not adjoint:
        features=SimpleNamespace(c=system.c.float(),rho=system.rho.float(),alpha=system.alpha.float(),
            g=g,frequency=system.frequency,k0=system.k0)
        initial=batched_gmres_cycle(operator,preconditioner.setup(features),rhs,warm_steps)
    else: warm_steps=0
    if method=='bicgstab':
        # A bounded learned preconditioner may be used throughout this solver;
        # the true residual controls acceptance independently of learning.
        if preconditioner is not None:
            chosen=learned_apply(preconditioner,system,adjoint=adjoint)
        else: chosen=fixed
        value,status=bicgstab(operator,chosen,rhs,initial,tolerance=tolerance)
        if status['reason']=='converged':
            traces.append({'steps':status['steps']+warm_steps,'residual':status['residual']}); reason='converged'
        else:
            # Fall back to the known reference algorithm without changing A.
            if not bool(torch.isfinite(value).all()): value=initial
            value,reason=gmres(operator,fixed,rhs,value,lambda n,u,r:traces.append({'steps':n+status['steps']+warm_steps,'residual':r}),
                restart=24,max_cycles=max_cycles,tolerance=tolerance)
    else:
        chosen=learned_apply(preconditioner,system,adjoint=adjoint) if method=='gmres_learned' and preconditioner is not None else fixed
        value,reason=gmres(operator,chosen,rhs,initial,lambda n,u,r:traces.append({'steps':n+warm_steps,'residual':r}),
            restart=24,max_cycles=max_cycles,tolerance=tolerance)
    residual=operator(value)-rhs
    individual=residual.flatten(1).norm(dim=1)/rhs.flatten(1).norm(dim=1).clamp_min(1e-30)
    info={'reason':reason,'steps':traces[-1]['steps'],'residual':traces[-1]['residual'],
          'max_angle_residual':float(individual.max()),'seconds':time.perf_counter()-start,'dtype':str(rhs.dtype),'method':method}
    return value,info

class ImplicitChannels(torch.autograd.Function):
    @staticmethod
    def forward(ctx,medium,alpha,data,bin_index,angles,preconditioner,tolerance):
        # Double precision avoids FP32 stagnation for weak candidate scattering.
        s,rhs,pref,line=data.assemble(medium,alpha,bin_index,angles,dtype=torch.float64)
        u,info=solve_system(s,rhs,preconditioner,tolerance=tolerance,mixed=True,method='bicgstab' if preconditioner is not None else 'gmres')
        if info['reason']!='converged' and preconditioner is not None:
            u,info=solve_system(s,rhs,None,tolerance=tolerance,mixed=True,method='gmres')
        if info['reason']!='converged': raise RuntimeError('Forward failed to converge: '+str(info))
        ctx.data=data; ctx.bin_index=bin_index; ctx.angles=angles; ctx.tolerance=tolerance; ctx.info=info; ctx.preconditioner=preconditioner
        ctx.save_for_backward(medium,alpha,u,pref,line)
        return channels(data.geometry,u)*receiver_response(s.frequency)
    @staticmethod
    def backward(ctx,gradient):
        medium,alpha,u,pref,line=ctx.saved_tensors; data=ctx.data; g=data.geometry
        with torch.no_grad():
            s=ConditionedFrequencySystem(g,medium.double(),alpha.double(),ctx.bin_index*40e6/2401)
            rhs=torch.zeros_like(u); rhs[...,g.face,g.rx_cols]=gradient*receiver_response(s.frequency)
            lam,info=solve_system(s,rhs,ctx.preconditioner,adjoint=True,tolerance=ctx.tolerance,mixed=True,
                method='bicgstab' if ctx.preconditioner is not None else 'gmres')
            if info['reason']!='converged' and ctx.preconditioner is not None:
                lam,info=solve_system(s,rhs,None,adjoint=True,tolerance=ctx.tolerance,mixed=True,method='gmres')
            if info['reason']!='converged': raise RuntimeError('Adjoint failed to converge: '+str(info))
        with torch.enable_grad():
            material=medium.detach().double().requires_grad_(True)
            physical=ConditionedFrequencySystem(g,material,alpha.double(),ctx.bin_index*40e6/2401)
            residual=physical(g.mask*(u+pref))-source_rhs(physical,line)
            scalar=-(lam.conj()*g.mask*residual).real.sum()
            derivative=torch.autograd.grad(scalar,material)[0].to(medium.dtype)
        return derivative,None,None,None,None,None,None

def implicit_channels(medium,alpha,data,bin_index,angles,preconditioner=None,tolerance=1e-5):
    return ImplicitChannels.apply(medium,alpha,data,bin_index,angles,preconditioner,tolerance)
