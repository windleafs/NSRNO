"""Converged frequency reference for the independent NSRNO wave operator.

This is a numerical reference, not a fast neural surrogate. Implicit-adjoint
input gradients are not implemented; inverse optimization must not use it yet.
"""
import time
import torch

def gmres(operator,precondition,rhs,x,callback,restart=32,max_cycles=80,tolerance=1e-4):
    shape=rhs.shape; n=rhs.numel(); scale=rhs.norm().clamp_min(1e-30)
    basis=torch.empty((restart+1,n),dtype=rhs.dtype,device=rhs.device)
    directions=torch.empty((restart,n),dtype=rhs.dtype,device=rhs.device)
    applications=0
    for cycle in range(max_cycles):
        residual=rhs-operator(x); beta=residual.norm(); relative=float(beta/scale)
        callback(applications,x,relative)
        if relative<tolerance: return x,'converged'
        basis[0]=residual.flatten()/beta
        h=torch.zeros((restart+1,restart),dtype=rhs.dtype,device=rhs.device)
        used=restart
        for j in range(restart):
            z=precondition(basis[j].reshape(shape)); directions[j]=z.flatten()
            w=operator(z).flatten(); applications+=1
            for _ in range(2):
                coefficients=basis[:j+1].conj()@w
                w=w-basis[:j+1].T@coefficients
                h[:j+1,j]+=coefficients
            h[j+1,j]=w.norm()
            if float(h[j+1,j].abs())<1e-12:
                used=j+1; break
            basis[j+1]=w/h[j+1,j]
        q=torch.zeros(used+1,dtype=rhs.dtype,device=rhs.device); q[0]=beta
        y=torch.linalg.lstsq(h[:used+1,:used].cpu().to(torch.complex128),q.cpu().to(torch.complex128)).solution.to(rhs.device,rhs.dtype)
        x=x+(directions[:used].T@y).reshape(shape)
    callback(applications,x,float((rhs-operator(x)).norm()/scale))
    return x,'iteration_budget'

@torch.no_grad()
def solve_frequency(model,medium,frequency_hz,angle_index,tolerance=1e-4,max_iterations=2560,restart=32):
    """Return complex receiver spectrum and measured convergence status.

    Uses only the existing model and c/rho input. No true attenuation or
    receiver target is read. One medium and one frequency/angle per solve.
    """
    if medium.requires_grad:
        raise ValueError('This reference has no implicit-adjoint backward; use the neural forward for input gradients.')
    if max_iterations<restart or max_iterations%restart:
        raise ValueError('max_iterations must be a positive multiple of restart')
    started=time.perf_counter(); model.eval()
    data,gain,multiplier=model.setup(medium,frequency_hz)
    incident=model.incident(frequency_hz,angle_index)
    ax,az,b,k0=data
    rhs=-model.apply_operator(incident,(ax-1,az-1,b-(1-1j*model.sponge),k0))
    wave=torch.zeros_like(incident)
    for _ in range(model.iterations): wave=model._iteration(wave,rhs,*data,gain,multiplier)
    traces=[]
    def monitor(count,field,residual): traces.append(dict(iterations=count,relative_residual=residual))
    # The learned boundary taper is excluded from the reference preconditioner:
    # it suppresses directions needed to solve residuals in the absorbing layer.
    def precondition(residual):
        return torch.fft.ifft2(multiplier*torch.fft.fft2(residual,s=multiplier.shape))[:model.H,:model.W]
    wave,reason=gmres(lambda p:model.apply_operator(p,data),precondition,rhs,wave,monitor,
                     restart=restart,max_cycles=max_iterations//restart,tolerance=tolerance)
    prediction=wave[model.face,model.rx_cols]*model.source_factor(frequency_hz,angle_index)
    return prediction,dict(reason=reason,initial_neural_iterations=model.iterations,
        solver_iterations=traces[-1]['iterations'],relative_residual=traces[-1]['relative_residual'],
        seconds=time.perf_counter()-started,trace=traces)
