"""Hold the trained PDE fixed and independently solve its linear system."""
from pathlib import Path
from argparse import ArgumentParser
import json,time
import numpy as np
import torch
from nsrno.wave_operator import load_wave_operator
from nsrno.train_wave import sample
from nsrno.dataset import read_json,write_json

def gmres(operator,precondition,rhs,x,callback,restart=32,max_cycles=80,tolerance=1e-4):
    shape=rhs.shape; n=rhs.numel(); scale=rhs.norm()
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
        # Small least-squares solve in double precision; no normal equations.
        y=torch.linalg.lstsq(h[:used+1,:used].cpu().to(torch.complex128),q.cpu().to(torch.complex128)).solution.to(rhs.device,rhs.dtype)
        x=x+(directions[:used].T@y).reshape(shape)
    callback(applications,x,float((rhs-operator(x)).norm()/scale))
    return x,'iteration_budget'

def main():
    parser=ArgumentParser(); parser.add_argument('--preconditioner',choices=('learned','spectral'),default='learned'); args=parser.parse_args()
    torch.set_num_threads(2); out=Path('out/v5/wave_diagnosis_v1'); out.mkdir(parents=True,exist_ok=True)
    model=load_wave_operator('out/v5/forward_round5_wave_relax_v1/best.pt','cuda')
    cache=Path('out/v5/forward_cache_v1'); index=read_json(cache/'index.json')
    row=next(r for r in index['records'] if r['id']=='train_200'); frequency=360*40e6/2401
    medium,target=sample(row,cache,5,360); history=[]; timer=time.perf_counter()
    with torch.no_grad():
        data,gain,multiplier=model.setup(medium,frequency); incident=model.incident(frequency,5)
        ax,az,b,k0=data; rhs=-model.apply_operator(incident,(ax-1,az-1,b-(1-1j*model.sponge),k0))
        u=torch.zeros_like(incident)
        for _ in range(8): u=model._iteration(u,rhs,*data,gain,multiplier)
        predicted,_,_=model(medium,frequency,5)
        value=u[model.face,model.rx_cols]; factor=(value.conj()*predicted).sum()/value.abs().square().sum()
        def metric(steps,field,residual):
            prediction=field[model.face,model.rx_cols]*factor
            fitted=(prediction.conj()*target).sum()/prediction.abs().square().sum().clamp_min(1e-20)
            record=dict(steps=steps,pde_relative_residual=residual,
                        channel_relative_rms=float((prediction-target).norm()/target.norm()),
                        channel_rms_after_oracle_scalar=float((fitted*prediction-target).norm()/target.norm()),
                        seconds=time.perf_counter()-timer)
            history.append(record); write_json(out/f'convergence_{args.preconditioner}_history.json',history); print(json.dumps(record),flush=True)
        metric(8,u,float((rhs-model.apply_operator(u,data)).norm()/rhs.norm()))
        np.save(out/'initial_wave.npy',u.cpu().numpy()); np.save(out/'incident_wave.npy',incident.cpu().numpy())
        def precondition(r): return (gain if args.preconditioner=='learned' else 1)*torch.fft.ifft2(multiplier*torch.fft.fft2(r))
        solution,reason=gmres(lambda u:model.apply_operator(u,data),precondition,rhs,u,metric)
        np.save(out/f'solution_{args.preconditioner}.npy',solution.cpu().numpy())
        write_json(out/f'convergence_{args.preconditioner}_completion.json',dict(reason=reason,last=history[-1],frequency_hz=frequency,
                   source_factor=[float(factor.real),float(factor.imag)],model_spec=model.spec))

if __name__=='__main__': main()
