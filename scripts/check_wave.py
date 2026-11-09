import json,time
from pathlib import Path
import numpy as np
import torch
from nsrno.wave_operator import WaveResidualOperator
from nsrno.train_wave import sample
from nsrno.dataset import read_json,write_json

torch.set_num_threads(2); torch.manual_seed(20261105)
cache=Path('out/v5/forward_cache_v1'); index=read_json(cache/'index.json'); row=index['records'][0]
spec={**index['grid'],'angles_deg':index['acquisition']['angles_deg'],'source_tref_s':row['source_tref_s']}
model=WaveResidualOperator(spec).cuda(); medium,target=sample(row,cache,5,360); medium.requires_grad_(True)
t=time.perf_counter(); model.train(); prediction,physics,trace=model(medium,360*40e6/2401,5,True,True)
loss=(prediction-target).abs().square().mean()/target.abs().square().mean()+.1*physics
loss.backward(); torch.cuda.synchronize()
grad_norm=[float(medium.grad[:,i].norm()) for i in range(2)]
assert all(np.isfinite(grad_norm)) and min(grad_norm)>0
assert all(b<=a+1e-5 for a,b in zip(trace,trace[1:]))
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
result=dict(seconds=time.perf_counter()-t,loss=float(loss),relative_channel_rms=float(((prediction-target).abs().square().sum()/target.abs().square().sum()).sqrt()),
    relative_pde_residual=trace,input_gradient_norm_c_rho=grad_norm,peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
with torch.no_grad():
    data,_,_=model.setup(medium,6e6); ax,az,b,k0=data; real_data=(ax,az,b.real,k0)
    u=torch.randn_like(b)+1j*torch.randn_like(b); v=torch.randn_like(u)
    left=(u.conj()*model.apply_operator(v,real_data)).sum(); right=(model.apply_operator(u,real_data).conj()*v).sum()
    result['discrete_self_adjoint_relative_error']=float((left-right).abs()/(left.abs()+right.abs()).clamp_min(1e-12))
    assert result['discrete_self_adjoint_relative_error']<1e-4
write_json(Path('out/v5/forward_round5_checks.json'),result); print(json.dumps(result),flush=True)
