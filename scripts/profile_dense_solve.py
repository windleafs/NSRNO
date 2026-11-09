from pathlib import Path
import time,json
import torch
from nsrno.dense_wave import DenseData,solve_system,channels,implicit_channels
from nsrno.conditioned_data import receiver_response
torch.set_num_threads(2); data=DenseData(reference_limit=12); row=next(r for r in data.rows if r['id']=='train_200')
medium,alpha=data.material(row); angles=list(range(11)); results={}
with torch.no_grad():
    s,rhs,pref,line=data.assemble(medium,alpha,360,angles,dtype=torch.float64)
    u,info=solve_system(s,rhs,tolerance=1e-5,mixed=True)
    prediction=channels(data.geometry,u)*receiver_response(s.frequency); target=data.target(row,360,angles)
    results['all11_angles']={**info,'rf_relative_rms':float((prediction-target).norm()/target.norm())}
material=medium.double().requires_grad_(True)
start=time.perf_counter(); prediction=implicit_channels(material,alpha.double(),data,360,[5],None,1e-6)
target=data.target(row,360,[5]); loss=(prediction-target).abs().square().sum()/target.abs().square().sum()
gradient=torch.autograd.grad(loss,material)[0]
g=data.geometry; xx=torch.arange(g.w,device='cuda')*g.dx+g.spec['x0_m']; zz=torch.arange(g.h,device='cuda')*g.dx+g.spec['z0_m']
window=torch.exp(-((xx[None,:]-.002)/.004)**2-((zz[:,None]-.012)/.006)**2)*(zz[:,None]>=.001)
direction=torch.stack((2*window,1.5*window)).double(); derivative=float((gradient*direction).sum()); values=[]; h=.002
with torch.no_grad():
    for sign in (-1,1):
        output=implicit_channels(material.detach()+sign*h*direction,alpha.double(),data,360,[5],None,1e-6)
        values.append(float((output-target).abs().square().sum()/target.abs().square().sum()))
finite=(values[1]-values[0])/(2*h)
results['implicit_gradient']={'analytic':derivative,'finite_difference':finite,
    'relative_difference':abs(derivative-finite)/max(abs(derivative),abs(finite),1e-10),'seconds':time.perf_counter()-start}
results['passed']=results['all11_angles']['reason']=='converged' and results['implicit_gradient']['relative_difference']<.03
Path('out/v5/conditioned_dense_v1/mixed_precision_check.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2),flush=True)
if not results['passed']: raise RuntimeError('Mixed solve/gradient check failed')
