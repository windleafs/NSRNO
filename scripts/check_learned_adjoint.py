from pathlib import Path
import json,time
import torch
from nsrno.dense_wave import DenseData,learned_apply,implicit_channels,solve_system,channels
from nsrno.conditioned_wave import LearnedPreconditioner
from nsrno.conditioned_data import receiver_response
torch.set_num_threads(2); torch.manual_seed(7); data=DenseData(reference_limit=12)
row=next(r for r in data.rows if r['id']=='train_200'); medium,alpha=data.material(row)
network=LearnedPreconditioner().cuda(); network.load_state_dict(torch.load('out/v5/forward_round8_dense/last.pt',map_location='cpu',weights_only=True)['model']); network.eval(); network.requires_grad_(False)
result={}
with torch.no_grad():
    s,rhs,_,_=data.assemble(medium,alpha,360,list(range(11)),dtype=torch.float64)
    x=torch.randn_like(rhs)*s.g.mask; y=torch.randn_like(rhs)*s.g.mask
    p=learned_apply(network,s); ph=learned_apply(network,s,adjoint=True)
    left=(y.conj()*p(x)).sum(); right=(ph(y).conj()*x).sum()
    result['learned_adjoint_error']=float(abs(left-right)/max(abs(left),abs(right)))
    value,info=solve_system(s,rhs,network,tolerance=1e-5,mixed=True,method='bicgstab')
    pred=channels(s.g,value)*receiver_response(s.frequency); target=data.target(row,360,list(range(11)))
    result['all11_forward']={**info,'rf_relative_rms':float((pred-target).norm()/target.norm())}
material=medium.double().requires_grad_(True); start=time.perf_counter()
pred=implicit_channels(material,alpha.double(),data,360,[5],network,1e-6)
target=data.target(row,360,[5]); loss=(pred-target).abs().square().sum()/target.abs().square().sum()
gradient=torch.autograd.grad(loss,material)[0]
g=data.geometry; xx=torch.arange(g.w,device='cuda')*g.dx+g.spec['x0_m']; zz=torch.arange(g.h,device='cuda')*g.dx+g.spec['z0_m']
window=torch.exp(-((xx[None,:]-.002)/.004)**2-((zz[:,None]-.012)/.006)**2)*(zz[:,None]>=.001)
direction=torch.stack((2*window,1.5*window)).double(); analytic=float((gradient*direction).sum()); h=.002; values=[]
with torch.no_grad():
    for sign in (-1,1):
        pred=implicit_channels(material.detach()+sign*h*direction,alpha.double(),data,360,[5],network,1e-6)
        values.append(float((pred-target).abs().square().sum()/target.abs().square().sum()))
finite=(values[1]-values[0])/(2*h)
result['gradient']={'analytic':analytic,'finite_difference':finite,'relative_difference':abs(analytic-finite)/max(abs(analytic),abs(finite),1e-10),'seconds':time.perf_counter()-start}
result['passed']=result['learned_adjoint_error']<1e-5 and result['gradient']['relative_difference']<.03
Path('out/v5/conditioned_dense_v1/learned_adjoint_check.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
if not result['passed']: raise RuntimeError('Learned adjoint check failed')
