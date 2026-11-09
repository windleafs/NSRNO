"""Necessary reference, adjoint, and implicit material derivative checks."""
from pathlib import Path
import json,time
import numpy as np
import torch
from nsrno.dense_wave import DenseData,source_rhs,projected_operator,adjoint_operator,implicit_channels,solve_system,channels,ROOT
from nsrno.conditioned_wave import ConditionedFrequencySystem
from nsrno.conditioned_data import receiver_response

torch.set_num_threads(2); torch.manual_seed(20261105); data=DenseData(); g=data.geometry; checks={}
with torch.no_grad():
    uniform=torch.empty(2,g.h,g.w,device='cuda',dtype=torch.float64); uniform[0]=1540.; uniform[1]=1000.
    for b in (270,360,420):
        pref=data.reference(b,[5]).to(torch.complex128)
        s=ConditionedFrequencySystem(g,uniform,torch.full_like(uniform[0],.002),b*40e6/2401)
        rhs=source_rhs(s,data.sources[b-241,[5]].to(torch.complex128))
        old=np.load(f'out/v5/conditioned_wave_v1/reference/b{b}_a5.npz')['pressure']
        exact=g.pad_field(torch.from_numpy(old).cuda()).to(torch.complex128)
        checks['reference_'+str(b)]={'field_relative_difference':float((pref[0]-exact).norm()/exact.norm()),
            'relative_equation_defect':float((s(pref)-rhs).norm()/rhs.norm())}
    row=next(r for r in data.rows if r['id']=='train_200'); medium,alpha=data.material(row)
    s,rhs,pref,line=data.assemble(medium,alpha,360,[5],dtype=torch.float64)
    x=torch.randn_like(rhs)*g.mask; y=torch.randn_like(rhs)*g.mask
    left=(y.conj()*projected_operator(s,x)).sum(); right=(adjoint_operator(s,y).conj()*x).sum()
    error=float(abs(left-right)/max(abs(left),abs(right)))
    checks['adjoint_identity_relative_error']=error
    if error>1e-9: raise RuntimeError('Adjoint identity failed '+str(error))
    solution,info=solve_system(s,rhs,tolerance=1e-6)
    prediction=channels(g,solution)*receiver_response(s.frequency); target=data.target(row,360,[5])
    checks['physical_forward']={**info,'rf_relative_rms':float((prediction-target).norm()/target.norm())}
    if info['reason']!='converged': raise RuntimeError('Reference forward did not converge')

# One deterministic nontrivial joint material direction checks the new backward.
material=medium.detach().double().requires_grad_(True)
pred=implicit_channels(material,alpha.double(),data,360,[5],None,1e-7)
target=data.target(row,360,[5]); loss=(pred-target).abs().square().sum()/target.abs().square().sum()
gradient=torch.autograd.grad(loss,material)[0]
xx=torch.arange(g.w,device='cuda')*g.dx+g.spec['x0_m']; zz=torch.arange(g.h,device='cuda')*g.dx+g.spec['z0_m']
window=torch.exp(-((xx[None,:]-.002)/.004)**2-((zz[:,None]-.012)/.006)**2)*(zz[:,None]>=.001)
direction=torch.stack((2.*window,1.5*window)).double()
analytic=float((gradient*direction).sum()); h=.002
values=[]
with torch.no_grad():
    for sign in (-1,1):
        value=implicit_channels(material.detach()+sign*h*direction,alpha.double(),data,360,[5],None,1e-7)
        values.append(float((value-target).abs().square().sum()/target.abs().square().sum()))
finite=(values[1]-values[0])/(2*h)
checks['material_directional_derivative']={'analytic':analytic,'finite_difference':finite,
    'relative_difference':abs(analytic-finite)/max(abs(analytic),abs(finite),1e-10),'step':h}
checks['passed']=error<1e-9 and checks['material_directional_derivative']['relative_difference']<.02
(ROOT/'physics_checks.json').write_text(json.dumps(checks,indent=2)); print(json.dumps(checks,indent=2),flush=True)
if not checks['passed']: raise RuntimeError('Implicit derivative check failed')
