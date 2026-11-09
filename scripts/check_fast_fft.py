from pathlib import Path
import json,time
import numpy as np
import torch
from nsrno.dense_wave import DenseData,solve_system,channels,adjoint_operator,projected_operator,implicit_channels
from nsrno.conditioned_wave import LearnedPreconditioner
from nsrno.conditioned_data import receiver_response
torch.set_num_threads(2); out=Path('out/v5/reconstruction_round2_dense_train_200')
data=DenseData(root='out/v5/conditioned_dense_full_v1'); fast=DenseData(root='out/v5/conditioned_dense_full_v1',fast_fft=True)
state=torch.load(out/'resume.pt',map_location='cuda',weights_only=True)
alpha=torch.from_numpy(np.load(out/'known_alpha.npy')).cuda()
g=data.geometry; z=torch.arange(g.h,device='cuda')*g.dx+g.spec['z0_m']; support=(z>=.001-1e-7)[:,None].expand(g.h,g.w).float()
from nsrno.reconstruct_material import build_medium
medium=build_medium(state['q'],support)
network=LearnedPreconditioner().cuda(); network.load_state_dict(torch.load('out/v5/forward_round8_dense_full/last.pt',map_location='cpu',weights_only=True)['model']); network.eval(); network.requires_grad_(False)
result={}; outputs=[]
with torch.no_grad():
    # Same native arbitrary field, different FFT padding; both represent the
    # same compact8th-order staggered derivatives with zero exterior pressure.
    native=torch.randn(g.h,g.w,device='cuda',dtype=torch.complex128)
    systems=[]
    for label,loader in [('original',data),('fast',fast)]:
        system,rhs,_,_=loader.assemble(medium,alpha,360,[5],dtype=torch.float64)
        systems.append(system)
        operator_value=loader.geometry.native(projected_operator(system,loader.geometry.pad_field(native)))
        outputs.append(operator_value)
        field,info=solve_system(system,rhs,network,tolerance=1e-7,mixed=True,method='bicgstab')
        result[label]={**info,'channels':channels(loader.geometry,field).cpu().numpy()}
    result['native_operator_relative_difference']=float((outputs[0]-outputs[1]).norm()/outputs[0].norm())
    a=result['original'].pop('channels'); b=result['fast'].pop('channels')
    result['channel_relative_difference']=float(np.linalg.norm(a-b)/np.linalg.norm(a))
    s=systems[1]; mask=fast.geometry.mask
    x=torch.randn_like(mask,dtype=torch.complex128)*mask; y=torch.randn_like(x)*mask
    left=(y.conj()*projected_operator(s,x)).sum(); right=(adjoint_operator(s,y).conj()*x).sum()
    result['fast_adjoint_identity_error']=float(abs(left-right)/max(abs(left),abs(right)))
result['passed']=result['native_operator_relative_difference']<2e-6 and result['channel_relative_difference']<.001 and result['fast_adjoint_identity_error']<1e-10
(out/'fast_fft_check.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
if not result['passed']: raise RuntimeError('Fast FFT equivalence check failed')
