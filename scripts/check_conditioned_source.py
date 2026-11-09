"""A single implementation check against the existing round-6 capture."""
from pathlib import Path
import json,time
import numpy as np
import torch
from nsrno.conditioned_wave import FrequencyGeometry,ConditionedFrequencySystem
from nsrno.wave_solver import gmres

torch.set_num_threads(2)
root=Path('out/v5/conditioned_wave_v1'); old=Path('out/v5/wave_diagnosis_v1')
spec=json.load(open(root/'spec.json')); g=FrequencyGeometry(spec).cuda()
f=360*40e6/2401; line=torch.from_numpy(np.load(root/'sources.npz')['b360_a5']).cuda()
result={}
with torch.no_grad():
    for kind in ('reference','material'):
        archive=np.load(old/f'train_200_{kind}_field.npz')
        medium=torch.from_numpy(np.stack((archive['original_c'],archive['original_rho']))).cuda()
        alpha=torch.from_numpy(archive['original_alpha']).cuda()
        s=ConditionedFrequencySystem(g,medium,alpha,f)
        rhs=s.source(line); captured=g.pad_field(torch.from_numpy(archive['pressure']).cuda())
        history=[]; start=time.perf_counter()
        solution,reason=gmres(s,s.fixed_preconditioner(),rhs,torch.zeros_like(rhs),
            lambda n,u,r:history.append({'steps':n,'residual':r}),restart=32,max_cycles=100,tolerance=1e-5)
        channels=g.channels(solution); true=torch.from_numpy(archive['receiver_dft']).cuda()
        row={'reason':reason,'last':history[-1],'seconds':time.perf_counter()-start,
             'captured_equation_residual':float((s(captured)-rhs).norm()/rhs.norm()),
             'field_error':float((solution-captured).norm()/captured.norm()),
             'native_channel_error':float((channels-true).norm()/true.norm()),
             'source_norm':float(rhs.norm())}
        result[kind]=row
        np.save(root/f'check_{kind}_solution.npy',g.native(solution).cpu().numpy())
        print(json.dumps({kind:row}),flush=True)
    ref=np.load(root/'check_reference_solution.npy'); mat=np.load(root/'check_material_solution.npy')
    a=np.load(old/'train_200_material_field.npz'); b=np.load(old/'train_200_reference_field.npz')
    cols=g.rx_cols.cpu().numpy()-g.px; face=g.face-g.pz
    pred=(mat-ref)[face,cols]; target=a['receiver_dft']-b['receiver_dft']
    result['scattered_native_channel_error']=float(np.linalg.norm(pred-target)/np.linalg.norm(target))
    if result['reference']['native_channel_error']>.01 or result['scattered_native_channel_error']>.01:
        result['accepted']=False
    else: result['accepted']=True
    (root/'source_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if not result['accepted']: raise RuntimeError('Explicit source/operator disagrees with captured reference')
