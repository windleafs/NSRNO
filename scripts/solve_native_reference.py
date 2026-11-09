"""Resolve coefficient and boundary errors using the generator's coefficients."""
from pathlib import Path
import json,time
import numpy as np
import torch
from nsrno.wave_operator import load_wave_operator
from nsrno.train_wave import sample
from nsrno.dataset import read_json,write_json
from nsrno.native_frequency_reference import NativeFrequencyReference
from scripts.solve_wave_reference import gmres

torch.set_num_threads(2); out=Path('out/v5/wave_diagnosis_v1'); cache=Path('out/v5/forward_cache_v1')
meta=read_json(out/'train_200_material_field.json'); f=meta['frequency_hz']
model=load_wave_operator('out/v5/forward_round5_wave_relax_v1/best.pt','cuda')
row=next(r for r in read_json(cache/'index.json')['records'] if r['id']=='train_200')
medium,target=sample(row,cache,5,360); results={}
with torch.no_grad():
    material=np.load(out/'train_200_material_field.npz'); reference=np.load(out/'train_200_reference_field.npz')
    physical=NativeFrequencyReference(model,material,meta,f); uniform=NativeFrequencyReference(model,reference,meta,f)
    pref=uniform.pad(reference['pressure']).to(torch.complex128); truth=physical.pad(material['pressure']).to(torch.complex128)-pref
    endpoints=physical.endpoint()-uniform.endpoint()
    data,gain,multiplier=model.setup(medium,f)
    exact_mass=physical.mass.clone()
    for name in ('native_true_coefficients','native_estimated_tissue_absorption'):
        if name=='native_estimated_tissue_absorption':
            tissue=(model.z[:,None]>=0)&(physical.mask>0)
            physical.mass=torch.where(tissue,data[2]+1j*model.sponge,exact_mass)
        else: physical.mass=exact_mass
        rhs=uniform(pref)-physical(pref)+endpoints
        initial_defect=float((physical(truth)-rhs).norm()/rhs.norm())
        history=[]; timer=time.perf_counter()
        def callback(steps,p,residual):
            channels=p[model.face,model.rx_cols]
            raw=torch.from_numpy(material['receiver_dft']-reference['receiver_dft']).cuda()
            row=dict(steps=steps,pde_relative_residual=residual,
                relative_rms_to_native_channels=float((channels-raw).norm()/raw.norm()),
                relative_rms_to_filtered_target=float((channels-target).norm()/target.norm()),
                seconds=time.perf_counter()-timer)
            history.append(row)
        def precondition(r): return physical.mask*torch.fft.ifft2(multiplier*torch.fft.fft2(r))
        solution,reason=gmres(physical,precondition,rhs,torch.zeros_like(pref),callback,max_cycles=100,tolerance=1e-7)
        results[name]=dict(field_equation_defect=initial_defect,reason=reason,last=history[-1],history=history)
        np.save(out/f'{name}_fp64_solution.npy',solution.cpu().numpy())
        write_json(out/'native_reference_fp64_results.json',results)
        print(json.dumps({name:{k:v for k,v in results[name].items() if k!='history'}}),flush=True)
