"""Separate tissue attenuation approximation from the known edge absorber."""
from pathlib import Path
import json,time
import numpy as np
import torch
from nsrno.wave_operator import load_wave_operator,C0
from nsrno.dataset import read_json,write_json
from nsrno.native_frequency_reference import NativeFrequencyReference
from nsrno.wave_solver import gmres

torch.set_num_threads(2); out=Path('out/v5/wave_diagnosis_v1'); meta=read_json(out/'train_200_material_field.json'); f=meta['frequency_hz']
model=load_wave_operator('out/v5/forward_round5_wave_relax_v1/best.pt','cuda'); results={}
with torch.no_grad():
    material=np.load(out/'train_200_material_field.npz'); reference=np.load(out/'train_200_reference_field.npz')
    op=NativeFrequencyReference(model,material,meta,f); ref=NativeFrequencyReference(model,reference,meta,f)
    pref=ref.pad(reference['pressure']).to(torch.complex128); true_scattered=op.pad(material['pressure']).to(torch.complex128)-pref
    native=torch.from_numpy(material['receiver_dft']-reference['receiver_dft']).cuda()
    endpoints=op.endpoint()-ref.endpoint(); original_mass=op.mass.clone()
    medium=torch.from_numpy(np.stack((material['original_c'],material['original_rho'])))[None].cuda()
    _,_,multiplier=model.setup(medium,f)
    estimated=model.alpha_log.exp()*(1-torch.exp(-((op.c/C0).log()/.03).square()-(op.a.log()/.05).square()))
    boundary_x=model.x.clamp(-.028+24*model.dx,.028-24*model.dx)
    known_edge=6*((boundary_x.abs()*1000-20.2)/7.8).clamp(0,1).square()[None,:]
    tissue=(model.z[:,None]>=0)&(op.mask>0)
    for name,alpha in [('same_c_rho_estimated_alpha',estimated),('estimated_alpha_plus_known_edge',estimated+known_edge)]:
        eta=[alpha*op.c/C0*v for v in meta['fit']['eta_per_unit_alpha']]
        amplitude=(1+(eta[0]+eta[1])/2).square()
        H=sum(e/(amplitude*(1+t*(op.D-1)/op.dt)) for e,t in zip(eta,meta['fit']['taus_s']))
        modulus=amplitude*(1-H)/(1-(1-op.q)*H)
        mass=op.a*(C0/op.c).square()/modulus
        op.mass=torch.where(tissue,mass,original_mass)
        rhs=ref(pref)-op(pref)+endpoints
        trace=[]; started=time.perf_counter()
        def monitor(count,w,residual): trace[:]=[count,residual]
        def precondition(r): return op.mask*torch.fft.ifft2(multiplier*torch.fft.fft2(r))
        wave,reason=gmres(op,precondition,rhs,torch.zeros_like(pref),monitor,tolerance=1e-7,max_cycles=100)
        result=dict(reason=reason,iterations=trace[0],residual=trace[1],
            native_channel_relative_rms=float((wave[model.face,model.rx_cols]-native).norm()/native.norm()),seconds=time.perf_counter()-started)
        results[name]=result; print(json.dumps({name:result}),flush=True); write_json(out/'absorption_controls.json',results)
