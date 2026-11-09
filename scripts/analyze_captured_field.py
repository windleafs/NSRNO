"""Test the PDE against actual V5 fields, including finite-window endpoints."""
from pathlib import Path
import cmath,json,math
import numpy as np
import torch
from torch.nn import functional as F
from nsrno.wave_operator import load_wave_operator,C0,RHO0
from nsrno.train_wave import sample
from nsrno.dataset import read_json,write_json
from nsrno.rf_representation import decode

torch.set_num_threads(2); out=Path('out/v5/wave_diagnosis_v1'); cache=Path('out/v5/forward_cache_v1')
model=load_wave_operator('out/v5/forward_round5_wave_relax_v1/best.pt','cuda')
meta=read_json(out/'train_200_material_field.json'); f=meta['frequency_hz']; dt=meta['dt_s']; duration=2401/40e6
row=next(r for r in read_json(cache/'index.json')['records'] if r['id']=='train_200')
medium,target=sample(row,cache,5,360); data,_,_=model.setup(medium,f)
D=cmath.exp(2j*math.pi*f*dt); q=1/D; end_factor=q**meta['nt']*dt/duration
k0=2*math.sin(math.pi*f*dt)/(dt*C0)
physical=(model.z[:,None]>.001)&(model.z[:,None]<.040)&(model.x[None,:].abs()<.020)

def pad(a): return F.pad(torch.from_numpy(np.array(a)).cuda(),model.pad)
def coefficients(record):
    # Replicate material coefficients outside the stored domain; residuals are
    # assessed >8 stencil points from its edges, so this extension is immaterial.
    def field(key): return F.pad(torch.from_numpy(record[key]).cuda()[None,None],model.pad,mode='replicate')[0,0]
    c,rho,eta1,eta2=(field(k) for k in ('c','rho','eta1','eta2'))
    a=RHO0/rho; amplitude=(1+(eta1+eta2)/2).square()
    hs=[e/(amplitude*(1+t*(D-1)/dt)) for e,t in zip((eta1,eta2),meta['fit']['taus_s'])]
    H=sum(hs); modulus=amplitude*(1-H)/(1-(1-q)*H)
    ax=2*a*torch.roll(a,-1,-1)/(a+torch.roll(a,-1,-1)); az=2*a*torch.roll(a,-1,-2)/(a+torch.roll(a,-1,-2))
    return (ax,az,a*(C0/c).square()/modulus,k0),(a,c,amplitude,H)

def endpoint(record,coeff,internal):
    a,c,amplitude,H=internal
    ep,evx,evz,erx,erz,es1,es2=[pad(record[k+'_final'])*end_factor for k in ('p','vx','vy','rhox','rhoy','S1','S2')]
    es=es1+es2; H0=H/q
    memory=-H0*(ep+es)-sum((1-dt/t)*s/(1-(1-dt/t)*q) for s,t in zip((es1,es2),meta['fit']['taus_s']))
    e_density=(q*memory/(1-H)-es)/(c.square()*amplitude)
    ax,az,_,_=coeff
    div_ep=model.apply_operator(ep,(ax,az,torch.zeros_like(ep),k0))
    div_v=torch.fft.ifft2(-model.Dx.conj()*torch.fft.fft2(evx)-model.Dz.conj()*torch.fft.fft2(evz))
    return div_ep/q-RHO0*div_v/(dt*q*k0*k0)+a*(1-q)*(erx+erz)/(dt*dt*q*k0*k0)-a*C0*C0*e_density

def ratio(residual,denominator):
    return float(residual[physical].norm()/denominator[physical].norm().clamp_min(1e-20))

with torch.no_grad():
    material=np.load(out/'train_200_material_field.npz'); reference=np.load(out/'train_200_reference_field.npz')
    p=pad(material['pressure']); pref=pad(reference['pressure']); scattered=p-pref
    oracle,internals=coefficients(material); ref_oracle,ref_internals=coefficients(reference)
    finite=endpoint(material,oracle,internals); finite_ref=endpoint(reference,ref_oracle,ref_internals)
    approximate_ref=(torch.ones_like(data[0]),torch.ones_like(data[1]),1-1j*model.sponge,data[-1])
    results={}
    for name,L,Lref in [('learned_coefficients',data,approximate_ref),('generator_coefficients',oracle,ref_oracle)]:
        rhs=-(model.apply_operator(pref,L)-model.apply_operator(pref,Lref))
        residual=model.apply_operator(scattered,L)-rhs
        results[name]=dict(relative_scattered_equation_defect=ratio(residual,rhs),
            defect_after_endpoint_correction=ratio(residual-(finite-finite_ref),rhs),
            endpoint_term_relative_to_rhs=ratio(finite-finite_ref,rhs))
    results['oracle_total_field']=dict(local_equation_defect_relative_pressure=ratio(model.apply_operator(p,oracle),p),
        after_endpoint_relative_pressure=ratio(model.apply_operator(p,oracle)-finite,p))
    incident=model.incident(f,5); scale=(incident[physical].conj()*pref[physical]).sum()/incident[physical].abs().square().sum()
    results['incident_shape_error_after_one_complex_scalar']=ratio(scale*incident-pref,pref)
    native=torch.from_numpy(material['receiver_dft']-reference['receiver_dft']).cuda()
    scale_native=(native.conj()*target).sum()/native.abs().square().sum()
    results['raw_vs_filtered_spectrum']=dict(relative_error=float((native-target).norm()/target.norm()),
        after_scalar=float((native*scale_native-target).norm()/target.norm()),scalar=[float(scale_native.real),float(scale_native.imag)])
    iq=torch.from_numpy(np.array(np.load(cache/row['iq'])[5]))[None,None].cuda()
    archived=decode(iq)[0,0]; replay=torch.from_numpy(np.load(out/'train_200_processed_rf.npy')).cuda()
    results['instrumented_replay_relative_rf_rms']=float((replay-archived).norm()/archived.norm())
    write_json(out/'field_consistency.json',results); print(json.dumps(results,indent=2),flush=True)
    np.savez(out/'field_residual_maps.npz',p=p.cpu().numpy(),reference=pref.cpu().numpy(),
             oracle_residual=(model.apply_operator(p,oracle)-finite).cpu().numpy(),
             approximate_residual=(model.apply_operator(p,data)-finite).cpu().numpy(),mask=physical.cpu().numpy())
