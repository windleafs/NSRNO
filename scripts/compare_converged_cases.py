"""Check the convergence mechanism across both cases, frequencies and angles."""
from pathlib import Path
import json,time
import numpy as np
import torch
from nsrno.wave_operator import load_wave_operator
from nsrno.forward_model import load_forward_model
from nsrno.train_wave import sample
from nsrno.rf_representation import decode
from nsrno.dataset import read_json,write_json
from scripts.solve_wave_reference import gmres

torch.set_num_threads(2); out=Path('out/v5/wave_diagnosis_v1'); cache=Path('out/v5/forward_cache_v1')
model=load_wave_operator('out/v5/forward_round5_wave_relax_v1/best.pt','cuda')
previous=load_forward_model(Path('out/v5/forward_round4_material/best.pt'),'cuda'); previous.eval()
by_id={r['id']:r for r in read_json(cache/'index.json')['records']}; results=[]
with torch.no_grad():
    for sid in ('train_000','train_200'):
        shared=Path('out/v5/forward_ray_shared_fp32_v1')
        def array(path): return torch.from_numpy(np.load(path)).cuda()
        bg=array(shared/'background'/f'{sid}.npy')[None]
        base=array(shared/'baseline'/f'{sid}.npy')[None]
        features=array(Path('out/v5/forward_ray_material_v1/features')/f'{sid}.npy')[None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            legacy=previous.predict_cached(bg,torch.arange(11,device='cuda')[None],features,base)
        legacy=torch.fft.rfft(decode(legacy)[0],dim=-1,norm='forward')
        for angle in (0,5,10):
            for bin_index in (270,360,420):
                timer=time.perf_counter(); f=bin_index*40e6/2401
                medium,target=sample(by_id[sid],cache,angle,bin_index)
                data,gain,multiplier=model.setup(medium,f); incident=model.incident(f,angle)
                ax,az,b,k0=data; rhs=-model.apply_operator(incident,(ax-1,az-1,b-(1-1j*model.sponge),k0))
                u=torch.zeros_like(incident)
                for _ in range(8): u=model._iteration(u,rhs,*data,gain,multiplier)
                pred8,_,_=model(medium,f,angle); value=u[model.face,model.rx_cols]
                factor=(value.conj()*pred8).sum()/value.abs().square().sum()
                state=[]
                def callback(steps,p,residual): state[:]=[steps,residual]
                def precondition(r): return torch.fft.ifft2(multiplier*torch.fft.fft2(r))
                solution,reason=gmres(lambda p:model.apply_operator(p,data),precondition,rhs,u,callback)
                pred=solution[model.face,model.rx_cols]*factor
                def error(p): return float((p-target).norm()/target.norm())
                record=dict(id=sid,case=by_id[sid]['case'],frequency_hz=f,angle_index=angle,
                    wave8_rms=error(pred8),converged_rms=error(pred),round4_rms=error(legacy[angle,:,bin_index]),
                    target_energy=float(target.abs().square().sum()),reason=reason,steps=state[0],
                    pde_relative_residual=state[1],seconds=time.perf_counter()-timer)
                results.append(record); write_json(out/'converged_case_comparison.json',results); print(json.dumps(record),flush=True)
    summary={}
    for case in ('Neg_07_Left','Neg_35_Left','both'):
        rows=[r for r in results if r['case']==case or case=='both']; total=sum(r['target_energy'] for r in rows)
        summary[case]={key:(sum(r[key]**2*r['target_energy'] for r in rows)/total)**.5 for key in ('wave8_rms','converged_rms','round4_rms')}
        summary[case]['all_converged']=all(r['reason']=='converged' for r in rows)
    write_json(out/'converged_case_summary.json',summary); print(json.dumps(summary),flush=True)
