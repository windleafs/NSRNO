"""Does the inverse result fit RF under the converged physical equation?"""
from pathlib import Path
from argparse import ArgumentParser
import json,time,math
import numpy as np
import torch
from nsrno.conditioned_wave import ConditionedFrequencySystem
from nsrno.conditioned_data import ConditionedData,receiver_response,key
from nsrno.wave_solver import gmres
from nsrno.reconstruct_material import DEFAULT_OUT,write_json

def main():
    parser=ArgumentParser(); parser.add_argument('--out',type=Path,default=DEFAULT_OUT); args=parser.parse_args()
    torch.set_num_threads(2); out=args.out; data=ConditionedData(); g=data.geometry
    if not (out/'completion.json').exists(): raise RuntimeError('Complete inversion before final evaluation')
    settings=json.load(open(out/'settings.json')); row=next(r for r in data.rows if r['id']==settings['id'])
    # Weak scattered RHS in a nearly uniform candidate amplifies FP32 residual
    # roundoff. Evaluate with complex128 fields and float64 material coefficients.
    alpha=torch.from_numpy(np.load(out/'known_alpha.npy')).cuda().double()
    fit=torch.tensor(settings['fit_receivers'],device='cuda'); heldout=torch.tensor(settings['heldout_receivers'],device='cuda')
    target={tuple(pair):data.target(row,*pair) for pair in settings['pairs']}
    results={}; summary={}
    with torch.no_grad():
        for name,filename in [('initial','initial_medium.npy'),('reconstruction','reconstructed_medium.npy')]:
            medium=torch.from_numpy(np.load(out/filename)).cuda().double(); rows=[]; spectra={}
            for b,a in settings['pairs']:
                start=time.perf_counter(); frequency=b*40e6/2401
                s=ConditionedFrequencySystem(g,medium,alpha,frequency); ref=data.reference(b,a)
                uniform=torch.empty_like(medium); uniform[0]=1540.; uniform[1]=1000.
                s0=ConditionedFrequencySystem(g,uniform,torch.full_like(alpha,.002),frequency)
                pressure=ref['pressure'].to(torch.complex128); line=data.sources[key(b,a)].to(torch.complex128)
                rhs=s.source(line)-s0.source(line)-s(pressure)+s0(pressure)
                trace=[]
                u,reason=gmres(s,s.fixed_preconditioner(),rhs,torch.zeros_like(rhs),
                    lambda n,u,r:trace.append({'steps':n,'residual':r}),restart=32,max_cycles=100,tolerance=1e-5)
                channels=g.channels(u)*receiver_response(frequency); true=target[b,a]
                item={'bin':b,'angle':a,'reason':reason,'last':trace[-1],'seconds':time.perf_counter()-start,'precision':'complex128',
                    'fit_error_sq':float((channels[fit]-true[fit]).abs().square().sum()),'fit_target_sq':float(true[fit].abs().square().sum()),
                    'heldout_error_sq':float((channels[heldout]-true[heldout]).abs().square().sum()),'heldout_target_sq':float(true[heldout].abs().square().sum())}
                rows.append(item); spectra[key(b,a)]=channels.cpu().numpy()
                print(json.dumps({'medium':name,**item}),flush=True)
                if reason!='converged': raise RuntimeError('Physical evaluation did not converge')
            results[name]=rows
            summary[name]={'fit_rms':math.sqrt(sum(r['fit_error_sq'] for r in rows)/sum(r['fit_target_sq'] for r in rows)),
                'heldout_rms':math.sqrt(sum(r['heldout_error_sq'] for r in rows)/sum(r['heldout_target_sq'] for r in rows)),
                'all_rms':math.sqrt(sum(r['fit_error_sq']+r['heldout_error_sq'] for r in rows)/sum(r['fit_target_sq']+r['heldout_target_sq'] for r in rows)),
                'equal_acquisition_fit_rms':math.sqrt(np.mean([r['fit_error_sq']/r['fit_target_sq'] for r in rows])),
                'max_residual':max(r['last']['residual'] for r in rows),'mean_steps':float(np.mean([r['last']['steps'] for r in rows]))}
            np.savez(out/(name+'_converged_spectra.npz'),**spectra)
            write_json(out/'physical_evaluation.json',{'summary':summary,'observations':results})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
