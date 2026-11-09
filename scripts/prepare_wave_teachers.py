"""Converged fields for 48 fixed IDs, three frequencies, three angles."""
from argparse import ArgumentParser
from pathlib import Path
import json,time
import numpy as np
import torch
from nsrno.conditioned_wave import ConditionedFrequencySystem
from nsrno.conditioned_data import ConditionedData,ROOT,key,field_path
from nsrno.wave_solver import gmres

def save_json(path,value): path.write_text(json.dumps(value,indent=2))
def main():
    parser=ArgumentParser(); parser.add_argument('--part',type=int,default=0); parser.add_argument('--parts',type=int,default=1)
    args=parser.parse_args(); torch.set_num_threads(2)
    if not json.load(open(ROOT/'source_check.json'))['accepted']: raise RuntimeError('Source check failed')
    data=ConditionedData(); g=data.geometry
    (ROOT/'reference').mkdir(exist_ok=True); (ROOT/'teachers').mkdir(exist_ok=True)
    with torch.no_grad():
        # Each worker owns one frequency and therefore distinct reference files.
        for fi,bin_index in enumerate(data.spec['bins']):
            if fi%args.parts!=args.part: continue
            f=bin_index*40e6/2401
            medium=torch.empty(2,g.h,g.w,device='cuda'); medium[0]=1540.; medium[1]=1000.
            s0=ConditionedFrequencySystem(g,medium,torch.full_like(medium[0],.002),f)
            for angle in data.spec['angle_indices']:
                name=key(bin_index,angle); ref_path=ROOT/'reference'/(name+'.npz')
                if ref_path.exists(): continue
                source=s0.source(data.sources[name]); trace=[]; start=time.perf_counter()
                p,reason=gmres(s0,s0.fixed_preconditioner(),source,torch.zeros_like(source),
                    lambda n,u,r:trace.append({'steps':n,'residual':r}),restart=32,max_cycles=80,tolerance=1e-4)
                if reason!='converged': raise RuntimeError('Homogeneous reference failed '+name+str(trace[-1]))
                np.savez(ref_path,pressure=g.native(p).cpu().numpy(),operator_pressure=g.native(s0(p)).cpu().numpy(),source=g.native(source).cpu().numpy())
                receipt={'reason':reason,'last':trace[-1],'seconds':time.perf_counter()-start}
                save_json(ref_path.with_suffix('.json'),receipt)
                print(json.dumps({'reference':name,**receipt}),flush=True)
            for row in data.rows:
                for angle in data.spec['angle_indices']:
                    path=field_path(row,bin_index,angle)
                    if path.exists() and path.with_suffix('.json').exists(): continue
                    start=time.perf_counter(); system,rhs,_=data.assemble(row,bin_index,angle); trace=[]
                    u,reason=gmres(system,system.fixed_preconditioner(),rhs,torch.zeros_like(rhs),
                        lambda n,u,r:trace.append({'steps':n,'residual':r}),restart=32,max_cycles=80,tolerance=1e-4)
                    if reason!='converged': raise RuntimeError(row['id']+' '+key(bin_index,angle)+' '+str(trace[-1]))
                    channels=g.channels(u); target=data.target(row,bin_index,angle)
                    receipt={'id':row['id'],'role':row['role'],'bin':bin_index,'angle':angle,
                        'reason':reason,'last':trace[-1],'seconds':time.perf_counter()-start,
                        'rf_error_sq':float((channels-target).abs().square().sum()),
                        'rf_target_sq':float(target.abs().square().sum()),
                        'rf_relative_rms':float((channels-target).norm()/target.norm()),
                        'field_norm_sq':float(u.abs().square().sum()),'trace':trace}
                    np.save(path,g.native(u).cpu().numpy()); save_json(path.with_suffix('.json'),receipt)
                    print(json.dumps({k:v for k,v in receipt.items() if k!='trace'}),flush=True)
    save_json(ROOT/f'teachers_complete_{args.part}.json',{'part':args.part,'parts':args.parts,'completed':True})

if __name__=='__main__': main()
