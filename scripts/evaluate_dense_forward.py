"""Complete dense-band validation: no favorable-frequency subsampling."""
from pathlib import Path
from argparse import ArgumentParser
import json,time,math
import numpy as np
import torch
from nsrno.dense_wave import DenseData,batched_gmres_cycle,channels,solve_system,ROOT
from nsrno.conditioned_wave import LearnedPreconditioner
from nsrno.conditioned_data import receiver_response

def write(path,value):
    temp=path.with_suffix('.partial.json'); temp.write_text(json.dumps(value,indent=2)); temp.replace(path)

def main():
    p=ArgumentParser(); p.add_argument('--part',type=int,required=True); p.add_argument('--parts',type=int,default=3)
    p.add_argument('--out',type=Path,default=Path('out/v5/forward_round8_dense_full'))
    p.add_argument('--root',type=Path,default=Path('out/v5/conditioned_dense_full_v1')); args=p.parse_args()
    torch.set_num_threads(2); data=DenseData(reference_limit=12,root=args.root); angles=list(range(11))
    networks={}
    for name,path in [('before','out/v5/forward_round7_conditioned/last.pt'),('after',args.out/'last.pt')]:
        model=LearnedPreconditioner().cuda(); model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']); model.eval(); networks[name]=model
    path=args.out/f'forward_part{args.part}.json'; rows=[]
    if path.exists(): rows=json.load(open(path))
    completed={(r['id'],r['bin'],r['method']) for r in rows}
    with torch.no_grad():
        for sid in ('train_200','train_000'):
            row=next(r for r in data.rows if r['id']==sid); medium,alpha=data.material(row)
            for bin_index in data.spec['bins']:
                if (bin_index-data.first)%args.parts!=args.part: continue
                target=data.target(row,bin_index,angles)
                system,rhs,_,_=data.assemble(medium,alpha,bin_index,angles)
                neural_initial=None
                methods=('before','after','after64','converged') if sid=='train_200' else ('before','after')
                for name in methods:
                    if (sid,bin_index,name) in completed: continue
                    torch.cuda.synchronize(); start=time.perf_counter(); info={}
                    if name=='converged':
                        s,r,_,_=data.assemble(medium,alpha,bin_index,angles,dtype=torch.float64)
                        field,info=solve_system(s,r,networks['after'],tolerance=1e-5,mixed=True,method='bicgstab',initial=neural_initial)
                        actual=float((s(field)-r).norm()/r.norm())
                    else:
                        selected=networks['after' if name=='after64' else name]
                        field=batched_gmres_cycle(system,selected.setup(system),rhs,64 if name=='after64' else 32)
                        if name=='after64': neural_initial=field.detach()
                        actual=float((system(field)-rhs).norm()/rhs.norm())
                    prediction=channels(data.geometry,field)*receiver_response(system.frequency)
                    torch.cuda.synchronize(); seconds=time.perf_counter()-start
                    squared=(prediction-target).abs().square().sum(dim=-1); norm=target.abs().square().sum(dim=-1)
                    record={'id':sid,'case':row['case'],'bin':bin_index,'frequency_hz':system.frequency,'method':name,
                        'rf_error_sq':float(squared.sum()),'target_sq':float(norm.sum()),
                        'angle_rf_error_sq':squared.cpu().tolist(),'angle_target_sq':norm.cpu().tolist(),
                        'rf_relative_rms':float((squared.sum()/norm.sum()).sqrt()),'pde_residual':actual,'seconds':seconds,'solver':info,
                        'full_rf_target_energy':float(data.full_energies[row['id']].sum())}
                    rows.append(record); completed.add((sid,bin_index,name)); write(path,rows)
                    np.save(args.out/f'prediction_{sid}_{name}_b{bin_index}.npy',prediction.cpu().numpy())
                    print(json.dumps({k:v for k,v in record.items() if not k.startswith('angle_')}),flush=True)
                    if name=='converged' and info['reason']!='converged': raise RuntimeError('Dense physical solve failed')
    write(args.out/f'forward_complete_{args.part}.json',{'completed':True,'rows':len(rows)})

if __name__=='__main__': main()
