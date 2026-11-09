"""Complete-band physical RF evaluation of initial and reconstructed maps."""
from pathlib import Path
from argparse import ArgumentParser
import json,time,math
import numpy as np
import torch
from nsrno.dense_wave import DenseData,solve_system,channels
from nsrno.conditioned_wave import LearnedPreconditioner
from nsrno.conditioned_data import receiver_response
from nsrno.reconstruct_material import write_json

p=ArgumentParser(); p.add_argument('--part',type=int,required=True); args=p.parse_args()
torch.set_num_threads(2); out=Path('out/v5/reconstruction_round2_dense_train_200')
if not (out/'completion.json').exists(): raise RuntimeError('Dense reconstruction not finished')
data=DenseData(root='out/v5/conditioned_dense_full_v1',reference_limit=12); row=next(r for r in data.rows if r['id']=='train_200')
alpha=torch.from_numpy(np.load(out/'known_alpha.npy')).cuda(); angles=list(range(11))
network=LearnedPreconditioner().cuda(); network.load_state_dict(torch.load('out/v5/forward_round8_dense_full/last.pt',map_location='cpu',weights_only=True)['model']); network.eval(); network.requires_grad_(False)
path=out/f'physical_part{args.part}.json'; rows=json.load(open(path)) if path.exists() else []
completed={(r['state'],r['bin']) for r in rows}
with torch.no_grad():
    for state,filename in [('initial','initial_medium.npy'),('reconstruction','reconstructed_medium.npy')]:
        medium=torch.from_numpy(np.load(out/filename)).cuda()
        for b in data.spec['bins']:
            if (b-data.first)%3!=args.part or (state,b) in completed: continue
            system,rhs,_,_=data.assemble(medium,alpha,b,angles,dtype=torch.float64)
            value,info=solve_system(system,rhs,network,tolerance=1e-5,mixed=True,method='bicgstab')
            if info['reason']!='converged':
                value,info=solve_system(system,rhs,None,tolerance=1e-5,mixed=True)
            if info['reason']!='converged': raise RuntimeError('Dense final physical evaluation failed '+str(info))
            pred=channels(data.geometry,value)*receiver_response(system.frequency); target=data.target(row,b,angles)
            record={'state':state,'bin':b,'frequency_hz':system.frequency,'error_sq':float((pred-target).abs().square().sum()),
                'target_sq':float(target.abs().square().sum()),'full_rf_target_energy':float(data.full_energies[row['id']].sum()),'solver':info}
            rows.append(record); write_json(path,rows)
            np.save(out/f'prediction_{state}_b{b}.npy',pred.cpu().numpy())
            print(json.dumps(record),flush=True)
write_json(out/f'physical_complete_{args.part}.json',{'completed':True,'rows':len(rows)})
