from pathlib import Path
import json,time
import torch
from nsrno.dense_wave import DenseData,solve_system,channels
from nsrno.conditioned_wave import LearnedPreconditioner
from nsrno.conditioned_data import receiver_response
torch.set_num_threads(2); data=DenseData(reference_limit=12); row=next(r for r in data.rows if r['id']=='train_200')
medium,alpha=data.material(row); network=LearnedPreconditioner().cuda()
path=Path('out/v5/forward_round8_dense/last.pt')
if not path.exists(): path=Path('out/v5/forward_round7_conditioned/last.pt')
network.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']); network.eval()
results={}
with torch.no_grad():
    system,rhs,_,_=data.assemble(medium,alpha,360,list(range(11)),dtype=torch.float64)
    for name,net in [('fixed',None),('learned',network)]:
        value,info=solve_system(system,rhs,net,tolerance=1e-5,mixed=True,method='bicgstab')
        pred=channels(data.geometry,value)*receiver_response(system.frequency); target=data.target(row,360,list(range(11)))
        results[name]={**info,'rf_relative_rms':float((pred-target).norm()/target.norm())}
        print(json.dumps({name:results[name]}),flush=True)
Path('out/v5/conditioned_dense_v1/bicgstab_profile.json').write_text(json.dumps(results,indent=2))
