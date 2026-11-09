"""Decision check: short-horizon improvement versus convergence acceleration."""
from pathlib import Path
import json,time,math
import numpy as np
import torch
from nsrno.conditioned_wave import LearnedPreconditioner,differentiable_gmres
from nsrno.conditioned_data import ConditionedData,receiver_response
from nsrno.train_preconditioner import errors,aggregate
from nsrno.wave_solver import gmres

torch.set_num_threads(2); data=ConditionedData(); out=Path('out/v5/forward_round7_conditioned')
checkpoint=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
model=LearnedPreconditioner().cuda(); model.load_state_dict(checkpoint['model']); model.eval()
val=[r for r in data.rows if r['role']=='validation']
monitor=[(val[j],b,5) for j in (0,5,10,15) for b in data.spec['bins']]
results={}; summary={}
with torch.no_grad():
    # Learned gain and spectral changes are examined separately, without refit.
    for mode in ('spectral_only','spatial_only'):
        rows=[]
        for row in val:
            for b in data.spec['bins']:
                for a in data.spec['angle_indices']:
                    s,rhs,_=data.assemble(row,b,a); truth=data.teacher(row,b,a); target=data.target(row,b,a)
                    pre=model.setup(s,spatial=mode=='spatial_only',spectral=mode=='spectral_only')
                    u=differentiable_gmres(s,pre,rhs,8); _,_,metrics=errors(s,rhs,u,truth,target)
                    rows.append({'id':row['id'],'bin':b,'angle':a,**metrics})
        results[mode]=rows; summary[mode]=aggregate(rows)
        print(json.dumps({mode:summary[mode]}),flush=True)
    # Same residual tolerance and restart for both; no RF-based stopping.
    for mode in ('fixed_converged','learned_converged'):
        rows=[]
        for row,b,a in monitor:
            truth=data.teacher(row,b,a); target=data.target(row,b,a)
            torch.cuda.synchronize(); start=time.perf_counter()
            s,rhs,_=data.assemble(row,b,a)
            pre=model.setup(s) if mode.startswith('learned') else s.fixed_preconditioner()
            trace=[]; calls=[0]
            def counted(p): calls[0]+=1; return s(p)
            u,reason=gmres(counted,pre,rhs,torch.zeros_like(rhs),lambda n,u,r:trace.append({'steps':n,'residual':r}),
                           restart=32,max_cycles=80,tolerance=1e-4)
            torch.cuda.synchronize(); seconds=time.perf_counter()-start
            _,_,metrics=errors(s,rhs,u,truth,target)
            rows.append({'id':row['id'],'bin':b,'angle':a,'seconds':seconds,'reason':reason,
                'steps':trace[-1]['steps'],'operator_calls':calls[0]+1,**metrics})
            print(json.dumps({'mode':mode,**rows[-1]}),flush=True)
        results[mode]=rows; summary[mode]={**aggregate(rows),
            'mean_steps':float(np.mean([r['steps'] for r in rows])),
            'mean_operator_calls':float(np.mean([r['operator_calls'] for r in rows])),
            'all_converged':all(r['reason']=='converged' for r in rows)}
        (out/'mechanism.json').write_text(json.dumps({'summary':summary,'rows':results},indent=2))
    print(json.dumps(summary,indent=2),flush=True)
