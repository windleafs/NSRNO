"""Use the short-horizon learned solver as initialization, not a permanent P."""
from pathlib import Path
import json,time,math
import numpy as np
import torch
from nsrno.conditioned_data import ConditionedData
from nsrno.conditioned_wave import LearnedPreconditioner,differentiable_gmres
from nsrno.wave_solver import gmres
from nsrno.train_preconditioner import errors,aggregate

torch.set_num_threads(2); out=Path('out/v5/forward_round7_conditioned'); data=ConditionedData()
saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
model=LearnedPreconditioner().cuda(); model.load_state_dict(saved['model']); model.eval()
val=[r for r in data.rows if r['role']=='validation']
examples=[(val[j],b,5) for j in (0,5,10,15) for b in data.spec['bins']]
results={}; summary={}
with torch.no_grad():
    for name in ('fixed8_warm_start','learned8_warm_start'):
        rows=[]
        for row,b,a in examples:
            truth=data.teacher(row,b,a); target=data.target(row,b,a)
            torch.cuda.synchronize(); start=time.perf_counter()
            s,rhs,_=data.assemble(row,b,a); fixed=s.fixed_preconditioner(); calls=[0]
            def counted(p): calls[0]+=1; return s(p)
            initial=model.setup(s) if name.startswith('learned') else fixed
            u0=differentiable_gmres(counted,initial,rhs,8); trace=[]
            u,reason=gmres(counted,fixed,rhs,u0,lambda n,u,r:trace.append({'steps':n,'residual':r}),
                           restart=32,max_cycles=80,tolerance=1e-4)
            torch.cuda.synchronize(); seconds=time.perf_counter()-start
            _,_,metrics=errors(s,rhs,u,truth,target)
            row_result={'id':row['id'],'bin':b,'angle':a,'seconds':seconds,'reason':reason,
                'steps':trace[-1]['steps']+8,'operator_calls':calls[0]+1,**metrics}
            rows.append(row_result); print(json.dumps({'mode':name,**row_result}),flush=True)
        results[name]=rows; summary[name]={**aggregate(rows),'mean_steps':float(np.mean([r['steps'] for r in rows])),
            'mean_operator_calls':float(np.mean([r['operator_calls'] for r in rows])),
            'all_converged':all(r['reason']=='converged' for r in rows)}
        (out/'hybrid.json').write_text(json.dumps({'summary':summary,'rows':results},indent=2))
    print(json.dumps(summary,indent=2),flush=True)
