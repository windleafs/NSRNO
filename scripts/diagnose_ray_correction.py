"""One scalar-gain oracle on existing predictions; no fitted gain is deployed."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import json
import numpy as np
import torch
from nsrno.forward_model import load_forward_model
from nsrno.train_ray import loader
from nsrno.rf_representation import RF_SAMPLES,RF_FS,decode,time_windows

torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark=True
root=Path('out/v5/forward_cache_v1'); out=Path('out/v5/forward_round4_material')
index=json.loads((root/'index.json').read_text())
args=SimpleNamespace(cache=root,features=Path('out/v5/forward_ray_material_v1'),
                     shared=Path('out/v5/forward_ray_shared_fp32_v1'),batch=4,eval_batch=4,
                     workers=4,angles_per_record=2,seed=20261105)
model=load_forward_model(out/'best.pt','cuda')
windows=time_windows(RF_SAMPLES,1e6*RF_SAMPLES/RF_FS)

@torch.no_grad()
def measure(rows):
    groups=['all',*sorted({r['variant'] for r in rows})]
    sums={g:torch.zeros((4,4),dtype=torch.float64,device='cuda') for g in groups}
    position=0; first_prediction=None
    for batch in loader(rows,args,fixed=tuple(range(11))):
        bg,indices,features,baseline,target=(v.cuda(non_blocking=True) for v in batch)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            iq=model.predict_cached(bg,indices,features,baseline)
        truth,pred,frozen=decode(target),decode(iq),decode(baseline)
        if first_prediction is None: first_prediction=pred[:1].clone()
        correction=pred-frozen; residual=truth-frozen
        membership={'all':list(range(len(bg)))}
        membership.update({g:[i for i,r in enumerate(rows[position:position+len(bg)]) if r['variant']==g] for g in groups[1:]})
        for j,window in enumerate(windows):
            r,d,y=residual[...,window],correction[...,window],truth[...,window]
            terms=torch.stack((r.square().sum((1,2,3),dtype=torch.float64),
                               d.square().sum((1,2,3),dtype=torch.float64),
                               (r*d).sum((1,2,3),dtype=torch.float64),
                               y.square().sum((1,2,3),dtype=torch.float64)))
            for group,selected in membership.items():
                if selected: sums[group][:,j]+=terms[:,selected].sum(1)
        position+=len(bg)
    result={}
    for group,values in sums.items():
        er,ed,cross,ey=values.cpu().numpy()
        gain=np.divide(cross,ed,out=np.zeros_like(cross),where=ed>1e-20)
        cosine=np.divide(cross,np.sqrt(er*ed),out=np.zeros_like(cross),where=er*ed>1e-20)
        best=np.maximum(0,er-np.divide(cross**2,ed,out=np.zeros_like(cross),where=ed>1e-20))
        result[group]={'optimal_scalar_gain':gain.tolist(),'correction_residual_cosine':cosine.tolist(),
                       'correction_rms_over_residual_rms':np.sqrt(ed/er).tolist(),
                       'baseline_relative_rf_rms':np.sqrt(er/ey).tolist(),
                       'actual_relative_rf_rms':np.sqrt((er+ed-2*cross)/ey).tolist(),
                       'oracle_relative_rf_rms':np.sqrt(best/ey).tolist(),
                       'oracle_error_energy_reduction_vs_baseline':(1-best/er).tolist()}
    return result,first_prediction

result={'description':'Post hoc scalar gain fit per time window, using labels only for diagnosis. Not applied to the model or final reported performance.',
        'checkpoint':'best.pt','best_epoch':70,'angles':11}
for role in ('train','validation'):
    rows=[r for r in index['records'] if r['role']==role and r['variant'] in ('oa_breast_original','high_echo_circles_v1')]
    result[role],first=measure(rows)
    if role=='validation':
        medium=torch.from_numpy(np.load(root/rows[0]['medium'])).cuda()[None]
        torch.cuda.synchronize(); start=perf_counter()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            live=model(medium)
        torch.cuda.synchronize(); seconds=perf_counter()-start
        relative=float((live-first).square().sum().sqrt()/live.square().sum().sqrt())
        result['online_inference']={'record':rows[0]['id'],'angles':11,'shape':list(live.shape),
                   'seconds_single_record':seconds,'relative_rf_rms_vs_cached':relative,
                   'window_relative_difference':[float((live[...,w]-first[...,w]).square().sum().sqrt()/live[...,w].square().sum().sqrt().clamp_min(1e-15)) for w in windows]}
        assert torch.isfinite(live).all() and relative<.003
(out/'correction_alignment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
