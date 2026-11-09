from pathlib import Path
import json
import numpy as np
import torch
from nsrno.conditioned_surrogate import ConditionedSurrogate
from nsrno.conditioned_data import ConditionedData
torch.set_num_threads(2); out=Path('out/v5/forward_round7_conditioned')
data=ConditionedData(); model=ConditionedSurrogate(out/'last.pt','out/v5/conditioned_wave_v1','cuda')
stored=json.load(open(out/'validation.json'))['learned8']['rows'][0]
row=next(r for r in data.rows if r['id']==stored['id'])
medium,alpha=data.material(row); target=data.target(row,stored['bin'],stored['angle'])
with torch.no_grad():
    result=model(medium,alpha,stored['bin'],stored['angle'])
    error=float((result['channels']-target).abs().square().sum())
    relative_difference=abs(error-stored['rf_error_sq'])/max(stored['rf_error_sq'],1e-20)
    check={'id':row['id'],'bin':stored['bin'],'angle':stored['angle'],'relative_error_metric_difference':relative_difference,
           'pde_relative_residual':float(result['relative_residual']),'passed':relative_difference<1e-5}
    (out/'api_check.json').write_text(json.dumps(check,indent=2)); print(json.dumps(check))
    if not check['passed']: raise RuntimeError('Inference API differs from evaluated model')
