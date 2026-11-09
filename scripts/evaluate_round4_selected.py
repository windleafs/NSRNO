from pathlib import Path
import json,math
import numpy as np
import torch
from nsrno.forward_model import load_forward_model
from nsrno.conditioned_data import ConditionedData
from nsrno.rf_representation import decode

torch.set_num_threads(2); data=ConditionedData()
model=load_forward_model(Path('out/v5/forward_round4_material/best.pt'),'cuda').eval()
shared=Path('out/v5/forward_ray_shared_fp32_v1'); results=[]
with torch.no_grad():
    for row in data.rows:
        if row['role']!='validation': continue
        sid=row['id']
        def array(path): return torch.from_numpy(np.load(path)).cuda()
        bg=array(shared/'background'/f'{sid}.npy')[None]
        base=array(shared/'baseline'/f'{sid}.npy')[None]
        features=array(Path('out/v5/forward_ray_material_v1/features')/f'{sid}.npy')[None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model.predict_cached(bg,torch.arange(11,device='cuda')[None],features,base)
        pred=torch.fft.rfft(decode(pred)[0],dim=-1,norm='forward')
        for b in data.spec['bins']:
            for a in data.spec['angle_indices']:
                target=data.target(row,b,a)
                results.append({'id':sid,'bin':b,'angle':a,
                    'rf_error_sq':float((pred[a,:,b]-target).abs().square().sum()),
                    'rf_target_sq':float(target.abs().square().sum())})
summary={'observations':len(results),'rf_relative_rms':math.sqrt(sum(r['rf_error_sq'] for r in results)/sum(r['rf_target_sq'] for r in results))}
Path('out/v5/forward_round7_conditioned/round4_validation.json').write_text(json.dumps({'summary':summary,'rows':results},indent=2))
print(json.dumps(summary),flush=True)
