"""One mechanism check: does solving the learned PDE further improve RF?"""
from pathlib import Path
import json,time
import numpy as np
import torch
from nsrno.wave_operator import WaveResidualOperator
from nsrno.train_wave import sample
from nsrno.dataset import read_json,write_json

torch.set_num_threads(2); root=Path('out/v5/forward_round5_wave_v1'); cache=Path('out/v5/forward_cache_v1')
saved=torch.load(root/'best.pt',map_location='cpu',weights_only=True); model=WaveResidualOperator(saved['spec']).cuda()
model.load_state_dict(saved['model']); model.eval(); index=read_json(cache/'index.json')
ids=read_json(root/'run.json')['validation_probe']; by_id={r['id']:r for r in index['records']}
result=[]
with torch.no_grad():
    for iterations in (8,32,128):
        model.iterations=iterations; error=energy=0.; residual=[]
        for i,sid in enumerate(ids):
            angle=(0,5,10)[i%3]; bin_index=(270,300,330,360,390,420)[i%6]
            medium,target=sample(by_id[sid],cache,angle,bin_index)
            pred,physics,_=model(medium,bin_index*40e6/2401,angle)
            error+=float((pred-target).abs().square().sum()); energy+=float(target.abs().square().sum())
            residual.append(float(physics.sqrt()))
        row=dict(iterations=iterations,relative_channel_spectrum_rms=(error/energy)**.5,pde_relative_residual_mean=float(np.mean(residual)))
        result.append(row); print(json.dumps(row),flush=True)
write_json(root/'iteration_diagnosis.json',result)
