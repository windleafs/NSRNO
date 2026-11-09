from pathlib import Path
import json
import torch
from nsrno.dataset import read_json,write_json
from nsrno.train_wave import sample
from nsrno.wave_operator import load_wave_operator
from nsrno.wave_solver import solve_frequency

torch.set_num_threads(2); cache=Path('out/v5/forward_cache_v1')
row=next(r for r in read_json(cache/'index.json')['records'] if r['id']=='train_200')
medium,target=sample(row,cache,5,360); model=load_wave_operator('out/v5/forward_round5_wave_relax_v1/best.pt','cuda')
with torch.no_grad(): initial,_,_=model(medium,360*40e6/2401,5)
prediction,status=solve_frequency(model,medium,360*40e6/2401,5)
report=dict(initial_rms=float((initial-target).norm()/target.norm()),reference_rms=float((prediction-target).norm()/target.norm()),
            **{k:v for k,v in status.items() if k!='trace'})
assert abs(report['initial_rms']-.7341397358)<1e-5
assert abs(report['reference_rms']-.2455874626)<2e-4
assert status['reason']=='converged'
write_json(Path('out/v5/wave_diagnosis_v1/reference_api_check.json'),report); print(json.dumps(report))
