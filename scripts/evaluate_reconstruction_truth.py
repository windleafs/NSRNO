"""Post-hoc counterexample: the surrogate objective evaluated at true maps."""
from argparse import ArgumentParser
from pathlib import Path
import json,math
import numpy as np
import torch
from nsrno.conditioned_surrogate import ConditionedSurrogate
from nsrno.conditioned_data import ConditionedData,receiver_response
from nsrno.reconstruct_material import DEFAULT_OUT,regularization,write_json

parser=ArgumentParser(); parser.add_argument('--out',type=Path,default=DEFAULT_OUT); out=parser.parse_args().out
if not (out/'completion.json').exists(): raise RuntimeError('No truth evaluation before inversion completes')
torch.set_num_threads(2); settings=json.load(open(out/'settings.json'))
data=ConditionedData(); row=next(r for r in data.rows if r['id']==settings['id'])
model=ConditionedSurrogate('out/v5/forward_round7_conditioned/last.pt','out/v5/conditioned_wave_v1','cuda')
medium,alpha=data.material(row); fit=torch.tensor(settings['fit_receivers'],device='cuda'); held=torch.tensor(settings['heldout_receivers'],device='cuda')
rows=[]; teacher_rows=[]
with torch.no_grad():
    for b,a in settings['pairs']:
        prediction=model(medium,alpha,b,a,iterations=settings['neural_iterations'])['channels']
        target=data.target(row,b,a)
        teacher=data.geometry.channels(data.teacher(row,b,a))*receiver_response(b*40e6/2401)
        for container,value in [(rows,prediction),(teacher_rows,teacher)]:
            container.append({'bin':b,'angle':a,'fit_error_sq':float((value[fit]-target[fit]).abs().square().sum()),
                'fit_target_sq':float(target[fit].abs().square().sum()),
                'heldout_error_sq':float((value[held]-target[held]).abs().square().sum()),
                'heldout_target_sq':float(target[held].abs().square().sum())})
    q=torch.log(medium/torch.tensor([1540.,1000.],device='cuda')[:,None,None])/torch.tensor([.06,.08],device='cuda')[:,None,None]
    z=torch.arange(data.geometry.h,device='cuda')*data.geometry.dx+data.spec['z0_m']
    free=(z>=settings.get('fixed_gel_mm',0)*.001-1e-7) if settings.get('fixed_gel_mm',0)>0 else (z>1e-7)
    mask=free[:,None].expand(data.geometry.h,data.geometry.w).float()
    penalty,tv,l2=regularization(q,mask)
def summarize(items):
    return {'fit_rms':math.sqrt(sum(r['fit_error_sq'] for r in items)/sum(r['fit_target_sq'] for r in items)),
        'heldout_rms':math.sqrt(sum(r['heldout_error_sq'] for r in items)/sum(r['heldout_target_sq'] for r in items)),
        'equal_acquisition_fit_rms':math.sqrt(np.mean([r['fit_error_sq']/r['fit_target_sq'] for r in items]))}
result={'true_material_neural':summarize(rows),'true_material_converged_reference':summarize(teacher_rows),
    'regularization_at_truth':float(penalty),'true_material_objective':float(np.mean([r['fit_error_sq']/r['fit_target_sq'] for r in rows])+penalty),
    'c_truth_range':[float(medium[0].min()),float(medium[0].max())],
    'rho_truth_range':[float(medium[1].min()),float(medium[1].max())],
    'truth_fraction_outside_bounds':{'c':float(((medium[0]<1400)|(medium[0]>1700)).float().mean()),
        'rho':float(((medium[1]<850)|(medium[1]>1200)).float().mean())},
    'note':'Truth is used for post-hoc diagnosis only, never initialization, iteration choice or gradients.'}
write_json(out/'truth_objective.json',result); print(json.dumps(result,indent=2))
