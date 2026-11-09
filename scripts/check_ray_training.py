"""One real cached batch: initialization, optimizer gradients, and live/cache parity."""
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch
from nsrno.forward_model import make_forward_model
from nsrno.ray_model import make_background
from nsrno.train_ray import loader,loss_fn

torch.set_num_threads(2); torch.manual_seed(20261105)
torch.set_float32_matmul_precision('high'); torch.backends.cudnn.benchmark=True
root=Path('out/v5/forward_cache_v1')
feature=Path('out/v5/forward_ray_material_v1')
index=json.loads((root/'index.json').read_text())
settings=json.loads((feature/'settings.json').read_text())
old=torch.load('out/v5/forward_round3_slowness/best.pt',map_location='cpu',weights_only=True)
spec={**settings['spec'],'architecture':'ray_arrival_v1','base_spec':old['spec'],
      'ray_delay':'material','ray_feature_scales':[.5,.5]}
model=make_forward_model(spec); model.base.load_state_dict(old['model']); model.cuda()
args=SimpleNamespace(cache=root,features=feature,shared=Path('out/v5/forward_ray_shared_fp32_v1'),
                     batch=4,eval_batch=4,workers=0,angles_per_record=2,seed=20261105)
rows=index['records'][:4]
batch=next(iter(loader(rows,args,fixed=(0,5))))
bg,ang,features,base,target=[v.cuda() for v in batch]
params=[p for p in model.parameters() if p.requires_grad]
opt=torch.optim.AdamW(params,lr=3e-4)
timings=[]
for step in range(3):
    torch.cuda.synchronize(); start=perf_counter(); opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16): prediction=model.predict_cached(bg,ang,features,base)
    if step==0:
        zero_error=float((prediction-base).abs().max()); assert zero_error==0.
    loss=loss_fn(prediction,target,spec['output_window_scales']); loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
    pulse_grad=float(model.pulse_real.grad.norm()); assert pulse_grad>0
    torch.nn.utils.clip_grad_norm_(params,2.); opt.step(); torch.cuda.synchronize()
    timings.append(perf_counter()-start)
medium=torch.from_numpy(np.load(root/rows[0]['medium'])).cuda()[None]
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    live=model.predict_iq(medium,[0,5])
    cached=model.predict_cached(bg[:1],ang[:1],features[:1],base[:1])
relative=float((live-cached).square().sum().sqrt()/live.square().sum().sqrt())
with torch.no_grad():
    direct_features=model.propagation(medium,[0,5])
    direct_background=make_background(medium,spec)
    direct_base=model.base.predict_iq(medium,[0,5])
    all_base=model.base.predict_iq(medium)[:,[0,5]]
def rel(a,b): return float((a-b).square().sum().sqrt()/a.square().sum().sqrt().clamp_min(1e-15))
print(json.dumps({'live_cache':relative,'features':rel(direct_features,features[:1].float()),
                  'background':rel(direct_background,bg[:1]),'base':rel(direct_base,base[:1]),
                  'all_base':rel(all_base,base[:1]),'all_vs_subset_base':rel(all_base,direct_base),
                  'live_rms':float(live.square().mean().sqrt()),'cached_rms':float(cached.square().mean().sqrt()),
                  'loss':float(loss),'timings':timings}),flush=True)
assert relative<.003
result={'zero_initial_correction_max':zero_error,'finite_nonzero_pulse_gradient':pulse_grad,
        'three_update_seconds':timings,'last_batch_loss':float(loss),'live_cache_relative_iq_rms':relative,
        'trainable_parameters':sum(p.numel() for p in params)}
Path('out/v5/forward_round4_checks/training_batch.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
