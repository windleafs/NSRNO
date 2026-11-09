"""Match native physical RF and measure the checkpointed full-grid adjoint."""
from pathlib import Path
from argparse import ArgumentParser
import json,time
import numpy as np
import torch
from .time_acquisition import TimeAcquisition,truth,ROOT

def main():
    p=ArgumentParser();p.add_argument('--source',default='lf');p.add_argument('--sample',default='train_200')
    p.add_argument('--angle',type=int,default=5);p.add_argument('--gradient',action='store_true');p.add_argument('--chunk',type=int,default=128)
    a=p.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    acq=TimeAcquisition(a.source,a.sample,chunk=a.chunk); material=truth(a.sample)
    if a.gradient:material.requires_grad_()
    torch.cuda.reset_peak_memory_stats();start=time.time();pred=acq.predict(material,a.angle)
    target=acq.observed[a.angle];diff=pred-target
    report={'source':a.source,'sample':a.sample,'angle':a.angle,'forward_seconds':time.time()-start,'relative_rms':float(diff.norm()/target.norm()),'window_relative_rms':{}}
    for begin,end in [(0,8),(8,18),(18,36),(36,90 if a.source=='point3' else 60)]:
        sl=slice(round(begin*40),round(end*40));report['window_relative_rms'][f'{begin}-{end}']=float(diff[:,sl].norm()/target[:,sl].norm())
    if a.gradient:
        start=time.time();(pred.square().sum()/target.square().sum()).backward()
        report.update(backward_seconds=time.time()-start,gradient_norms=[float(g.norm()) for g in material.grad])
        np.save(ROOT/f'first_gradient_{a.source}.npy',material.grad.cpu().numpy())
    report['memory_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30;report['memory_reserved_GiB']=torch.cuda.max_memory_reserved()/2**30
    np.savez(ROOT/f'check_{a.source}_{a.sample}_{a.angle}.npz',prediction=pred.detach().cpu().numpy(),target=target.cpu().numpy())
    path=ROOT/f'check_{a.source}_{a.sample}_{a.angle}.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__': main()
