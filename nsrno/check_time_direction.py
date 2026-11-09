"""One full-grid direction verifies the acquisition/material/adjoint composition."""
import json
import numpy as np
import torch
from .time_acquisition import TimeAcquisition,truth,ROOT

torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
a=TimeAcquisition('lf');m=truth();g=torch.from_numpy(np.load(ROOT/'first_gradient_lf.npy')).to('cuda')
z,x=torch.meshgrid(torch.arange(m.shape[-2],device='cuda')*50e-6-.003,torch.arange(m.shape[-1],device='cuda')*50e-6-.028,indexing='ij')
g[:,z<.001]=0
d=g/g.abs().amax((-2,-1),keepdim=True)*torch.tensor([20,20,.1],device='cuda')[:,None,None]
adj=float((g*d).sum());h=.005;values=[]
with torch.no_grad():
    for s in [-1,1]:
        p=a.predict(m+s*h*d,5);values.append(p.double())
base=torch.from_numpy(np.load(ROOT/'check_lf_train_200_5.npz')['prediction']).to('cuda').double()
fd=float(((values[1]-values[0])*2*base).sum()/(2*h*a.observed[5].double().square().sum()))
r={'adjoint_direction':adj,'finite_difference':fd,'relative_error':abs(fd-adj)/max(abs(fd),abs(adj)),'step':h}
(ROOT/'full_direction.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
