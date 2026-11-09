"""Diagnose off-truth forward bias using existing time-domain replay, no truth maps."""
from pathlib import Path
from argparse import ArgumentParser
import json
import numpy as np
from scipy import fft
import torch
from nsrno.dense_wave import DenseData,solve_system,channels
from nsrno.conditioned_wave import LearnedPreconditioner
from nsrno.conditioned_data import receiver_response

p=ArgumentParser(); p.add_argument('--bin',type=int,required=True); args=p.parse_args()
torch.set_num_threads(2)
out=Path('out/v5/reconstruction_round2_dense_train_200')
data=DenseData(root='out/v5/conditioned_dense_full_v1',fast_fft=True,reference_limit=12)
alpha=torch.from_numpy(np.load(out/'known_alpha.npy')).cuda()
network=LearnedPreconditioner().cuda()
network.load_state_dict(torch.load('out/v5/forward_round8_dense_full/last.pt',map_location='cpu',weights_only=True)['model'])
network.eval(); network.requires_grad_(False)
observed=fft.rfft(np.load(out/'observed_rf.npy'),axis=-1,norm='forward')[:,:,args.bin]
scale=np.linalg.norm(observed)
result={'frequency_bin':args.bin,'frequency_hz':args.bin*40e6/2401,'angles':11,'truth_read':False,'states':{}}
with torch.no_grad():
    for name,mfile,rfile in [('initial','initial_medium.npy','initial_rf.npy'),('reconstruction','reconstructed_medium.npy','reconstruction_rf.npy')]:
        medium=torch.from_numpy(np.load(out/mfile)).cuda()
        system,rhs,_,_=data.assemble(medium,alpha,args.bin,list(range(11)),dtype=torch.float64)
        u,info=solve_system(system,rhs,network,tolerance=1e-5,mixed=True,method='bicgstab')
        if info['reason']!='converged': raise RuntimeError(str(info))
        predicted=(channels(data.geometry,u)*receiver_response(system.frequency)).cpu().numpy()
        timed=fft.rfft(np.load(out/rfile),axis=-1,norm='forward')[:,:,args.bin]
        result['states'][name]={'pde_residual':info['residual'],'solve_seconds':info['seconds'],
            'frequency_vs_time_relative_to_time':float(np.linalg.norm(predicted-timed)/np.linalg.norm(timed)),
            'frequency_vs_time_relative_to_observed':float(np.linalg.norm(predicted-timed)/scale),
            'frequency_data_relative_rms':float(np.linalg.norm(predicted-observed)/scale),
            'time_data_relative_rms':float(np.linalg.norm(timed-observed)/scale)}
        np.save(out/f'diagnostic_{name}_spectrum_b{args.bin}.npy',predicted)
(out/f'offtruth_forward_b{args.bin}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
