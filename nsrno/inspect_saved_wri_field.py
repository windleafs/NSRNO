"""True normal residual of an already completed saved frequency block.

Reads X and the field-stage Y_base, reconstructing the augmented RHS without
rerunning any propagation. The result is local to this frequency, not proof
that the entire acquisition or reconstruction has converged.
"""
from pathlib import Path
import argparse,ctypes,json,time,subprocess
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
import numpy as np
import torch
from .conditioned_wave import FrequencyGeometry,ConditionedFrequencySystem,LearnedPreconditioner
from .dense_wave import learned_apply
from .first_order_wavefield import FirstOrderWavefield
from .lf_frequency_data import receiver_response


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--position',type=int,default=89)
    p.add_argument('--device',type=int,default=0);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();start=time.time();folder=a.root/f'step{a.step:03d}'
    if a.out.exists():raise FileExistsError('Preserve existing saved-field residual measurement')
    run=json.loads((a.root/'run.json').read_text())
    if run['step']!=a.step or run['status']!='field_sweep':raise RuntimeError('Require an unchanging live field phase')
    if not (folder/'moments'/f'lf_{a.position:03d}.npz').exists():raise RuntimeError('This frequency has not finished all11 shots')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(s.split(',')[0]):int(s.split(',')[1]) for s in rows.splitlines()}
    if used[a.device]+1.1*1024+768>16384:raise RuntimeError('Whole-card16GiB admission failed')
    torch.set_num_threads(1);torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=f'cuda:{a.device}';torch.cuda.set_per_process_memory_fraction(1.1*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
    spec=json.loads(Path('out/v5/conditioned_lf_full_v1/index.json').read_text())['spec']
    g=FrequencyGeometry(spec).to(device);frequency=spec['bins'][a.position]*40e6/spec['rf_samples']
    material=torch.from_numpy(np.load(folder/'old_material.npy')).to(device)
    system=ConditionedFrequencySystem(g,material[:2],material[2],frequency);op=FirstOrderWavefield(system)
    network=LearnedPreconditioner().to(device)
    network.load_state_dict(torch.load('out/v5/forward_round8_dense_full/last.pt',map_location='cpu',weights_only=True)['model'])
    network.requires_grad_(False);network.eval()
    X=np.load(a.root/'banks'/'lf_X.npy',mmap_mode='r');Y=np.load(a.root/'banks'/'lf_Y.npy',mmap_mode='r')
    yd=np.load(a.root/'banks'/'lf_Ydata.npy',mmap_mode='r')
    penalty=float(np.load(a.root/'banks'/'lf_penalty.npy',mmap_mode='r')[a.position]);response=receiver_response(frequency)
    def norm(value):return float(value.abs().square().sum(dtype=torch.float64).sqrt())
    rows=[]
    with torch.no_grad():
        _,adjoint=op.preconditioners(learned_apply(network,system),learned_apply(network,system,adjoint=True))
        for shot in range(11):
            x=g.pad_field(torch.from_numpy(np.array(X[a.position,shot])).to(device))
            y=g.pad_field(torch.from_numpy(np.array(Y[a.position,shot])).to(device))
            d=torch.from_numpy(np.array(yd[a.position,shot])).to(device)
            # Y_base = shifted_RHS - B X. These are the exact field-problem RHSs.
            bx=op(x);cx=op.readout(x,response)
            gradient=op.readout_adjoint(d,response)+penalty*op.adjoint(y)
            global_right=op.readout_adjoint(d+cx,response)+penalty*op.adjoint(y+bx)
            pre_gradient=adjoint(gradient);pre_right=adjoint(global_right)
            rows.append({'shot':shot,'physical_normal_residual':norm(gradient)/max(norm(global_right),1e-30),
                'preconditioned_global_normal_residual':norm(pre_gradient)/max(norm(pre_right),1e-30),
                'normal_gradient_norm':norm(gradient),'normal_right_norm':norm(global_right)})
            del x,y,d,bx,cx,gradient,global_right,pre_gradient,pre_right
    after=json.loads((a.root/'run.json').read_text())
    if after['step']!=a.step or after['status']!='field_sweep':raise RuntimeError('Field phase changed; do not use this snapshot')
    result={'step':a.step,'source':'lf','frequency_position':a.position,'frequency_hz':frequency,
        'shots':rows,'normalization':'full augmented normal-equation RHS, not current correction RHS',
        'all11_shots':True,'propagation_runs':0,'material_truth_read':False,'banks_modified':False,
        'GPU_peak_GiB':torch.cuda.max_memory_allocated()/2**30,'seconds':time.time()-start,
        'scope':'One already-computed frequency; not a full450-frequency convergence claim'}
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
