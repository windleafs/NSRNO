"""Locate a pressure-preconditioner defect using saved fields, with one LU.

Compare Ac^-1 and Ac^-1 Q on the same saved physical pressure. No material
derivatives, truth, new propagation, or live bank changes are involved.
"""
from pathlib import Path
import ctypes,json,time,subprocess,argparse
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
import numpy as np
import torch
from .conditioned_wave import FrequencyGeometry,ConditionedFrequencySystem,LearnedPreconditioner
from .dense_wave import learned_apply
from .native_compact_weighted_pressure import NativeCompactPressureCorrection


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',type=int,default=1);a=p.parse_args();start=time.time()
    if a.out.exists():raise FileExistsError('Preserve inverse-defect measurement')
    checkpoint=json.loads((a.root/'material_pending_step003.json').read_text())
    if checkpoint['status']!='retired_at_material_pending':raise RuntimeError('Require the preserved no-writer field checkpoint')
    used={int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
    if used[a.device]+1.5*1024+768>16384:raise RuntimeError('Whole-card budget')
    torch.set_num_threads(1);torch.cuda.set_device(a.device)
    torch.cuda.set_per_process_memory_fraction(1.5*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=f'cuda:{a.device}';spec=json.loads(Path('out/v5/conditioned_lf_full_v1/index.json').read_text())['spec']
    g=FrequencyGeometry(spec).to(device);frequency=spec['bins'][89]*40e6/spec['rf_samples']
    material=torch.from_numpy(np.load(a.root/'step003/old_material.npy')).to(device)
    system=ConditionedFrequencySystem(g,material[:2],material[2],frequency)
    network=LearnedPreconditioner().to(device)
    network.load_state_dict(torch.load('out/v5/forward_round8_dense_full/last.pt',map_location='cpu',weights_only=True)['model'])
    network.requires_grad_(False);network.eval();rows=[]
    with torch.no_grad():
        fine=learned_apply(network,system);correction=NativeCompactPressureCorrection(system)
        bank=np.load(a.root/'banks/lf_X.npy',mmap_mode='r')
        for shot in [0,10]:
            p=g.pad_field(torch.from_numpy(np.array(bank[89,shot,0])).to(device));rhs=system(p)
            first=fine(rhs);remaining=rhs-system(first)
            for name,apply in [('unweighted',correction.unweighted),('weighted_Q',correction)]:
                value=first+apply(remaining);error=g.native(value-p);reference=g.native(p)
                regions={};inside=torch.zeros_like(reference.real);inside[24:-24,24:-24]=1
                for label,mask in [('native',torch.ones_like(inside)),('interior',inside),('PML_rim',1-inside)]:
                    regions[label]=float((error.abs().square()*mask).sum(dtype=torch.float64).sqrt()/
                        (reference.abs().square()*mask).sum(dtype=torch.float64).sqrt().clamp_min(1e-30))
                residual=system(value)-rhs
                rows.append({'shot':shot,'inverse':name,'pressure_relative_error':regions,
                    'relative_equation_defect':float(residual.abs().square().sum(dtype=torch.float64).sqrt()/rhs.abs().square().sum(dtype=torch.float64).sqrt())})
    result={'frequency_hz':frequency,'rows':rows,'pressure_solver':correction.metadata,'seconds':time.time()-start,
        'GPU_peak_GiB':torch.cuda.max_memory_allocated()/2**30,'truth_read':False,'banks_changed':False,'new_propagation':0}
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
