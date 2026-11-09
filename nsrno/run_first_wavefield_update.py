"""First real field block, with fixed RF material; not joint reconstruction."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import json,subprocess,time,os
import numpy as np
import torch
from .lf_frequency_data import LowFrequencyData,receiver_response
from .conditioned_wave import ConditionedFrequencySystem,LearnedPreconditioner
from .dense_wave import learned_apply
from .first_order_wavefield import FirstOrderWavefield
from .wavefield_step import penalty_scale,update_wavefield


def main():
    used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    cards={int(s.split(',')[0]):int(s.split(',')[1]) for s in used.splitlines()}
    device_index=1;cap=1.1
    if cards[device_index]+cap*1024+768>16*1024:raise RuntimeError('No room for the field block within 16 GiB')
    out=Path('out/v5/first_wavefield_update_v1_train_200');out.mkdir(exist_ok=False)
    start=time.time();torch.set_num_threads(1);torch.cuda.set_device(device_index)
    torch.cuda.set_per_process_memory_fraction(cap*2**30/torch.cuda.get_device_properties(device_index).total_memory,device_index)
    device=f'cuda:{device_index}'
    progress=json.loads(Path('out/v5/conditioned_lf_full_v1/progress.json').read_text())
    if '0:64:96' not in progress['completed']:
        raise RuntimeError('The requested actual reference frequency has not been written yet')
    data=LowFrequencyData(device=device,allow_partial=True)
    fbin,angle=180,0;frequency=fbin*40e6/2401
    material=torch.from_numpy(np.load('out/v5/reconstruction_geometry_v1_train_200/snapshots/stage0_step064/material.npy')).to(device)
    with torch.no_grad():
        uniform=material.clone();uniform[0].fill_(1540);uniform[1].fill_(1000);uniform[2].fill_(.002)
        system=ConditionedFrequencySystem(data.geometry,material[:2],material[2],frequency)
        reference_system=ConditionedFrequencySystem(data.geometry,uniform[:2],uniform[2],frequency)
        op,refop=FirstOrderWavefield(system),FirstOrderWavefield(reference_system)
        pref=data.reference(fbin,angle);reference=refop.from_pressure(pref)
        rhs=op.scattering_rhs(refop,reference)
        checkpoint=torch.load('out/v5/forward_round8_dense_full/last.pt',map_location='cpu',weights_only=True)
        network=LearnedPreconditioner().to(device);network.load_state_dict(checkpoint['model']);network.requires_grad_(False);network.eval()
        forward,adjoint=op.preconditioners(learned_apply(network,system),learned_apply(network,system,adjoint=True))
        response=receiver_response(frequency)
        penalty,eigenvalue=penalty_scale(op,forward,adjoint,response)
        current=torch.zeros_like(reference);dual=torch.zeros_like(reference);target=data.target(fbin,angle)
        torch.cuda.reset_peak_memory_stats()
        value,result=update_wavefield(op,rhs,target,response,current,dual,torch.zeros_like(target),
                                     penalty,forward,adjoint,maximum=32,tolerance=1e-3)
        np.save(out/'scattered_field.npy',data.geometry.native(value).cpu().numpy())
    result.update({'scope':'one actual LF field update at fixed RF-derived material; no material optimization or truth access',
                   'material_updated':False,'truth_read':False,'sample':'train_200','frequency_hz':frequency,'angle':angle,
                   'penalty':penalty,'estimated_data_preconditioner_norm_squared':eigenvalue,
                   'GPU_peak_GiB':torch.cuda.max_memory_allocated()/2**30,'card_before_MiB':cards,
                   'device':device_index,'cap_GiB':cap,'pid':os.getpid(),'elapsed_seconds':time.time()-start})
    (out/'run.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
