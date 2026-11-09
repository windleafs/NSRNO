"""Actual all11-shot block continuation of one saved field problem."""
from pathlib import Path
import argparse,ctypes,json,time,subprocess,hashlib
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
import numpy as np
import torch
from .conditioned_wave import FrequencyGeometry,ConditionedFrequencySystem,LearnedPreconditioner
from .dense_wave import learned_apply
from .first_order_wavefield import FirstOrderWavefield
from .batched_first_order_wavefield import BatchedFirstOrderWavefield
from .native_compact_weighted_pressure import two_level_pressure
from .receiver_normal_preconditioner_batch import ReceiverNormalPreconditioner
from .block_wavefield_step import update_block
from .lf_frequency_data import receiver_response


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--position',type=int,default=89)
    p.add_argument('--device',type=int,default=0);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=32);p.add_argument('--cap-gib',type=float,default=10.)
    a=p.parse_args();start=time.time();folder=a.root/f'step{a.step:03d}'
    if a.out.exists():raise FileExistsError('Preserve existing block continuation')
    run=json.loads((a.root/'run.json').read_text());checkpoint=json.loads((a.root/f'material_pending_step{a.step:03d}.json').read_text())
    retired=checkpoint['status']=='retired_at_material_pending'
    if run['step']!=a.step or run['status']!=('solver_upgrade_checkpoint' if retired else 'material_update') or checkpoint['status'] not in ['held_reversibly','held_reversibly_extended','retired_at_material_pending']:raise RuntimeError('Require the typed field-complete checkpoint')
    if not retired and '\nState:\tT' not in Path(f"/proc/{checkpoint['source_pid']}/status").read_text():raise RuntimeError('Coordinator is not held')
    snapshot={}
    for name in ['X','Y','Ydata']:
        bank=np.load(a.root/'banks'/f'lf_{name}.npy',mmap_mode='r');snapshot[name]=np.array(bank[a.position]);del bank
    penalty=float(np.load(a.root/'banks'/'lf_penalty.npy',mmap_mode='r')[a.position])
    used={int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
    if used[a.device]+a.cap_gib*1024+768>16384:raise RuntimeError('Whole-card16GiB admission failed')
    torch.set_num_threads(1);torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(a.cap_gib*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
    device=f'cuda:{a.device}';spec=json.loads(Path('out/v5/conditioned_lf_full_v1/index.json').read_text())['spec']
    g=FrequencyGeometry(spec).to(device);frequency=spec['bins'][a.position]*40e6/spec['rf_samples']
    material=torch.from_numpy(np.load(folder/'old_material.npy')).to(device)
    system=ConditionedFrequencySystem(g,material[:2],material[2],frequency)
    op=FirstOrderWavefield(system);batch_op=BatchedFirstOrderWavefield(system)
    network=LearnedPreconditioner().to(device)
    network.load_state_dict(torch.load('out/v5/forward_round8_dense_full/last.pt',map_location='cpu',weights_only=True)['model'])
    network.requires_grad_(False);network.eval();response=receiver_response(frequency)
    with torch.no_grad():
        pressure,pressure_adj,meta=two_level_pressure(system,learned_apply(network,system),learned_apply(network,system,adjoint=True))
        normal_pre=ReceiverNormalPreconditioner(op,pressure_adj,response,penalty)
        print(json.dumps({'preconditioners_complete':True,'pressure':meta,'receiver':normal_pre.metadata,'seconds':time.time()-start}),flush=True)
        forward,adjoint=batch_op.preconditioners(pressure,pressure_adj)
        x=g.pad_field(torch.from_numpy(snapshot['X']).to(device))
        y=g.pad_field(torch.from_numpy(snapshot['Y']).to(device))
        yd=torch.from_numpy(snapshot['Ydata']).to(device)
        b=y+batch_op(x);d=yd+batch_op.readout(x,response);del y,yd
        final,record,eq,rx=update_block(batch_op,b,d,response,x,penalty,forward,adjoint,normal_pre,maximum=a.iterations,tolerance=.003)
        del final,eq,rx
    result={'frequency_hz':frequency,'step':a.step,'source':'lf','position':a.position,'all11_shots':True,
        'record':record,'pressure_preconditioner':meta,'receiver_preconditioner':normal_pre.metadata,
        'seconds':time.time()-start,'GPU_peak_GiB':torch.cuda.max_memory_allocated()/2**30,
        'truth_read':False,'banks_modified':False,'new_propagation_runs':0,
        'code_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in
            ['continue_saved_wri_field_weighted.py','block_wavefield_step.py','batched_first_order_wavefield.py',
             'native_compact_weighted_pressure.py','receiver_normal_preconditioner_batch.py']}}
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
