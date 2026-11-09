"""Continue one saved, fully acquired WRI frequency with a two-level inverse.

Only an immutable in-memory copy of the saved augmented field problem is
used. No bank, material, multiplier, or physical observation is modified.
"""
from pathlib import Path
import argparse,ctypes,json,time,subprocess,hashlib
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
import numpy as np
import torch
from .conditioned_wave import FrequencyGeometry,ConditionedFrequencySystem,LearnedPreconditioner
from .dense_wave import learned_apply
from .first_order_wavefield import FirstOrderWavefield
from .native_compact_pressure_preconditioner import two_level_pressure
from .lf_frequency_data import receiver_response
from .wavefield_step import dot
from .wavefield_step_v3 import update_wavefield
from .receiver_normal_preconditioner import ReceiverNormalPreconditioner


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--position',type=int,default=89)
    p.add_argument('--device',type=int,default=0);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=64);p.add_argument('--cap-gib',type=float,default=1.5)
    p.add_argument('--shots',type=int,default=2)
    a=p.parse_args();start=time.time();folder=a.root/f'step{a.step:03d}'
    if a.out.exists():raise FileExistsError('Preserve the existing continuation result')
    def phase():
        run=json.loads((a.root/'run.json').read_text())
        live_field=run['step']==a.step and run['status']=='field_sweep'
        held_material=False
        marker=a.root/f'material_pending_step{a.step:03d}.json'
        if run['step']==a.step and run['status']=='material_update' and marker.exists():
            c=json.loads(marker.read_text());state=Path(f"/proc/{c['source_pid']}/status")
            held_material=c['status']=='held_reversibly' and state.exists() and '\nState:\tT' in state.read_text()
        if not (live_field or held_material):raise RuntimeError('Snapshot requires an unchanging field or typed held-material phase')
        return run
    run=phase();moment=folder/'moments'/f'lf_{a.position:03d}.npz'
    if not moment.exists():raise RuntimeError('All11 shots of this frequency must already have finished')
    snapshot={}
    for name in ['X','Y','Ydata']:
        bank=np.load(a.root/'banks'/f'lf_{name}.npy',mmap_mode='r')
        snapshot[name]=np.array(bank[a.position]);del bank
    penalty=float(np.load(a.root/'banks'/'lf_penalty.npy',mmap_mode='r')[a.position])
    phase();snapshot_time=time.time()
    used={int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))/2**20
    if used[a.device]+a.cap_gib*1024+768>16384:raise RuntimeError('Whole-card16GiB admission failed')
    if available<32:raise RuntimeError('Insufficient host memory for a coarse sparse factorization')
    torch.set_num_threads(1);torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(a.cap_gib*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
    device=f'cuda:{a.device}'
    spec=json.loads(Path('out/v5/conditioned_lf_full_v1/index.json').read_text())['spec']
    g=FrequencyGeometry(spec).to(device);frequency=spec['bins'][a.position]*40e6/spec['rf_samples']
    material=torch.from_numpy(np.load(folder/'old_material.npy')).to(device)
    system=ConditionedFrequencySystem(g,material[:2],material[2],frequency);op=FirstOrderWavefield(system)
    network=LearnedPreconditioner().to(device)
    network.load_state_dict(torch.load('out/v5/forward_round8_dense_full/last.pt',map_location='cpu',weights_only=True)['model'])
    network.requires_grad_(False);network.eval()
    def norm(value):return float(dot(value,value).real.sqrt())
    def stationarity(x,b,d,adjoint,response):
        eq=b-op(x);rx=d-op.readout(x,response)
        gradient=op.readout_adjoint(rx,response)+penalty*op.adjoint(eq)
        full=op.readout_adjoint(d,response)+penalty*op.adjoint(b)
        return {'physical_global_normal_residual':norm(gradient)/max(norm(full),1e-30),
            'preconditioned_global_normal_residual':norm(adjoint(gradient))/max(norm(adjoint(full)),1e-30),
            'augmented_cost':float(dot(rx,rx).real+penalty*dot(eq,eq).real)}
    with torch.no_grad():
        fine=learned_apply(network,system);fine_adj=learned_apply(network,system,adjoint=True)
        pressure,pressure_adj,metadata=two_level_pressure(system,fine,fine_adj)
        forward,adjoint=op.preconditioners(pressure,pressure_adj)
        print(json.dumps({'factor_complete':metadata,'seconds':time.time()-start}),flush=True)
        response=receiver_response(frequency);rows=[]
        normal_preconditioner=ReceiverNormalPreconditioner(op,pressure_adj,response,penalty)
        print(json.dumps({'receiver_preconditioner_complete':normal_preconditioner.metadata}),flush=True)
        for shot in range(a.shots):
            shot_start=time.time()
            x=g.pad_field(torch.from_numpy(snapshot['X'][shot]).to(device))
            y=g.pad_field(torch.from_numpy(snapshot['Y'][shot]).to(device))
            yd=torch.from_numpy(snapshot['Ydata'][shot]).to(device)
            b=y+op(x);d=yd+op.readout(x,response)
            before=stationarity(x,b,d,adjoint,response)
            final,record=update_wavefield(op,b,d,response,x,torch.zeros_like(x),torch.zeros_like(d),
                penalty,forward,adjoint,maximum=a.iterations,tolerance=.003,normal_preconditioner=normal_preconditioner)
            after=stationarity(final,b,d,adjoint,response)
            row={'shot':shot,'before':before,'after':after,'field_record':record,'seconds':time.time()-shot_start}
            rows.append(row);print(json.dumps(row),flush=True)
            del x,y,yd,b,d,final
    result={'step':a.step,'source':'lf','frequency_position':a.position,'frequency_hz':frequency,
        'all11_shots':a.shots==11,'shots_tested':a.shots,'snapshot_unix_s':snapshot_time,'source_main_pid':run['pid'],
        'problem':'saved augmented quadratic, not a physical RF replay','coarse_correction':metadata,'shots':rows,
        'receiver_preconditioner':normal_preconditioner.metadata,
        'propagation_runs':0,'material_truth_read':False,'banks_modified':False,'material_updated':False,
        'GPU_peak_GiB':torch.cuda.max_memory_allocated()/2**30,'seconds':time.time()-start,
        'code_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['native_compact_pressure_preconditioner.py','coarse_pressure_preconditioner.py','continue_saved_wri_field_receiver.py','wavefield_step_v3.py','receiver_normal_preconditioner.py']}}
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps({'complete':True,'out':str(a.out),'seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
