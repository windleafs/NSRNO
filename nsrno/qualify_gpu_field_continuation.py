"""Full all11 field integration check on the immutable original WRI RHS.

No new physical propagation or material update. Required once before a
backend change, because full receiver workspaces must fit the16GiB card.
"""
from pathlib import Path
import argparse,ctypes,json,time,hashlib,threading,subprocess,traceback
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
import numpy as np
import torch
from .conditioned_wave import FrequencyGeometry,ConditionedFrequencySystem,LearnedPreconditioner
from .dense_wave import learned_apply
from .first_order_wavefield import FirstOrderWavefield
from .batched_first_order_wavefield import BatchedFirstOrderWavefield
from .native_compact_weighted_pressure_gpu import two_level_pressure
from .receiver_normal_preconditioner_batch import ReceiverNormalPreconditioner
from .block_wavefield_step import update_block
from .frequency_group_data import FrequencyGroupData,receiver_response
from .reconstruct_wri_gpu import validate_checkpoint


def cards():
    return {int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',type=int,default=0);p.add_argument('--cap-gib',type=float,default=14.5);p.add_argument('--iterations',type=int,default=32)
    a=p.parse_args();start=time.time()
    if a.out.exists():raise FileExistsError('Preserve the GPU field integration report')
    cp,index=validate_checkpoint(a.checkpoint);root=Path(cp['root']);folder=root/f"step{cp['step']:03d}";ledger=Path(cp['fixed_rhs_directory'])
    before=cards()
    if before[a.device]+a.cap_gib*1024+768>16384:raise RuntimeError('Whole-card16GiB admission failed')
    torch.set_num_threads(1);torch.cuda.set_device(a.device)
    torch.cuda.set_per_process_memory_fraction(a.cap_gib*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=f'cuda:{a.device}';maximum=before.copy();stop=threading.Event()
    def monitor():
        while not stop.wait(1):
            try:
                for d,v in cards().items():maximum[d]=max(maximum.get(d,0),v)
            except Exception:pass
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    result={'all11_shots':True,'device':a.device,'material_updated':False,'truth_read':False,'banks_modified':False,'new_propagation_runs':0,
        'source_checkpoint':str(a.checkpoint.resolve()),'material_sha256':cp['material_sha256'],
        'code_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
            ['native_compact_weighted_pressure_gpu.py','gpu_sparse_lu.py','native_compact_weighted_pressure.py',
             'block_wavefield_step.py','batched_first_order_wavefield.py','first_order_wavefield.py',
             'receiver_normal_preconditioner_batch.py','dense_wave.py','wri_worker_block_gpu.py','frequency_group_data.py']}}
    try:
        data=FrequencyGroupData('lf',cp['sample'],device);g=data.geometry;spec=data.spec
        point_data=FrequencyGroupData('point3',cp['sample'],device,g)
        frequency=spec['bins'][89]*40e6/spec['rf_samples'];old=torch.from_numpy(np.load(folder/'old_material.npy')).to(device)
        system=ConditionedFrequencySystem(g,old[:2],old[2],frequency);op=FirstOrderWavefield(system);batchop=BatchedFirstOrderWavefield(system)
        # Retain the same live reference/source buffers as the production
        # worker. A smaller isolated solve would understate its memory peak.
        uniform=old.clone();uniform[0].fill_(1540);uniform[1].fill_(1000);uniform[2].fill_(.002)
        refsystem=ConditionedFrequencySystem(g,uniform[:2],uniform[2],frequency);refop=FirstOrderWavefield(refsystem)
        zz=np.clip(np.arange(g.H)-g.pz,24,g.h-25);xx=np.clip(np.arange(g.W)-g.px,24,g.w-25)
        index_tensor=torch.from_numpy((zz[:,None]*g.w+xx[None,:]).astype(np.int64)).to(device)
        ix=torch.roll(index_tensor,-1,-1);iz=torch.roll(index_tensor,-1,-2)
        net=LearnedPreconditioner().to(device)
        net.load_state_dict(torch.load('out/v5/forward_round8_dense_full/last.pt',map_location='cpu',weights_only=True)['model'])
        net.requires_grad_(False);net.eval();response=receiver_response(frequency)
        penalty=float(np.load(root/'banks/lf_penalty.npy',mmap_mode='r')[89])
        with torch.no_grad():
            pressure,pressure_adj,meta=two_level_pressure(system,learned_apply(net,system),learned_apply(net,system,adjoint=True))
            normal=ReceiverNormalPreconditioner(op,pressure_adj,response,penalty)
            print(json.dumps({'preconditioners_ready':True,'pressure':meta,'receiver':normal.metadata,'seconds':time.time()-start}),flush=True)
            forward,adjoint=batchop.preconditioners(pressure,pressure_adj)
            x=g.pad_field(torch.from_numpy(np.array(np.load(root/'banks/lf_X.npy',mmap_mode='r')[89])).to(device))
            b=g.pad_field(torch.from_numpy(np.array(np.load(ledger/'lf_rhs.npy',mmap_mode='r')[89])).to(device))
            d=torch.from_numpy(np.array(np.load(ledger/'lf_data.npy',mmap_mode='r')[89])).to(device)
            references=torch.stack([refop.from_pressure(data.reference(89,a)) for a in range(11)])
            physical_rhs=torch.stack([op.scattering_rhs(refop,references[a]) for a in range(11)])
            target=torch.stack([data.target(89,a) for a in range(11)])
            final,record,eq,rx=update_block(batchop,b,d,response,x,penalty,forward,adjoint,normal,maximum=a.iterations,tolerance=.003)
            data_errors=(target-batchop.readout(final,response)).abs().square().sum(-1,dtype=torch.float64).tolist()
            equation_errors=(physical_rhs-batchop(final)).abs().square().flatten(1).sum(-1,dtype=torch.float64).tolist()
            # Save the actual completed result for reproducibility; active
            # reconstruction banks are not overwritten by this qualification.
            np.save(a.out.with_suffix('.field.npy'),g.native(final).cpu().numpy())
            np.save(a.out.with_suffix('.Ybase.npy'),g.native(eq).cpu().numpy());np.save(a.out.with_suffix('.Ydata.npy'),rx.cpu().numpy())
            result.update(status='complete',frequency_hz=frequency,source='lf',position=89,record=record,
                pressure_preconditioner=meta,receiver_preconditioner=normal.metadata,
                production_reference_and_source_buffers_retained=True,
                physical_data_error_squared=data_errors,physical_equation_error_squared=equation_errors,
                passed=record['converged'] and maximum[a.device]<=16384,
                GPU_peak_GiB=torch.cuda.max_memory_allocated(a.device)/2**30)
    except BaseException:
        result.update(status='failed',passed=False,error=traceback.format_exc());print(result['error'],flush=True)
    finally:
        stop.set();watcher.join(timeout=2)
        for d,v in cards().items():maximum[d]=max(maximum.get(d,0),v)
        result.update(sampled_maximum_whole_card_MiB=maximum,seconds=time.time()-start,resource_sampling_interval_s=1,
            own_allocator_cap_GiB=a.cap_gib,whole_card_limit_GiB=16)
        if maximum[a.device]>16384:result['passed']=False
        a.out.write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'passed':result['passed'],'out':str(a.out),'seconds':result['seconds']}),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
