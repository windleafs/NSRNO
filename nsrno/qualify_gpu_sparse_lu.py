"""Once-only CPU/GPU backend consistency and cost check, without propagation.

Uses the existing frozen model's actual pressure LU and saved pressure RHSs.
No material perturbations, training, reconstruction, or bank writes occur.
"""
from pathlib import Path
import argparse,ctypes,os,json,time,subprocess,threading,traceback,hashlib,gc
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
import numpy as np
import torch
from .conditioned_wave import FrequencyGeometry,ConditionedFrequencySystem
from .native_compact_weighted_pressure import NativeCompactPressureCorrection
from .gpu_sparse_lu import GpuSuperLU,_api


def used_cards():
    return {int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',type=int,default=2);p.add_argument('--cap-gib',type=float,default=3.5)
    p.add_argument('--nrhs',type=int,nargs='+',default=[8,11])
    a=p.parse_args();start=time.time()
    if a.out.exists():raise FileExistsError('Preserve the existing GPU LU qualification')
    before=used_cards()
    if before[a.device]+a.cap_gib*1024+768>16384:raise RuntimeError('Whole-card16GiB admission failed')
    # The production worker has an11GiB allocator cap. Reserve its possible
    # growth and both CUDA contexts, even if this admission sample is lower.
    if 11+a.cap_gib+1.5>16:raise RuntimeError('Known production cap plus both context allowances exceeds16GiB')
    torch.set_num_threads(1);torch.cuda.set_device(a.device)
    torch.cuda.set_per_process_memory_fraction(a.cap_gib*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=f'cuda:{a.device}';maximum=before.copy();stop=threading.Event();violation=threading.Event()
    def sample():
        while not stop.wait(1):
            current=used_cards()
            for d,m in current.items():maximum[d]=max(maximum.get(d,0),m)
            if current[a.device]>16384:violation.set()
    monitor=threading.Thread(target=sample,daemon=True);monitor.start();solver=None;rows=[]
    result={'pid':os.getpid(),'main_reconstruction_not_modified':True,'material_truth_read':False,
        'new_propagation_runs':0,'live_banks_changed':False,'whole_card_limit_GiB':16,'own_cap_GiB':a.cap_gib,
        'GPU_before_MiB':before,'source_code_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['gpu_sparse_lu.py','qualify_gpu_sparse_lu.py','native_compact_weighted_pressure.py']}}
    try:
        api=_api()
        for key in ['CUSPARSE_OPERATION_NON_TRANSPOSE','CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE','CUDA_C_32F',
            'CUSPARSE_INDEX_32I','CUSPARSE_INDEX_BASE_ZERO','CUSPARSE_ORDER_COL','CUSPARSE_FILL_MODE_LOWER',
            'CUSPARSE_FILL_MODE_UPPER','CUSPARSE_DIAG_TYPE_UNIT','CUSPARSE_DIAG_TYPE_NON_UNIT',
            'CUSPARSE_SPMAT_FILL_MODE','CUSPARSE_SPMAT_DIAG_TYPE','CUSPARSE_SPSM_ALG_DEFAULT']:api.k(key)
        spec=json.loads(Path('out/v5/conditioned_lf_full_v1/index.json').read_text())['spec']
        g=FrequencyGeometry(spec).to(device);frequency=spec['bins'][89]*40e6/spec['rf_samples']
        material=torch.from_numpy(np.load(a.root/'step003/old_material.npy')).to(device)
        system=ConditionedFrequencySystem(g,material[:2],material[2],frequency)
        with torch.no_grad():pressure=NativeCompactPressureCorrection(system)
        print(json.dumps({'CPU_factor_ready':pressure.metadata,'seconds':time.time()-start}),flush=True)
        solver=GpuSuperLU(pressure.lu,device)
        print(json.dumps({'GPU_factors_ready':solver.metadata,'seconds':time.time()-start}),flush=True)
        bank=np.load(a.root/'banks/lf_X.npy',mmap_mode='r');all_rhs=np.array(bank[89,:,0]).reshape(11,-1);del bank
        for count in a.nrhs:
            if not 1<=count<=11:raise ValueError('Qualification RHS count must be between1 and11')
            raw=np.ascontiguousarray(all_rhs[:count]);q=np.ascontiguousarray(raw[::-1].conj())
            input_tensor=torch.from_numpy(raw).to(device);outputs={};row={'nrhs':count,'directions':{}};rows.append(row)
            for adjoint in [False,True]:
                source=q if adjoint else raw;input_tensor.copy_(torch.from_numpy(source))
                t=time.time();reference=pressure.lu.solve(np.asfortranarray(source.T),trans='H' if adjoint else 'N').T.copy()
                cpu_seconds=time.time()-t
                t=time.time();warm=solver.solve(input_tensor,adjoint=adjoint,out=input_tensor);torch.cuda.synchronize(a.device)
                warm_seconds=time.time()-t;actual=warm.cpu().numpy().copy();del warm
                rel=float(np.linalg.norm(actual.astype(np.complex128)-reference)/max(np.linalg.norm(reference.astype(np.complex128)),1e-30))
                timings=[]
                for repeat in range(2):
                    input_tensor.copy_(torch.from_numpy(source))
                    torch.cuda.synchronize(a.device);t=time.time();value=solver.solve(input_tensor,adjoint=adjoint,out=input_tensor)
                    torch.cuda.synchronize(a.device);timings.append(time.time()-t);del value
                outputs[adjoint]=actual
                row['directions']['adjoint' if adjoint else 'forward']={'relative_L2_difference_to_CPU':rel,
                    'CPU_solve_seconds':cpu_seconds,'GPU_first_solve_including_analysis_seconds':warm_seconds,
                    'GPU_repeated_seconds':timings,'speedup_CPU_over_mean_GPU':cpu_seconds/(sum(timings)/len(timings)),
                    'workspace':solver.workspace_metadata(),'caller_output_reuses_input':True}
                if violation.is_set():raise RuntimeError('Sampled card use exceeded16GiB; stop auxiliary qualification')
                # Avoid retaining both pairs of830MiB workspaces beside the
                # live11GiB production worker. The single-pair result does not
                # qualify a future production cache of all directions/batches.
                solver.clear_plans();gc.collect();torch.cuda.empty_cache()
            left=np.vdot(outputs[False].astype(np.complex128),q.astype(np.complex128))
            right=np.vdot(raw.astype(np.complex128),outputs[True].astype(np.complex128))
            row['complex_adjoint_relative_error']=float(abs(left-right)/max(abs(left),abs(right),1e-30))
            row['qualification_cache_policy']='only one direction at a time; production full-cache memory not qualified'
            print(json.dumps(row),flush=True)
            del input_tensor,outputs;solver.clear_plans();gc.collect();torch.cuda.empty_cache()
        result.update(status='complete',backend=solver.metadata,rows=rows,requested_nrhs=a.nrhs,
            passed=all(d['relative_L2_difference_to_CPU']<=3e-4 for r in rows for d in r['directions'].values()) and
                all(r['complex_adjoint_relative_error']<=1e-3 for r in rows) and not violation.is_set(),
            interpretation='Linear algebra transport only; not RF surrogate accuracy or three-parameter reconstruction')
    except BaseException:
        result.update(status='failed',error=traceback.format_exc(),rows=rows,passed=False)
        print(result['error'],flush=True)
    finally:
        if solver is not None:
            try:solver.close()
            except Exception:pass
        stop.set();monitor.join(timeout=2)
        current=used_cards()
        for d,m in current.items():maximum[d]=max(maximum.get(d,0),m)
        result.update(seconds=time.time()-start,GPU_peak_allocated_GiB=torch.cuda.max_memory_allocated(a.device)/2**30,
            sampled_maximum_whole_card_MiB=maximum,own_card_observed_above16GiB=maximum[a.device]>16384,
            resource_sampling_interval_seconds=1,resource_sampling_not_continuous=True)
        a.out.write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'passed':result['passed'],'out':str(a.out),'seconds':result['seconds']}),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
