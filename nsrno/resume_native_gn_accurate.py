"""Continue a saved fixed native GN problem with accurate CPU QP proposals.

Earlier native HVPs and gradient are reused exactly. Physical line searches
must improve the already accepted incumbent, not just the older reference.
"""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import numpy as np
from .native_gn_saved_state import SavedQuadratic
from .refine_native_gn_cpu import solve_accurately,atomic,save_array
from .native_gn_coordinates import dot
from .reconstruct_native_gn import Pool
from .pressure_resources import PressureResources

def no_source_writer(root,run):
    record=root/'worker_processes.json'
    pids=[run['pid']]
    if record.exists():pids+=json.loads(record.read_text()).get('worker_pids',[])
    for pid in pids:
        if (Path('/proc')/str(pid)).exists():
            raise RuntimeError(f'Source PID {pid} still exists; inspect it, do not start a second GPU run')
    if run['status'] not in ['planned_native_GN_steps_finished','cpu_subproblem_not_converged']:
        raise RuntimeError('Require a complete source result or preserved CPU-only continuation boundary')

def maps(args,folder,monitor):
    with open(folder/'maps.log','w') as log:
        proc=subprocess.Popen([sys.executable,'-m','nsrno.evaluate_time_material',
            str(folder/'material.npy'),'--sample',args.sample,'--out',str(folder/'maps'),
            '--estimate-label','Native GN with refined constraints'],stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:monitor.sample('post_hoc_refined_GN_maps_only')
        if proc.returncode:raise RuntimeError('Post-hoc map process failed; inspect maps.log')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--source-step',type=int,default=1)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--extra-hvps',type=int,default=8)
    p.add_argument('--cpu-maximum',type=int,default=1200)
    p.add_argument('--cpu-tolerance',type=float,default=.01)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    p.add_argument('--warm-total-direction',type=Path)
    args=p.parse_args();args.source_root=args.source_root.resolve()
    source_run=json.loads((args.source_root/'run.json').read_text())
    no_source_writer(args.source_root,source_run)
    if args.out.exists():raise RuntimeError('Use a new output directory; never restart on an observation timeout')
    if args.cap_gib>14:raise ValueError('Keep room for CUDA context under the 16-GiB limit')
    for name,expected in source_run['source_sha256'].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Source runtime changed: {name}')
    state=SavedQuadratic(args.source_root,args.source_step)
    args.sample=state.sample
    reused=len(state.paths)
    args.out.mkdir(parents=True);(args.out/'jobs').mkdir()
    folder=args.out/'step001';folder.mkdir()
    source_manifest=dict(source_run['source_sha256'])
    for name in ['native_gn_saved_state.py','refine_native_gn_cpu.py','resume_native_gn_accurate.py']:
        source_manifest[name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    run={'status':'restoring_saved_fixed_quadratic','pid':os.getpid(),'sample':args.sample,
        'solver_revision':'saved_native_GN_with_explicit_CPU_KKT_control',
        'source_root':str(args.source_root),'source_step':args.source_step,
        'source_sha256':source_manifest,'preconditioner_statistics':str(state.preconditioner),
        'devices':args.devices,'allocator_cap_GiB':args.cap_gib,'model_per_card_limit_GiB':16,
        'reused_complete_HVPs':reused,'new_gradient_wave_solves':0,
        'extra_HVP_budget':args.extra_hvps,'CPU_relative_KKT_target':args.cpu_tolerance,
        'true_QP_relative_KKT_target':state.reference['true_QP_target_relative'],
        'fixed_linearization_material':str(state.origin/'old_material.npy'),
        'accepted_incumbent_material':str(state.source_folder/'material.npy'),
        'old_material_SHA256':hashlib.sha256((state.origin/'old_material.npy').read_bytes()).hexdigest(),
        'truth_read_by_optimizer':False,'joint_imaging_success_claimed':False,
        'created_unix_s':time.time(),'completed_step':0,'step':1}
    atomic(args.out/'run.json',run)
    atomic(args.out/'worker_processes.json',{'coordinator_pid':os.getpid(),'worker_pids':[],
        'devices':args.devices,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
    atomic(args.out/'fixed_model_reference.json',state.reference)
    atomic(folder/'fixed_HVP_manifest.json',{'paths':[str(p) for p in state.paths]})
    atomic(folder/'incumbent_RF.json',state.best_rf)
    save_array(folder/'best_log_direction.npy',state.best_u)
    save_array(folder/'material.npy',state.best_material)
    save_array(folder/'stage0_initial_material.npy',state.best_material)
    incumbent_material=state.best_material.copy();incumbent_cost=state.best_cost
    atomic(folder/'restored.json',{'reused_HVP_count':reused,'true_model_change':state.value(),
        'projected_KKT_relative':state.norm()/max(state.initial_norm,1e-300),
        'fixed_reference_RF_loss':state.baseline_rf['loss'],'incumbent_RF_loss':state.best_rf['loss']})
    monitor=PressureResources(args.out);pool=None;cpu_failure=False;new_rows=[]
    try:
        for extra in range(args.extra_hvps):
            relative=state.norm()/max(state.initial_norm,1e-300)
            if relative<=state.reference['true_QP_target_relative']:break
            index=len(state.paths)
            cpu_folder=folder/f'cpu{index:03d}';cpu_folder.mkdir()
            run.update(status='CPU_refined_box_QP',fixed_HVP_count=index)
            atomic(args.out/'run.json',run)
            initial=np.zeros_like(state.u)
            if extra==0 and args.warm_total_direction is not None:
                initial=np.clip(np.load(args.warm_total_direction)-state.u,
                    state.low-state.u,state.high-state.u)
                if dot(state.gradient,initial)+.5*dot(initial,state.model.apply(initial))>0:
                    initial.fill(0)
            def progress(row,d,save):
                atomic(cpu_folder/'progress.json',row)
                atomic(folder/'direction_progress.json',dict(row,phase='CPU_refined_box_QP',
                    hvp_count=len(state.paths),true_QP_KKT_relative=relative))
                if save:save_array(cpu_folder/'direction_checkpoint.npy',d)
            d,cpu=solve_accurately(state.model,state.gradient,state.low-state.u,state.high-state.u,
                initial,maximum=args.cpu_maximum,tolerance=args.cpu_tolerance,callback=progress)
            atomic(cpu_folder/'result.json',cpu);save_array(cpu_folder/'direction.npy',d)
            save_array(cpu_folder/'total_candidate_log_direction.npy',state.u+d)
            if not cpu['tolerance_met']:
                cpu_failure=True
                run['status']='cpu_subproblem_not_converged'
                break
            if dot(state.gradient,d)>=0:
                run['numerical_stop_reason']='no further feasible CPU descent'
                break
            if pool is None:pool=Pool(args,monitor)
            run['status']='native_full_GN_direction';atomic(args.out/'run.json',run)
            _,data_image=pool.evaluate(state.material,'hvp',f'fixed_hvp{index:03d}',d)
            row=state.append(d,data_image,folder/f'hvp{index:03d}',cpu)
            new_rows.append(row)
            atomic(folder/'fixed_HVP_manifest.json',{'paths':[str(p) for p in state.paths]})
            atomic(folder/'direction_progress.json',row)
        save_array(folder/'log_direction.npy',state.u)
        save_array(folder/'cumulative_total_image.npy',state.Hu)
        total=dot(state.u,state.Hu);tvq=dot(state.u,state.h_tv(state.u));dq=dot(state.u,state.damping*state.u)
        report={'method':run['solver_revision'],'reused_HVPs':reused,
            'new_HVPs':len(state.paths)-reused,'true_QP_KKT_relative':state.norm()/max(state.initial_norm,1e-300),
            'true_QP_tolerance_met':state.norm()<=state.reference['true_QP_target_relative']*state.initial_norm,
            'QP_exact_solution_claimed':False,'model_change':state.value(),
            'linear_slope':dot(state.gradient0,state.u),'total_quadratic':total,
            'data_quadratic':total-tvq-dq,'TV_quadratic':tvq,'damping_quadratic':dq,
            'CPU_subproblem_failure':cpu_failure,'new_history':new_rows,'truth_read':False}
        atomic(folder/'direction.json',report)
        if cpu_failure:
            atomic(args.out/'run.json',run)
            return
        slope=dot(state.gradient0,state.u)
        base_cost=state.baseline_rf['loss']+state.tv
        lines=[];accepted=False
        for line in range(8 if new_rows else 0):
            if pool is None:pool=Pool(args,monitor)
            alpha=.5**line;trial=state.co.material(alpha*state.u)
            run.update(status='native_material_line_search',line=line);atomic(args.out/'run.json',run)
            rf,_=pool.evaluate(trial,'forward',f'refined_line{line:02d}')
            tv=state.co.regularization(alpha*state.u,state.reference['tv_weight'])[0]
            cost=rf['loss']+tv
            armijo=cost<=base_cost+1e-4*alpha*slope
            improves=cost<incumbent_cost-max(1e-12,abs(incumbent_cost)*1e-8)
            accepted=bool(armijo and improves)
            row={'alpha':alpha,'native_RF':rf,'TV':tv,'physical_objective':cost,
                'change_from_fixed_reference':cost-base_cost,
                'change_from_accepted_incumbent':cost-incumbent_cost,
                'fixed_reference_Armijo':bool(armijo),'improves_incumbent':bool(improves),
                'accepted':accepted}
            lines.append(row);atomic(folder/'line_search.json',{'candidates':lines})
            if accepted:
                state.best_material=trial;state.best_u=alpha*state.u;state.best_rf=rf;state.best_cost=cost
                break
        save_array(folder/'material.npy',state.best_material)
        save_array(folder/'best_log_direction.npy',state.best_u)
        save_array(args.out/'current_material.npy',state.best_material)
        atomic(folder/'incumbent_RF.json',state.best_rf)
        report.update(accepted=accepted,native_data_before=json.loads(
            (state.source_folder/'incumbent_RF.json').read_text())['loss']
            if (state.source_folder/'incumbent_RF.json').exists() else
            json.loads((state.source_folder/'material_update.json').read_text())['native_data_after'],
            native_data_after=state.best_rf['loss'],
            physical_RMS_change_from_incumbent=np.sqrt(np.mean(
                (state.best_material.astype(np.float64)-incumbent_material)[:,state.co.mask]**2,axis=1)).tolist())
        if accepted:report['accepted_alpha']=alpha
        atomic(folder/'material_update.json',report)
        if pool is not None:pool.close();pool=None
        run.update(status='post_hoc_maps_only',completed_step=1);atomic(args.out/'run.json',run)
        maps(args,folder,monitor)
        run.update(status='planned_native_GN_steps_finished',finished_unix_s=time.time())
        atomic(args.out/'run.json',run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',run);raise
    finally:
        if pool is not None:pool.close()
        monitor.log.close()

if __name__=='__main__':main()
