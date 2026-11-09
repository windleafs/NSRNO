"""Low-memory nonlinear joint inversion with fresh source-block sketches.

Every accepted outer step changes the reference material. New gradients and
HVPs are evaluated there; old-reference secants never become exact new data.
The RF-conditioned spectral symbol is only an approximate full-rank base.
"""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import traceback
import numpy as np
from .native_gn_block_pool import Pool
from .native_gn_fresh_state import FreshSourceQuadratic
from .native_gn_json import atomic
from .native_gn_coordinates import dot
from .refine_native_gn_cpu import solve_accurately,save_array
from .reconstruct_native_gn import source_manifest
from .resume_native_gn_accurate import maps
from .pressure_resources import PressureResources


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--symbol-snapshot',type=Path,required=True)
    p.add_argument('--preconditioner-statistics',type=Path,required=True)
    p.add_argument('--calibration-record',type=Path,required=True)
    p.add_argument('--sample',default='train_200')
    p.add_argument('--outer-steps',type=int,default=2)
    p.add_argument('--inner-hvps',type=int,default=2)
    p.add_argument('--CPU-maximum',type=int,default=1200)
    p.add_argument('--CPU-tolerance',type=float,default=.01)
    p.add_argument('--true-QP-forcing',type=float,default=.3)
    p.add_argument('--trust',type=float,nargs=3,default=[.02,.10,.35])
    p.add_argument('--tv-weight',type=float,default=.001)
    p.add_argument('--damping-relative',type=float,default=.01)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    args=p.parse_args()
    args.out=args.out.resolve()
    if args.out.exists():raise RuntimeError('Inspect the existing run; no implicit restart')
    if args.cap_gib>14:raise ValueError('Keep CUDA context below the 16-GiB per-card limit')
    material=np.load(args.seed)
    if material.shape!=(3,961,1121):raise ValueError('Full native V5 material required')
    edge=np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    with np.load(args.preconditioner_statistics) as data:blocks=data['blocks'].astype(np.float64)
    calibration=json.loads(args.calibration_record.read_text())['local_data_preconditioner_scale']
    symbol=np.load(args.symbol_snapshot,mmap_mode='r')
    names=['native_gn_block_worker.py','native_gn_block_pool.py','native_gn_block_metric.py',
        'native_gn_fresh_state.py','reconstruct_native_gn_nonlinear.py','native_gn_spectral.py',
        'native_gn_multisecant.py','native_gn_secant.py','native_gn_json.py','refine_native_gn_cpu.py']
    manifest=source_manifest()
    for name in names:manifest[name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    args.out.mkdir(parents=True)
    (args.out/'jobs').mkdir()
    run={'status':'starting_fresh_nonlinear_source_inversion','pid':os.getpid(),
        'sample':args.sample,'seed':str(args.seed.resolve()),
        'seed_SHA256':hashlib.sha256(args.seed.read_bytes()).hexdigest(),
        'solver_revision':'fresh_native_source_block_Nystrom_each_nonlinear_outer_step',
        'source_sha256':manifest,'symbol_snapshot':str(args.symbol_snapshot.resolve()),
        'symbol_SHA256':hashlib.sha256(args.symbol_snapshot.read_bytes()).hexdigest(),
        'preconditioner_statistics':str(args.preconditioner_statistics.resolve()),
        'old_exact_secants_reused_at_new_reference':False,
        'spectral_base_is_only_approximate':True,'unknowns':'independent full-grid c/rho/tissue-alpha',
        'shape':list(material.shape),'native_spacing_m':50e-6,
        'shots':'11LF + 11point3','receivers':192,'all_retained_RF':True,
        'trust_log_c_log_Z_log_alpha':args.trust,'outer_step_budget':args.outer_steps,
        'inner_HVP_budget_per_reference':args.inner_hvps,'devices':args.devices,
        'allocator_cap_GiB':args.cap_gib,'model_per_card_limit_GiB':16,
        'truth_read_by_optimizer':False,'joint_imaging_success_claimed':False,
        'created_unix_s':time.time(),'completed_step':0}
    atomic(args.out/'run.json',run)
    save_array(args.out/'initial_material.npy',material)
    monitor=PressureResources(args.out)
    pool=None
    try:
        pool=Pool(args,monitor)
        for step in range(1,args.outer_steps+1):
            folder=args.out/f'step{step:03d}'
            folder.mkdir()
            save_array(folder/'old_material.npy',material)
            save_array(folder/'stage0_initial_material.npy',material)
            run.update(status='native_gradient_at_new_material',step=step)
            atomic(args.out/'run.json',run)
            baseline,gradient=pool.evaluate(material,'gradient',f'outer{step:03d}_gradient')
            atomic(folder/'baseline_RF.json',baseline)
            save_array(folder/'data_gradient.npy',gradient)
            state=FreshSourceQuadratic(material,edge,gradient,baseline,symbol,blocks,
                calibration,args.tv_weight,args.damping_relative,args.trust)
            atomic(folder/'fresh_metric.json',state.model.report)
            for inner in range(args.inner_hvps):
                relative=state.norm()/max(state.initial_norm,1e-300)
                if relative<=args.true_QP_forcing:break
                run.update(status='CPU_fresh_source_box_QP',inner=inner,
                    current_true_QP_KKT_relative=relative)
                atomic(args.out/'run.json',run)
                cpu_folder=folder/f'cpu{inner:03d}'
                cpu_folder.mkdir()
                def progress(row,d,save):
                    atomic(cpu_folder/'progress.json',row)
                    atomic(folder/'direction_progress.json',dict(row,phase='fresh_source_CPU',
                        native_HVP_count=len(state.history),true_QP_KKT_relative=relative))
                    if save:save_array(cpu_folder/'direction_checkpoint.npy',d)
                d,cpu=solve_accurately(state.model,state.gradient,state.low-state.u,
                    state.high-state.u,np.zeros_like(state.u),maximum=args.CPU_maximum,
                    tolerance=args.CPU_tolerance,callback=progress)
                atomic(cpu_folder/'result.json',cpu)
                save_array(cpu_folder/'direction.npy',d)
                if not cpu['tolerance_met']:
                    run['inner_stop_reason']='CPU accuracy budget reached; retain previously certified segments'
                    break
                if dot(state.gradient,d)>=0:break
                run['status']='native_HVP_at_current_material'
                atomic(args.out/'run.json',run)
                job,data_image=pool.evaluate(material,'hvp',f'outer{step:03d}_hvp{inner:03d}',d)
                row=state.append(d,data_image,job['source_curvature'],folder/f'hvp{inner:03d}',cpu)
                atomic(folder/'direction_progress.json',row)
            save_array(folder/'log_direction.npy',state.u)
            save_array(folder/'cumulative_total_image.npy',state.Hu)
            slope=dot(state.gradient0,state.u)
            lines=[];accepted=False
            before=material.copy()
            if slope<0:
                for line in range(8):
                    alpha=.5**line
                    trial=state.co.material(alpha*state.u)
                    run.update(status='native_nonlinear_material_line_search',line=line)
                    atomic(args.out/'run.json',run)
                    rf,_=pool.evaluate(trial,'forward',f'outer{step:03d}_line{line:02d}')
                    tv=state.co.regularization(alpha*state.u,args.tv_weight)[0]
                    change=rf['loss']+tv-baseline['loss']-state.tv
                    accepted=bool(change<=1e-4*alpha*slope)
                    predicted=state.predicted_physical_change(alpha)
                    lines.append({'alpha':alpha,'native_RF':rf,'TV':tv,
                        'physical_objective_change':change,'predicted_change':predicted,
                        'actual_to_predicted_reduction':change/predicted if predicted<0 else None,
                        'accepted':accepted})
                    atomic(folder/'line_search.json',{'candidates':lines})
                    if accepted:
                        material=trial
                        break
            save_array(folder/'material.npy',material)
            save_array(args.out/'current_material.npy',material)
            report={'accepted':accepted,'native_data_before':baseline['loss'],
                'native_data_after':lines[-1]['native_RF']['loss'] if accepted else baseline['loss'],
                'new_HVPs':len(state.history),'true_QP_KKT_relative':state.norm()/max(state.initial_norm,1e-300),
                'model_change':state.value(),'fresh_reference_each_outer_step':True,
                'old_exact_secants_reused':False,'trust_log_c_log_Z_log_alpha':args.trust,
                'physical_RMS_change_c_rho_alpha':np.sqrt(np.mean(
                    (material.astype(np.float64)-before)[:,state.co.mask]**2,axis=1)).tolist(),
                'truth_read':False}
            atomic(folder/'material_update.json',report)
            run.update(status='post_hoc_maps_only',completed_step=step)
            atomic(args.out/'run.json',run)
            maps(args,folder,monitor)
            if not accepted:
                run['status']='no_native_descent_at_current_reference'
                break
        else:run['status']='planned_nonlinear_steps_finished'
        run['finished_unix_s']=time.time()
        atomic(args.out/'run.json',run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',run)
        raise
    finally:
        if pool is not None:pool.close()
        monitor.log.close()


if __name__=='__main__':main()
