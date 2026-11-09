"""Native joint reconstruction with ROI/global block Krylov allocation."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from .native_gn_spatial_state import NativeSpatialState,digest
from .native_gn_coordinates import dot
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def array(path,value):
    temporary=path.with_name(path.stem+'.partial.npy')
    np.save(temporary,value);temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--extra-hvps',type=int,default=6)
    p.add_argument('--max-seconds',type=float,default=3600)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    a=p.parse_args()
    if a.out.exists():
        raise RuntimeError('Preserve output; no implicit restart')
    if a.extra_hvps<1 or a.max_seconds<=0 or not 0<a.cap_gib<=14:
        raise ValueError('Invalid fixed budget')
    state=NativeSpatialState(a.source);a.sample=state.sample
    from .reconstruct_native_gn import Pool
    hashes=dict(state.source_run['source_sha256'])
    for name in ['native_gn_spatial_state.py','reconstruct_native_spatial_krylov.py']:
        hashes[name]=digest(Path(__file__).parent/name)
    a.out.mkdir();(a.out/'jobs').mkdir();folder=a.out/'step001';folder.mkdir()
    for name in hashes:
        (a.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    run={'status':'restored_joint_spatial_state','pid':os.getpid(),'created_unix_s':time.time(),
        'source_root':str(a.source.resolve()),'sample':a.sample,
        'solver_revision':'true_Galerkin_with_ROI_ROI_global_proposals',
        'source_sha256':hashes,'devices':a.devices,'allocator_cap_GiB':a.cap_gib,'model_per_card_limit_GiB':16,
        'new_gradient_wave_solves':0,'reused_complete_HVPs':6,'extra_HVP_budget':a.extra_hvps,
        'walltime_budget_seconds':a.max_seconds,'walltime_stop_at_completed_HVP_boundary':True,
        'fixed_reference_material':str((state.origin/'stage0_initial_material.npy').resolve()),
        'fixed_reference_SHA256':state.fixed_SHA256,'fixed_reference_RF_loss':state.baseline_rf['loss'],
        'accepted_incumbent_material':str((state.origin/'material.npy').resolve()),
        'incumbent_material_SHA256':state.incumbent_SHA256,'incumbent_RF_loss':state.incumbent_rf['loss'],
        'all_three_coordinates_free':True,'shape':list(state.material.shape),'native_spacing_m':50e-6,
        'ROI_lateral_mm':[-19.1,19.1],'ROI_depth_mm':[1,43],
        'proposal_block_schedule':['ROI','ROI','global'],'no_permanent_exterior_freezing':True,
        'full_projected_KKT_target_relative':.001,'ROI_residual_does_not_replace_full_residual':True,
        'all_retained_RF':True,'shots':'11LF + 11point3','receivers':192,
        'unknowns':'independent full native c/rho/alpha_tissue',
        'truth_read_by_optimizer':False,'joint_imaging_success_claimed':False,
        'reserved_test_case_unread':'Neg_47_Left','new_HVPs':0,
        'line_search':'incumbent-anchored; true cost decrease divided by known quadratic prediction',
        'line_acceptance_minimum_ratio':.05,
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    atomic(a.out/'run.json',run)
    atomic(a.out/'fixed_model_reference.json',state.reference)
    for name,value in [('old_material',state.material),('linear_reference_material',state.material),
        ('stage0_initial_material',state.incumbent),('material',state.incumbent),
        ('anchor_log_direction',state.anchor_u),('anchor_total_image',state.anchor_Hu)]:
        array(folder/(name+'.npy'),value)
    array(a.out/'current_material.npy',state.incumbent)
    shutil.copy2(state.origin/'data_gradient.npy',folder/'data_gradient.npy')
    atomic(folder/'baseline_RF.json',state.baseline_rf);atomic(folder/'incumbent_RF.json',state.incumbent_rf)
    atomic(folder/'restored.json',{**state.initial_report,'provenance':state.provenance,
        'exact_reference_SHA256':state.fixed_SHA256,'current_incumbent_SHA256':state.incumbent_SHA256})
    paths=list(state.paths)
    atomic(folder/'fixed_HVP_manifest.json',{'paths':[str(v) for v in paths]})
    atomic(folder/'line_search.json',{'candidates':[]})
    monitor=PressureResources(a.out);pool=None;child=None;completed=0;started=time.time()
    def freeze():
        for name,expected in hashes.items():
            if digest(Path(__file__).parent/name)!=expected:
                raise RuntimeError('Frozen source changed: '+name)
        if digest(state.origin/'stage0_initial_material.npy')!=state.fixed_SHA256:
            raise RuntimeError('Fixed physical reference changed')
    try:
        for extra in range(a.extra_hvps):
            if state.span.projected_norm()<=.001*state.initial_norm:
                break
            freeze();direction,proposal=state.propose(extra)
            direction=direction.astype(np.float32).astype(np.float64)
            index=len(paths);checkpoint=folder/f'hvp{index:03d}';checkpoint.mkdir()
            array(checkpoint/'direction.npy',direction);atomic(checkpoint/'proposal.json',proposal)
            run.update(status='native_spatial_HVP',new_HVPs=completed,fixed_HVP_count=len(paths),
                next_HVP=index,requested_spatial_block=proposal['requested_spatial_block'],
                actual_spatial_block=proposal['proposal_block'])
            atomic(a.out/'run.json',run)
            if pool is None:
                pool=Pool(a,monitor)
            job,data_image=pool.evaluate(state.material,'hvp',f'spatial_{index:03d}_{proposal["proposal_block"]}',direction)
            array(checkpoint/'data_image.npy',data_image)
            total_image=data_image+state.h_tv(direction)+state.damping*direction
            array(checkpoint/'total_image.npy',total_image)
            old_value=dot(state.gradient,state.current_u())+.5*dot(state.current_u(),state.current_Hu())
            run['status']='spatial_true_subspace_QP';atomic(a.out/'run.json',run)
            normal=state.span.append(direction,total_image)
            report={**normal,**state.span.solve(),**state.residual_report()}
            if report['model_change']>old_value+1e-8:
                raise RuntimeError('Enlarged exact subspace lost decrease')
            completed+=1;paths.append(checkpoint.resolve())
            array(checkpoint/'cumulative_log_direction.npy',state.current_u())
            array(checkpoint/'cumulative_total_image.npy',state.current_Hu())
            report.update(iteration=index+1,new_HVPs=completed,reused_complete_HVPs=6,
                requested_spatial_block=proposal['requested_spatial_block'],
                actual_spatial_block=proposal['proposal_block'],true_HVP_job=job['tag'],
                local_data_preconditioner_scale=state.scale,seconds_from_start=time.time()-started)
            atomic(checkpoint/'progress.json',report);atomic(folder/'progress.json',report)
            atomic(folder/'fixed_HVP_manifest.json',{'paths':[str(v) for v in paths]})
            if time.time()-started>=a.max_seconds:
                break
        u=state.current_u();Hu=state.current_Hu()
        array(folder/'log_direction.npy',u);array(folder/'cumulative_total_image.npy',Hu)
        direction_report={'method':run['solver_revision'],'reused_HVPs':6,'new_HVPs':completed,
            'total_directions':len(paths),'model_change':dot(state.gradient,u)+.5*dot(u,Hu),
            **state.residual_report(),'QP_exact_solution_claimed':False,
            'full_KKT_target_met':bool(state.span.projected_norm()<=.001*state.initial_norm),'truth_read':False}
        atomic(folder/'direction.json',direction_report)
        delta=u-state.anchor_u;Hdelta=Hu-state.anchor_Hu
        model_slope=dot(state.gradient+state.anchor_Hu,delta);model_curvature=dot(delta,Hdelta)
        if model_curvature<-1e-8*max(abs(model_slope),1e-30):
            raise RuntimeError('Known image increment has negative curvature')
        accepted=False;lines=[];best=state.incumbent.copy();best_rf=state.incumbent_rf
        incumbent_cost=best_rf['loss']+state.incumbent_tv
        for line in range(8 if completed else 0):
            alpha=.5**line
            predicted=alpha*model_slope+.5*alpha*alpha*model_curvature
            if predicted>=-1e-12:
                break
            freeze();trial_u=state.anchor_u+alpha*delta;trial=state.co.material(trial_u)
            run.update(status='native_material_line_search',line=line,new_HVPs=completed)
            atomic(a.out/'run.json',run)
            rf,_=pool.evaluate(trial,'forward',f'spatial_line{line:02d}')
            tv=state.co.regularization(trial_u,.001)[0];cost=rf['loss']+tv
            actual_change=cost-incumbent_cost;ratio=actual_change/predicted
            accepted=bool(ratio>=.05 and cost<incumbent_cost-max(1e-12,abs(incumbent_cost)*1e-8))
            lines.append({'alpha':alpha,'native_RF':rf,'TV':tv,'physical_objective':cost,
                'actual_objective_change_from_incumbent':actual_change,'predicted_change_from_known_image_anchor':predicted,
                'actual_to_predicted_decrease_ratio':ratio,'accepted':accepted,
                'not_claimed_as_exact_gradient_Armijo_at_incumbent':True})
            atomic(folder/'line_search.json',{'candidates':lines})
            if accepted:
                best=trial;best_rf=rf;array(folder/'accepted_log_direction.npy',trial_u)
                array(folder/'accepted_total_image.npy',state.anchor_Hu+alpha*Hdelta)
                break
        array(folder/'material.npy',best);array(a.out/'current_material.npy',best)
        update={**direction_report,'accepted':accepted,'native_data_before':state.incumbent_rf['loss'],
            'native_data_after':best_rf['loss'],'material_SHA256':digest(folder/'material.npy'),
            'fixed_reference_native_RF':state.baseline_rf['loss'],
            'physical_RMS_change_from_incumbent':np.sqrt(np.mean((best.astype(np.float64)-state.incumbent)[:,state.co.mask]**2,axis=1)).tolist(),
            'FOV_physical_RMS_change_from_incumbent':np.sqrt(np.mean((best.astype(np.float64)-state.incumbent)[:,state.roi]**2,axis=1)).tolist()}
        atomic(folder/'material_update.json',update);atomic(folder/'incumbent_RF.json',best_rf)
        if pool is not None:
            pool.close();pool=None
        run.update(status='post_hoc_maps_only',new_HVPs=completed,accepted=accepted)
        atomic(a.out/'run.json',run)
        with (folder/'maps.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m','nsrno.evaluate_time_material',str(folder/'material.npy'),
                '--sample',a.sample,'--out',str(folder/'maps'),'--estimate-label','Spatial-block joint GN'],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            run['child_pid']=child.pid;atomic(a.out/'run.json',run);code=child.wait()
        if code:
            raise RuntimeError('Post-hoc maps failed; inspect preserved maps.log')
        monitor.sample('native_spatial_Krylov_finished')
        run.update(status='native_spatial_Krylov_finished',finished_unix_s=time.time(),
            native_data_after=best_rf['loss'],material_SHA256=update['material_SHA256'])
        atomic(a.out/'run.json',run);print(json.dumps(run),flush=True)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(a.out/'run.json',run);raise
    finally:
        if pool is not None:
            pool.close()
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill();child.wait()
        monitor.log.close()


if __name__=='__main__':
    main()
