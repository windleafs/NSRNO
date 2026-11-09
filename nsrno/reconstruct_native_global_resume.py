"""Recover one interrupted native global GN model within its original budget."""
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
from .native_gn_recovered_state import NativeRecoveredGlobalState
from .native_gn_spatial_state import digest
from .native_gn_coordinates import dot
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def array(path,value):
    temporary=path.with_name(path.stem+'.partial.npy')
    np.save(temporary,value);temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--extra-hvps',type=int,default=48)
    parser.add_argument('--max-seconds',type=float,default=18000)
    parser.add_argument('--check-every',type=int,default=6)
    parser.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--chunk',type=int,default=128)
    parser.add_argument('--cap-gib',type=float,default=10.)
    args=parser.parse_args()
    if args.out.exists():
        raise RuntimeError('Preserve output; no implicit restart')
    if args.extra_hvps<1 or args.max_seconds<=0 or args.check_every<1 or not 0<args.cap_gib<=14:
        raise ValueError('Invalid fixed budget')
    # This restores all saved images and the physical anchor on CPU before Pool.
    state=NativeRecoveredGlobalState(args.source);args.sample=state.sample
    if args.extra_hvps>state.original_total_budget-state.original_new_offset:
        raise ValueError('Recovery would enlarge original HVP budget')
    args.max_seconds=max(state.original_deadline-time.time(),1e-6)
    from .reconstruct_native_gn import Pool
    hashes=dict(state.source_run['source_sha256'])
    for name in ['native_gn_recovered_state.py','reconstruct_native_global_resume.py']:
        hashes[name]=digest(Path(__file__).parent/name)
    args.out.mkdir();(args.out/'jobs').mkdir();folder=args.out/'step001';folder.mkdir()
    checks=args.out/'physical_checkpoints';checks.mkdir()
    for name in hashes:
        (args.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    initial_material=state.incumbent.copy();initial_rf=state.incumbent_rf
    initial_tv=state.incumbent_tv;paths=list(state.paths);reused=len(paths)
    run={'status':'restored_global_continuation_state','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        'created_unix_s':time.time(),'source_root':str(args.source.resolve()),
        'reference_source_root':str(state.reference_source_root),'sample':args.sample,
        'solver_revision':'recovered_global_true_Galerkin_with_unchanged_reference_and_original_budget',
        'source_sha256':hashes,'devices':args.devices,'allocator_cap_GiB':args.cap_gib,
        'model_per_card_limit_GiB':16,'new_gradient_wave_solves':0,
        'reused_complete_HVPs':reused,'extra_HVP_budget':args.extra_hvps,
        'walltime_budget_seconds':args.max_seconds,'walltime_stop_at_completed_HVP_boundary':True,
        'physical_check_every_new_HVPs':args.check_every,
        'fixed_reference_material':str(state.fixed_reference_path),
        'fixed_reference_SHA256':state.fixed_SHA256,'fixed_reference_RF_loss':state.baseline_rf['loss'],
        'accepted_incumbent_material':str((state.origin/'material.npy').resolve()),
        'incumbent_material_SHA256':state.incumbent_SHA256,'incumbent_RF_loss':initial_rf['loss'],
        'all_three_coordinates_free':True,'shape':list(state.material.shape),'native_spacing_m':50e-6,
        'ROI_lateral_mm':[-19.1,19.1],'ROI_depth_mm':[1,43],
        'proposal_block_schedule':['global'],'no_new_hard_FOV_mask':True,
        'full_projected_KKT_target_relative':.001,'ROI_residual_does_not_replace_full_residual':True,
        'all_retained_RF':True,'shots':'11LF + 11point3','receivers':192,
        'unknowns':'independent full native c/rho/alpha_tissue','truth_read_by_optimizer':False,
        'joint_imaging_success_claimed':False,'reserved_test_case_unread':'Neg_47_Left',
        'new_HVPs':0,'accepted_physical_checks':0,'physical_checks':0,'child_pid':None,
        'line_search':'incumbent-anchored true RF+TV decrease / fixed-model predicted decrease',
        'line_acceptance_minimum_ratio':.05,'refresh_reference_ratio_below':.25,
        'refresh_reference_step_below':.25,'disk_reserve_GiB':2.,
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    run.update(original_total_new_HVP_budget=state.original_total_budget,
        original_new_HVP_offset=state.original_new_offset,original_new_HVPs=state.original_new_offset,
        original_budget_start_unix_s=state.original_budget_start,original_deadline_unix_s=state.original_deadline,
        original_initial_root=state.original_initial_root,original_source_physical_root=state.original_source_physical_root,
        recovery_evidence=state.recovery_evidence,material_SHA256=state.incumbent_SHA256,
        native_data_after=initial_rf['loss'])
    atomic(args.out/'run.json',run);atomic(args.out/'fixed_model_reference.json',state.reference)
    for name,value in [('old_material',state.material),('linear_reference_material',state.material),
        ('stage0_initial_material',initial_material),('material',initial_material),
        ('anchor_log_direction',state.anchor_u),('anchor_total_image',state.anchor_Hu),
        ('accepted_log_direction',state.anchor_u),('accepted_total_image',state.anchor_Hu),
        ('log_direction',state.current_u()),('cumulative_total_image',state.current_Hu())]:
        array(folder/(name+'.npy'),value)
    array(args.out/'current_material.npy',initial_material)
    shutil.copy2(state.origin/'data_gradient.npy',folder/'data_gradient.npy')
    atomic(folder/'baseline_RF.json',state.baseline_rf);atomic(folder/'incumbent_RF.json',initial_rf)
    atomic(folder/'restored.json',{**state.initial_report,'provenance':state.provenance,
        'exact_reference_SHA256':state.fixed_SHA256,'current_incumbent_SHA256':state.incumbent_SHA256})
    atomic(folder/'fixed_HVP_manifest.json',{'paths':[str(v) for v in paths]})
    atomic(folder/'line_search.json',{'checkpoints':[]})
    monitor=PressureResources(args.out);pool=None;child=None;completed=0
    last_assessed=0;started=time.time();physical_history=[]

    def freeze():
        for name,expected in hashes.items():
            if digest(Path(__file__).parent/name)!=expected:
                raise RuntimeError('Frozen source changed: '+name)
        if digest(state.fixed_reference_path)!=state.fixed_SHA256:
            raise RuntimeError('Fixed linear reference changed')

    def direction_report():
        u=state.current_u();Hu=state.current_Hu()
        return {'method':run['solver_revision'],'reused_HVPs':reused,'new_HVPs':completed,'original_new_HVPs':completed+state.original_new_offset,
            'total_directions':len(paths),'model_change':dot(state.gradient,u)+.5*dot(u,Hu),
            **state.residual_report(),'full_space_exact_solution_claimed':False,
            'full_KKT_target_met':bool(state.span.projected_norm()<=.001*state.initial_norm),
            'truth_read':False}

    def overall_update():
        change=state.incumbent.astype(np.float64)-initial_material
        update={**direction_report(),'accepted':bool(run['accepted_physical_checks']),
            'native_data_before':initial_rf['loss'],'native_data_after':state.incumbent_rf['loss'],
            'initial_physical_TV':initial_tv,'returned_physical_TV':state.incumbent_tv,
            'material_SHA256':digest(folder/'material.npy'),
            'fixed_reference_native_RF':state.baseline_rf['loss'],
            'physical_RMS_change_from_incumbent':np.sqrt(np.mean(change[:,state.co.mask]**2,axis=1)).tolist(),
            'FOV_physical_RMS_change_from_incumbent':np.sqrt(np.mean(change[:,state.roi]**2,axis=1)).tolist(),
            'physical_checks':run['physical_checks'],'accepted_physical_checks':run['accepted_physical_checks']}
        atomic(folder/'material_update.json',update);atomic(folder/'direction.json',direction_report())
        return update

    def maps(destination,label):
        nonlocal child
        freeze();run.update(status='post_hoc_maps_only',map_folder=str(destination))
        atomic(args.out/'run.json',run)
        with (destination/'maps.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m','nsrno.evaluate_time_material',
                str(destination/'material.npy'),'--sample',args.sample,'--out',str(destination/'maps'),
                '--estimate-label',label],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            run.update(child_pid=child.pid,
                child_start_ticks=int(Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19]))
            atomic(args.out/'run.json',run)
            while child.poll() is None:
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:monitor.sample('post_hoc_long_GN_maps_only')
            code=child.returncode;last_pid=child.pid;child=None
        run.update(child_pid=None,last_reaped_child_pid=last_pid,last_child_exit_code=code)
        atomic(args.out/'run.json',run)
        if code:
            raise RuntimeError('Post-hoc maps failed; inspect preserved maps.log')

    def physical_check():
        nonlocal last_assessed,pool
        if completed==last_assessed:
            return None
        checkpoint=checks/f'new{completed+state.original_new_offset:03d}';checkpoint.mkdir()
        before=state.incumbent.copy();before_rf=state.incumbent_rf;before_tv=state.incumbent_tv
        array(checkpoint/'stage0_initial_material.npy',before)
        array(checkpoint/'QP_log_direction.npy',state.current_u())
        array(checkpoint/'QP_total_image.npy',state.current_Hu())
        array(checkpoint/'QP_coefficients.npy',state.span.coefficients)
        atomic(checkpoint/'direction.json',direction_report())
        delta=state.current_u()-state.anchor_u;Hdelta=state.current_Hu()-state.anchor_Hu
        slope=dot(state.gradient+state.anchor_Hu,delta);curvature=dot(delta,Hdelta)
        if curvature<-1e-8*max(abs(slope),1e-30):
            raise RuntimeError('Known image increment has negative curvature')
        accepted=False;lines=[];refresh=None
        for line in range(8):
            alpha=.5**line;predicted=alpha*slope+.5*alpha*alpha*curvature
            if predicted>=-1e-12:
                refresh='fixed_reference_no_predicted_descent';break
            freeze();trial_u=state.anchor_u+alpha*delta;trial=state.co.material(trial_u)
            run.update(status='native_material_line_search',line=line,
                physical_check_new_HVPs=completed,new_HVPs=completed,original_new_HVPs=completed+state.original_new_offset)
            atomic(args.out/'run.json',run)
            if pool is None:
                pool=Pool(args,monitor)
            rf,_=pool.evaluate(trial,'forward',f'global_check{completed+state.original_new_offset:03d}_line{line:02d}')
            tv=state.co.regularization(trial_u,.001)[0];cost=rf['loss']+tv
            change=cost-before_rf['loss']-before_tv;ratio=change/predicted
            accepted=bool(ratio>=.05 and change<-max(1e-12,abs(before_rf['loss']+before_tv)*1e-8))
            lines.append({'alpha':alpha,'native_RF':rf,'TV':tv,'physical_objective':cost,
                'actual_objective_change_from_incumbent':change,
                'predicted_change_from_known_image_anchor':predicted,
                'actual_to_predicted_decrease_ratio':ratio,'accepted':accepted,
                'not_claimed_as_exact_gradient_Armijo_at_incumbent':True})
            atomic(checkpoint/'line_search.json',{'candidates':lines})
            if accepted:
                state.incumbent=trial;state.incumbent_rf=rf;state.incumbent_tv=tv
                state.anchor_u=trial_u;state.anchor_Hu=state.anchor_Hu+alpha*Hdelta
                run['accepted_physical_checks']+=1
                if ratio<.25 or alpha<.25:
                    refresh='accepted_step_requires_reference_refresh'
                break
        if not accepted and refresh is None:
            refresh='fixed_reference_no_actual_descent'
        # Save returned fields and their exact known-image anchor before truth maps.
        for destination in [checkpoint,folder]:
            array(destination/'material.npy',state.incumbent)
            array(destination/'accepted_log_direction.npy',state.anchor_u)
            array(destination/'accepted_total_image.npy',state.anchor_Hu)
            atomic(destination/'incumbent_RF.json',state.incumbent_rf)
        array(args.out/'current_material.npy',state.incumbent)
        difference=state.incumbent.astype(np.float64)-before
        update={**direction_report(),'accepted':accepted,'native_data_before':before_rf['loss'],
            'native_data_after':state.incumbent_rf['loss'],'physical_TV_before':before_tv,
            'physical_TV_after':state.incumbent_tv,'material_SHA256':digest(checkpoint/'material.npy'),
            'physical_RMS_change':np.sqrt(np.mean(difference[:,state.co.mask]**2,axis=1)).tolist(),
            'FOV_physical_RMS_change':np.sqrt(np.mean(difference[:,state.roi]**2,axis=1)).tolist(),
            'reference_refresh_reason':refresh}
        atomic(checkpoint/'material_update.json',update)
        atomic(checkpoint/'line_search.json',{'candidates':lines})
        run['physical_checks']+=1;last_assessed=completed
        physical_history.append({'checkpoint':str(checkpoint.resolve()),'new_HVPs':completed,'original_new_HVPs':completed+state.original_new_offset,
            'accepted':accepted,'reference_refresh_reason':refresh,'update':update,'candidates':lines})
        atomic(folder/'line_search.json',{'checkpoints':physical_history})
        overall_update()
        run.update(material_SHA256=digest(folder/'material.npy'),native_data_after=state.incumbent_rf['loss'],
            latest_physical_checkpoint=str(checkpoint.resolve()))
        atomic(args.out/'run.json',run)
        maps(checkpoint,f'Recovered global GN: {completed+state.original_new_offset} total additional HVPs')
        return refresh

    try:
        overall_update();stop=None
        for extra in range(args.extra_hvps):
            if state.span.projected_norm()<=.001*state.initial_norm:
                stop='full_projected_KKT_target_met';break
            if time.time()>=state.original_deadline:
                stop='walltime_budget_at_completed_HVP_boundary';break
            # Three full saved HVP arrays plus space for a physical check and latest arrays.
            required=2*2**30+16*state.material.size*8
            if shutil.disk_usage(args.out).free<required:
                stop='disk_reserve_at_completed_HVP_boundary';break
            freeze();direction,proposal=state.propose_global()
            direction=direction.astype(np.float32).astype(np.float64)
            index=len(paths);checkpoint=folder/f'hvp{index:03d}';checkpoint.mkdir()
            array(checkpoint/'direction.npy',direction);atomic(checkpoint/'proposal.json',proposal)
            run.update(status='native_global_HVP',new_HVPs=completed,original_new_HVPs=completed+state.original_new_offset,fixed_HVP_count=len(paths),next_HVP=index)
            atomic(args.out/'run.json',run)
            if pool is None:
                pool=Pool(args,monitor)
            job,data_image=pool.evaluate(state.material,'hvp',f'global_{index:03d}',direction)
            array(checkpoint/'data_image.npy',data_image)
            total_image=data_image+state.h_tv(direction)+state.damping*direction
            array(checkpoint/'total_image.npy',total_image)
            # No expensive propagation is lost if the subsequent CPU QP fails.
            atomic(checkpoint/'image_provenance.json',{'true_HVP_job':job['tag'],
                'fixed_reference_SHA256':state.fixed_SHA256,'all_three_arrays_saved':True})
            old_value=dot(state.gradient,state.current_u())+.5*dot(state.current_u(),state.current_Hu())
            run['status']='global_true_subspace_QP';atomic(args.out/'run.json',run)
            normal=state.span.append(direction,total_image)
            report={**normal,**state.span.solve(),**state.residual_report()}
            if report['model_change']>old_value+1e-8:
                raise RuntimeError('Enlarged exact subspace lost decrease')
            completed+=1;paths.append(checkpoint.resolve())
            array(folder/'log_direction.npy',state.current_u())
            array(folder/'cumulative_total_image.npy',state.current_Hu())
            array(folder/'QP_coefficients.npy',state.span.coefficients)
            report.update(iteration=len(paths),new_HVPs=completed,original_new_HVPs=completed+state.original_new_offset,reused_complete_HVPs=reused,
                actual_spatial_block='global',true_HVP_job=job['tag'],
                local_data_preconditioner_scale=state.scale,seconds_from_start=time.time()-started)
            atomic(checkpoint/'progress.json',report);atomic(folder/'progress.json',report)
            atomic(folder/'fixed_HVP_manifest.json',{'paths':[str(v) for v in paths]})
            atomic(folder/'direction.json',direction_report())
            run.update(new_HVPs=completed,original_new_HVPs=completed+state.original_new_offset,fixed_HVP_count=len(paths),
                full_projected_KKT_relative=report['full_projected_KKT_relative'])
            atomic(args.out/'run.json',run)
            if (completed+state.original_new_offset)%args.check_every==0:
                stop=physical_check()
                if stop:break
        else:
            stop='new_HVP_budget_at_completed_HVP_boundary'
        if completed!=last_assessed:
            refresh=physical_check()
            if refresh:stop=refresh
        if state.span.projected_norm()<=.001*state.initial_norm and not stop:
            stop='full_projected_KKT_target_met'
        update=overall_update()
        if pool is not None:
            pool.close();pool=None
        # This comparison always uses the physical material accepted before this round.
        maps(folder,'Recovered global GN, returned physical material')
        monitor.sample('native_long_GN_finished')
        run.update(status='native_long_GN_finished',finished_unix_s=time.time(),
            stop_reason=stop or 'completed_boundary',new_HVPs=completed,original_new_HVPs=completed+state.original_new_offset,fixed_HVP_count=len(paths),
            native_data_after=state.incumbent_rf['loss'],material_SHA256=update['material_SHA256'],
            full_projected_KKT_relative=update['full_projected_KKT_relative'],
            full_KKT_target_met=update['full_KKT_target_met'],child_pid=None)
        atomic(args.out/'run.json',run);print(json.dumps(run),flush=True)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',run);raise
    finally:
        if pool is not None:
            pool.close()
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        monitor.log.close()


if __name__=='__main__':
    main()
