"""Recover the same conditional GN reference after an external interruption."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from .native_gn_coordinates import Coordinates,physical_blocks,dot
from .native_gn_galerkin import TrueImageSpan
from .native_gn_json import atomic
from .reconstruct_native_gn import Pool,source_manifest
from .pressure_resources import PressureResources


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(path,value):
    temporary=path.with_name(path.stem+'.partial.npy');np.save(temporary,value);temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--resume-from',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--block',choices=['impedance','speed','joint'],default='impedance')
    p.add_argument('--maximum-hvps',type=int,default=6)
    p.add_argument('--max-seconds',type=float,default=3600)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--cap-gib',type=float,default=10.)
    p.add_argument('--chunk',type=int,default=128)
    a=p.parse_args();source=json.loads((a.source_root/'run.json').read_text())
    interrupted=json.loads((a.resume_from/'run.json').read_text())
    if a.resume_from.resolve()==a.out.resolve():raise RuntimeError('Preserve interrupted output')
    if interrupted['status']!='native_conditional_HVP':raise RuntimeError('Only interrupted HVP stages supported')
    if interrupted['block']!=a.block or interrupted['maximum_new_HVPs']!=a.maximum_hvps:
        raise RuntimeError('Keep the interrupted block and total HVP budget')
    if Path(interrupted['source_root']).resolve()!=a.source_root.resolve():raise RuntimeError('Changed physical source')
    old_workers=json.loads((a.resume_from/'worker_processes.json').read_text())
    current_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if old_workers['boot_id']==current_boot:raise RuntimeError('This recovery entry requires a confirmed server reboot')
    old_pids=[interrupted['pid'],interrupted.get('child_pid'),*old_workers['worker_pids']]
    if any(pid and Path('/proc',str(pid)).exists() for pid in old_pids):raise RuntimeError('Old process identity is ambiguous')
    for name,expected in interrupted['source_sha256'].items():
        if digest(Path(__file__).parent/name)!=expected:raise RuntimeError('Interrupted source changed: '+name)
    if source['status'] not in ['planned_native_recycled_GN_steps_finished','native_block_GN_stage_finished']:
        raise RuntimeError('Require a completed physical source')
    pids=[source['pid'],source.get('child_pid')]
    workers=a.source_root/'worker_processes.json'
    if workers.exists():pids+=json.loads(workers.read_text()).get('worker_pids',[])
    if any(pid and Path('/proc',str(pid)).exists() for pid in pids):raise RuntimeError('Source process still exists')
    if a.out.exists():raise RuntimeError('Preserve current output; no implicit restart')
    if a.maximum_hvps<1 or a.max_seconds<=0 or not 0<a.cap_gib<=14:raise ValueError('Invalid fixed budget')
    seed_path=a.source_root/'step001/material.npy';seed=np.load(seed_path)
    if seed.shape!=(3,961,1121) or seed.dtype!=np.float32:raise ValueError('Full native float32 materials required')
    if digest(seed_path)!=interrupted['source_material_SHA256']:raise RuntimeError('Fixed reference changed')
    if digest(a.resume_from/'step001/stage0_initial_material.npy')!=digest(seed_path):
        raise RuntimeError('Interrupted fixed reference changed')
    if not np.array_equal(seed,np.load(a.resume_from/'current_material.npy')):raise RuntimeError('A nonlinear update already occurred')
    source_update=json.loads((a.source_root/'step001/material_update.json').read_text())
    source_rf=source_update['native_data_after'];a.sample=source.get('sample','train_200')
    statistics=Path('out/v5/reconstruction_pressure_v1_train_200/step002/joint_normal.npz')
    if digest(statistics)!='acd29954a613f156e95530c9dd67d413e6f1ea9a1b1492ec51eb58a354bfa961':
        raise RuntimeError('Preconditioner statistics changed')
    calibration=Path('out/v5/reconstruction_native_gn_secant_v1_train_200/step001/hvp00/progress.json')
    scale=json.loads(calibration.read_text())['local_data_preconditioner_scale']
    co=Coordinates(seed,np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'))
    low,high=co.bounds(trust=(1.,1.,10.));active={'impedance':[1],'speed':[0],'joint':[0,1,2]}[a.block]
    for j in set(range(3))-set(active):low[j]=0;high[j]=0
    zero=np.zeros_like(seed,dtype=np.float64);tv,g_tv,h_tv,tv_diagonal=co.regularization(zero,.001)
    with np.load(statistics) as data:blocks=physical_blocks(data['blocks'],seed,co.edge)
    damping=.01*scale*blocks[:3];damping[:,~co.mask]=0
    diagonal=np.maximum(scale*blocks[:3]+tv_diagonal+damping,1e-30);diagonal[:,~co.mask]=1
    hashes=source_manifest()
    for name,expected in source['source_sha256'].items():
        if name in hashes and hashes[name]!=expected:raise RuntimeError('Native source changed: '+name)
    for name in ['reconstruct_native_block_gn.py','resume_native_block_gn.py','native_gn_galerkin.py','native_span_qp.py',
        'native_gn_json.py','evaluate_time_material.py','v5_materials.py']:
        hashes[name]=digest(Path(__file__).parent/name)
    a.out.mkdir();(a.out/'jobs').mkdir();folder=a.out/'step001';folder.mkdir()
    for name in hashes:(a.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    tolerance=.001 if a.block=='joint' else .01
    run={'status':'restoring_interrupted_native_block','pid':os.getpid(),'created_unix_s':time.time(),
        'interrupted_root':str(a.resume_from.resolve()),'interrupted_boot_id':old_workers['boot_id'],
        'current_boot_id':current_boot,'same_reference_HVP_recovery':True,
        'original_total_HVP_budget_unchanged':True,'interrupted_original_created_unix_s':interrupted['created_unix_s'],
        'block':a.block,'temporarily_active_coordinates':active,'coordinate_names':['scaled_log_c','scaled_log_Z','scaled_log_alpha_tissue'],
        'source_root':str(a.source_root.resolve()),'source_material_SHA256':digest(seed_path),
        'source_native_RF':source_rf,'sample':a.sample,'maximum_new_HVPs':a.maximum_hvps,
        'walltime_budget_seconds':a.max_seconds,'walltime_stop_at_complete_HVP_boundary':True,
        'conditional_block_relative_target':tolerance,'conditional_residual_is_not_full_joint_residual':a.block!='joint',
        'source_sha256':hashes,'preconditioner_statistics':str(statistics.resolve()),'preconditioner_scale':scale,
        'old_HVPs_not_reused_at_new_reference':True,'native_full_gradient_jobs_planned':0,'sources_per_gradient_job':22,
        'devices':a.devices,'allocator_cap_GiB':a.cap_gib,'model_per_card_limit_GiB':16,
        'shape':list(seed.shape),'native_spacing_m':50e-6,'depth_scope_mm':[1,43],
        'shots':'11LF + 11point3','receivers':192,'all_retained_RF':True,
        'truth_read_by_optimizer':False,'reserved_test_case_unread':'Neg_47_Left','joint_imaging_success_claimed':False,
        'stage_is_initialization_only':a.block!='joint','new_HVPs':0}
    atomic(a.out/'run.json',run)
    for name,value in [('old_material',seed),('stage0_initial_material',seed),('material',seed)]:array(folder/(name+'.npy'),value)
    array(a.out/'current_material.npy',seed)
    atomic(folder/'line_search.json',{'candidates':[]});monitor=PressureResources(a.out);pool=None;child=None;started=time.time();completed=0
    def freeze():
        for name,expected in hashes.items():
            if digest(Path(__file__).parent/name)!=expected:raise RuntimeError('Frozen source changed: '+name)
    try:
        baseline=json.loads((a.resume_from/'step001/baseline_RF.json').read_text())
        gradient_data=np.load(a.resume_from/'step001/data_gradient.npy')
        if gradient_data.shape!=seed.shape or not np.isfinite(gradient_data).all():raise RuntimeError('Incomplete saved gradient')
        if abs(baseline['loss']-source_rf)>1e-5*max(abs(source_rf),1e-12):raise RuntimeError('Current baseline disagrees with accepted source')
        atomic(folder/'baseline_RF.json',baseline);array(folder/'data_gradient.npy',gradient_data)
        gradient=gradient_data+g_tv;span=TrueImageSpan(gradient,diagonal,low,high);initial_norm=span.projected_norm()
        atomic(a.out/'fixed_model_reference.json',{'origin_step_folder':str(folder.resolve()),'tv_weight':.001,
            'damping_relative':.01,'active_coordinates':active,'conditional_relative_target':tolerance})
        paths=[];checks=[];restored=int(interrupted['new_HVPs'])
        if not 0<restored<a.maximum_hvps:raise RuntimeError('No interrupted remainder to recover')
        old_folder=a.resume_from/'step001'
        old_jobs={json.loads(path.read_text())['tag']:json.loads(path.read_text())
            for path in (a.resume_from/'jobs').glob('*.json')}
        for index in range(restored):
            checkpoint=old_folder/f'hvp{index:03d}'
            direction=np.load(checkpoint/'direction.npy');data_image=np.load(checkpoint/'data_image.npy')
            image=np.load(checkpoint/'total_image.npy')
            if any(value.shape!=seed.shape or not np.isfinite(value).all() for value in [direction,data_image,image]):
                raise RuntimeError('Incomplete completed HVP '+str(index))
            computed=data_image+h_tv(direction)+damping*direction
            image_error=float(np.linalg.norm(computed-image)/max(np.linalg.norm(image),1e-300))
            tag=f'{a.block}_hvp{index:03d}';job=old_jobs[tag]
            curvature=dot(direction,data_image)
            curvature_error=abs(curvature-job['adjoint_quadratic'])/max(abs(curvature),1e-300)
            if image_error>1e-10 or curvature_error>1e-10:raise RuntimeError('Saved HVP integrity failed')
            span.append(direction,image);span.solve()
            u_error=float(np.linalg.norm(span.u-np.load(checkpoint/'cumulative_log_direction.npy').reshape(-1))
                /max(np.linalg.norm(span.u),1e-300))
            h_error=float(np.linalg.norm(span.Hu-np.load(checkpoint/'cumulative_total_image.npy').reshape(-1))
                /max(np.linalg.norm(span.Hu),1e-300))
            if max(u_error,h_error)>1e-7:raise RuntimeError('Saved Krylov state did not reconstruct')
            new_checkpoint=folder/checkpoint.name;shutil.copytree(checkpoint,new_checkpoint)
            paths.append(str(new_checkpoint.resolve()))
            checks.append({'index':index,'data_curvature_error':curvature_error,'total_image_error':image_error,
                'cumulative_direction_error':u_error,'cumulative_image_error':h_error,
                'direction_SHA256':digest(checkpoint/'direction.npy'),'total_image_SHA256':digest(checkpoint/'total_image.npy')})
        partial=old_folder/f'hvp{restored:03d}'
        if any((partial/name).exists() for name in ['data_image.npy','total_image.npy','progress.json']):
            raise RuntimeError('Additional completed native work needs separate recovery, do not discard')
        next_direction=span.next_direction()[0].astype(np.float32).astype(np.float64)
        if (partial/'direction.npy').exists():
            old_next=np.load(partial/'direction.npy')
            next_error=float(np.linalg.norm(next_direction-old_next)/max(np.linalg.norm(old_next),1e-300))
            if next_error>1e-7:raise RuntimeError('Next saved direction did not reconstruct')
        else:next_error=None
        for path in (a.resume_from/'jobs').glob('*.json'):shutil.copy2(path,a.out/'jobs'/path.name)
        last=json.loads((old_folder/'progress.json').read_text())
        uptime=float(Path('/proc/uptime').read_text().split()[0])
        elapsed_prior=max(last['seconds_from_start'],time.time()-uptime-interrupted['created_unix_s'])
        started=time.time()-elapsed_prior;completed=restored
        atomic(a.out/'restored.json',{'restored_complete_HVPs':restored,'checks':checks,
            'next_direction_relative_error':next_error,'same_native_reference_material_SHA256':digest(seed_path),
            'restored_conditional_block_relative':span.projected_norm()/max(initial_norm,1e-300),
            'counted_elapsed_before_interruption_seconds':elapsed_prior,'downtime_excluded_from_active_budget':True,
            'gradient_reused_without_wave_solves':True,'old_completed_HVPs_not_recomputed':True})
        atomic(folder/'fixed_HVP_manifest.json',{'paths':paths})
        atomic(folder/'progress.json',last)
        shutil.copy2(a.resume_from/'resource_summary.json',a.out/'prior_resource_summary.json')
        run.update(status='restored_native_block_ready',new_HVPs=completed,restored_HVPs=restored,
            additional_completed_HVPs=0,counted_elapsed_before_interruption_seconds=elapsed_prior)
        atomic(a.out/'run.json',run);freeze();pool=Pool(a,monitor)
        for index in range(completed,a.maximum_hvps):
            if initial_norm<1e-20 or span.projected_norm()<=tolerance*initial_norm:break
            freeze();direction,proposal=span.next_direction();direction=direction.astype(np.float32).astype(np.float64)
            checkpoint=folder/f'hvp{index:03d}';checkpoint.mkdir();array(checkpoint/'direction.npy',direction)
            atomic(checkpoint/'proposal.json',proposal)
            run.update(status='native_conditional_HVP',new_HVPs=completed,next_HVP=index)
            atomic(a.out/'run.json',run)
            job,data_image=pool.evaluate(seed,'hvp',f'{a.block}_hvp{index:03d}',direction)
            # Preserve completed native work before any small CPU solve can fail.
            array(checkpoint/'data_image.npy',data_image)
            image=data_image+h_tv(direction)+damping*direction;array(checkpoint/'total_image.npy',image)
            old_value=float(gradient.reshape(-1)@span.u+.5*span.u@span.Hu)
            run['status']='conditional_true_box_QP';atomic(a.out/'run.json',run)
            normal=span.append(direction,image);report=span.solve()
            if report['model_change']>old_value+1e-8:raise RuntimeError('Conditional true subspace lost decrease')
            completed+=1;paths.append(str(checkpoint.resolve()))
            run['additional_completed_HVPs']=completed-restored
            norm=report.pop('full_projected_norm')
            report.update(normal,conditional_block_projected_norm=norm,
                conditional_block_relative=norm/max(initial_norm,1e-300),initial_conditional_norm=initial_norm,
                new_HVPs=completed,iteration=index+1,active_coordinates=active,
                conditional_residual_not_full_joint_residual=a.block!='joint',
                local_data_preconditioner_scale=scale,seconds_from_start=time.time()-started,
                true_HVP_job=job['tag'])
            array(checkpoint/'cumulative_log_direction.npy',span.u.reshape(seed.shape))
            array(checkpoint/'cumulative_total_image.npy',span.Hu.reshape(seed.shape))
            atomic(checkpoint/'progress.json',report);atomic(folder/'progress.json',report)
            atomic(folder/'fixed_HVP_manifest.json',{'paths':paths})
            if time.time()-started>=a.max_seconds:break
        u=span.u.reshape(seed.shape);Hu=span.Hu.reshape(seed.shape)
        array(folder/'log_direction.npy',u);array(folder/'cumulative_total_image.npy',Hu)
        report={'block':a.block,'active_coordinates':active,'new_HVPs':completed,
            'conditional_block_relative':span.projected_norm()/max(initial_norm,1e-300),
            'conditional_block_target':tolerance,'conditional_residual_not_full_joint_residual':a.block!='joint',
            'model_change':dot(gradient,u)+.5*dot(u,Hu),'QP_exact_solution_claimed':False,
            'initialization_not_joint_reconstruction':a.block!='joint'}
        atomic(folder/'direction.json',report)
        accepted=False;best=seed.copy();best_rf=baseline;lines=[];slope=dot(gradient,u);base_cost=baseline['loss']+tv
        for line in range(8 if completed else 0):
            freeze();alpha=.5**line;trial=co.material(alpha*u)
            if a.block=='impedance' and (not np.array_equal(trial[0],seed[0]) or not np.array_equal(trial[2],seed[2])):
                raise RuntimeError('Impedance initialization changed an inactive field')
            if a.block=='speed':
                old_Z=seed[0].astype(np.float64)*seed[1];new_Z=trial[0].astype(np.float64)*trial[1]
                if np.max(abs(new_Z-old_Z)/old_Z)>5e-7 or not np.array_equal(trial[2],seed[2]):
                    raise RuntimeError('Speed block changed inactive impedance or attenuation')
            run.update(status='native_material_line_search',line=line,new_HVPs=completed);atomic(a.out/'run.json',run)
            rf,_=pool.evaluate(trial,'forward',f'{a.block}_line{line:02d}')
            candidate_tv=co.regularization(alpha*u,.001)[0];cost=rf['loss']+candidate_tv
            armijo=cost<=base_cost+1e-4*alpha*slope
            accepted=bool(armijo and cost<base_cost-max(1e-12,abs(base_cost)*1e-8))
            lines.append({'alpha':alpha,'native_RF':rf,'TV':candidate_tv,'physical_objective':cost,'accepted':accepted})
            atomic(folder/'line_search.json',{'candidates':lines})
            if accepted:best=trial;best_rf=rf;array(folder/'accepted_log_direction.npy',alpha*u);break
        array(folder/'material.npy',best);array(a.out/'current_material.npy',best)
        atomic(folder/'incumbent_RF.json',best_rf)
        report.update(accepted=accepted,native_data_before=baseline['loss'],native_data_after=best_rf['loss'],
            physical_RMS_change=np.sqrt(np.mean((best.astype(np.float64)-seed)[:,co.mask]**2,axis=1)).tolist(),
            material_SHA256=digest(folder/'material.npy'))
        atomic(folder/'material_update.json',report)
        pool.close();pool=None;run.update(status='post_hoc_maps_only',new_HVPs=completed,accepted=accepted)
        atomic(a.out/'run.json',run)
        with (folder/'maps.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m','nsrno.evaluate_time_material',str(folder/'material.npy'),
                '--sample',a.sample,'--out',str(folder/'maps'),'--estimate-label',a.block+' block GN'],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            run['child_pid']=child.pid;atomic(a.out/'run.json',run);code=child.wait()
        if code:raise RuntimeError('Post-hoc maps failed; inspect maps.log')
        monitor.sample('native_block_GN_stage_finished')
        run.update(status='native_block_GN_stage_finished',new_HVPs=completed,finished_unix_s=time.time(),
            native_data_after=best_rf['loss'],material_SHA256=digest(folder/'material.npy'))
        atomic(a.out/'run.json',run);print(json.dumps(run),flush=True)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time());atomic(a.out/'run.json',run);raise
    finally:
        if pool is not None:pool.close()
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        monitor.log.close()


if __name__=='__main__':main()
