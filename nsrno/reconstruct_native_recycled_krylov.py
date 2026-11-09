"""True-image recycled Galerkin GN; no approximate-Hessian QP proposals."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import numpy as np
from .native_gn_recycled_state import RecycledState,digest
from .native_gn_coordinates import dot
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def array(path,value):
    temporary=path.with_name(path.stem+'.partial.npy');np.save(temporary,value);temporary.replace(path)


def require_previous_exit():
    root=Path('out/v5/raw_rf_window_balanced_experiment_v1')
    run=json.loads((root/'run.json').read_text());model=Path(run['model'])
    records=[run,*[json.loads((model/p).read_text()) for p in ['run.json','evaluation/run.json','window_evaluation/run.json']]]
    if [r['status'] for r in records]!=['raw_RF_inverse_training_and_evaluations_finished','training_finished',
        'evaluation_finished','window_evaluation_finished']:raise RuntimeError('Finish current neural experiment first')
    for record in records:
        for key in ['pid','child_pid']:
            if record.get(key) and Path('/proc',str(record[key])).exists():raise RuntimeError('Existing neural process has not exited')
    return run


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('out/v5/reconstruction_native_gn_enriched_v1_train_200'))
    p.add_argument('--prepared',type=Path,default=Path('out/v5/native_recycled_krylov_preparation_v1'))
    p.add_argument('--incumbent',type=Path,default=Path('out/v5/reconstruction_native_gn_nonlinear_finish_v1_train_200'))
    p.add_argument('--out',type=Path,default=Path('out/v5/reconstruction_native_recycled_krylov_v1_train_200'))
    p.add_argument('--extra-hvps',type=int,default=8)
    p.add_argument('--max-seconds',type=float,default=7200)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    a=p.parse_args();previous=require_previous_exit()
    if a.out.exists():raise RuntimeError('Preserve output; never restart after an observation timeout')
    if a.extra_hvps<1 or a.max_seconds<=0 or a.cap_gib>14:raise ValueError('Invalid fixed budget')
    preparation=json.loads((a.prepared/'run.json').read_text())
    if preparation['status']!='recycled_direction_CPU_prepared' or Path('/proc',str(preparation['pid'])).exists():
        raise RuntimeError('CPU direction preparation must have finished and exited')
    for name,expected in preparation['source_sha256'].items():
        if digest(Path(__file__).parent/name)!=expected:raise RuntimeError('Prepared algebra source changed: '+name)
    incumbent_run=json.loads((a.incumbent/'run.json').read_text())
    if incumbent_run['status']!='saved_native_checkpoint_finished' or Path('/proc',str(incumbent_run['pid'])).exists():
        raise RuntimeError('Require completed physical incumbent')
    for source in [a.source,a.incumbent]:
        old=json.loads((source/'run.json').read_text());pids=[old['pid'],old.get('child_pid')]
        workers=source/'worker_processes.json'
        if workers.exists():pids+=json.loads(workers.read_text()).get('worker_pids',[])
        if any(pid and Path('/proc',str(pid)).exists() for pid in pids):
            raise RuntimeError('Old native source still has a live process')
    incumbent_folder=a.incumbent/'step001';incumbent_path=incumbent_folder/'material.npy'
    if digest(incumbent_path)!='18a88a238709b33bf29a802bc8469deeafa45819241f07ec129bea15879f415b':
        raise RuntimeError('Accepted physical incumbent changed')
    incumbent_update=json.loads((incumbent_folder/'material_update.json').read_text())
    incumbent_rf=next(r['native_RF'] for r in reversed(json.loads((incumbent_folder/'line_search.json').read_text())['candidates']) if r['accepted'])
    if incumbent_rf['loss']!=incumbent_update['native_data_after']:raise RuntimeError('Incumbent RF record disagrees')
    # Importing native workers allocates no GPU; Pool is created after restore.
    from .reconstruct_native_gn import Pool,source_manifest
    hashes=source_manifest()
    old_native=json.loads((a.source/'run.json').read_text())
    for name,expected in old_native['source_sha256'].items():
        if name in hashes and hashes[name]!=expected:
            raise RuntimeError('Native operator differs from saved true images: '+name)
    for name in ['reconstruct_native_recycled_krylov.py','native_gn_recycled_state.py','native_gn_galerkin.py',
        'native_span_qp.py','native_gn_secant.py','native_gn_json.py','evaluate_time_material.py','v5_materials.py']:
        hashes[name]=digest(Path(__file__).parent/name)
    a.out.mkdir();(a.out/'jobs').mkdir();folder=a.out/'step001';folder.mkdir()
    for name in hashes:(a.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    run={'status':'restoring_true_native_images','pid':os.getpid(),'created_unix_s':time.time(),
        'solver_revision':'fully_corrected_true_Galerkin_deflated_residual',
        'source_root':str(a.source.resolve()),'prepared_direction':str(a.prepared.resolve()),
        'accepted_incumbent_material':str(incumbent_path.resolve()),'incumbent_RF_loss':incumbent_rf['loss'],
        'source_sha256':hashes,'devices':a.devices,'allocator_cap_GiB':a.cap_gib,'model_per_card_limit_GiB':16,
        'new_gradient_wave_solves':0,'reused_complete_HVPs':15,'extra_HVP_budget':a.extra_hvps,
        'walltime_budget_seconds':a.max_seconds,'full_space_optimality_claimed':False,
        'walltime_stop_at_completed_HVP_boundary':True,
        'unknowns':'independent full native c/rho/alpha_tissue; no permanent field subspace restriction',
        'pixelwise_search_direction_magnitude_clipping':False,'physical_boxes_inside_coefficient_QP':True,
        'truth_read_by_optimizer':False,'reserved_test_case_unread':'Neg_47_Left','joint_imaging_success_claimed':False,
        'previous_neural_experiment':str(Path(previous['model']).resolve()),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    atomic(a.out/'run.json',run);monitor=PressureResources(a.out);pool=None;child=None;completed=0;started=time.time()
    def freeze():
        for name,expected in hashes.items():
            if digest(Path(__file__).parent/name)!=expected:raise RuntimeError('Frozen source changed: '+name)
    try:
        state=RecycledState(a.source,callback=lambda item:atomic(folder/'progress.json',item));a.sample=state.sample
        run.update(sample=a.sample,fixed_reference_material=str(state.origin/'old_material.npy'),
            fixed_reference_SHA256=digest(state.origin/'old_material.npy'),fixed_reference_RF_loss=state.baseline_rf['loss'])
        atomic(a.out/'run.json',run);atomic(a.out/'fixed_model_reference.json',state.reference)
        incumbent=np.load(incumbent_path);incumbent_u=state.coordinate(incumbent)
        incumbent_tv=state.co.regularization(incumbent_u,state.reference['tv_weight'])[0]
        incumbent_cost=incumbent_rf['loss']+incumbent_tv
        array(folder/'stage0_initial_material.npy',incumbent);array(folder/'material.npy',incumbent)
        atomic(folder/'incumbent_RF.json',incumbent_rf)
        paths=list(state.paths);atomic(folder/'fixed_HVP_manifest.json',{'paths':[str(v) for v in paths]})
        atomic(folder/'restored.json',{**state.initial_report,'full_projected_KKT_relative':state.span.projected_norm()/state.initial_norm,
            'current_incumbent_RF_loss':incumbent_rf['loss'],'fixed_reference_RF_loss':state.baseline_rf['loss']})
        for extra in range(a.extra_hvps):
            if state.span.projected_norm()<=1e-3*state.initial_norm:break
            freeze();direction,proposal=state.span.next_direction();direction=direction.astype(np.float32).astype(np.float64)
            if extra==0:
                prepared=np.load(a.prepared/'next_direction.npy')
                if not np.array_equal(direction,prepared.astype(np.float64)):raise RuntimeError('First true residual direction differs from CPU preparation')
            index=len(paths);checkpoint=folder/f'hvp{index:03d}';checkpoint.mkdir()
            array(checkpoint/'direction.npy',direction);atomic(checkpoint/'proposal.json',proposal)
            run.update(status='native_HVP',fixed_HVP_count=len(paths),new_HVPs=completed,next_HVP=index)
            atomic(a.out/'run.json',run)
            if pool is None:pool=Pool(a,monitor)
            job,data_image=pool.evaluate(state.material,'hvp',f'deflated_hvp{index:03d}',direction)
            old_value=float(state.gradient.reshape(-1)@state.span.u+.5*state.span.u@state.span.Hu)
            run.update(status='true_subspace_box_QP');atomic(a.out/'run.json',run)
            total_image,report=state.append(direction,data_image)
            if report['model_change']>old_value+1e-8:raise RuntimeError('Enlarged true-image span lost quadratic decrease')
            completed+=1;paths.append(checkpoint.resolve())
            for name,value in [('data_image',data_image),('total_image',total_image),
                ('cumulative_log_direction',state.current_u()),('cumulative_total_image',state.current_Hu())]:array(checkpoint/(name+'.npy'),value)
            report.update(iteration=index+1,new_HVPs=completed,local_data_preconditioner_scale=state.scale,
                true_HVP_job=job['tag'],seconds_from_start=time.time()-started)
            atomic(checkpoint/'progress.json',report);atomic(folder/'progress.json',report)
            atomic(folder/'fixed_HVP_manifest.json',{'paths':[str(v) for v in paths]})
            if time.time()-started>=a.max_seconds:break
        u=state.current_u();Hu=state.current_Hu();array(folder/'log_direction.npy',u);array(folder/'cumulative_total_image.npy',Hu)
        direction_report={'method':run['solver_revision'],'reused_HVPs':15,'new_HVPs':completed,
            'model_change':dot(state.gradient,u)+.5*dot(u,Hu),'true_QP_KKT_relative':state.span.projected_norm()/state.initial_norm,
            'QP_exact_solution_claimed':False,'true_QP_tolerance_met':bool(state.span.projected_norm()<=1e-3*state.initial_norm),
            'total_directions':len(paths),'truth_read':False}
        atomic(folder/'direction.json',direction_report)
        accepted=False;lines=[];best=incumbent.copy();best_rf=incumbent_rf;slope=dot(state.gradient,u)
        base_cost=state.baseline_rf['loss']+state.tv
        for line in range(8 if completed else 0):
            freeze();alpha=.5**line;trial=state.co.material(alpha*u)
            run.update(status='native_material_line_search',line=line);atomic(a.out/'run.json',run)
            rf,_=pool.evaluate(trial,'forward',f'galerkin_line{line:02d}')
            tv=state.co.regularization(alpha*u,state.reference['tv_weight'])[0];cost=rf['loss']+tv
            armijo=cost<=base_cost+1e-4*alpha*slope;improves=cost<incumbent_cost-max(1e-12,abs(incumbent_cost)*1e-8)
            accepted=bool(armijo and improves)
            row={'alpha':alpha,'native_RF':rf,'TV':tv,'physical_objective':cost,
                'fixed_reference_Armijo':bool(armijo),'improves_current_incumbent':bool(improves),'accepted':accepted}
            lines.append(row);atomic(folder/'line_search.json',{'candidates':lines})
            if accepted:best=trial;best_rf=rf;array(folder/'accepted_log_direction.npy',alpha*u);break
        array(folder/'material.npy',best);array(a.out/'current_material.npy',best)
        atomic(folder/'material_update.json',{**direction_report,'accepted':accepted,
            'native_data_before':incumbent_rf['loss'],'native_data_after':best_rf['loss'],
            'physical_RMS_change_from_incumbent':np.sqrt(np.mean((best.astype(np.float64)-incumbent)[:,state.co.mask]**2,axis=1)).tolist()})
        atomic(folder/'incumbent_RF.json',best_rf)
        if pool is not None:pool.close();pool=None
        run.update(status='post_hoc_maps_only',new_HVPs=completed,accepted=accepted);atomic(a.out/'run.json',run)
        with (folder/'maps.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m','nsrno.evaluate_time_material',str(folder/'material.npy'),
                '--sample',a.sample,'--out',str(folder/'maps'),'--estimate-label','True recycled Krylov GN'],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            run.update(child_pid=child.pid);atomic(a.out/'run.json',run)
            code=child.wait()
        if code:raise RuntimeError('CPU maps failed; inspect preserved maps.log')
        monitor.sample('native_recycled_Krylov_finished')
        run.update(status='planned_native_recycled_GN_steps_finished',finished_unix_s=time.time())
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
