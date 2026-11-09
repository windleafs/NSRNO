"""Long full-field nonlinear continuation using fresh native gradients on CPU L-BFGS."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
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
import scipy
from scipy.optimize import Bounds,minimize
from .reconstruct_native_gn import Pool,source_manifest
from .native_gn_coordinates import Coordinates,physical_blocks
from .native_gn_json import atomic
from .pressure_resources import PressureResources


class BudgetBoundary(Exception):
    pass


class ProjectedTarget(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(path,value):
    temporary=path.with_name(path.stem+'.partial.npy')
    np.save(temporary,value);temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=32)
    p.add_argument('--maximum-evaluations',type=int,default=64)
    p.add_argument('--max-seconds',type=float,default=18000)
    p.add_argument('--check-every',type=int,default=4)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    a=p.parse_args();a.out=a.out.resolve();source=a.source.resolve();origin=source/'step001'
    if a.out.exists():raise RuntimeError('Preserve output; no implicit restart')
    if min(a.iterations,a.maximum_evaluations,a.check_every)<1 or a.max_seconds<=0 or not 0<a.cap_gib<=14:
        raise ValueError('Invalid resource budget')
    old=json.loads((source/'run.json').read_text())
    if old['status']!='native_long_GN_finished':raise RuntimeError('Wait for the completed long GN source')
    w=json.loads((source/'worker_processes.json').read_text())
    if any(pid and Path('/proc',str(pid)).exists() for pid in [old['pid'],old.get('child_pid'),*w['worker_pids']]):
        raise RuntimeError('Source parent or child still exists')
    for name,expected in old['source_sha256'].items():
        if sha(Path(__file__).parent/name)!=expected:raise RuntimeError('Source changed: '+name)
    update=json.loads((origin/'material_update.json').read_text())
    if sha(origin/'material.npy')!=old['material_SHA256'] or update['material_SHA256']!=old['material_SHA256']:
        raise RuntimeError('Source physical material changed')
    seed=np.load(origin/'material.npy');edge=np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    if seed.shape!=(3,961,1121) or seed.dtype!=np.float32:raise ValueError('Full native float32 material required')
    baseline_rf=json.loads((origin/'incumbent_RF.json').read_text())
    if baseline_rf['loss']!=update['native_data_after']:raise RuntimeError('Source physical RF evidence disagrees')
    a.sample=old['sample'];co=Coordinates(seed,edge);zero=np.zeros_like(seed,dtype=np.float64)
    low,high=co.bounds(trust=(1.,1.,10.));tv0,_,_,tv_diagonal=co.regularization(zero,.001)
    reference_run=json.loads((Path(old['reference_source_root'])/'run.json').read_text())
    preconditioner=Path(reference_run['preconditioner_statistics'])
    if sha(preconditioner)!='acd29954a613f156e95530c9dd67d413e6f1ea9a1b1492ec51eb58a354bfa961':
        raise RuntimeError('Fixed variable scaling statistics changed')
    with np.load(preconditioner) as data:blocks=physical_blocks(data['blocks'],seed,edge)
    diagonal=np.maximum(reference_run['preconditioner_scale']*blocks[:3]+tv_diagonal,1e-30)
    metric=np.sqrt(diagonal);metric[:,~co.mask]=1
    lower=(low*metric).reshape(-1);upper=(high*metric).reshape(-1)
    a.out.mkdir();(a.out/'jobs').mkdir();(a.out/'evaluations').mkdir();(a.out/'physical_checkpoints').mkdir()
    final=a.out/'step001';final.mkdir()
    hashes=source_manifest()
    for name in ['reconstruct_native_long_lbfgs.py','native_gn_json.py']:
        hashes[name]=sha(Path(__file__).parent/name)
    for name in hashes:(a.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    run={'status':'starting_long_nonlinear_continuation','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'created_unix_s':time.time(),
        'source_root':str(source),'source_material_SHA256':old['material_SHA256'],'sample':a.sample,
        'solver':'bounded CPU L-BFGS with fresh full native RF and gradient at every trial',
        'scipy_version':scipy.__version__,'source_sha256':hashes,'devices':a.devices,
        'allocator_cap_GiB':a.cap_gib,'model_per_card_limit_GiB':16,
        'iteration_budget':a.iterations,'gradient_evaluation_budget':a.maximum_evaluations,
        'walltime_budget_seconds':a.max_seconds,'physical_check_every_accepted_steps':a.check_every,
        'L_BFGS_history_pairs':8,'old_reference_exact_HVPs_reused':False,'new_HVPs':0,
        'variable_scaling_statistics':str(preconditioner),'fixed_scaling':True,
        'unknowns':'independent full-native c/rho/tissue alpha','all_three_fields_free':True,
        'native_spacing_m':50e-6,'shape':list(seed.shape),'full_FOV_depth_mm':[1,43],
        'shots':'11LF + 11point3','receivers':192,'all_retained_RF':True,
        'source_physical_RF_loss':baseline_rf['loss'],'source_physical_TV':tv0,
        'surface_network_initialization_used':False,'truth_read_by_optimizer':False,
        'reserved_test_case_unread':'Neg_47_Left','joint_imaging_success_claimed':False,
        'full_projected_gradient_relative_target':.001,'child_pid':None,
        'accepted_steps':0,'native_gradient_evaluations':0}
    atomic(a.out/'run.json',run)
    for name in ['initial_material','current_material','best_material']:array(a.out/(name+'.npy'),seed)
    array(final/'stage0_initial_material.npy',seed)
    atomic(a.out/'source_RF.json',baseline_rf)
    monitor=PressureResources(a.out);pool=None;child=None;started=time.time()
    count=0;accepted=0;last_mapped=0;initial_norm=None;cache={};accepted_cache={};best={'objective':baseline_rf['loss']+tv0,
        'native_RF':baseline_rf,'TV':tv0,'evaluation':0,'optimizer_acceptance_confirmed':True}
    stop_reason=None
    atomic(a.out/'best.json',best)

    def freeze():
        for name,expected in hashes.items():
            if sha(Path(__file__).parent/name)!=expected:raise RuntimeError('Frozen source changed: '+name)
        if sha(origin/'material.npy')!=old['material_SHA256']:raise RuntimeError('Source material changed')

    def maps(destination,label):
        nonlocal child
        run.update(status='post_hoc_maps_only',map_folder=str(destination));atomic(a.out/'run.json',run)
        with (destination/'maps.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m','nsrno.evaluate_time_material',str(destination/'material.npy'),
                '--sample',a.sample,'--out',str(destination/'maps'),'--estimate-label',label],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            run['child_pid']=child.pid;atomic(a.out/'run.json',run)
            while child.poll() is None:
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:monitor.sample('long_LBFGS_post_hoc_maps')
            code=child.returncode;pid=child.pid;child=None
        run.update(child_pid=None,last_reaped_child_pid=pid,last_child_exit_code=code);atomic(a.out/'run.json',run)
        if code:raise RuntimeError('Post-hoc map process failed')

    def physical_check():
        nonlocal last_mapped
        if accepted==last_mapped:return
        destination=a.out/'physical_checkpoints'/f'accepted{accepted:03d}';destination.mkdir()
        array(destination/'material.npy',accepted_cache['material']);array(destination/'stage0_initial_material.npy',seed)
        array(destination/'log_coordinates.npy',accepted_cache['u'])
        array(destination/'scaled_gradient.npy',accepted_cache['gradient'].reshape(seed.shape))
        atomic(destination/'material_update.json',{**accepted_cache['record'],'accepted_step':accepted,
            'native_data_before':baseline_rf['loss'],'native_data_after':accepted_cache['record']['native_RF']['loss'],
            'material_SHA256':sha(destination/'material.npy'),'truth_read':False})
        maps(destination,f'Long nonlinear L-BFGS: {accepted} accepted steps');last_mapped=accepted

    def evaluate(x):
        nonlocal count,initial_norm,stop_reason
        if count>=a.maximum_evaluations:stop_reason='native_gradient_budget';raise BudgetBoundary()
        if time.time()-started>=a.max_seconds:stop_reason='walltime_budget_at_saved_evaluation_boundary';raise BudgetBoundary()
        if shutil.disk_usage(a.out).free<2*2**30+400*2**20:
            stop_reason='disk_reserve_at_saved_evaluation_boundary';raise BudgetBoundary()
        freeze();count+=1
        u=x.reshape(seed.shape)/metric;material=co.material(u)
        run.update(status='native_RF_and_gradient_at_trial',native_gradient_evaluations=count,accepted_steps=accepted)
        atomic(a.out/'run.json',run)
        rf,g=pool.evaluate(material,'gradient',f'nonlinear_eval{count:03d}')
        tv,g_tv,_,_=co.regularization(u,.001);objective=rf['loss']+tv
        gradient=((g+g_tv)/metric).reshape(-1)
        if not np.isfinite(objective) or not np.isfinite(gradient).all():raise RuntimeError('Nonfinite objective or gradient')
        norm=float(np.linalg.norm(x-np.clip(x-gradient,lower,upper)))
        if initial_norm is None:
            initial_norm=norm
            if abs(rf['loss']-baseline_rf['loss'])>1e-7*max(abs(baseline_rf['loss']),1e-20):
                raise RuntimeError('Fresh source RF does not reproduce its protected physical evidence')
        record={'evaluation':count,'native_RF':rf,'TV':tv,'objective':objective,
            'full_projected_gradient_norm':norm,'initial_projected_gradient_norm':initial_norm,
            'full_projected_gradient_relative':norm/max(initial_norm,1e-300),
            'seconds_from_start':time.time()-started,'truth_read':False,'current_material_native_gradient':True}
        atomic(a.out/'evaluations'/f'eval{count:03d}.json',record)
        array(a.out/'latest_trial_material.npy',material);array(a.out/'latest_trial_log_coordinates.npy',u)
        array(a.out/'latest_trial_scaled_gradient.npy',gradient.reshape(seed.shape))
        cache.update(x=x.copy(),material=material,u=u,gradient=gradient,record=record)
        if objective<best['objective']:
            best.update(objective=objective,native_RF=rf,TV=tv,evaluation=count,optimizer_acceptance_confirmed=False,
                full_projected_gradient_relative=record['full_projected_gradient_relative'],
                full_projected_gradient_norm=norm,gradient_evaluation=count)
            array(a.out/'best_material.npy',material);array(a.out/'best_log_coordinates.npy',u)
            array(a.out/'best_scaled_gradient.npy',gradient.reshape(seed.shape))
            atomic(a.out/'best.json',best);array(a.out/'current_material.npy',material)
        if count==1:
            if not np.array_equal(material,seed):raise RuntimeError('Zero coordinates changed source material')
            if best['evaluation']==0:
                best.update(full_projected_gradient_relative=1.,full_projected_gradient_norm=norm,gradient_evaluation=1)
                atomic(a.out/'best.json',best)
            array(a.out/'initial_data_gradient.npy',g);atomic(a.out/'initial_evaluation.json',record)
        atomic(a.out/'latest_evaluation.json',record)
        return objective,gradient

    def callback(x):
        nonlocal accepted,stop_reason
        if not np.array_equal(cache['x'],x):raise RuntimeError('Accepted point differs from last evaluated point')
        accepted+=1;record={**cache['record'],'accepted_step':accepted}
        accepted_cache.update(material=cache['material'].copy(),u=cache['u'].copy(),
            gradient=cache['gradient'].copy(),record=record)
        atomic(a.out/'evaluations'/f'accepted{accepted:03d}.json',record)
        array(a.out/'accepted_log_coordinates.npy',cache['u'])
        array(a.out/'accepted_scaled_gradient.npy',cache['gradient'].reshape(seed.shape))
        array(a.out/'accepted_material.npy',cache['material'])
        if best['evaluation']==count:
            best['optimizer_acceptance_confirmed']=True;atomic(a.out/'best.json',best)
        run.update(accepted_steps=accepted,native_gradient_evaluations=count,
            full_projected_gradient_relative=record['full_projected_gradient_relative'])
        atomic(a.out/'run.json',run)
        if accepted%a.check_every==0:physical_check()
        if record['full_projected_gradient_relative']<=.001:
            stop_reason='full_nonlinear_projected_gradient_target';raise ProjectedTarget()

    try:
        pool=Pool(a,monitor);termination=None
        try:
            result=minimize(evaluate,zero.reshape(-1),method='L-BFGS-B',jac=True,
                bounds=Bounds(lower,upper),callback=callback,
                options={'maxiter':a.iterations,'maxfun':a.maximum_evaluations,'maxcor':8,
                    'maxls':8,'ftol':1e-13,'gtol':1e-14})
            termination={'message':str(result.message),'software_success':bool(result.success),
                'iterations':int(result.nit),'evaluations':int(result.nfev)}
            stop_reason='optimizer_returned'
        except (BudgetBoundary,ProjectedTarget):
            termination={'message':stop_reason,'software_success':False,'iterations':accepted,'evaluations':count}
        if accepted!=last_mapped:physical_check()
        returned=np.load(a.out/'best_material.npy');array(final/'material.npy',returned)
        array(a.out/'current_material.npy',returned)
        atomic(final/'incumbent_RF.json',best['native_RF'])
        change=returned.astype(np.float64)-seed
        z=(np.arange(961)-60)*.05;x=(np.arange(1121)-560)*.05
        roi=(abs(x)[None,:]<=19.1)&(z[:,None]>=1)&(z[:,None]<=43)
        report={'accepted':best['evaluation']>0,'native_data_before':baseline_rf['loss'],
            'native_data_after':best['native_RF']['loss'],'physical_TV_before':tv0,'physical_TV_after':best['TV'],
            'returned_evaluation':best['evaluation'],'returned_point_LBFGS_accepted':best['optimizer_acceptance_confirmed'],
            'returned_full_projected_gradient_relative':best.get('full_projected_gradient_relative'),
            'returned_point_gradient_evaluation':best.get('gradient_evaluation'),
            'material_SHA256':sha(final/'material.npy'),'accepted_steps':accepted,'native_gradient_evaluations':count,
            'physical_RMS_change':np.sqrt(np.mean(change[:,co.mask]**2,axis=1)).tolist(),
            'FOV_physical_RMS_change':np.sqrt(np.mean(change[:,roi]**2,axis=1)).tolist(),
            'truth_read':False,'joint_imaging_success_claimed':False}
        atomic(final/'material_update.json',report)
        if pool is not None:pool.close();pool=None
        maps(final,'Long nonlinear L-BFGS, lowest true evaluated objective')
        monitor.sample('native_long_LBFGS_finished')
        run.update(status='native_long_LBFGS_finished',finished_unix_s=time.time(),stop_reason=stop_reason,
            termination=termination,accepted_steps=accepted,native_gradient_evaluations=count,
            material_SHA256=report['material_SHA256'],native_data_after=report['native_data_after'],
            returned_evaluation=best['evaluation'],returned_point_LBFGS_accepted=best['optimizer_acceptance_confirmed'])
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


if __name__=='__main__':
    main()
