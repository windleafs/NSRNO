"""Conditional low-memory three-field pilot using full-RF vector transport."""
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
from .native_transport_pool import Pool
from .interval_lbfgs import solve
from .native_gn_coordinates import Coordinates,physical_blocks
from .native_gn_json import atomic
from .reconstruct_native_gn import source_manifest
from .pressure_resources import PressureResources


class BudgetBoundary(Exception):pass


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(path,value):
    temporary=path.with_name(path.stem+'.partial.npy');np.save(temporary,value);temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--transport-lambda',type=float,required=True)
    parser.add_argument('--inner-tolerance',type=float,default=1e-6)
    parser.add_argument('--inner-maximum',type=int,default=12000)
    parser.add_argument('--smoothing-fraction',type=float,default=.02)
    parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--maximum-evaluations',type=int,default=24)
    parser.add_argument('--max-seconds',type=float,default=14400)
    parser.add_argument('--check-every',type=int,default=4)
    parser.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--chunk',type=int,default=128);parser.add_argument('--cap-gib',type=float,default=10.)
    args=parser.parse_args();source=args.source.resolve();args.out=args.out.resolve();origin=source/'step001'
    if args.out.exists():raise RuntimeError('Preserve outputs; never restart implicitly')
    if min(args.iterations,args.maximum_evaluations,args.check_every,args.inner_maximum)<1:
        raise ValueError('Positive iteration/evaluation budget required')
    if not args.devices or len(args.devices)!=len(set(args.devices)):
        raise ValueError('One source worker per distinct GPU is required')
    if not 0<args.cap_gib<=14 or not 0<args.inner_tolerance<1 or min(args.transport_lambda,args.max_seconds,args.smoothing_fraction)<=0:
        raise ValueError('Invalid physical scale or resource budget')
    old=json.loads((source/'run.json').read_text());worker_record=json.loads((source/'worker_processes.json').read_text())
    if old['status']!='native_long_GN_finished':raise RuntimeError('Wait for the completed global source')
    if any(pid and Path('/proc',str(pid)).exists() for pid in [old['pid'],old.get('child_pid'),*worker_record['worker_pids']]):
        raise RuntimeError('Source parent or worker still exists; no concurrent GPU experiment')
    for name,expected in old['source_sha256'].items():
        if sha(Path(__file__).parent/name)!=expected:raise RuntimeError('Source reference code changed: '+name)
    update=json.loads((origin/'material_update.json').read_text())
    if sha(origin/'material.npy')!=old['material_SHA256'] or update['material_SHA256']!=old['material_SHA256']:
        raise RuntimeError('Protected physical source material changed')
    seed=np.load(origin/'material.npy');edge=np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    if seed.shape!=(3,961,1121) or seed.dtype!=np.float32:raise ValueError('Full native float32 source required')
    source_RF=json.loads((origin/'incumbent_RF.json').read_text())
    if source_RF['loss']!=update['native_data_after']:raise RuntimeError('Source RF lineage disagrees')
    args.sample=old['sample'];co=Coordinates(seed,edge);zero=np.zeros_like(seed,dtype=np.float64)
    lo,hi=co.bounds(trust=(1.,1.,10.));tv0,_,_,tv_diag=co.regularization(zero,.001)
    reference_run=json.loads((Path(old['reference_source_root'])/'run.json').read_text())
    statistics=Path(reference_run['preconditioner_statistics'])
    if sha(statistics)!='acd29954a613f156e95530c9dd67d413e6f1ea9a1b1492ec51eb58a354bfa961':
        raise RuntimeError('Protected variable scaling statistics changed')
    with np.load(statistics) as data:blocks=physical_blocks(data['blocks'],seed,edge)
    diagonal=np.maximum(reference_run['preconditioner_scale']*blocks[:3]+tv_diag,1e-30)
    metric=np.sqrt(diagonal);metric[:,~co.mask]=1.
    lower=(lo*metric).reshape(-1);upper=(hi*metric).reshape(-1)
    args.out.mkdir();(args.out/'jobs').mkdir();(args.out/'evaluations').mkdir();(args.out/'physical_checkpoints').mkdir()
    final=args.out/'step001';final.mkdir()
    hashes=source_manifest()
    for name in ['reconstruct_native_transport.py','native_transport_worker.py','native_transport_pool.py',
        'rf_transport_objective.py','rf_vector_transport.py','rf_weighted_transport_scaled.py','interval_lbfgs.py','native_gn_json.py']:
        hashes[name]=sha(Path(__file__).parent/name)
    for name in hashes:(args.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    run={'status':'starting_native_transport_pilot','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'created_unix_s':time.time(),
        'source_root':str(source),'source_material_SHA256':old['material_SHA256'],'sample':args.sample,
        'source_sha256':hashes,'devices':args.devices,'allocator_cap_GiB':args.cap_gib,'per_card_limit_GiB':16,
        'solver':'CPU free-set L-BFGS, fresh native waves, certified RF-interval decrease',
        'iteration_budget':args.iterations,'gradient_evaluation_budget':args.maximum_evaluations,
        'walltime_budget_seconds':args.max_seconds,'physical_check_every_accepted_steps':args.check_every,
        'L_BFGS_history_pairs':8,'new_HVPs':0,'old_reference_HVPs_reused':False,
        'unknowns':'independent full-native c/rho/tissue alpha','all_three_fields_free':True,
        'native_shape':list(seed.shape),'spacing_m':50e-6,'full_FOV_depth_mm':[1,43],
        'source_acquisition':'11LF + 11point3','receivers':192,'all_retained_RF':True,
        'transport_lambda':args.transport_lambda,'inner_tolerance':args.inner_tolerance,
        'inner_maximum':args.inner_maximum,'smoothing_fraction_of_weakest_observed_window_RMS':args.smoothing_fraction,
        'source_original_RF_loss':source_RF['loss'],'source_TV':tv0,'TV_weight':.001,
        'variable_scaling_statistics':str(statistics),'old_RF_scaling_is_not_new_exact_curvature':True,
        'gradient_has_certified_nonzero_RF_inner_error':True,'raw_RF_and_new_objective_reported_separately':True,
        'approximate_projected_gradient_is_not_a_convergence_stop':True,
        'truth_read_by_optimizer':False,'reserved_test_case_unread':'Neg_47_Left','joint_imaging_success_claimed':False,
        'child_pid':None,'accepted_steps':0,'attempted_native_gradient_evaluations':0,'completed_native_gradient_evaluations':0}
    atomic(args.out/'run.json',run);atomic(args.out/'source_original_RF.json',source_RF)
    for name in ['initial_material','current_material']:array(args.out/(name+'.npy'),seed)
    array(final/'stage0_initial_material.npy',seed)
    monitor=PressureResources(args.out);pool=None;child=None;started=time.time()
    attempted=completed=accepted=last_mapped=0;accepted_cache={};stop_reason=None

    def freeze():
        for name,expected in hashes.items():
            if sha(Path(__file__).parent/name)!=expected:raise RuntimeError('Frozen source changed: '+name)
        if sha(origin/'material.npy')!=old['material_SHA256']:raise RuntimeError('Source material changed')

    def maps(destination,label):
        nonlocal child
        run.update(status='post_hoc_maps_only',map_folder=str(destination));atomic(args.out/'run.json',run)
        with (destination/'maps.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m','nsrno.evaluate_time_material',str(destination/'material.npy'),
                '--sample',args.sample,'--out',str(destination/'maps'),'--estimate-label',label],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            run['child_pid']=child.pid;atomic(args.out/'run.json',run)
            while child.poll() is None:
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:monitor.sample('native_transport_post_hoc_maps')
            code=child.returncode;pid=child.pid;child=None
        run.update(child_pid=None,last_reaped_child_pid=pid,last_child_exit_code=code);atomic(args.out/'run.json',run)
        if code:raise RuntimeError('Post-hoc material maps failed')

    def checkpoint():
        nonlocal last_mapped
        if accepted==last_mapped:return
        destination=args.out/'physical_checkpoints'/f'accepted{accepted:03d}';destination.mkdir()
        array(destination/'stage0_initial_material.npy',seed);array(destination/'material.npy',accepted_cache['material'])
        array(destination/'log_coordinates.npy',accepted_cache['u'])
        atomic(destination/'material_update.json',{**accepted_cache['record'],'accepted_step':accepted,
            'material_SHA256':sha(destination/'material.npy'),'truth_read':False,'joint_imaging_success_claimed':False})
        maps(destination,f'Native vector transport, {accepted} accepted steps');last_mapped=accepted

    def oracle(x):
        nonlocal attempted,completed,stop_reason
        if completed>=args.maximum_evaluations:stop_reason='native_gradient_evaluation_budget';raise BudgetBoundary()
        if time.time()-started>=args.max_seconds:stop_reason='walltime_at_saved_evaluation_boundary';raise BudgetBoundary()
        if shutil.disk_usage(args.out).free<2*2**30+400*2**20:
            stop_reason='disk_reserve_at_saved_evaluation_boundary';raise BudgetBoundary()
        freeze();attempted+=1;u=x.reshape(seed.shape)/metric;material=co.material(u)
        array(args.out/'latest_trial_material.npy',material);array(args.out/'latest_trial_log_coordinates.npy',u)
        run.update(status='native_transport_RF_and_gradient',attempted_native_gradient_evaluations=attempted,
            completed_native_gradient_evaluations=completed,accepted_steps=accepted)
        atomic(args.out/'run.json',run)
        rf,g=pool.evaluate(material,'gradient',f'transport_eval{attempted:03d}');completed+=1
        tv,g_tv,_,_=co.regularization(u,.001);gradient=((g+g_tv)/metric).reshape(-1)
        if not np.isfinite(gradient).all():raise RuntimeError('Nonfinite material gradient')
        if completed==1:
            if not np.array_equal(material,seed):raise RuntimeError('Zero coordinates changed source material')
            if abs(rf['raw_RF_loss']-source_RF['loss'])>1e-6*max(abs(source_RF['loss']),1e-20):
                raise RuntimeError('Source raw RF no longer reproduces the protected physical evidence')
        record={'evaluation':completed,'native_RF_transport':rf,'TV':tv,'lower':rf['lower']+tv,
            'upper':rf['upper']+tv,'approximate_objective':rf['loss']+tv,'original_RF_loss':rf['raw_RF_loss'],
            'seconds_from_start':time.time()-started,'truth_read':False,
            'fresh_current_material_native_propagation':True,'RF_inner_gradient_error_not_zero':True}
        atomic(args.out/'evaluations'/f'eval{completed:03d}.json',record)
        atomic(args.out/'latest_evaluation.json',record);array(args.out/'latest_trial_scaled_gradient.npy',gradient.reshape(seed.shape))
        run['completed_native_gradient_evaluations']=completed;atomic(args.out/'run.json',run)
        return {'lower':record['lower'],'upper':record['upper'],'gradient':gradient,'record':record,
                'material':material,'u':u}

    def callback(x,value,point):
        nonlocal accepted
        accepted=point['accepted_step'];record={**value['record'],**point}
        accepted_cache.update(material=value['material'].copy(),u=value['u'].copy(),
            gradient=value['gradient'].copy(),record=record)
        array(args.out/'accepted_material.npy',accepted_cache['material'])
        array(args.out/'accepted_log_coordinates.npy',accepted_cache['u'])
        array(args.out/'accepted_scaled_gradient.npy',accepted_cache['gradient'].reshape(seed.shape))
        array(args.out/'current_material.npy',accepted_cache['material'])
        atomic(args.out/'accepted_point.json',record)
        atomic(args.out/'evaluations'/f'accepted{accepted:03d}.json',record)
        run.update(accepted_steps=accepted,approximate_projected_gradient_relative=point['projected_gradient_relative'])
        atomic(args.out/'run.json',run)
        if accepted and accepted%args.check_every==0:checkpoint()

    try:
        pool=Pool(args,monitor)
        try:
            _,_,termination=solve(oracle,zero.reshape(-1),lower,upper,iterations=args.iterations,history=8,
                maximum_lines=8,callback=callback,target=None)
            stop_reason=termination['stop']
        except BudgetBoundary:
            termination={'stop':stop_reason,'last_fully_accepted_step':accepted,
                'latest_trial_is_not_implicitly_selected':True}
        if not accepted_cache:raise RuntimeError('No completed initial evaluation; inspect preserved trial')
        if accepted!=last_mapped:checkpoint()
        returned=accepted_cache['material'];array(final/'material.npy',returned)
        array(args.out/'current_material.npy',returned)
        record=accepted_cache['record'];raw=record['native_RF_transport']
        atomic(final/'incumbent_RF.json',{'loss':raw['raw_RF_loss'],'groups':raw['groups'],'truth_read':False})
        atomic(final/'incumbent_transport.json',raw)
        report={'accepted_steps':accepted,'completed_native_gradient_evaluations':completed,
            'native_data_before':source_RF['loss'],'native_data_after':raw['raw_RF_loss'],
            'transport_lower':record['lower'],'transport_upper':record['upper'],
            'physical_TV_after':record['TV'],'material_SHA256':sha(final/'material.npy'),
            'returned_evaluation':record['evaluation'],'returned_point_was_interval_certified_accepted':True,
            'returned_projected_gradient_relative_is_approximate':record['projected_gradient_relative'],
            'truth_read':False,'joint_imaging_success_claimed':False}
        atomic(final/'material_update.json',report);atomic(args.out/'termination.json',termination)
        pool.close();pool=None;maps(final,'Native vector transport, final accepted material')
        monitor.sample('native_transport_pilot_finished')
        run.update(status='native_transport_pilot_finished',finished_unix_s=time.time(),stop_reason=stop_reason,
            accepted_steps=accepted,completed_native_gradient_evaluations=completed,
            native_data_after=raw['raw_RF_loss'],material_SHA256=report['material_SHA256'])
        atomic(args.out/'run.json',run);print(json.dumps(run),flush=True)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time());atomic(args.out/'run.json',run);raise
    finally:
        if pool is not None:pool.close()
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        monitor.log.close()


if __name__=='__main__':main()
