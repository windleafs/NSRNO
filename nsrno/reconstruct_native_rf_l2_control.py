"""Matched-baseline four-update original-RF control after spatial metric correction."""
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
from .native_rf_l2_pool import Pool
from .interval_lbfgs_adaptive import solve
from .native_gn_coordinates import Coordinates
from .native_gn_json import atomic
from .reconstruct_native_gn import source_manifest
from .pressure_resources import PressureResources


class BudgetBoundary(Exception):pass


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array(path,value):
    temporary=path.with_name(path.stem+'.partial.npy')
    np.save(temporary,value);temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=4)
    parser.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--chunk',type=int,default=128)
    parser.add_argument('--cap-gib',type=float,default=10.)
    args=parser.parse_args()
    source=args.source.resolve();comparison=args.comparison.resolve();args.out=args.out.resolve()
    if args.out.exists():raise RuntimeError('Preserve existing outputs; never restart implicitly')
    if not 1<=args.iterations<=4 or len(args.devices)!=len(set(args.devices)) or not args.devices:
        raise ValueError('At most four updates and distinct source worker GPUs required')
    if not 0<args.cap_gib<=14:raise ValueError('Allocator budget must remain below total per-card limit')
    previous=read(comparison/'run.json');previous_workers=read(comparison/'worker_processes.json')
    if previous['status']!='native_transport_pilot_finished' or previous['accepted_steps']!=4:
        raise RuntimeError('Wait for the actual completed four-update constant-metric comparison')
    if previous.get('metric_kind')!='spatially_constant_diagonal_in_log_c_log_Z_log_tissue_alpha':
        raise RuntimeError('Expected the completed spatially constant metric experiment')
    if any(pid and Path('/proc',str(pid)).exists() for pid in
           [previous['pid'],previous.get('child_pid'),*previous_workers['worker_pids']]):
        raise RuntimeError('The comparison parent, workers or map child still exists')
    if previous['devices']!=args.devices or previous['allocator_cap_GiB']!=args.cap_gib:
        raise RuntimeError('Use the same source streaming and GPU memory limit')
    if Path(previous['source_root']).resolve()!=source:
        raise RuntimeError('Use exactly the same physical starting material')
    for name,expected in previous['source_sha256'].items():
        if sha(comparison/name)!=expected or sha(Path(__file__).parent/name)!=expected:
            raise RuntimeError('Comparison dependency changed: '+name)
    deadline=previous['original_deadline_unix_s']
    maximum=previous['gradient_evaluation_budget']
    if time.time()>=deadline or previous['attempted_native_gradient_evaluations']>=maximum:
        raise RuntimeError('The original shared resource budget is exhausted')
    origin=source/'step001';source_run=read(source/'run.json')
    if source_run['status']!='native_long_GN_finished':raise RuntimeError('Physical baseline is not completed')
    if sha(origin/'material.npy')!=previous['source_material_SHA256']:
        raise RuntimeError('Protected matched baseline changed')
    seed=np.load(origin/'material.npy')
    if seed.shape!=(3,961,1121) or seed.dtype!=np.float32:
        raise ValueError('Full native float32 three-field baseline required')
    edge=np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    co=Coordinates(seed,edge);zero=np.zeros_like(seed,dtype=np.float64)
    lo,hi=co.bounds(trust=(1.,1.,10.))
    constants=np.asarray(previous['constant_diagonal_values'],np.float64)
    if constants.shape!=(3,) or not np.isfinite(constants).all() or np.any(constants<=0):
        raise ValueError('Identical positive constant metric required')
    metric=np.broadcast_to(np.sqrt(constants)[:,None,None],seed.shape).copy();metric[:,~co.mask]=1.
    lower=(lo*metric).reshape(-1);upper=(hi*metric).reshape(-1)
    source_RF=read(origin/'incumbent_RF.json')
    args.sample=previous['sample']
    args.out.mkdir();(args.out/'jobs').mkdir();(args.out/'evaluations').mkdir()
    (args.out/'physical_checkpoints').mkdir();final=args.out/'step001';final.mkdir()
    hashes=source_manifest()
    for name in ['reconstruct_native_rf_l2_control.py','native_rf_l2_worker.py','native_rf_l2_pool.py',
                 'interval_lbfgs_adaptive.py','native_gn_json.py']:
        hashes[name]=sha(Path(__file__).parent/name)
    for name in hashes:(args.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    attempted=previous['attempted_native_gradient_evaluations']
    completed=previous['completed_native_gradient_evaluations']
    previous_completed=completed;accepted=last_mapped=0
    started=previous['original_created_unix_s']
    run={'status':'starting_original_RF_L2_control','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'created_unix_s':time.time(),
        'sample':args.sample,'source_root':str(source),'source_material_SHA256':sha(origin/'material.npy'),
        'completed_comparison_root':str(comparison),'budget_predecessor_root':str(comparison),
        'comparison_final_material_is_not_the_control_start':True,
        'original_created_unix_s':started,'original_deadline_unix_s':deadline,
        'walltime_budget_seconds':previous['walltime_budget_seconds'],'gradient_evaluation_budget':maximum,
        'original_time_and_evaluation_budgets_not_reset':True,
        'previous_attempted_evaluations':attempted,'previous_completed_evaluations':completed,
        'attempted_native_gradient_evaluations':attempted,'completed_native_gradient_evaluations':completed,
        'accepted_steps':0,'iteration_budget':args.iterations,'physical_check_every_accepted_steps':4,
        'devices':args.devices,'allocator_cap_GiB':args.cap_gib,'per_card_limit_GiB':16,
        'native_shape':list(seed.shape),'spacing_m':50e-6,'full_FOV_depth_mm':[1,43],
        'source_acquisition':'11LF + 11point3','receivers':192,'all_retained_RF':True,
        'unknowns':'independent full-native c/rho/tissue alpha','all_three_fields_free':True,
        'metric_kind':previous['metric_kind'],'constant_diagonal_values':constants.tolist(),
        'all_exterior_material_cells_still_free':True,
        'objective_kind':'original_geometry_normalized_RF_L2_plus_TV',
        'TV_weight':.001,'source_original_RF_loss':source_RF['loss'],
        'solver':'same adaptive CPU free-set L-BFGS; original RF loss and its native gradient',
        'L_BFGS_history_pairs':8,'no_RF_inner_optimization':True,
        'data_loss_values_are_floating_point_not_interval_error_certificates':True,
        'baseline_gradient_must_be_new_for_changed_RF_objective':True,
        'initial_raw_RF_gradient_actually_completed':False,'transport_gradient_not_reused_as_RF_gradient':True,
        'RF_archive_policy':'Every native evaluation; workers/rankXX/RF/tokenNNNN',
        'source_sha256':hashes,'truth_read_by_optimizer':False,'reserved_test_case_unread':'Neg_47_Left',
        'joint_imaging_success_claimed':False,'child_pid':None,
        'projected_gradient_is_diagnostic_not_completion':True}
    atomic(args.out/'run.json',run);atomic(args.out/'source_original_RF.json',source_RF)
    for name in ['initial_material','current_material']:array(args.out/(name+'.npy'),seed)
    array(final/'stage0_initial_material.npy',seed)
    monitor=PressureResources(args.out);pool=None;child=None;cache={};stop_reason=None

    def freeze():
        for name,expected in hashes.items():
            if sha(Path(__file__).parent/name)!=expected:raise RuntimeError('Frozen source changed: '+name)
        if sha(origin/'material.npy')!=run['source_material_SHA256']:
            raise RuntimeError('The protected baseline changed')

    def maps(folder,label):
        nonlocal child
        run.update(status='post_hoc_maps_only',map_folder=str(folder));atomic(args.out/'run.json',run)
        with (folder/'maps.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m','nsrno.evaluate_time_material',str(folder/'material.npy'),
                '--sample',args.sample,'--out',str(folder/'maps'),'--estimate-label',label],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            run['child_pid']=child.pid;atomic(args.out/'run.json',run)
            while child.poll() is None:
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:monitor.sample('original_RF_post_hoc_maps')
            code=child.returncode;pid=child.pid;child=None
        run.update(child_pid=None,last_reaped_child_pid=pid,last_child_exit_code=code)
        atomic(args.out/'run.json',run)
        if code:raise RuntimeError('Actual material maps failed')

    def checkpoint():
        nonlocal last_mapped
        if accepted==last_mapped:return
        folder=args.out/'physical_checkpoints'/f'accepted{accepted:03d}';folder.mkdir()
        array(folder/'stage0_initial_material.npy',seed);array(folder/'material.npy',cache['material'])
        array(folder/'log_coordinates.npy',cache['u'])
        atomic(folder/'material_update.json',{**cache['record'],'accepted_step':accepted,
            'material_SHA256':sha(folder/'material.npy'),'truth_read':False})
        maps(folder,f'Native original RF L2, {accepted} accepted steps');last_mapped=accepted

    def oracle(x):
        nonlocal attempted,completed,stop_reason
        if attempted>=maximum:stop_reason='original_total_attempt_budget';raise BudgetBoundary()
        if time.time()>=deadline:stop_reason='original_deadline_at_saved_evaluation_boundary';raise BudgetBoundary()
        if shutil.disk_usage(args.out).free<2*2**30+400*2**20:
            stop_reason='disk_reserve_at_saved_evaluation_boundary';raise BudgetBoundary()
        freeze();attempted+=1;u=x.reshape(seed.shape)/metric;material=co.material(u)
        array(args.out/'latest_trial_material.npy',material);array(args.out/'latest_trial_log_coordinates.npy',u)
        run.update(status='native_RF_L2_and_gradient',attempted_native_gradient_evaluations=attempted,
                   completed_native_gradient_evaluations=completed,accepted_steps=accepted)
        atomic(args.out/'run.json',run)
        rf,g=pool.evaluate(material,'gradient',f'original_RF_eval{attempted:03d}');completed+=1
        tv,gtv,_,_=co.regularization(u,.001);gradient=((g+gtv)/metric).reshape(-1)
        if not np.isfinite(gradient).all():raise RuntimeError('Nonfinite original RF material gradient')
        if completed==previous_completed+1:
            if not np.array_equal(material,seed):raise RuntimeError('Initial oracle did not use the matched seed')
            if abs(rf['loss']-source_RF['loss'])>1e-6*max(abs(source_RF['loss']),1e-20):
                raise RuntimeError('Necessary initial RF evaluation no longer matches the protected physical baseline')
            run['initial_raw_RF_gradient_actually_completed']=True
        value=rf['loss']+tv
        record={'evaluation':completed,'attempted_evaluation':attempted,'native_RF_quadratic':rf,
            'original_RF_loss':rf['loss'],'TV':tv,'lower':value,'upper':value,'approximate_objective':value,
            'seconds_from_start':time.time()-started,'no_RF_inner_optimization':True,
            'loss_bounds_are_identical_computed_floating_point_values':True,
            'fresh_current_material_native_propagation':True,'truth_read':False}
        atomic(args.out/'evaluations'/f'eval{completed:03d}.json',record)
        atomic(args.out/'latest_evaluation.json',record)
        array(args.out/'latest_trial_scaled_gradient.npy',gradient.reshape(seed.shape))
        run['completed_native_gradient_evaluations']=completed;atomic(args.out/'run.json',run)
        return {'lower':value,'upper':value,'gradient':gradient,'record':record,'material':material,'u':u}

    def callback(x,value,point):
        nonlocal accepted
        accepted=point['accepted_step'];record={**value['record'],**point}
        cache.update(material=value['material'].copy(),u=value['u'].copy(),
                     gradient=value['gradient'].copy(),record=record)
        for name,key in [('accepted_material','material'),('accepted_log_coordinates','u')]:
            array(args.out/(name+'.npy'),cache[key])
        array(args.out/'accepted_scaled_gradient.npy',cache['gradient'].reshape(seed.shape))
        array(args.out/'current_material.npy',cache['material'])
        atomic(args.out/'accepted_point.json',record)
        atomic(args.out/'evaluations'/f'accepted{accepted:03d}.json',record)
        run.update(accepted_steps=accepted,approximate_projected_gradient_relative=point['projected_gradient_relative'])
        atomic(args.out/'run.json',run)
        if accepted and accepted%4==0:checkpoint()

    try:
        pool=Pool(args,monitor)
        try:
            _,_,termination=solve(oracle,zero.reshape(-1),lower,upper,iterations=args.iterations,
                history=8,maximum_lines=8,callback=callback,target=None)
            stop_reason=termination['stop']
        except BudgetBoundary:
            termination={'stop':stop_reason,'last_fully_accepted_step':accepted,
                         'latest_rejected_trial_is_not_selected':True}
        if not cache:raise RuntimeError('No complete necessary initial RF gradient; inspect preserved trial')
        if accepted!=last_mapped:checkpoint()
        array(final/'material.npy',cache['material']);array(args.out/'current_material.npy',cache['material'])
        record=cache['record'];rf=record['native_RF_quadratic']
        atomic(final/'incumbent_RF.json',{'loss':rf['loss'],'groups':rf['groups'],'truth_read':False})
        report={'accepted_steps':accepted,'completed_native_gradient_evaluations':completed,
            'native_data_before':source_RF['loss'],'native_data_after':rf['loss'],
            'physical_TV_after':record['TV'],'material_SHA256':sha(final/'material.npy'),
            'returned_evaluation':record['evaluation'],'returned_point_was_accepted':True,
            'objective_kind':'original_RF_L2_plus_TV','truth_read':False,'joint_imaging_success_claimed':False}
        atomic(final/'material_update.json',report);atomic(args.out/'termination.json',termination)
        pool.close();pool=None
        if accepted:
            folder=args.out/'physical_checkpoints'/f'accepted{accepted:03d}'
            if sha(folder/'material.npy')!=sha(final/'material.npy'):
                raise RuntimeError('Reused final maps do not match the actual accepted material')
            for suffix in ['png','metrics.json']:shutil.copy2(folder/('maps.'+suffix),final/('maps.'+suffix))
            run['final_maps_reused_from_identical_checkpoint']=str(folder)
        else:maps(final,'Native original RF L2, unchanged baseline')
        monitor.sample('native_RF_L2_control_finished')
        run.update(status='native_RF_L2_control_finished',finished_unix_s=time.time(),
            stop_reason=stop_reason,accepted_steps=accepted,completed_native_gradient_evaluations=completed,
            native_data_after=rf['loss'],material_SHA256=report['material_SHA256'])
        atomic(args.out/'run.json',run);print(json.dumps(run),flush=True)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',run);raise
    finally:
        if pool is not None:pool.close()
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        monitor.log.close()


if __name__=='__main__':main()
