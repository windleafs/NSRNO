"""Test RF-conditioned superficial initialization with full native L-BFGS.

The forward kernel, full RF objective, independent native fields and TV are
unchanged. Old local statistics only scale optimization variables. Gradients
are evaluated at every trial material; no approximate normal QP or HVP runs.
"""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
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
import scipy
from scipy.optimize import Bounds,minimize
from .reconstruct_native_gn import Pool,source_manifest
from .native_gn_coordinates import Coordinates,physical_blocks
from .native_gn_json import atomic
from .refine_native_gn_cpu import save_array
from .pressure_resources import PressureResources


class EvaluationBudget(Exception):pass


def post_hoc(args,folder,monitor):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    with open(folder/'maps.log','w') as log:
        proc=subprocess.Popen([sys.executable,'-m','nsrno.evaluate_time_material',
            str(folder/'material.npy'),'--sample',args.sample,'--out',str(folder/'maps'),
            '--estimate-label','RF surface initialization + native L-BFGS'],
            stdout=log,stderr=subprocess.STDOUT,env=env)
        while proc.poll() is None:
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:monitor.sample('post_hoc_surface_LBFGS_maps')
        if proc.returncode:raise RuntimeError('Post-hoc material maps failed; inspect maps.log')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--incumbent-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preconditioner-statistics',type=Path,required=True)
    p.add_argument('--calibration-record',type=Path,required=True)
    p.add_argument('--sample',default='train_200')
    p.add_argument('--iterations',type=int,default=4)
    p.add_argument('--maximum-evaluations',type=int,default=12)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    p.add_argument('--tv-weight',type=float,default=.001)
    args=p.parse_args();args.out=args.out.resolve();args.incumbent_root=args.incumbent_root.resolve()
    if args.out.exists():raise RuntimeError('Inspect existing run; no implicit restart')
    if not 0<args.cap_gib<=14:raise ValueError('Keep full process-tree VRAM below 16 GiB per card')
    old=json.loads((args.incumbent_root/'run.json').read_text())
    if old['status']!='saved_native_checkpoint_finished':raise RuntimeError('Require completed physical incumbent')
    processes=json.loads((args.incumbent_root/'worker_processes.json').read_text())
    for pid in [old['pid']]+processes['worker_pids']:
        if Path(f'/proc/{pid}').exists():raise RuntimeError(f'Prior source process {pid} still exists')
    seed=np.load(args.seed);incumbent=np.load(args.incumbent_root/'step001/material.npy')
    if seed.shape!=(3,961,1121) or seed.dtype!=np.float32:raise ValueError('Require full native float32 fields')
    edge=np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    co=Coordinates(seed,edge)
    low,high=co.bounds(trust=(1.,1.,10.))
    zero=np.zeros_like(seed,dtype=np.float64)
    _,_,_,tv_diagonal=co.regularization(zero,args.tv_weight)
    with np.load(args.preconditioner_statistics) as data:primitive=data['blocks'].astype(np.float64)
    scale=json.loads(args.calibration_record.read_text())['local_data_preconditioner_scale']
    diagonal=scale*physical_blocks(primitive,seed,co.edge)[:3]+tv_diagonal
    metric=np.sqrt(np.maximum(diagonal,1e-30));metric[:,~co.mask]=1.
    old_rf=json.loads((args.incumbent_root/'step001/material_update.json').read_text())['native_data_after']
    old_tv=Coordinates(incumbent,edge).regularization(zero,args.tv_weight)[0]
    incumbent_cost=old_rf+old_tv
    args.out.mkdir();(args.out/'jobs').mkdir();(args.out/'evaluations').mkdir()
    manifest=source_manifest()
    for name in ['reconstruct_native_surface_lbfgs.py','native_gn_json.py','refine_native_gn_cpu.py']:
        manifest[name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    run={'status':'starting_surface_initialized_native_LBFGS','pid':os.getpid(),
        'sample':args.sample,'seed':str(args.seed.resolve()),
        'seed_SHA256':hashlib.sha256(args.seed.read_bytes()).hexdigest(),
        'incumbent_root':str(args.incumbent_root),'incumbent_native_RF':old_rf,
        'solver':'SciPy bounded L-BFGS with exact current-material native gradients',
        'scipy_version':scipy.__version__,'source_sha256':manifest,
        'preconditioner_statistics':str(args.preconditioner_statistics.resolve()),
        'preconditioner_role':'fixed diagonal change of optimization variables only',
        'old_exact_secants_reused':False,'normal_QP_solves':0,'HVP_budget':0,
        'unknowns':'independent native c/rho/tissue-alpha at every unknown pixel',
        'shape':list(seed.shape),'native_spacing_m':50e-6,'depth_scope_mm':[1,43],
        'shots':'11LF + 11point3','receivers':192,'all_retained_RF':True,
        'coarse_material_parameterization':False,'surface_variables_frozen':False,
        'global_bounds_c_m_s':[1400,1680],'global_bounds_log_Z_relative':[-.25,.25],
        'devices':args.devices,'allocator_cap_GiB':args.cap_gib,'model_per_card_limit_GiB':16,
        'iteration_budget':args.iterations,'gradient_evaluation_budget':args.maximum_evaluations,
        'truth_read_by_optimizer':False,'joint_imaging_success_claimed':False,
        'development_case_only':True,'completed_step':0,'created_unix_s':time.time()}
    atomic(args.out/'run.json',run)
    save_array(args.out/'seed_material.npy',seed)
    save_array(args.out/'original_incumbent_material.npy',incumbent)
    save_array(args.out/'current_material.npy',incumbent)
    monitor=PressureResources(args.out);pool=None
    counter=0;accepted=0;cache={};best={'objective':float('inf')}

    def save_branch_point(index,material,record):
        folder=args.out/f'step{index:03d}';folder.mkdir()
        save_array(folder/'material.npy',material)
        save_array(folder/'stage0_initial_material.npy',incumbent)
        atomic(folder/'material_update.json',record)
        run.update(status='post_hoc_maps_only',completed_step=index)
        atomic(args.out/'run.json',run)
        post_hoc(args,folder,monitor)

    def retain(material,record):
        if record['objective']<best['objective']:
            best.update(objective=record['objective'],native_RF=record['native_RF'],evaluation=record['evaluation'])
            save_array(args.out/'branch_best_material.npy',material)
            atomic(args.out/'branch_best.json',best)
        improves=record['objective']<incumbent_cost
        if improves:
            save_array(args.out/'current_material.npy',np.load(args.out/'branch_best_material.npy'))
        return improves

    try:
        pool=Pool(args,monitor)
        def evaluate(x):
            nonlocal counter
            if counter>=args.maximum_evaluations:raise EvaluationBudget()
            counter+=1
            u=x.reshape(seed.shape)/metric;material=co.material(u)
            run.update(status='native_RF_and_gradient_at_trial',evaluation=counter,accepted_steps=accepted)
            atomic(args.out/'run.json',run)
            rf,g=pool.evaluate(material,'gradient',f'lbfgs_eval{counter:03d}')
            tv,g_tv,_,_=co.regularization(u,args.tv_weight)
            objective=rf['loss']+tv
            gradient=((g+g_tv)/metric).ravel()
            if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
                raise RuntimeError('Nonfinite native objective or gradient')
            record={'evaluation':counter,'native_RF':rf['loss'],'TV':tv,'objective':objective,
                'scaled_gradient_norm':float(np.linalg.norm(gradient)),'RF_groups':rf['groups'],
                'truth_read':False,'current_material_native_gradient':True,'HVPs':0}
            atomic(args.out/'evaluations'/f'eval{counter:03d}.json',record)
            save_array(args.out/'latest_trial_material.npy',material)
            save_array(args.out/'latest_trial_log_coordinates.npy',u)
            save_array(args.out/'latest_trial_scaled_gradient.npy',gradient.reshape(seed.shape))
            cache.update(x=x.copy(),material=material,record=record,u=u)
            if counter==1:
                improves=retain(material,record)
                record=dict(record,kind='RF_conditioned_initialization',accepted_global_incumbent=improves,
                    native_data_before=old_rf,native_data_after=rf['loss'])
                atomic(args.out/'initial_native_RF.json',record)
                save_branch_point(0,material,record)
            return objective,gradient

        def callback(x):
            nonlocal accepted
            if not np.array_equal(cache['x'],x):raise RuntimeError('Accepted L-BFGS point differs from last evaluated point')
            accepted+=1
            record=dict(cache['record'],accepted_step=accepted,kind='accepted_native_LBFGS_step')
            record['accepted_global_incumbent']=retain(cache['material'],record)
            save_array(args.out/'accepted_scaled_coordinates.npy',x.reshape(seed.shape))
            save_array(args.out/'accepted_log_coordinates.npy',cache['u'])
            save_branch_point(accepted,cache['material'],record)

        result=None
        try:
            result=minimize(evaluate,zero.ravel(),method='L-BFGS-B',jac=True,
                bounds=Bounds((low*metric).ravel(),(high*metric).ravel()),callback=callback,
                options={'maxiter':args.iterations,'maxfun':args.maximum_evaluations,
                    'maxcor':8,'maxls':6,'ftol':1e-10,'gtol':1e-7})
            termination={'message':str(result.message),'software_success':bool(result.success),
                'iterations':int(result.nit),'evaluations':int(result.nfev)}
        except EvaluationBudget:
            termination={'message':'Explicit native gradient evaluation budget reached',
                'software_success':False,'iterations':accepted,'evaluations':counter}
        run.update(status='planned_native_LBFGS_budget_finished',accepted_steps=accepted,
            native_gradient_evaluations=counter,termination=termination,
            best_branch_native_RF=best.get('native_RF'),best_branch_objective=best['objective'],
            branch_improves_original_incumbent=best['objective']<incumbent_cost,
            finished_unix_s=time.time())
        atomic(args.out/'run.json',run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',run);raise
    finally:
        if pool is not None:pool.close()
        monitor.log.close()


if __name__=='__main__':main()
