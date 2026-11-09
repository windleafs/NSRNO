"""Finish physical RF acceptance/maps from complete saved native HVPs.

This repairs a failed report-writing tail without replaying gradient/HVPs.
The original failed source and its runtime hashes remain unchanged.
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
from .native_gn_saved_state import SavedQuadratic
from .native_gn_json import atomic
from .refine_native_gn_cpu import save_array
from .native_gn_coordinates import dot
from .reconstruct_native_gn import Pool
from .resume_native_gn_accurate import maps
from .pressure_resources import PressureResources


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    args=p.parse_args()
    args.source_root=args.source_root.resolve()
    args.out=args.out.resolve()
    source=json.loads((args.source_root/'run.json').read_text())
    if args.out.exists():raise RuntimeError('Use a new output, not a restart on a timeout')
    if args.cap_gib>14:raise ValueError('Keep room for context below 16 GiB')
    if source['status']!='failed' or 'is not JSON serializable' not in source.get('error',''):
        raise RuntimeError('This recovery requires the recorded serialization failure')
    pids=[source['pid']]+json.loads((args.source_root/'worker_processes.json').read_text())['worker_pids']
    for pid in pids:
        if (Path('/proc')/str(pid)).exists():raise RuntimeError(f'Source PID {pid} still alive')
    for name,expected in source['source_sha256'].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Original runtime changed: {name}')
    state=SavedQuadratic(args.source_root)
    if not np.array_equal(np.load(state.source_folder/'log_direction.npy'),state.u):
        raise RuntimeError('Final saved primal direction differs from complete HVP state')
    if not np.array_equal(np.load(state.source_folder/'cumulative_total_image.npy'),state.Hu):
        raise RuntimeError('Final cumulative HVP differs from complete checkpoints')
    args.sample=state.sample
    args.out.mkdir(parents=True)
    (args.out/'jobs').mkdir()
    folder=args.out/'step001'
    folder.mkdir()
    manifest=dict(source['source_sha256'])
    for name in ['native_gn_json.py','finish_native_gn_checkpoint.py']:
        manifest[name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    run={'status':'restored_complete_native_direction','pid':os.getpid(),'sample':args.sample,
        'solver_revision':'complete_saved_source_resolved_step_after_report_serialization_failure',
        'source_root':str(args.source_root),'source_sha256':manifest,
        'preconditioner_statistics':str(state.preconditioner),'devices':args.devices,
        'allocator_cap_GiB':args.cap_gib,'model_per_card_limit_GiB':16,
        'reused_complete_HVPs':len(state.paths),'new_HVPs':0,'new_gradient_wave_solves':0,
        'truth_read_by_optimizer':False,'joint_imaging_success_claimed':False,
        'created_unix_s':time.time(),'completed_step':0,'step':1}
    atomic(args.out/'run.json',run)
    atomic(args.out/'fixed_model_reference.json',state.reference)
    atomic(folder/'fixed_HVP_manifest.json',{'paths':[str(p) for p in state.paths]})
    atomic(folder/'source_failure.json',{'original_error':source['error'],
        'source_run':str(args.source_root/'run.json'),'original_runtime_unchanged':True,
        'all_source_PIDs_gone':True,'repeated_gradient_or_HVPs':0})
    save_array(folder/'stage0_initial_material.npy',state.best_material)
    save_array(folder/'log_direction.npy',state.u)
    save_array(folder/'cumulative_total_image.npy',state.Hu)
    before=state.best_material.copy()
    incumbent_rf=state.best_rf['loss']
    incumbent_cost=state.best_cost
    monitor=PressureResources(args.out)
    pool=None
    try:
        pool=Pool(args,monitor)
        slope=dot(state.gradient0,state.u)
        if slope>=0:raise RuntimeError('Saved true direction is not descent')
        reference=state.baseline_rf['loss']+state.tv
        accepted=False
        lines=[]
        for line in range(8):
            alpha=.5**line
            trial=state.co.material(alpha*state.u)
            run.update(status='native_material_line_search',line=line)
            atomic(args.out/'run.json',run)
            rf,_=pool.evaluate(trial,'forward',f'finish_line{line:02d}')
            tv=state.co.regularization(alpha*state.u,state.reference['tv_weight'])[0]
            cost=rf['loss']+tv
            armijo=bool(cost<=reference+1e-4*alpha*slope)
            improves=bool(cost<incumbent_cost-max(1e-12,abs(incumbent_cost)*1e-8))
            accepted=bool(armijo and improves)
            lines.append({'alpha':alpha,'native_RF':rf,'TV':tv,'physical_objective':cost,
                'change_from_fixed_reference':cost-reference,
                'change_from_accepted_incumbent':cost-incumbent_cost,
                'fixed_reference_Armijo':armijo,'improves_incumbent':improves,'accepted':accepted})
            atomic(folder/'line_search.json',{'candidates':lines})
            if accepted:
                state.best_material=trial
                state.best_u=alpha*state.u
                state.best_rf=rf
                state.best_cost=cost
                break
        for path,value in [(folder/'material.npy',state.best_material),
            (folder/'best_log_direction.npy',state.best_u),
            (args.out/'current_material.npy',state.best_material)]:save_array(path,value)
        atomic(folder/'incumbent_RF.json',state.best_rf)
        report={'method':run['solver_revision'],'accepted':accepted,
            'native_data_before':incumbent_rf,'native_data_after':state.best_rf['loss'],
            'accepted_alpha':alpha if accepted else None,'reused_HVPs':len(state.paths),
            'new_HVPs':0,'true_QP_KKT_relative':state.norm()/state.initial_norm,
            'true_QP_tolerance_met':bool(state.norm()<=state.reference['true_QP_target_relative']*state.initial_norm),
            'QP_exact_solution_claimed':False,'model_change':state.value(),
            'physical_RMS_change_from_incumbent':np.sqrt(np.mean(
                (state.best_material.astype(np.float64)-before)[:,state.co.mask]**2,axis=1)).tolist(),
            'truth_read':False}
        atomic(folder/'direction.json',report)
        atomic(folder/'material_update.json',report)
        pool.close()
        pool=None
        run.update(status='post_hoc_maps_only',completed_step=1)
        atomic(args.out/'run.json',run)
        maps(args,folder,monitor)
        run.update(status='planned_native_GN_steps_finished',finished_unix_s=time.time())
        atomic(args.out/'run.json',run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',run)
        raise
    finally:
        if pool is not None:pool.close()
        monitor.log.close()


if __name__=='__main__':main()
