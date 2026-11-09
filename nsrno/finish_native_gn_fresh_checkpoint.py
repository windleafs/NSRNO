"""Finish an immutable fresh-reference native checkpoint without replaying HVPs."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import traceback
import numpy as np
from .native_gn_fresh_state import FreshSourceQuadratic
from .native_gn_block_pool import Pool
from .native_gn_json import atomic
from .native_gn_coordinates import dot
from .refine_native_gn_cpu import save_array
from .resume_native_gn_accurate import maps
from .pressure_resources import PressureResources


def restore(source,step):
    run=json.loads((source/'run.json').read_text());folder=source/f'step{step:03d}'
    for name,value in run['source_sha256'].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=value:
            raise RuntimeError(f'Original runtime changed: {name}')
    material=np.load(folder/'old_material.npy')
    baseline=json.loads((folder/'baseline_RF.json').read_text())
    with np.load(run['preconditioner_statistics']) as data:blocks=data['blocks'].astype(np.float64)
    # This is the original fixed damping scale, not a new data calibration.
    calibration=json.loads(Path('out/v5/reconstruction_native_gn_secant_v1_train_200/step001/hvp00/progress.json').read_text())['local_data_preconditioner_scale']
    state=FreshSourceQuadratic(material,np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'),
        np.load(folder/'data_gradient.npy'),baseline,np.load(run['symbol_snapshot']),blocks,
        calibration,trust=run['trust_log_c_log_Z_log_alpha'])
    paths=sorted(p.parent for p in folder.glob('hvp*/progress.json'))
    if not paths:raise RuntimeError('No complete native HVP to finish')
    for checkpoint in paths:
        row=json.loads((checkpoint/'progress.json').read_text())
        d=np.load(checkpoint/'direction.npy');image=np.load(checkpoint/'total_image.npy')
        if not np.allclose(image,np.load(checkpoint/'data_image.npy')+state.h_tv(d)+state.damping*d,rtol=1e-10,atol=1e-14):
            raise RuntimeError('Saved native total image changed')
        state.model.pending_partials=json.loads((checkpoint/'source_curvature.json').read_text())['shots']
        state.model.update(d,image)
        state.u=np.load(checkpoint/'cumulative_log_direction.npy')
        state.Hu=np.load(checkpoint/'cumulative_total_image.npy')
        state.gradient=state.gradient0+state.Hu
        state.history.append(row)
        if abs(state.value()-row['model_change'])>1e-10:raise RuntimeError('Saved true quadratic did not reproduce')
        if abs(state.norm()/state.initial_norm-row['projected_KKT_relative'])>1e-8:
            raise RuntimeError('Saved true projected KKT did not reproduce')
    report={'source':str(source),'source_step':step,'complete_HVPs':len(paths),
        'source_checkpoint_paths':[str(p) for p in paths],
        'true_model_change':state.value(),'true_QP_KKT_relative':state.norm()/state.initial_norm,
        'new_gradient_wave_solves':0,'new_HVPs':0,'CPU_only_restore':True,
        'fixed_reference_SHA256':hashlib.sha256((folder/'old_material.npy').read_bytes()).hexdigest(),
        'truth_read':False}
    return run,state,report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--source-step',type=int,default=1)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--out',type=Path)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    args=p.parse_args();args.source=args.source.resolve()
    source_run,state,restored=restore(args.source,args.source_step)
    if args.prepare_only:
        print(json.dumps(restored),flush=True);return
    if args.out is None:raise ValueError('Require output directory')
    args.out=args.out.resolve();args.sample=source_run['sample']
    if args.out.exists():raise RuntimeError('Inspect prior output; never restart on a timeout')
    processes=json.loads((args.source/'worker_processes.json').read_text())
    for pid in [source_run['pid']]+processes['worker_pids']:
        if Path(f'/proc/{pid}').exists():raise RuntimeError(f'Source process {pid} still exists')
    args.out.mkdir();(args.out/'jobs').mkdir();folder=args.out/'step001';folder.mkdir()
    run={'status':'finishing_saved_native_direction','pid':os.getpid(),
        'sample':args.sample,'source':str(args.source),'source_step':args.source_step,
        'source_status':source_run['status'],'reused_native_HVPs':len(state.history),
        'new_gradient_wave_solves':0,'new_HVPs':0,'devices':args.devices,
        'allocator_cap_GiB':args.cap_gib,'model_per_card_limit_GiB':16,
        'truth_read_by_optimizer':False,'joint_imaging_success_claimed':False,
        'created_unix_s':time.time(),'completed_step':0,
        'source_sha256':{**source_run['source_sha256'],
            'finish_native_gn_fresh_checkpoint.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    atomic(args.out/'run.json',run);atomic(args.out/'restored.json',restored)
    save_array(folder/'old_material.npy',state.material)
    save_array(folder/'stage0_initial_material.npy',state.material)
    save_array(folder/'log_direction.npy',state.u)
    save_array(folder/'cumulative_total_image.npy',state.Hu)
    atomic(folder/'baseline_RF.json',state.baseline)
    monitor=PressureResources(args.out);pool=None
    try:
        pool=Pool(args,monitor);before=state.material.copy();material=before.copy()
        slope=dot(state.gradient0,state.u);accepted=False;lines=[]
        for line in range(8 if slope<0 else 0):
            alpha=.5**line;trial=state.co.material(alpha*state.u)
            run.update(status='native_RF_line_search',line=line);atomic(args.out/'run.json',run)
            rf,_=pool.evaluate(trial,'forward',f'saved_line{line:02d}')
            tv=state.co.regularization(alpha*state.u,state.tv_weight)[0]
            change=rf['loss']+tv-state.baseline['loss']-state.tv
            predicted=state.predicted_physical_change(alpha)
            accepted=bool(change<=1e-4*alpha*slope)
            lines.append({'alpha':alpha,'native_RF':rf,'TV':tv,'physical_objective_change':change,
                'predicted_change':predicted,'actual_to_predicted_reduction':change/predicted if predicted<0 else None,
                'accepted':accepted})
            atomic(folder/'line_search.json',{'candidates':lines})
            if accepted:material=trial;break
        save_array(folder/'material.npy',material);save_array(args.out/'current_material.npy',material)
        atomic(folder/'material_update.json',dict(restored,accepted=accepted,
            native_data_before=state.baseline['loss'],
            native_data_after=lines[-1]['native_RF']['loss'] if accepted else state.baseline['loss'],
            physical_RMS_change_c_rho_alpha=np.sqrt(np.mean((material.astype(np.float64)-before)[:,state.co.mask]**2,axis=1)).tolist()))
        pool.close();pool=None
        run.update(status='post_hoc_maps_only',completed_step=1);atomic(args.out/'run.json',run)
        maps(args,folder,monitor)
        run.update(status='saved_native_checkpoint_finished',finished_unix_s=time.time())
        atomic(args.out/'run.json',run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',run);raise
    finally:
        if pool is not None:pool.close()
        monitor.log.close()


if __name__=='__main__':main()
