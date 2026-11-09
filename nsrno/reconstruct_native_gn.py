"""Low-memory, all-shot native physical GN reconstruction of c/rho/alpha.

The optimization imports no material-truth interface. Post-hoc maps run in a
separate subprocess after a material step; their results do not enter updates.
"""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
import traceback
import numpy as np
from .native_gn_coordinates import Coordinates,physical_blocks,bounded_direction,dot
from .native_gn_worker import worker
from .pressure_resources import PressureResources
from .wri_worker import atomic
from .rf_depth_weights import WEIGHTS,NAMES

class Pool:
    def __init__(self,args,monitor):
        self.args=args;self.monitor=monitor;self.token=0
        context=mp.get_context('spawn')
        self.jobs=[context.Queue() for _ in args.devices]
        self.results=context.Queue();self.workers=[]
        monitor.wait_budget(args.devices,args.cap_gib,'before_native_GN_workers')
        for rank,device in enumerate(args.devices):
            proc=context.Process(target=worker,args=(rank,device,self.jobs[rank],self.results,
                args.sample,args.chunk,args.cap_gib,str(args.out/'workers'/f'rank{rank:02d}'),len(args.devices)))
            proc.start();self.workers.append(proc)
        atomic(args.out/'worker_processes.json',{'coordinator_pid':os.getpid(),
            'worker_pids':[p.pid for p in self.workers],'devices':args.devices,
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})

    def evaluate(self,material,kind,tag,direction=None):
        self.token+=1;token=self.token;started=time.time()
        job={'token':token,'kind':kind,'material':np.asarray(material,np.float32)}
        if direction is not None:job['direction']=np.asarray(direction,np.float32)
        atomic(self.args.out/'active_job.json',{'tag':tag,'token':token,'kind':kind,
            'started_unix_s':started,'truth_read':False})
        for jobs in self.jobs:jobs.put(job)
        rows=[]
        while len(rows)<len(self.workers):
            try:row=self.results.get(timeout=20)
            except queue.Empty:
                self.monitor.sample(tag)
                dead=[(i,p.exitcode) for i,p in enumerate(self.workers) if p.exitcode is not None]
                if dead:raise RuntimeError(f'Native GN workers exited during {tag}: {dead}')
                continue
            if row.get('error'):raise RuntimeError(f'Worker {row["rank"]}, {tag}:\n{row["error"]}')
            if row['token']!=token or any(r['rank']==row['rank'] for r in rows):
                raise RuntimeError('Unexpected or duplicate native GN worker result')
            rows.append(row)
            self.monitor.sample(tag)
        result={'tag':tag,'token':token,'kind':kind,'loss':sum(r['loss'] for r in rows),
            'seconds':time.time()-started,'peak_Torch_GiB':max(r['peak_Torch_GiB'] for r in rows),
            'groups':{},'truth_read':False}
        if kind in ['hvp','gradient']:
            gradient=sum((r['gradient'] for r in rows),np.zeros_like(material,dtype=np.float64))
        else:gradient=None
        if kind=='hvp':
            result['weighted_tangent_energy']=sum(r['tangent_energy'] for r in rows)
            result['adjoint_quadratic']=dot(direction,gradient)
            den=max(abs(result['weighted_tangent_energy']),abs(result['adjoint_quadratic']),1e-30)
            result['routine_curvature_identity_relative_error']=abs(
                result['weighted_tangent_energy']-result['adjoint_quadratic'])/den
            if result['routine_curvature_identity_relative_error']>2e-3:
                raise RuntimeError('HVP curvature disagrees with its own weighted tangent RF energy')
        else:
            for source in ['lf','point3']:
                values=[r['groups'][source] for r in rows]
                den=np.array(values[0]['window_energy'])
                numer=sum((np.array(v['window_error_squared']) for v in values),np.zeros(5))
                error=sum(v['error_squared'] for v in values)
                energy=values[0]['energy']
                result['groups'][source]={'relative_RMS':float(np.sqrt(error/energy)),
                    'geometry_window_relative_RMS':np.sqrt(numer/den).tolist(),
                    'geometry_loss':.5*float(np.dot(WEIGHTS,numer/den)),
                    'window_error_squared':numer.tolist(),'window_energy':den.tolist()}
            check=sum(v['geometry_loss'] for v in result['groups'].values())/2
            if abs(check-result['loss'])>1e-5*max(abs(check),1e-20):
                raise RuntimeError('Full native objective disagrees with window sufficient statistics')
        atomic(self.args.out/'jobs'/f'{token:04d}_{tag}.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)
        return result,gradient

    def close(self):
        for jobs in self.jobs:jobs.put(None)
        for proc in self.workers:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate();proc.join(timeout=10)
        self.monitor.sample('native_GN_workers_closed')

def source_manifest():
    folder=Path(__file__).parent
    names=['reconstruct_native_gn.py','native_gn_worker.py','native_gn_coordinates.py',
        'native_tangent.py','native_tangent.cu','native_time.py','native_time.cu',
        'time_acquisition.py','rf_depth_weights.py','physical_box_direction.py',
        'pressure_resources.py','wri_worker.py','wri_material.py','coupled_coordinates.py',
        'v5_observations.py','rf_representation.py']
    return {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}

def post_hoc_maps(args,folder,monitor):
    np.save(folder/'stage0_initial_material.npy',np.load(args.seed))
    cmd=[sys.executable,'-m','nsrno.evaluate_time_material',str(folder/'material.npy'),
        '--sample',args.sample,'--out',str(folder/'maps'),
        '--estimate-label','Native full-Jacobian GN']
    with open(folder/'maps.log','w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:monitor.sample('post_hoc_native_GN_maps_only')
        if proc.returncode:raise RuntimeError(f'Post-hoc map process failed: {folder/"maps.log"}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--preconditioner-statistics',type=Path,required=True)
    p.add_argument('--sample',default='train_200')
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--outer-steps',type=int,default=4)
    p.add_argument('--inner-hvps',type=int,default=8)
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    p.add_argument('--tv-weight',type=float,default=.001)
    p.add_argument('--damping-relative',type=float,default=.01)
    args=p.parse_args()
    if args.out.exists():raise RuntimeError('Use a new output directory; no implicit partial restart')
    if args.cap_gib>14:raise ValueError('Keep CUDA context overhead below the 16-GiB process-tree limit')
    args.out.mkdir(parents=True);(args.out/'jobs').mkdir()
    material=np.load(args.seed)
    edge=np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    with np.load(args.preconditioner_statistics) as data:primitive_blocks=data['blocks'].astype(np.float64)
    run={'status':'starting','pid':os.getpid(),'sample':args.sample,
        'objective':'full-native RF, mean of LF and point3 geometry-normalized waveform losses',
        'sources':['lf','point3'],'shots_per_source':11,'receivers':192,
        'frequency_selection':'all retained native RF, existing source-specific linear response',
        'native_spacing_m':50e-6,'shape':list(material.shape),'parameterization':'independent log(c), log(rho*c), log(tissue alpha)',
        'unknown_first_row':80,'no_parameter_coarsening':True,'depth_window_names':NAMES,
        'depth_window_weights':WEIGHTS,'depth_window_caveat':'fixed arrival proxies, not unique scatterer depths',
        'devices':args.devices,'allocator_cap_GiB':args.cap_gib,'model_per_card_limit_GiB':16,
        'seed':str(args.seed.resolve()),'seed_SHA256':hashlib.sha256(args.seed.read_bytes()).hexdigest(),
        'preconditioner_statistics':str(args.preconditioner_statistics.resolve()),
        'preconditioner_SHA256':hashlib.sha256(args.preconditioner_statistics.read_bytes()).hexdigest(),
        'preconditioner_role':'local metric only; data Hessian uses new exact native JVP and adjoint',
        'inner_hvp_budget':args.inner_hvps,'outer_step_budget':args.outer_steps,
        'truth_read_by_optimizer':False,'joint_imaging_success_claimed':False,
        'source_sha256':source_manifest(),'created_unix_s':time.time(),'completed_step':0}
    atomic(args.out/'run.json',run);np.save(args.out/'initial_material.npy',material)
    monitor=PressureResources(args.out);pool=None
    try:
        pool=Pool(args,monitor)
        for step in range(1,args.outer_steps+1):
            folder=args.out/f'step{step:03d}';folder.mkdir()
            coordinates=Coordinates(material,edge)
            zero=np.zeros_like(material,dtype=np.float64)
            np.save(folder/'old_material.npy',material)
            run.update(status='native_gradient',step=step);atomic(args.out/'run.json',run)
            baseline,gradient=pool.evaluate(material,'gradient',f'step{step:03d}_gradient')
            np.save(folder/'data_gradient.npy',gradient)
            atomic(folder/'baseline_RF.json',baseline)
            blocks=physical_blocks(primitive_blocks,material,coordinates.edge)
            run['status']='native_full_GN_direction';atomic(args.out/'run.json',run)
            def hvp(direction,index):
                _,image=pool.evaluate(material,'hvp',f'step{step:03d}_hvp{index:02d}',direction)
                return image
            direction,record=bounded_direction(coordinates,gradient,hvp,blocks,
                maximum=args.inner_hvps,tv_weight=args.tv_weight,
                damping_relative=args.damping_relative,
                callback=lambda row:atomic(folder/'direction_progress.json',row))
            np.save(folder/'log_direction.npy',direction)
            atomic(folder/'direction.json',record)
            tv,g_tv,_,_=coordinates.regularization(zero,args.tv_weight)
            slope=dot(gradient+g_tv,direction)
            accepted=False;lines=[]
            if slope<0:
                for line in range(8):
                    alpha=.5**line
                    trial=coordinates.material(alpha*direction)
                    run.update(status='native_material_line_search',line=line)
                    atomic(args.out/'run.json',run)
                    rf,_=pool.evaluate(trial,'forward',f'step{step:03d}_line{line:02d}')
                    tv_trial=coordinates.regularization(alpha*direction,args.tv_weight)[0]
                    change=rf['loss']+tv_trial-baseline['loss']-tv
                    accepted=change<=1e-4*alpha*slope
                    row={'alpha':alpha,'native_data_change':rf['loss']-baseline['loss'],
                        'TV_change':tv_trial-tv,'objective_change':change,'armijo_bound':1e-4*alpha*slope,
                        'accepted':bool(accepted),'native_RF':rf}
                    lines.append(row)
                    atomic(folder/'line_search.json',{'candidates':lines})
                    if accepted:
                        change_material=trial.astype(np.float64)-material
                        record['accepted_physical_RMS_c_rho_alpha']=np.sqrt(
                            np.mean(change_material[:,coordinates.mask]**2,axis=1)).tolist()
                        record['accepted_alpha']=alpha
                        material=trial
                        break
            np.save(folder/'material.npy',material)
            np.save(args.out/'current_material.npy',material)
            record['accepted']=bool(accepted)
            record['native_data_before']=baseline['loss']
            record['native_data_after']=lines[-1]['native_RF']['loss'] if accepted else baseline['loss']
            atomic(folder/'material_update.json',record)
            run.update(status='post_hoc_maps_only',completed_step=step)
            atomic(args.out/'run.json',run)
            post_hoc_maps(args,folder,monitor)
            if not accepted:
                run['status']='no_native_descent_step'
                break
        else:run['status']='planned_native_GN_steps_finished'
        run['finished_unix_s']=time.time();atomic(args.out/'run.json',run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',run)
        raise
    finally:
        if pool is not None:pool.close()
        monitor.log.close()

if __name__=='__main__':main()
