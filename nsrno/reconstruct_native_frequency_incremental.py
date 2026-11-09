"""Bounded native source-batch inversion with all-source physical acceptance."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
from .native_frequency_incremental_pool import NativeFrequencyIncrementalPool,source_batches
from .native_gn_coordinates import Coordinates,SCALES
from .low_frequency_0p5 import atomic,digest,SAMPLE,CASE
from .pressure_resources import PressureResources
from .reconstruct_native_gn import source_manifest
from .reconstruct_frequency_multilevel import subbands,array


def same_process(record):
    p=Path('/proc',str(record['pid']))
    if not p.exists():
        return False
    return (Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
            and int((p/'stat').read_text().rsplit(')',1)[1].split()[19])==record['start_ticks'])


def native_RF_records(root,prior,families):
    final=root/'final/actual_native_acceptance.json'
    record=json.loads(final.read_text()) if final.exists() else {}
    if record.get('RF_source_records'):
        if record['material_SHA256']!=prior['material_SHA256']:
            raise ValueError('Cached full-source material identity changed')
        rows=record['RF_source_records']
    else:
        if prior['status']!='frequency_multilevel_stage_finished':
            raise ValueError('Explicit physical RF ancestry is required')
        if prior.get('native_candidate_accepted'):
            if record.get('material_SHA256')!=prior['material_SHA256']:
                raise ValueError('Parent native acceptance material changed')
            token=record['actual_native_RF']['token']
        else:
            if prior['material_SHA256']!=prior['native_anchor_material_SHA256']:
                raise ValueError('Unaccepted parent is not its measured native origin')
            token=1
        rows=[]
        for name in families:
            for shot in range(11):
                matches=list((root/'workers').glob(
                    f'rank*/fine_RF/token{token:04d}/{name}_{shot:02d}.npy'))
                if len(matches)!=1:
                    raise ValueError(f'Missing or repeated physical parent RF {name}/{shot}')
                rows.append({'family':name,'shot':shot,'path':str(matches[0].resolve()),
                             'SHA256':digest(matches[0])})
    expected={(f,s) for f in families for s in range(11)}
    if len(rows)!=44 or {(r['family'],r['shot']) for r in rows}!=expected:
        raise ValueError('All 44 matched native parent sources are required')
    for row in rows:
        if digest(row['path'])!=row['SHA256']:
            raise ValueError('Parent physical RF file changed')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--high-index',type=Path,required=True)
    p.add_argument('--anchor-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--high-band-hz',type=float,nargs=2,default=[1.5e6,2.6e6])
    p.add_argument('--high-split-hz',type=float,nargs='*',default=[2e6])
    p.add_argument('--low-weight',type=float,default=.75)
    p.add_argument('--cycles',type=int,default=2)
    p.add_argument('--max-seconds',type=float,default=3000.)
    p.add_argument('--seed',type=int,default=8035)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    a=p.parse_args()
    if not (1<=a.cycles<=2 and 600<=a.max_seconds<=3600 and 0<a.cap_gib<=10
            and 0<a.low_weight<1 and 1<=len(a.devices)<=3
            and len(a.devices)==len(set(a.devices))):
        raise ValueError('Explicit native source and memory budget required')
    for name in ['data','high_index','anchor_root','out']:
        value=getattr(a,name).resolve()
        if not value.is_relative_to(Path.cwd().resolve()):
            raise ValueError('All paths must remain in this independent nsrno project')
        setattr(a,name,value)
    if a.out.exists():
        raise RuntimeError('Inspect the existing job; never duplicate or overwrite')
    prior=json.loads((a.anchor_root/'run.json').read_text())
    if prior['status'] not in ['frequency_multilevel_stage_finished',
                               'native_frequency_incremental_stage_finished']:
        raise ValueError('Completed native matched-frequency predecessor required')
    if prior.get('sample')!=SAMPLE or prior.get('case')!=CASE or same_process(prior):
        raise ValueError('Fixed authorized case and inactive predecessor required')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                '--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Another GPU experiment is live')
    anchor=a.anchor_root/'final/material.npy'
    if digest(anchor)!=prior['material_SHA256']:
        raise ValueError('Accepted parent material changed')
    low_path=a.data/'index.json'
    low=json.loads(low_path.read_text())
    high=json.loads(a.high_index.read_text())
    if (low['status']!='paired_0p5MHz_generation_finished' or low['sample']!=SAMPLE
            or high['status']!='matched_existing_3MHz_full90_ready'
            or high['sample']!=SAMPLE or high['case']!=CASE
            or high['absorption']!=low['absorption']
            or high['low_0p5MHz_index_SHA256']!=digest(low_path)):
        raise ValueError('Use only the matched already-generated observations')
    if not 1.5e6<=a.high_band_hz[0]<a.high_band_hz[1]<=4.5e6:
        raise ValueError('High objective outside the acquired valid band')
    parent_plan_name=('anchor_plan.json' if prior['status']=='frequency_multilevel_stage_finished'
                      else 'native_plan.json')
    parent_plan=json.loads((a.anchor_root/parent_plan_name).read_text())
    if (parent_plan['data_index_SHA256']!=digest(low_path)
            or parent_plan['high_index_SHA256']!=digest(a.high_index)):
        raise ValueError('Cached parent RF belongs to another acquisition')
    for name in ['native_time.py','native_time.cu','time_acquisition.py',
                 'frequency_multilevel_acquisition.py','full90_3mhz_acquisition.py',
                 'acquisition_single_case_0p5.py','low_frequency_0p5.py',
                 'matrix_free_rf_response.py']:
        if prior['source_SHA256'].get(name)!=digest(Path(__file__).parent/name):
            raise ValueError('Cached physical source/receiver operator changed: '+name)
    families={}
    for name in ['lf','point3']:
        families[name]={'centre':'low','source':name,
            'bands_hz':subbands([.2e6,.75e6],[.325e6,.45e6,.6e6]),
            'coefficient':.25*a.low_weight}
    for name,source in [('lf3','lf'),('point3_3mhz','point3')]:
        families[name]={'centre':'high','source':source,
            'bands_hz':subbands(a.high_band_hz,a.high_split_hz),
            'coefficient':.25*(1-a.low_weight)}
    references=native_RF_records(a.anchor_root,prior,families)
    reserve=2*2**30+400*2**20
    if shutil.disk_usage('.').free<reserve+200*2**20:
        raise RuntimeError('Insufficient reserve for bounded native checkpoints')
    edge_path=Path('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    edge=np.load(edge_path)
    co=Coordinates(np.load(anchor),edge)
    z,x=np.ogrid[:961,:1121]
    mask=(z>=80)&(z<961-24)&(x>=24)&(x<1121-24)
    low_bound,high_bound=co.bounds(trust=(1.,1.,10.))
    low_bound[:,~mask]=high_bound[:,~mask]=0.
    a.out.mkdir()
    for name in ['jobs','batches','cycles','final']:
        (a.out/name).mkdir()
    plan={'data_root':str(a.data),'high_index':str(a.high_index),'families':families,
          'data_index_SHA256':digest(low_path),'high_index_SHA256':digest(a.high_index),
          'known_edge_SHA256':digest(edge_path),
          'native_anchor_material':str(anchor),'native_anchor_material_SHA256':digest(anchor),
          'reference_RF':references,'truth_read':False}
    atomic(a.out/'native_plan.json',plan)
    hashes=source_manifest()
    for name in [Path(__file__).name,'native_frequency_incremental_worker.py',
                 'native_frequency_incremental_pool.py','native_gn_coordinates.py',
                 'reconstruct_frequency_multilevel.py','frequency_multilevel_pool.py',
                 'frequency_multilevel_worker.py','frequency_multilevel_coordinates.py',
                 'frequency_multilevel_acquisition.py','full90_3mhz_acquisition.py',
                 'acquisition_single_case_0p5.py','low_frequency_0p5.py',
                 'matrix_free_rf_response.py','bounded_model_lbfgs.py']:
        hashes[name]=digest(Path(__file__).parent/name)
    for name in hashes:
        shutil.copy2(Path(__file__).parent/name,a.out/name)
    started=time.time()
    run={'status':'starting_native_frequency_incremental','pid':os.getpid(),
         'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
         'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         'created_unix_s':started,'deadline_unix_s':started+a.max_seconds,
         'sample':SAMPLE,'case':CASE,'families':families,'seed':a.seed,
         'parent_root':str(a.anchor_root),'native_anchor_material_SHA256':digest(anchor),
         'maximum_cycles':a.cycles,'maximum_gradient_jobs':6*a.cycles,
         'maximum_adjoint_source_shots':44*a.cycles,'maximum_lines_per_batch':3,
         'maximum_native_forward_checks':a.cycles,'gradient_jobs':0,
         'native_adjoint_source_shots':0,'native_forward_only_source_shots':0,
         'source_batches_accepted':0,'cycles_completed':0,'cycles_accepted':0,
         'new_native_forward_checks':0,'native_candidate_accepted':False,
         'full_native_shape':[3,961,1121],'native_spacing_m':50e-6,
         'source_shots_per_full_evaluation':44,'receivers':192,'record_us':90,
         'TV_weight':.001,'batch_log_step_caps':[.004,.006,.05],
         'devices':a.devices,'allocator_cap_GiB':a.cap_gib,'per_card_limit_GiB':16,
         'source_SHA256':hashes,'truth_read_by_optimizer':False,
         'reserved_case_unread':'Neg_47_Left','new_observation_generation':False,
         'old_phase_budgets_not_reset':True,
         'parent_counters_as_recorded':{k:prior.get(k) for k in [
             'old_native_gradients_consumed','new_native_gradients','ancestor_coarse_gradients',
             'new_coarse_gradients','ancestor_native_RF_only_evaluations',
             'new_native_forward_checks','native_adjoint_source_shots',
             'native_forward_only_source_shots']},
         'joint_imaging_success_claimed':False}
    atomic(a.out/'run.json',run)
    monitor=PressureResources(a.out)
    pool=None
    stopped=[0]
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    for sig in [signal.SIGTERM,signal.SIGINT]:
        signal.signal(sig,lambda value,frame:stopped.__setitem__(0,value))
    full_sources=[(f,s) for f in families for s in range(11)]
    u=np.zeros_like(co.m)
    best_u=u.copy()
    best_folder=None
    best_paths=references
    rejected_cycles=0

    def freeze():
        for name,expected in hashes.items():
            if digest(Path(__file__).parent/name)!=expected:
                raise RuntimeError('Frozen code changed: '+name)
        if (digest(anchor)!=plan['native_anchor_material_SHA256']
                or digest(low_path)!=plan['data_index_SHA256']
                or digest(a.high_index)!=plan['high_index_SHA256']
                or digest(edge_path)!=plan['known_edge_SHA256']):
            raise RuntimeError('Frozen acquisition or known material metadata changed')

    def checkpoint():
        array(a.out/'latest_batch_control.npy',u)
        array(a.out/'latest_batch_material.npy',co.material(u))
        run.update(**pool.counts)
        atomic(a.out/'run.json',run)

    def maps(folder,label):
        with (folder/'maps.log').open('x') as log:
            subprocess.run([sys.executable,'-m','nsrno.evaluate_time_material',
                str(folder/'material.npy'),'--sample',SAMPLE,'--out',str(folder/'maps'),
                '--estimate-label',label],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True,timeout=300)

    try:
        pool=NativeFrequencyIncrementalPool(a,monitor,plan)
        run['status']='reusing_actual_native_anchor_RF'
        atomic(a.out/'run.json',run)
        best_rf,_=pool.evaluate(None,'cached_forward',full_sources,'cached_native_baseline')
        best_tv=co.regularization(best_u,.001)[0]
        run.update(native_anchor_RF_loss=best_rf['loss'],initial_native_TV=best_tv,
                   status='updating_native_source_batches')
        atomic(a.out/'run.json',run)
        for cycle in range(a.cycles):
            if stopped[0] or time.time()>=run['deadline_unix_s']-300:
                break
            u=best_u.copy()
            step_caps=np.asarray([.004,.006,.05])[:,None,None]*.5**rejected_cycles/SCALES
            batches=source_batches(families,a.seed,cycle)
            complete=True
            for batch_index,sources in enumerate(batches):
                if stopped[0] or time.time()>=run['deadline_unix_s']-300:
                    complete=False
                    break
                freeze()
                tag=f'cycle{cycle+1:02d}_batch{batch_index+1:02d}'
                run.update(status='native_source_batch_gradient',active_cycle=cycle+1,
                           active_batch=batch_index+1)
                atomic(a.out/'run.json',run)
                rf,g=pool.evaluate(co.material(u),'gradient',sources,tag+'_gradient')
                tv,gtv,_,_=co.regularization(u,.001)
                g+=gtv
                g[:,~mask]=0.
                rms=np.sqrt(np.mean(g[:,mask]**2,axis=1))
                diagonal=np.maximum(rms,max(float(rms.max())*1e-3,1e-24))/.1
                direction=np.clip(-g/diagonal[:,None,None],-step_caps,step_caps)*mask
                lines=[]
                batch_accepted=False
                for line,fraction in enumerate([1.,.5,.25],1):
                    if stopped[0] or time.time()>=run['deadline_unix_s']-180:
                        break
                    proposal=np.clip(u+fraction*direction,low_bound,high_bound)
                    slope=float(np.sum(g*(proposal-u),dtype=np.float64))
                    if not slope<0:
                        lines.append({'fraction':fraction,'slope':slope,'propagation':False})
                        continue
                    run['status']='native_source_batch_line_search'
                    atomic(a.out/'run.json',run)
                    trial,_=pool.evaluate(co.material(proposal),'forward',sources,
                                          tag+f'_line{line:02d}')
                    proposed_tv=co.regularization(proposal,.001)[0]
                    ok=trial['loss']+proposed_tv<=rf['loss']+tv+1e-4*slope
                    lines.append({'fraction':fraction,'slope':slope,
                        'batch_RF_before':rf['loss'],'batch_RF_after':trial['loss'],
                        'TV_after':proposed_tv,'accepted':bool(ok),'token':trial['token']})
                    if ok:
                        u=proposal
                        batch_accepted=True
                        run['source_batches_accepted']+=1
                        break
                atomic(a.out/'batches'/(tag+'.json'),{'cycle':cycle+1,'batch':batch_index+1,
                    'sources':sources,'full_objective_source_scaling':11/(len(sources)/4),
                    'actual_native_batch_gradient':True,'lines':lines,'accepted':batch_accepted,
                    'not_yet_accepted_by_all_sources':True,'truth_read':False})
                checkpoint()
                del g,gtv,direction
            if complete:
                run['cycles_completed']+=1
            if stopped[0] or time.time()>=run['deadline_unix_s'] or np.array_equal(u,best_u):
                break
            folder=a.out/'cycles'/f'cycle{cycle+1:02d}'
            folder.mkdir()
            array(folder/'material.npy',co.material(u))
            array(folder/'control.npy',u)
            os.link(anchor,folder/'stage0_initial_material.npy')
            run['status']='actual_native_all_source_check'
            atomic(a.out/'run.json',run)
            rf,_=pool.evaluate(co.material(u),'forward',full_sources,
                                f'cycle{cycle+1:02d}_full_RF',save_RF=True)
            tv=co.regularization(u,.001)[0]
            ok=(rf['loss']<best_rf['loss']-1e-6*max(best_rf['loss'],1e-20)
                and rf['loss']+tv<best_rf['loss']+best_tv-1e-8*max(best_rf['loss']+best_tv,1e-20))
            paths=[]
            for f,s in full_sources:
                matches=list((a.out/'workers').glob(
                    f'rank*/fine_RF/token{rf["token"]:04d}/{f}_{s:02d}.npy'))
                if len(matches)!=1:
                    raise RuntimeError('Full native RF archive incomplete')
                paths.append({'family':f,'shot':s,'path':str(matches[0]),
                              'SHA256':digest(matches[0])})
            record={'actual_native_RF':rf,'native_RF_before':best_rf['loss'],
                'native_TV':tv,'native_objective_before':best_rf['loss']+best_tv,
                'native_objective_after':rf['loss']+tv,
                'accepted_using_actual_native_RF_only':bool(ok),
                'material_SHA256':digest(folder/'material.npy'),
                'RF_source_records':paths,'truth_read':False,'complete_cycle':complete}
            atomic(folder/'actual_native_acceptance.json',record)
            run['new_native_forward_checks']+=1
            if ok:
                best_u=u.copy()
                best_rf,best_tv,best_folder,best_paths=rf,tv,folder,paths
                run['cycles_accepted']+=1
                run['native_candidate_accepted']=True
                maps(folder,'Native source-batch 0.5 and 3 MHz, actual full RF accepted')
            else:
                rejected_cycles+=1
                u=best_u.copy()
            checkpoint()
        pool.close()
        pool=None
        final=a.out/'final'
        os.link(anchor,final/'stage0_initial_material.npy')
        os.link(anchor if best_folder is None else best_folder/'material.npy',final/'material.npy')
        for suffix in ['png','metrics.json']:
            shutil.copy2((a.anchor_root/'final' if best_folder is None else best_folder)/
                         ('maps.'+suffix),final/('maps.'+suffix))
        atomic(final/'actual_native_acceptance.json',{'actual_native_RF':best_rf,
            'native_TV':best_tv,'material_SHA256':digest(final/'material.npy'),
            'RF_source_records':best_paths,'truth_read':False,
            'accepted_using_actual_native_RF_only':bool(run['native_candidate_accepted'])})
        run.update(status='native_frequency_incremental_stage_finished',
                   finished_unix_s=time.time(),seconds=time.time()-started,
                   material_SHA256=digest(final/'material.npy'),
                   final_actual_native_RF_loss=best_rf['loss'])
        atomic(a.out/'run.json',run)
    except BaseException:
        run.update(status='native_frequency_incremental_interrupted_or_failed',
                   error=traceback.format_exc(),stopped_unix_s=time.time())
        atomic(a.out/'run.json',run)
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__=='__main__':
    main()

