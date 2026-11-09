"""Independent c/rho/tissue-alpha inversion with matched 0.5 and 3 MHz RF.

One necessary native gradient anchors a corrected coarse model. All accepted
materials are checked with the actual native receiving operators. No truth
arrays enter the optimizer.
"""
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
import torch
from .frequency_multilevel_coordinates import FrequencyMultilevelCoordinates
from .frequency_multilevel_pool import FrequencyMultilevelPool
from .bounded_model_lbfgs import solve
from .low_frequency_0p5 import atomic,digest,SAMPLE,CASE
from .native_gn_coordinates import SCALES
from .pressure_resources import PressureResources
from .reconstruct_native_gn import source_manifest


class BudgetBoundary(Exception):
    pass


def array(path,value):
    temporary=path.with_name(path.stem+'.partial.npy')
    np.save(temporary,value)
    temporary.replace(path)


def emit(value):
    try:
        print(json.dumps(value),flush=True)
    except BrokenPipeError:
        pass


def subbands(extent,splits):
    edges=[extent[0],*splits,extent[1]]
    if any(x>=y for x,y in zip(edges[:-1],edges[1:])):
        raise ValueError('Subband edges must increase')
    return list(zip(edges[:-1],edges[1:]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--high-index',type=Path)
    p.add_argument('--anchor-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--band-hz',type=float,nargs=2,default=[2e5,75e4])
    p.add_argument('--split-hz',type=float,nargs='*',default=[325e3,45e4,6e5])
    p.add_argument('--high-band-hz',type=float,nargs=2,default=[1.5e6,2.6e6])
    p.add_argument('--high-split-hz',type=float,nargs='*',default=[2e6])
    p.add_argument('--low-weight',type=float,default=.75)
    p.add_argument('--coarse-factor',type=int,choices=[2,4],default=2)
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--max-coarse-evaluations',type=int,default=24)
    p.add_argument('--max-seconds',type=float,default=5400)
    p.add_argument('--coarse-seconds',type=float,default=4200)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    a=p.parse_args()
    if a.out.exists():
        raise RuntimeError('Inspect the existing job; never duplicate or overwrite')
    if not (1<=a.steps<=32 and 1<=a.max_coarse_evaluations<=50 and
            0<a.coarse_seconds<=a.max_seconds-600 and a.max_seconds<=7200 and
            0<a.cap_gib<=10 and 0<a.low_weight<1):
        raise ValueError('Invalid explicit resource or data weights')
    if not a.devices or len(a.devices)!=len(set(a.devices)):
        raise ValueError('Distinct source-worker devices required')
    if a.high_index is not None and a.coarse_factor!=2:
        raise ValueError('This 3 MHz candidate requires the twofold propagation grid')
    prior=json.loads((a.anchor_root/'run.json').read_text())
    fine_parent=prior['status']=='single_case_low_frequency_reconstruction_finished'
    accepted_parent=prior['status'] in [
        'corrected_multilevel_low_frequency_stage_finished','frequency_multilevel_stage_finished']
    if not fine_parent and not (accepted_parent and prior.get('native_candidate_accepted')):
        raise ValueError('Use a physically accepted completed native material')
    if prior.get('sample')!=SAMPLE or prior.get('case')!=CASE:
        raise ValueError('Only the authorized development sample is admitted')
    if Path('/proc',str(prior['pid'])).exists():
        raise RuntimeError('The predecessor coordinator is still live')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                '--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Another GPU experiment is live')
    anchor=a.anchor_root/'final/material.npy'
    if digest(anchor)!=prior['material_SHA256']:
        raise ValueError('The accepted native anchor changed')
    data_index=a.data/'index.json'
    low=json.loads(data_index.read_text())
    if low['status']!='paired_0p5MHz_generation_finished' or low['sample']!=SAMPLE:
        raise ValueError('Fixed authorized 0.5 MHz observation required')
    low_bands=subbands(a.band_hz,a.split_hz)
    if not .1e6<=a.band_hz[0]<a.band_hz[1]<=1.25e6:
        raise ValueError('Low objective outside retained frequencies')
    families={}
    low_weight=1. if a.high_index is None else a.low_weight
    for name in ['lf','point3']:
        families[name]={'centre':'low','source':name,'bands_hz':low_bands,
                        'coefficient':.25*low_weight}
    high_hash=None
    if a.high_index is not None:
        high=json.loads(a.high_index.read_text())
        if (high['status']!='matched_existing_3MHz_full90_ready' or
                high['sample']!=SAMPLE or high['case']!=CASE or
                high['absorption']!=low['absorption'] or
                high['low_0p5MHz_index_SHA256']!=digest(data_index)):
            raise ValueError('Use the matched existing high-frequency observation')
        if not 1.5e6<=a.high_band_hz[0]<a.high_band_hz[1]<=4.5e6:
            raise ValueError('High objective outside valid 3 MHz receiving band')
        high_bands=subbands(a.high_band_hz,a.high_split_hz)
        for name,source in [('lf3','lf'),('point3_3mhz','point3')]:
            families[name]={'centre':'high','source':source,'bands_hz':high_bands,
                            'coefficient':.25*(1-low_weight)}
        high_hash=digest(a.high_index)
    edge=np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    co=FrequencyMultilevelCoordinates(np.load(anchor),edge,factor=a.coarse_factor)
    reserve=2*2**30+400*2**20
    if shutil.disk_usage('.').free<reserve+300*2**20:
        raise RuntimeError('Insufficient reserve for full multi-frequency RF and material checkpoints')
    a.out=a.out.resolve()
    a.data=a.data.resolve()
    a.anchor_root=a.anchor_root.resolve()
    if a.high_index is not None:
        a.high_index=a.high_index.resolve()
    a.out.mkdir()
    for n in ['jobs','evaluations','native_candidates','final']:
        (a.out/n).mkdir()
    old_native=(prior['completed_native_gradients'] if fine_parent else
                prior.get('old_native_gradients_consumed',0)+prior.get('new_native_gradients',0))
    old_coarse=prior.get('ancestor_coarse_gradients',0)+prior.get('new_coarse_gradients',0)
    old_checks=prior.get('ancestor_native_RF_only_evaluations',0)+prior.get('new_native_forward_checks',0)
    plan={'data_root':str(a.data),'data_index_SHA256':digest(data_index),
          'high_index':str(a.high_index) if a.high_index is not None else None,
          'high_index_SHA256':high_hash,'families':families,'coarse_factor':a.coarse_factor,
          'native_anchor_material':str(anchor.resolve()),
          'native_anchor_material_SHA256':digest(anchor),'native_anchor_RF':[],
          'native_anchor_RF_measured_in_necessary_initial_gradient':True,
          'parent_root':str(a.anchor_root),'parent_native_gradient_jobs':old_native,
          'parent_coarse_gradient_jobs':old_coarse,'parent_native_forward_checks':old_checks}
    atomic(a.out/'anchor_plan.json',plan)
    hashes=source_manifest()
    for name in [Path(__file__).name,'frequency_multilevel_worker.py',
                 'frequency_multilevel_pool.py','frequency_multilevel_coordinates.py',
                 'frequency_multilevel_acquisition.py','full90_3mhz_acquisition.py',
                 'acquisition_single_case_0p5.py','low_frequency_0p5.py',
                 'matrix_free_rf_response.py','bounded_model_lbfgs.py']:
        hashes[name]=digest(Path(__file__).parent/name)
    for name in hashes:
        shutil.copy2(Path(__file__).parent/name,a.out/name)
    started=time.time()
    run={'status':'starting_frequency_multilevel_inverse','pid':os.getpid(),
         'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
         'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         'created_unix_s':started,'deadline_unix_s':started+a.max_seconds,
         'coarse_deadline_unix_s':started+a.coarse_seconds,
         'maximum_coarse_gradients':a.max_coarse_evaluations,
         'maximum_coarse_accepted_steps':a.steps,'maximum_native_forward_checks':3,
         'new_native_gradients_planned':1,'new_native_gradients':0,
         'new_coarse_gradients':0,'coarse_accepted_steps':0,'new_native_forward_checks':0,
         'native_candidate_accepted':False,'sample':SAMPLE,'case':CASE,
         'families':families,'source_shots_per_complete_evaluation':11*len(families),
         'centres_hz':[.5e6,3e6] if a.high_index is not None else [.5e6],
         'record_us':90,'receivers':192,'native_shape':[3,961,1121],
         'coarse_info':co.info(),'coarse_factor':a.coarse_factor,
         'all_three_fields_independent':True,'truth_read_by_optimizer':False,
         'reserved_case_unread':'Neg_47_Left',
         'native_anchor_material_SHA256':plan['native_anchor_material_SHA256'],
         'parent_root':str(a.anchor_root),'source_SHA256':hashes,
         'devices':a.devices,'allocator_cap_GiB':a.cap_gib,'per_card_limit_GiB':16,
         'old_native_gradients_consumed':old_native,'ancestor_coarse_gradients':old_coarse,
         'ancestor_native_RF_only_evaluations':old_checks,'old_budgets_not_reset':True,
         'TV_weight':.001,'new_observation_generation':False,
         'physical_acceptance':'actual native joint RF and joint RF+TV both decrease',
         'joint_imaging_success_claimed':False}
    atomic(a.out/'run.json',run)
    monitor=PressureResources(a.out)
    pool=None
    stopped=[0]
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    for sig in [signal.SIGTERM,signal.SIGINT]:
        signal.signal(sig,lambda value,frame:stopped.__setitem__(0,value))
    count=accepted=0
    cache={}
    tau=metric=damp=None
    selected=None

    def freeze():
        for name,expected in hashes.items():
            if digest(Path(__file__).parent/name)!=expected:
                raise RuntimeError('Frozen source changed: '+name)
        if digest(anchor)!=plan['native_anchor_material_SHA256']:
            raise RuntimeError('Frozen native anchor changed')
        if digest(a.data/'index.json')!=plan['data_index_SHA256']:
            raise RuntimeError('Frozen low observation index changed')
        if a.high_index is not None and digest(a.high_index)!=plan['high_index_SHA256']:
            raise RuntimeError('Frozen high observation index changed')

    def model(u,rf,gdata):
        tv,gtv=co.regularization(u,.001)
        linear=float(np.sum(tau*u))
        penalty=.5*float(np.sum(damp*u*u))
        value=rf['loss']+tv+linear+penalty
        gradient=gdata+gtv+tau+damp*u
        record={'RF_model':rf,'native_TV':tv,'linear_first_order_correction':linear,
                'quadratic_model_damping':penalty,'computed_model_value':value,
                'not_a_native_RF_measurement':True}
        return {'lower':value,'upper':value,'gradient':(gradient/metric).reshape(-1),
                'line_search_objective_scale':rf['loss']+tv+penalty,
                'control':u.copy(),'record':record}

    def oracle(x):
        nonlocal count
        if stopped[0] or time.time()>=run['coarse_deadline_unix_s'] or count>=a.max_coarse_evaluations:
            raise BudgetBoundary('coarse_budget_or_stop_at_completed_evaluation')
        freeze()
        control=x.reshape(co.shape)/metric
        rf,gdata=pool.evaluate(control,'coarse_gradient',f'coarse{count+1:03d}')
        count+=1
        value=model(control,rf,gdata)
        run.update(status='optimizing_frequency_coarse_model',new_coarse_gradients=count)
        atomic(a.out/'evaluations'/f'coarse{count:03d}.json',value['record'])
        atomic(a.out/'run.json',run)
        return value

    def callback(x,value,point):
        nonlocal accepted
        accepted=point['accepted_step']
        cache.update(control=value['control'].copy(),record={**value['record'],**point})
        array(a.out/'accepted_coarse_control.npy',cache['control'])
        atomic(a.out/'accepted_coarse_point.json',cache['record'])
        pool.save_selected(value['record']['RF_model']['token'])
        run.update(coarse_accepted_steps=accepted,current_model_value=value['lower'],
                   current_coarse_corrected_RF_loss=value['record']['RF_model']['loss'])
        atomic(a.out/'run.json',run)
        emit({'stage':'coarse_accepted','accepted_steps':accepted,'coarse_gradients':count,
              'model_value':value['lower'],'corrected_coarse_RF_loss':value['record']['RF_model']['loss']})

    def maps(folder,label):
        with (folder/'maps.log').open('x') as log:
            subprocess.run([sys.executable,'-m','nsrno.evaluate_time_material',
                str(folder/'material.npy'),'--sample',SAMPLE,'--out',str(folder/'maps'),
                '--estimate-label',label],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True,timeout=300)

    try:
        torch.set_num_threads(1)
        pool=FrequencyMultilevelPool(a,monitor,plan,co)
        zero=np.zeros(co.shape,np.float64)
        run['status']='necessary_native_gradient_all_centres'
        atomic(a.out/'run.json',run)
        before_rf,fine_g=pool.evaluate(zero,'fine_gradient','necessary_native_anchor_gradient')
        tv0,gtv0=co.regularization(zero,.001)
        _,fine_gtv,_,_=co.fine.regularization(np.zeros_like(co.native_anchor),.001)
        fine_gradient=fine_g+fine_gtv
        array(a.out/'native_anchor_physical_gradient.npy',fine_gradient)
        reduced=co.transpose(fine_gradient)
        run.update(new_native_gradients=1,native_anchor_RF_loss=before_rf['loss'],
                   native_anchor_RF_groups=before_rf['groups'],initial_native_TV=tv0,
                   status='evaluating_actual_coarse_origin_all_centres')
        atomic(a.out/'run.json',run)
        rf0,gc0=pool.evaluate(zero,'coarse_gradient','coarse001_origin')
        count=1
        anchor_error=abs(rf0['loss']-before_rf['loss'])/max(before_rf['loss'],1e-30)
        if anchor_error>1e-10:
            raise RuntimeError('Fixed waveform correction did not retain the actual fine anchor')
        tau=reduced-(gc0+gtv0)
        rms=np.sqrt(np.mean(reduced[:,co.mask]**2,axis=1))
        diagonal=np.maximum(rms,max(float(rms.max())*1e-3,1e-20))/.2
        metric=np.broadcast_to(np.sqrt(diagonal)[:,None,None],co.shape).copy()
        metric[:,~co.mask]=1.
        damp=np.broadcast_to(.05*diagonal[:,None,None],co.shape)*co.mask
        trust=np.asarray([.04,.06,.35])[:,None,None]/SCALES
        low_bound=np.maximum(co.low,-trust)
        high_bound=np.minimum(co.high,trust)
        first=model(zero,rf0,gc0)
        identity=np.linalg.norm(first['gradient'].reshape(co.shape)*metric-reduced)/max(np.linalg.norm(reduced),1e-30)
        if identity>1e-10:
            raise RuntimeError('Initial computed gradient correction failed')
        run.update(status='optimizing_frequency_coarse_model',new_coarse_gradients=1,
                   initial_RF_anchor_relative_error=float(anchor_error),
                   initial_gradient_consistency_relative_error=float(identity),
                   field_metric_diagonal=diagonal.tolist(),
                   trust_region_log_c_Z_alpha=[.04,.06,.35],model_damping_relative=.05)
        atomic(a.out/'run.json',run)
        array(a.out/'first_order_gradient_correction.npy',tau)
        reused=[False]

        def first_oracle(x):
            if not reused[0] and not np.any(x):
                reused[0]=True
                return first
            return oracle(x)

        try:
            _,_,termination=solve(first_oracle,zero.reshape(-1),
                (low_bound*metric).reshape(-1),(high_bound*metric).reshape(-1),
                iterations=a.steps,history=8,maximum_lines=6,callback=callback,target=None)
        except BudgetBoundary as exc:
            termination={'stop':str(exc),'last_coarse_accepted_step':accepted}
        atomic(a.out/'coarse_termination.json',termination)
        if not cache:
            raise RuntimeError('No complete model point')
        before=before_rf['loss']+tv0
        for index,fraction in enumerate([1.,.5,.25],1):
            if stopped[0] or time.time()>=run['deadline_unix_s'] or not np.any(cache['control']):
                break
            freeze()
            control=fraction*cache['control']
            folder=a.out/'native_candidates'/f'candidate{index:02d}'
            folder.mkdir()
            array(folder/'material.npy',co.native_material(control))
            array(folder/'coarse_control.npy',control)
            os.link(anchor,folder/'stage0_initial_material.npy')
            run.update(status='actual_native_RF_check_all_centres',
                       active_native_candidate=index,active_control_fraction=fraction)
            atomic(a.out/'run.json',run)
            rf,_=pool.evaluate(control,'fine_forward',f'native_candidate{index:02d}')
            tv,_=co.regularization(control,.001)
            ok=(rf['loss']<before_rf['loss']-1e-6*max(before_rf['loss'],1e-20) and
                rf['loss']+tv<before-1e-8*max(before,1e-20))
            record={'fraction':fraction,'actual_native_RF':rf,
                    'native_RF_before':before_rf['loss'],'native_TV':tv,
                    'native_objective_before':before,'native_objective_after':rf['loss']+tv,
                    'accepted_using_actual_native_RF_only':ok,
                    'material_SHA256':digest(folder/'material.npy'),'truth_read':False}
            atomic(folder/'actual_native_acceptance.json',record)
            run.update(new_native_forward_checks=index,latest_native_RF_loss=rf['loss'])
            atomic(a.out/'run.json',run)
            emit({'stage':'native_RF_check','candidate':index,'fraction':fraction,
                  'before_RF':before_rf['loss'],'candidate_RF':rf['loss'],'accepted':ok})
            if ok:
                selected=(folder,record)
                maps(folder,'Joint 0.5 and 3 MHz, actual native RF accepted')
                break
        pool.close()
        pool=None
        final=a.out/'final'
        os.link(anchor,final/'stage0_initial_material.npy')
        if selected is None:
            os.link(anchor,final/'material.npy')
            for suffix in ['png','metrics.json']:
                shutil.copy2(a.anchor_root/'final'/('maps.'+suffix),final/('maps.'+suffix))
            final_rf=before_rf['loss']
            run['native_candidate_accepted']=False
        else:
            folder,record=selected
            os.link(folder/'material.npy',final/'material.npy')
            for suffix in ['png','metrics.json']:
                shutil.copy2(folder/('maps.'+suffix),final/('maps.'+suffix))
            atomic(final/'actual_native_acceptance.json',record)
            final_rf=record['actual_native_RF']['loss']
            run['native_candidate_accepted']=True
        run.update(status='frequency_multilevel_stage_finished',
                   finished_unix_s=time.time(),seconds=time.time()-started,
                   new_coarse_gradients=count,coarse_accepted_steps=accepted,
                   final_actual_native_RF_loss=final_rf,
                   material_SHA256=digest(final/'material.npy'))
        atomic(a.out/'run.json',run)
        emit({k:run[k] for k in ['status','native_candidate_accepted','new_native_gradients',
              'new_coarse_gradients','new_native_forward_checks','native_anchor_RF_loss',
              'final_actual_native_RF_loss','material_SHA256','seconds']})
    except BaseException:
        run.update(status='frequency_multilevel_interrupted_or_failed',
                   error=traceback.format_exc(),stopped_unix_s=time.time())
        atomic(a.out/'run.json',run)
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__=='__main__':
    main()
