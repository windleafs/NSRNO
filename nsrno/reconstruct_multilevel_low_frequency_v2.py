"""Staged first-order corrected multilevel low-frequency three-parameter reconstruction.

The model is exact at a saved actual fine-RF anchor and its computed gradient
is matched to the saved fine gradient. Every proposed final material is accepted
only after a real full-native RF evaluation. No truth enters optimization.
"""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0) != 0:
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
from .multilevel_low_coordinates import MultilevelCoordinates
from .multilevel_low_pool import MultilevelPool
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--anchor-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--band-hz',type=float,nargs=2,default=[.2e6,.325e6])
    p.add_argument('--split-hz',type=float,nargs='*',default=[])
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--max-coarse-evaluations',type=int,default=28)
    p.add_argument('--max-seconds',type=float,default=1800)
    p.add_argument('--coarse-seconds',type=float,default=1200)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    a=p.parse_args()
    if a.out.exists():
        raise RuntimeError('Inspect the existing phase; never restart or overwrite implicitly')
    if not (1<=a.steps<=32 and 1<=a.max_coarse_evaluations<=50 and
            0<a.coarse_seconds<=a.max_seconds-420 and a.max_seconds<=3600 and
            0<a.cap_gib<=10):
        raise ValueError('Exceeds the explicit candidate budget or native RF reserve')
    if not a.devices or len(a.devices)!=len(set(a.devices)):
        raise ValueError('Distinct source-worker devices required')
    prior=json.loads((a.anchor_root/'run.json').read_text())
    fine_parent=prior['status']=='single_case_low_frequency_reconstruction_finished'
    multilevel_parent=prior['status']=='corrected_multilevel_low_frequency_stage_finished'
    if not fine_parent and not multilevel_parent:
        raise RuntimeError('Complete an actual fine or multilevel anchor phase first')
    if multilevel_parent and not prior.get('native_candidate_accepted'):
        raise RuntimeError('Inspect a rejected coarse proposal before choosing another phase')
    if prior.get('sample')!=SAMPLE or prior.get('case')!=CASE:
        raise ValueError('Only the authorized development case is admitted')
    if Path('/proc',str(prior['pid'])).exists():
        raise RuntimeError('The native anchor coordinator is still live')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                '--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Another GPU experiment is still live')
    dataset=json.loads((a.data/'index.json').read_text())
    if dataset['status']!='paired_0p5MHz_generation_finished' or dataset['sample']!=SAMPLE:
        raise ValueError('Use only the actual authorized development observation')
    if not .1e6<=a.band_hz[0]<a.band_hz[1]<=1.25e6:
        raise ValueError('Frequency band outside the actual retained acquisition')
    edges=[a.band_hz[0],*a.split_hz,a.band_hz[1]]
    if any(x>=y for x,y in zip(edges[:-1],edges[1:])):
        raise ValueError('Objective subband edges must increase')
    bands=list(zip(edges[:-1],edges[1:]))
    parent_bands=prior.get('balanced_subbands_hz',[prior['band_hz']])
    same_objective=np.array_equal(np.asarray(parent_bands),np.asarray(bands))
    # A previous coarse stage's saved gradient belongs to its OLD anchor,
    # not its accepted native candidate. Always refresh at a new material.
    reuse_gradient=fine_parent and not a.split_hz and same_objective
    parent_native_gradients=(prior['completed_native_gradients'] if fine_parent else
        prior.get('old_native_gradients_consumed',0)+prior.get('new_native_gradients',0))
    parent_coarse_gradients=(prior.get('ancestor_coarse_gradients',0)+
                             prior.get('new_coarse_gradients',0))
    parent_native_RF_only=(prior.get('ancestor_native_RF_only_evaluations',0)+
                          prior.get('new_native_forward_checks',0))
    anchor=a.anchor_root/'final/material.npy'
    if digest(anchor)!=prior['material_SHA256']:
        raise ValueError('Actual native anchor changed')
    if fine_parent:
        point=json.loads((a.anchor_root/'final/accepted.json').read_text())
        if point['accepted_step']!=prior['accepted_steps']:
            raise ValueError('Anchor objective and final material are inconsistent')
        parent_steps=prior['accepted_steps']
        RF_archive='RF'
        gradient_path=a.anchor_root/'accepted_gradient.npy'
        gradient_hash=digest(gradient_path)
        diagonal=np.asarray(prior['global_coordinate_metric_diagonal'])
    else:
        accepted=json.loads((a.anchor_root/'final/actual_native_acceptance.json').read_text())
        if not accepted['accepted_using_actual_native_RF_only'] or accepted['material_SHA256']!=prior['material_SHA256']:
            raise ValueError('Require the actual physically accepted native candidate')
        point={'native_RF':accepted['actual_native_RF'],'RF_loss':accepted['actual_native_RF']['loss']}
        parent_steps=prior['coarse_accepted_steps']
        RF_archive='fine_RF'
        gradient_path=gradient_hash=diagonal=None
    g=None
    if reuse_gradient:
        g=np.load(gradient_path)*np.sqrt(diagonal)[:,None,None]
        g[:,:80]=0.
        if digest(gradient_path)!=gradient_hash:
            raise RuntimeError('Native gradient changed while being read')
    edge=np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    co=MultilevelCoordinates(np.load(anchor),edge)
    native_reduced_gradient=co.transpose(g) if reuse_gradient else None
    reserve=2*2**30+400*2**20
    if shutil.disk_usage('.').free<reserve+160*2**20:
        raise RuntimeError('Insufficient material and true-RF checkpoint reserve')
    a.out=a.out.resolve()
    a.data=a.data.resolve()
    a.anchor_root=a.anchor_root.resolve()
    a.out.mkdir()
    for n in ['jobs','evaluations','native_candidates','final']:
        (a.out/n).mkdir()
    RF=[]
    token=point['native_RF']['token']
    expected_loss=0.
    for family in ['lf','point3']:
        records=sorted([r for r in dataset['shots'] if r['family']==family and r['kind']==SAMPLE],
                       key=lambda r:r['shot'])
        frequency=np.asarray(dataset['acquisition']['retained_frequency_hz'])
        keep=(frequency>=a.band_hz[0])&(frequency<=a.band_hz[1])
        band_masks=[(frequency>=lo)&(frequency<=hi) for lo,hi in bands]
        error=np.zeros(len(bands),np.float64)
        energy=np.zeros(len(bands),np.float64)
        for shot,row in enumerate(records):
            candidates=list((a.anchor_root/'workers').glob(
                f'rank*/{RF_archive}/token{token:04d}/{family}_{shot:02d}.npy'))
            if len(candidates)!=1:
                raise ValueError('Expected exactly one actual fine anchor RF per shot')
            path=candidates[0]
            prediction=np.load(path)
            if digest(row['path'])!=row['SHA256']:
                raise ValueError('Observed RF changed')
            target=np.load(row['path'])
            for j,mask_band in enumerate(band_masks):
                error[j]+=float(np.sum(abs(prediction[:,mask_band]-target[:,mask_band])**2))
                energy[j]+=float(np.sum(abs(target[:,mask_band])**2))
            RF.append({'family':family,'shot':shot,'path':str(path.resolve()),
                       'SHA256':digest(path)})
        expected_loss+=.25*float(np.mean(error/energy))
    if same_objective and abs(expected_loss-point['RF_loss'])>1e-8*max(expected_loss,1e-20):
        raise RuntimeError('Cached actual waveform loss no longer matches the native anchor')
    plan={'data_root':str(a.data),'data_index_SHA256':digest(a.data/'index.json'),
          'native_anchor_material':str(anchor.resolve()),
          'native_anchor_material_SHA256':digest(anchor),
          'native_anchor_RF':RF,'band_hz':a.band_hz,'bands_hz':bands,
          'native_anchor_RF_loss':expected_loss,
          'parent_native_phase':str(a.anchor_root),
          'parent_native_complete_gradients':parent_native_gradients,
          'parent_coarse_gradients':parent_coarse_gradients,
          'parent_native_RF_only_evaluations':parent_native_RF_only,
          'parent_accepted_steps':parent_steps,
          'parent_step_kind':'fine_physical_steps' if fine_parent else 'coarse_model_steps_plus_native_acceptance',
          'parent_scaled_gradient_SHA256':gradient_hash,
          'parent_global_metric_diagonal':diagonal.tolist() if diagonal is not None else None,
          'cached_native_gradient_includes_TV':True,
          'native_gradient_reused_without_a_new_material_derivative':reuse_gradient,
          'changed_objective_requires_one_necessary_actual_native_gradient':not reuse_gradient}
    atomic(a.out/'anchor_plan.json',plan)
    if reuse_gradient:
        array(a.out/'native_anchor_physical_gradient.npy',g)
    hashes=source_manifest()
    for n in [Path(__file__).name,'multilevel_low_worker.py','multilevel_low_pool.py',
              'multilevel_low_coordinates.py','multilevel_low_acquisition.py',
              'acquisition_single_case_0p5.py','low_frequency_0p5.py',
              'matrix_free_rf_response.py','bounded_model_lbfgs.py']:
        hashes[n]=digest(Path(__file__).parent/n)
    for n in hashes:
        shutil.copy2(Path(__file__).parent/n,a.out/n)
    started=time.time()
    run={'status':'starting_corrected_multilevel_low_frequency_inverse',
         'pid':os.getpid(),
         'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
         'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         'created_unix_s':started,'deadline_unix_s':started+a.max_seconds,
         'coarse_deadline_unix_s':started+a.coarse_seconds,
         'maximum_coarse_gradients':a.max_coarse_evaluations,
         'maximum_coarse_accepted_steps':a.steps,'maximum_native_forward_checks':3,
         'new_native_gradients_planned':0 if reuse_gradient else 1,
         'new_native_gradients':0,
         'sample':SAMPLE,'case':CASE,'band_hz':a.band_hz,'balanced_subbands_hz':bands,
         'objective':'equal geometry and subband relative complex RF error, plus native TV',
         'native_anchor_material_SHA256':plan['native_anchor_material_SHA256'],
         'native_anchor_RF_loss':expected_loss,'all_three_fields_independent':True,
         'final_material_is_full_native':True,'coarse_corrections_are_only_initialization':True,
         'unknowns':'independent c, rho, positive tissue alpha',
         'native_physical_box_enforced_by_control_support_bounds':True,
         'truth_read_by_optimizer':False,'reserved_case_unread':'Neg_47_Left',
         'devices':a.devices,'allocator_cap_GiB':a.cap_gib,'per_card_limit_GiB':16,
         'source_SHA256':hashes,'coarse_info':co.info(),
         'gradient_consistency':'computed initial model gradient equals the reduced actual fine gradient in this objective',
         'coarse_RF_is_approximate_at_other_materials':True,
         'physical_acceptance':'actual native RF and native RF+TV both decrease',
         'old_native_phase_budget_not_reset':True,
         'old_native_gradients_consumed':parent_native_gradients,
         'ancestor_coarse_gradients':parent_coarse_gradients,
         'ancestor_native_RF_only_evaluations':parent_native_RF_only,
         'new_coarse_gradients':0,'new_native_forward_checks':0,
         'coarse_accepted_steps':0,'joint_imaging_success_claimed':False}
    atomic(a.out/'run.json',run)
    monitor=PressureResources(a.out)
    pool=None
    stopped=[0]
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    for sig in [signal.SIGTERM,signal.SIGINT]:
        signal.signal(sig,lambda value,frame:stopped.__setitem__(0,value))
    coarse_count=accepted=0
    cache={}
    tau=metric=damp=None
    selected=None

    def freeze():
        for name,expected in hashes.items():
            if digest(Path(__file__).parent/name)!=expected:
                raise RuntimeError('Frozen source changed: '+name)
        if digest(anchor)!=plan['native_anchor_material_SHA256']:
            raise RuntimeError('Locked native anchor changed')
        if digest(a.data/'index.json')!=plan['data_index_SHA256']:
            raise RuntimeError('Frozen observation index changed')

    def model_value(u,rf,data_gradient):
        tv,gtv=co.regularization(u,.001)
        linear=float(np.sum(tau*u))
        penalty=.5*float(np.sum(damp*u*u))
        value=rf['loss']+tv+linear+penalty
        gradient=data_gradient+gtv+tau+damp*u
        record={'RF_model':rf,'native_TV':tv,'linear_first_order_correction':linear,
                'quadratic_model_damping':penalty,'computed_model_value':value,
                'model_value_not_a_native_RF_measurement':True}
        return {'lower':value,'upper':value,
                'line_search_objective_scale':rf['loss']+tv+penalty,
                'gradient':(gradient/metric).reshape(-1),
                'control':u.copy(),'record':record}

    def oracle(x):
        nonlocal coarse_count
        if stopped[0]:
            raise BudgetBoundary('stop_requested_at_completed_model_evaluation')
        if coarse_count>=a.max_coarse_evaluations or time.time()>=run['coarse_deadline_unix_s']:
            raise BudgetBoundary('coarse_work_or_time_budget')
        freeze()
        u=x.reshape(co.shape)/metric
        rf,gc=pool.evaluate(u,'coarse_gradient',f'coarse{coarse_count+1:03d}')
        coarse_count+=1
        value=model_value(u,rf,gc)
        run.update(status='optimizing_corrected_coarse_model',new_coarse_gradients=coarse_count)
        atomic(a.out/'evaluations'/f'coarse{coarse_count:03d}.json',value['record'])
        atomic(a.out/'run.json',run)
        return value

    def callback(x,value,point_value):
        nonlocal accepted
        accepted=point_value['accepted_step']
        cache.update(control=value['control'].copy(),
                     record={**value['record'],**point_value})
        array(a.out/'accepted_coarse_control.npy',cache['control'])
        atomic(a.out/'accepted_coarse_point.json',cache['record'])
        pool.save_selected(value['record']['RF_model']['token'])
        run.update(coarse_accepted_steps=accepted,
                   current_model_value=value['lower'],
                   current_coarse_corrected_RF_loss=value['record']['RF_model']['loss'])
        atomic(a.out/'run.json',run)
        emit({'stage':'coarse_accepted','accepted_steps':accepted,
              'coarse_gradients':coarse_count,'model_value':value['lower'],
              'corrected_coarse_RF_loss':value['record']['RF_model']['loss']})

    def maps(folder,label):
        with (folder/'maps.log').open('x') as log:
            subprocess.run([sys.executable,'-m','nsrno.evaluate_time_material',
                str(folder/'material.npy'),'--sample',SAMPLE,'--out',str(folder/'maps'),
                '--estimate-label',label],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True,timeout=300)

    try:
        torch.set_num_threads(1)
        pool=MultilevelPool(a,monitor,plan,co)
        zero=np.zeros(co.shape,np.float64)
        tv0,gtv0=co.regularization(zero,.001)
        if not reuse_gradient:
            run.update(status='necessary_actual_native_gradient_for_changed_objective')
            atomic(a.out/'run.json',run)
            fine_rf,fine_g=pool.evaluate(zero,'fine_gradient','necessary_native_anchor_gradient')
            if abs(fine_rf['loss']-expected_loss)>1e-6*max(expected_loss,1e-20):
                raise RuntimeError('Changed-objective native RF does not match its actual saved anchor')
            _,native_gtv,_,_=co.fine.regularization(np.zeros_like(co.native_anchor),.001)
            g=fine_g+native_gtv
            native_reduced_gradient=co.transpose(g)
            array(a.out/'native_anchor_physical_gradient.npy',g)
            run.update(new_native_gradients=1)
            atomic(a.out/'run.json',run)
        run.update(status='evaluating_actual_coarse_model_origin')
        atomic(a.out/'run.json',run)
        rf0,gc0=pool.evaluate(zero,'coarse_gradient','coarse001_origin')
        coarse_count=1
        if abs(rf0['loss']-expected_loss)>1e-8*max(expected_loss,1e-20):
            raise RuntimeError('The fixed RF correction did not reproduce the actual native anchor')
        tau=native_reduced_gradient-(gc0+gtv0)
        rms=np.sqrt(np.mean(native_reduced_gradient[:,co.mask]**2,axis=1))
        diagonal=np.maximum(rms,max(float(rms.max())*1e-3,1e-20))/.2
        metric=np.broadcast_to(np.sqrt(diagonal)[:,None,None],co.shape).copy()
        metric[:,~co.mask]=1.
        damp=np.broadcast_to(.05*diagonal[:,None,None],co.shape)*co.mask
        trust=np.asarray([.04,.06,.35])[:,None,None]/SCALES
        low=np.maximum(co.low,-trust)
        high=np.minimum(co.high,trust)
        first=model_value(zero,rf0,gc0)
        identity=np.linalg.norm(first['gradient'].reshape(co.shape)*metric-native_reduced_gradient)
        identity/=max(np.linalg.norm(native_reduced_gradient),1e-30)
        if identity>1e-10:
            raise RuntimeError('Computed first-order gradient correction is inconsistent')
        run.update(status='optimizing_corrected_coarse_model',new_coarse_gradients=1,
                   initial_actual_native_RF_loss=expected_loss,initial_native_TV=tv0,
                   initial_computed_gradient_consistency_relative_error=float(identity),
                   initial_coarse_RF_anchor_relative_error=abs(rf0['loss']-expected_loss)/max(expected_loss,1e-30),
                   field_metric_diagonal=diagonal.tolist(),
                   trust_region_log_c_Z_alpha=[.04,.06,.35],
                   model_damping_relative=.05)
        atomic(a.out/'run.json',run)
        array(a.out/'first_order_gradient_correction.npy',tau)
        reused=[False]

        def reused_oracle(x):
            if not reused[0] and not np.any(x):
                reused[0]=True
                return first
            return oracle(x)

        try:
            _,_,termination=solve(reused_oracle,zero.reshape(-1),
                (low*metric).reshape(-1),(high*metric).reshape(-1),
                iterations=a.steps,history=8,maximum_lines=6,callback=callback,target=None)
        except BudgetBoundary as exc:
            termination={'stop':str(exc),'last_coarse_accepted_step':accepted}
        atomic(a.out/'coarse_termination.json',termination)
        if not cache:
            raise RuntimeError('No complete accepted model origin')
        before=expected_loss+tv0
        for index,fraction in enumerate([1.,.5,.25],1):
            if stopped[0] or time.time()>=run['deadline_unix_s'] or not np.any(cache['control']):
                break
            freeze()
            control=fraction*cache['control']
            folder=a.out/'native_candidates'/f'candidate{index:02d}'
            folder.mkdir()
            array(folder/'material.npy',co.native_material(control))
            os.link(anchor,folder/'stage0_initial_material.npy')
            array(folder/'coarse_control.npy',control)
            run.update(status='actual_native_RF_check',
                       active_native_candidate=index,active_control_fraction=fraction)
            atomic(a.out/'run.json',run)
            rf,_=pool.evaluate(control,'fine_forward',f'native_candidate{index:02d}')
            tv,_=co.regularization(control,.001)
            value=rf['loss']+tv
            ok=(rf['loss']<expected_loss-1e-6*max(expected_loss,1e-20) and
                value<before-1e-8*max(before,1e-20))
            record={'fraction':fraction,'actual_native_RF':rf,
                    'native_RF_before':expected_loss,'native_TV':tv,
                    'native_objective_before':before,'native_objective_after':value,
                    'accepted_using_actual_native_RF_only':ok,
                    'truth_read':False,'material_SHA256':digest(folder/'material.npy')}
            atomic(folder/'actual_native_acceptance.json',record)
            run.update(new_native_forward_checks=index,
                       latest_native_RF_loss=rf['loss'])
            atomic(a.out/'run.json',run)
            emit({'stage':'native_RF_check','candidate':index,'fraction':fraction,
                  'before_RF':expected_loss,'candidate_RF':rf['loss'],'accepted':ok})
            if ok:
                selected=(folder,record)
                maps(folder,'First-order multilevel model, accepted actual native RF')
                break
        pool.close()
        pool=None
        final=a.out/'final'
        os.link(anchor,final/'stage0_initial_material.npy')
        if selected is None:
            os.link(anchor,final/'material.npy')
            final_rf=expected_loss
            run['native_candidate_accepted']=False
            for suffix in ['png','metrics.json']:
                shutil.copy2(a.anchor_root/'final'/('maps.'+suffix),final/('maps.'+suffix))
            run['unchanged_anchor_maps_reused_from_identical_material']=str(a.anchor_root/'final')
        else:
            folder,record=selected
            os.link(folder/'material.npy',final/'material.npy')
            for suffix in ['png','metrics.json']:
                shutil.copy2(folder/('maps.'+suffix),final/('maps.'+suffix))
            atomic(final/'actual_native_acceptance.json',record)
            final_rf=record['actual_native_RF']['loss']
            run['native_candidate_accepted']=True
        run.update(status='corrected_multilevel_low_frequency_stage_finished',
                   finished_unix_s=time.time(),seconds=time.time()-started,
                   new_coarse_gradients=coarse_count,coarse_accepted_steps=accepted,
                   final_actual_native_RF_loss=final_rf,
                   material_SHA256=digest(final/'material.npy'))
        atomic(a.out/'run.json',run)
        emit({k:run[k] for k in ['status','native_candidate_accepted','new_coarse_gradients',
              'new_native_forward_checks','native_anchor_RF_loss',
              'final_actual_native_RF_loss','material_SHA256','seconds']})
    except BaseException:
        run.update(status='multilevel_stage_interrupted_or_failed',
                   error=traceback.format_exc(),stopped_unix_s=time.time())
        atomic(a.out/'run.json',run)
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__=='__main__':
    main()
