"""Coverage revision: add three current joint images, reserve later images."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import copy
import gc
import json
import os
import resource
import time
import traceback
import numpy as np
import torch
from .native_inverse_operator import SpatialSpectralInverse,binned_inverse_initialization,quality,coarse_balance,structural_check
from .native_gn_recycled_state import RecycledState,digest
from .native_gn_spatial_state import NativeSpatialState
from .native_gn_json import atomic


def collect(state):
    diagonal=state.diagonal.copy();root=np.sqrt(diagonal).reshape(-1)
    X=state.span.basis*root[None];Y=state.span.images/root[None]
    norms=np.linalg.norm(X,axis=1);X/=norms[:,None];Y/=norms[:,None]
    paths=list(state.paths)
    provenance=[{'path':str(p),'direction_SHA256':digest(p/'direction.npy'),
        'total_image_SHA256':digest(p/'total_image.npy')} for p in paths]
    return {'X':X,'Y':Y,'diagonal':diagonal,'paths':paths,'provenance':provenance,
        'mask':state.co.mask.copy(),'shape':state.material.shape}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=150)
    p.add_argument('--max-seconds',type=float,default=600)
    a=p.parse_args()
    if a.out.exists():
        raise RuntimeError('Preserve existing CPU model')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Explicit CPU-only environment required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(1732);rng=np.random.default_rng(1732)
    a.out.mkdir()
    source_names=['train_native_inverse_operator_v2.py','native_inverse_operator.py','native_gn_recycled_state.py',
        'native_gn_spatial_state.py','native_gn_coordinates.py','native_gn_galerkin.py','native_span_qp.py',
        'physical_box_direction.py','native_gn_json.py']
    hashes={n:digest(Path(__file__).parent/n) for n in source_names}
    for name in hashes:(a.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    criteria={'minimum_noncoarse_current_reference_holdouts':2,
        'each_group_median_balanced_cosine_at_least':.8,'each_group_minimum_balanced_cosine_at_least':0.,
        'each_group_mean_cosine_gain_over_best_baseline_at_least':.10,
        'holdout_groups':['old_reference_eight_new_Krylov_directions','current_reference_new_spatial_directions'],
        'no_target_rescaling_or_refitting_on_holdouts':True,
        'passing_only_supports_a_subsequent_native_trial_not_material_recovery':True}
    run={'status':'CPU_loading_saved_inverse_training_pairs','pid':os.getpid(),'created_unix_s':time.time(),
        'source_sha256':hashes,'criteria_fixed_before_fitting':criteria,'steps_budget':a.steps,
        'training_time_budget_seconds':a.max_seconds,'GPU_work':False,'new_wave_solves':0,
        'truth_read':False,'reserved_test_case_unread':'Neg_47_Left','joint_imaging_success_claimed':False,
        'revision':'current_reference_cross_material_coverage',
        'prior_v1_tested_first_three_spatial_images_now_used_as_development_training':True,
        'new_current_holdouts_are_later_spatial_images_not_the_reused_v1_tests':True}
    atomic(a.out/'run.json',run);started=time.time()
    try:
        algebra=structural_check();atomic(a.out/'operator_structure_check.json',algebra)
        state=RecycledState(Path('out/v5/reconstruction_native_gn_enriched_v1_train_200'))
        old=collect(state);old_reference=digest(state.origin/'old_material.npy')
        del state;gc.collect()
        state=NativeSpatialState(Path('out/v5/reconstruction_native_impedance_bootstrap_resume1_v1_train_200'))
        native_root=Path('out/v5/reconstruction_native_spatial_block_krylov_v1_train_200')
        native_run=json.loads((native_root/'run.json').read_text())
        snapshot=(native_root/'step001/fixed_HVP_manifest.json').read_bytes()
        current_prefix=[Path(v) for v in json.loads(snapshot)['paths']]
        if native_run['fixed_reference_SHA256']!=state.fixed_SHA256 or len(current_prefix)<9 or current_prefix[:6]!=state.paths:
            raise RuntimeError('Require the first three complete spatial images at the same reference')
        for path in current_prefix[6:9]:
            if not (path/'progress.json').exists():
                raise RuntimeError('Current training image is incomplete')
            direction=np.load(path/'direction.npy');image=np.load(path/'total_image.npy')
            data_image=np.load(path/'data_image.npy')
            expected=data_image+state.h_tv(direction)+state.damping*direction
            if np.linalg.norm(expected-image)>1e-10*max(np.linalg.norm(image),1e-300):
                raise RuntimeError('Additional current training image differs from native operator')
            state.span.append(direction,image);state.paths.append(path)
        (a.out/'current_training_prefix_snapshot.json').write_bytes(snapshot)
        current=collect(state);new_reference=state.fixed_SHA256
        del state;gc.collect()
        shape=old['shape']
        if shape!=current['shape'] or not np.array_equal(old['mask'],current['mask']):
            raise RuntimeError('Physical coordinate masks differ')
        if len(old['paths'])!=15 or len(current['paths'])!=9:
            raise RuntimeError('Use fifteen old and nine current known images')
        X=np.concatenate([old['X'],current['X']],axis=0).reshape(-1,*shape).astype(np.float32)
        Y=np.concatenate([old['Y'],current['Y']],axis=0).reshape(-1,*shape).astype(np.float32)
        atomic(a.out/'training_provenance.json',{'old_reference_SHA256':old_reference,
            'current_reference_SHA256':new_reference,'old_training_pairs':old['provenance'],
            'current_training_pairs':current['provenance'],
            'training_pairs_are_same_case_at_two_reference_materials_not_two_patients':True,
            'data_are_GN_plus_TV_and_damping_images_not_material_truth':True})
        run.update(status='CPU_binned_positive_inverse_initialization',training_pairs=len(X))
        atomic(a.out/'run.json',run)
        chol,gain,diagonal_gain,initialization=binned_inverse_initialization(X,Y,
            callback=lambda i,n:atomic(a.out/'progress.json',{'phase':'binned_inverse_fit','completed':i,'total':n}))
        model=SpatialSpectralInverse(old['mask'],gain);model.initialize_cholesky(chol)
        stationary=copy.deepcopy(model).eval()
        prior={name:p.detach().clone() for name,p in model.named_parameters()}
        torch.save(stationary.state_dict(),a.out/'stationary_initial.pt')
        optimizer=torch.optim.Adam(model.parameters(),lr=.02)
        tx=torch.from_numpy(X);ty=torch.from_numpy(Y)
        run.update(status='CPU_fitting_positive_nonlocal_inverse')
        atomic(a.out/'run.json',run);training_started=time.time();history=[]
        for step in range(a.steps):
            indices=rng.choice(len(X),2,replace=False)
            optimizer.zero_grad(set_to_none=True)
            output=model(ty[indices]);fit=(output-tx[indices]).square().flatten(1).sum(1).mean()
            regularization=torch.zeros(())
            for name,param in model.named_parameters():
                change=param-prior[name];scale=1+prior[name].square().mean()
                regularization=regularization+change.square().mean()/scale
                regularization=regularization+(change[...,1:,:]-change[...,:-1,:]).square().mean()/scale
                regularization=regularization+(change[...,:,1:]-change[...,:,:-1]).square().mean()/scale
            loss=fit+1e-4*regularization
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite CPU inverse fit')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
            row={'step':step+1,'minibatch_relative_squared_fit':float(fit.detach()),
                'regularization':float(regularization.detach()),'training_seconds':time.time()-training_started}
            history.append(row)
            if (step+1)%10==0:
                atomic(a.out/'progress.json',{'phase':'CPU_inverse_training',**row})
                print(json.dumps(row),flush=True)
            if time.time()-training_started>=a.max_seconds:
                break
        model.eval();model.requires_grad_(False);training_elapsed=time.time()-training_started
        for name,expected in hashes.items():
            if digest(Path(__file__).parent/name)!=expected:
                raise RuntimeError('CPU training code changed')
        torch.save(model.state_dict(),a.out/'weights.pt')
        weights_sha=digest(a.out/'weights.pt')
        frozen_at=time.time()
        atomic(a.out/'weight_freeze.json',{'weights_SHA256':weights_sha,'frozen_unix_s':frozen_at,
            'holdout_target_arrays_not_read_before_freeze':True,'steps':len(history)})
        atomic(a.out/'training_history.json',{'history':history,'initialization':initialization})
        def apply_model(which,value):
            with torch.no_grad():
                tensor=torch.from_numpy(np.asarray(value,np.float32))
                return which(tensor).numpy().astype(np.float64)
        train_metrics={}
        for name,which in [('stationary_initial',stationary),('learned_spatial_spectral',model)]:
            errors=[];cosines=[]
            for index in range(len(X)):
                value=apply_model(which,Y[index])
                row=quality(value,X[index]);errors.append(row['relative_error']);cosines.append(row['cosine'])
            train_metrics[name]={'mean_relative_error':float(np.mean(errors)),'median_cosine':float(np.median(cosines)),
                'old_reference_mean_relative_error':float(np.mean(errors[:15])),
                'current_reference_mean_relative_error':float(np.mean(errors[15:])),
                'old_reference_median_cosine':float(np.median(cosines[:15])),
                'current_reference_median_cosine':float(np.median(cosines[15:]))}
        atomic(a.out/'training_metrics.json',train_metrics)
        run.update(status='CPU_frozen_weights_waiting_existing_native_terminal',weights_SHA256=weights_sha,
            completed_steps=len(history),training_seconds=training_elapsed,
            waiting_on_existing_native_root=str(native_root.resolve()))
        atomic(a.out/'run.json',run)
        while True:
            observed=json.loads((native_root/'run.json').read_text())
            if observed['status'] in ['native_spatial_Krylov_finished','failed']:
                break
            if not Path('/proc',str(observed['pid'])).exists():
                break
            time.sleep(5)
        old_manifest=Path('out/v5/reconstruction_native_recycled_krylov_v1_train_200/step001/fixed_HVP_manifest.json')
        old_bytes=old_manifest.read_bytes();old_paths=[Path(v) for v in json.loads(old_bytes)['paths']]
        if old_paths[:15]!=old['paths'] or len(old_paths)!=23:
            raise RuntimeError('Old test manifest changed')
        native_root=Path('out/v5/reconstruction_native_spatial_block_krylov_v1_train_200')
        current_bytes=(native_root/'step001/fixed_HVP_manifest.json').read_bytes()
        current_paths=[Path(v) for v in json.loads(current_bytes)['paths']]
        if current_paths[:9]!=current['paths']:
            raise RuntimeError('Current tests use different coarse images')
        (a.out/'old_manifest_after_weight_freeze.json').write_bytes(old_bytes)
        (a.out/'current_manifest_after_weight_freeze.json').write_bytes(current_bytes)
        run.update(status='CPU_frozen_inverse_holdout_evaluation',weights_SHA256=weights_sha,
            completed_steps=len(history),training_seconds=training_elapsed)
        atomic(a.out/'run.json',run)
        groups=[('old_reference_eight_new_Krylov_directions',old,old_paths[15:]),
            ('current_reference_new_spatial_directions',current,current_paths[9:])]
        all_results={};group_summary={}
        for group,data,paths in groups:
            if any(path in data['paths'] for path in paths):
                raise RuntimeError('Coarse training vector leaked into holdouts')
            coarseX=data['X'];coarseY=data['Y'];gram=coarseX@coarseY.T;gram=.5*(gram+gram.T)
            if np.linalg.eigvalsh(gram)[0]<=0:
                raise RuntimeError('Known coarse curvature not positive')
            methods={'training_scalar_diagonal':lambda value:diagonal_gain*value,
                'stationary_inverse_fit':lambda value:apply_model(stationary,value),
                'learned_spatial_spectral':lambda value:apply_model(model,value)}
            rows=[];sqrtD=np.sqrt(data['diagonal'])
            for path in paths:
                direction=np.load(path/'direction.npy');image=np.load(path/'total_image.npy')
                target=sqrtD*direction;norm=np.linalg.norm(target);target/=norm
                rhs=image/sqrtD/norm;details={}
                for name,apply in methods.items():
                    base=apply(rhs)
                    balanced=coarse_balance(rhs,apply,coarseX,coarseY,gram)
                    details[name]={'base_inverse':quality(base,target),'fixed_true_coarse_balanced_inverse':quality(balanced,target)}
                rows.append({'path':str(path),'direction_SHA256':digest(path/'direction.npy'),
                    'total_image_SHA256':digest(path/'total_image.npy'),'models':details})
                atomic(a.out/'progress.json',{'phase':'frozen_holdouts','group':group,'completed':len(rows),'total':len(paths)})
            all_results[group]=rows
            stats={}
            for name in methods:
                values=[row['models'][name]['fixed_true_coarse_balanced_inverse']['cosine'] for row in rows]
                stats[name]={'mean_cosine':float(np.mean(values)) if values else None,
                    'median_cosine':float(np.median(values)) if values else None,
                    'minimum_cosine':float(np.min(values)) if values else None}
            enough=len(paths)>=(8 if group.startswith('old_') else 2)
            learned=stats['learned_spatial_spectral']
            advantage=(learned['mean_cosine']-max(stats[name]['mean_cosine']
                for name in ['training_scalar_diagonal','stationary_inverse_fit'])) if paths else None
            passed=bool(enough and learned['median_cosine']>=.8 and learned['minimum_cosine']>=0 and advantage>=.10)
            group_summary[group]={'count':len(paths),'models':stats,'mean_cosine_advantage':advantage,'criteria_passed':passed,
                'coarse_images_used_for_balance':len(data['paths'])}
        sufficient=all(row['criteria_passed'] for row in group_summary.values())
        if digest(a.out/'weights.pt')!=weights_sha:
            raise RuntimeError('Frozen inverse weights changed during evaluation')
        report={'weights_SHA256':weights_sha,'weight_freeze_unix_s':frozen_at,'initialization':initialization,
            'training_metrics':train_metrics,'heldout_groups':all_results,'group_summary':group_summary,
            'criteria_fixed_before_fitting':criteria,'supports_one_future_native_trial':sufficient,
            'not_material_reconstruction_success':True,'no_new_wave_solves':True,'GPU_work':False,
            'known_coarse_targets_excluded_from_holdout_scores':True,
            'same_frozen_coarse_basis_within_each_group':True,
            'later_test_directions_were_generated_with_larger_bases_not_fully_replayed':True,
            'seconds':time.time()-started,'peak_CPU_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        atomic(a.out/'metrics.json',report)
        run.update(status='CPU_inverse_operator_fit_and_holdouts_finished',finished_unix_s=time.time(),
            supports_one_future_native_trial=sufficient,seconds=time.time()-started)
        atomic(a.out/'run.json',run)
        print(json.dumps({'status':run['status'],'training_metrics':train_metrics,'group_summary':group_summary,
            'supports_one_future_native_trial':sufficient,'peak_CPU_RSS_GiB':report['peak_CPU_RSS_GiB']}),flush=True)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(a.out/'run.json',run);raise


if __name__=='__main__':
    main()
