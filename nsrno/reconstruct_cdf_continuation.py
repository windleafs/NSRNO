"""Bounded4low+4full actual native independent three-field reconstruction."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable THP')
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
from .cdf_continuation_pool import CDFContinuationPool
from .cdf_spatial_regularization import CDFSpatialRegularization
from .boxed_common_descent import common_box_direction
from .independent_physical_coordinates import IndependentPhysicalCoordinates
from .pressure_resources import PressureResources
from .low_frequency_0p5 import atomic,digest,SAMPLE,CASE
from .reconstruct_frequency_multilevel import array
from .reconstruct_native_frequency_incremental import same_process


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1])
    p.add_argument('--max-seconds',type=float,default=14400.)
    args=p.parse_args()
    project,source=Path.cwd().resolve(),Path(__file__).parent
    for key in ['inputs','out','artifact_root']:
        setattr(args,key,getattr(args,key).resolve())
    if (not args.inputs.is_relative_to(project/'out/v5') or not args.out.is_relative_to(project/'out/v5') or
        not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or args.out.exists() or
        args.artifact_root.exists() or args.devices!=[0,1] or args.max_seconds!=14400):
        raise ValueError('Fresh4low+4full,264adjoint,1100forward,14400s trajectory only')
    prepared=json.loads((args.inputs/'run.json').read_text())
    if prepared['status']!='actual_CDF_spatial_inverse_inputs_prepared' or same_process(prepared):
        raise RuntimeError('Actual CPUinput preparation must be finished and gone')
    parent=Path(prepared['parent_root'])
    old=json.loads((parent/'run.json').read_text())
    old_workers=json.loads((parent/'worker_processes.json').read_text())
    if old['status']!='learned_prior_saved_direction_native_stage_finished' or same_process(old) or any(
        Path('/proc',str(i)).exists() for i in old_workers['worker_pids']):
        raise RuntimeError('Predecessor must remain terminal;never duplicate a livejob')
    original_coordinate_receipt=json.loads((Path(old['parent_root'])/'run.json').read_text())
    if prepared['native_anchor_material_SHA256']!=original_coordinate_receipt['native_anchor_material_SHA256']:
        raise RuntimeError('Preserve the original native coordinate anchor identity')
    hashes={**old['source_SHA256'],**prepared['source_SHA256']}
    names=[Path(__file__).name,'cdf_continuation_pool.py','cdf_continuation_worker.py']
    hashes.update({name:digest(source/name) for name in names})
    for name,sha in hashes.items():
        if digest(source/name)!=sha:
            raise RuntimeError('Frozen input or executable changed: '+name)
    for path,sha in [(prepared['initial_material_path'],prepared['initial_material_SHA256']),
        (prepared['gradient_path'],prepared['gradient_SHA256']),
        (prepared['neural_path'],prepared['neural_SHA256']),(prepared['cdf_path'],prepared['cdf_SHA256'])]:
        if digest(path)!=sha:
            raise RuntimeError('Fixed initial point/prior product changed')
    acceptance=json.loads((parent/'final/actual_native_acceptance.json').read_text())
    plan=json.loads((parent/'native_plan.json').read_text())
    plan['objectives']=json.loads((args.inputs/'objective_specifications.json').read_text())
    if len(plan['objectives']['low_background']['sources'])!=22 or len(plan['objectives']['full_joint']['sources'])!=44:
        raise RuntimeError('Declared original-source sets changed')
    initial_refs=acceptance['RF_source_records']
    if len(initial_refs)!=44:
        raise RuntimeError('All44 cached starting predictions required')
    for row in initial_refs:
        if digest(row['path'])!=row['SHA256']:
            raise RuntimeError('Starting same-material RF cache changed')
    edge_path=project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    for path,sha in [(Path(plan['native_anchor_material']),prepared['native_anchor_material_SHA256']),
        (edge_path,plan['known_edge_SHA256']),(Path(plan['data_root'])/'index.json',plan['data_index_SHA256']),
        (Path(plan['high_index']),plan['high_index_SHA256'])]:
        if digest(path)!=sha:
            raise RuntimeError('Original acquisition/coordinate inputs changed')
    reserve=2*2**30+400*2**20
    if shutil.disk_usage(project).free<reserve+3*2**20 or shutil.disk_usage(Path.home()).free<reserve+3*2**30:
        raise RuntimeError('Retain old reserves and3GiB allowance for the full declared trajectory')
    args.out.mkdir()
    args.artifact_root.mkdir()
    for name in ['arrays','workers','steps','final','trajectory_final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name,target_is_directory=True)
    (args.out/'jobs').mkdir()
    for name in hashes:
        shutil.copy2(source/name,args.out/name)
    atomic(args.out/'native_plan.json',plan)
    for name,path in [('initial_material.npy',prepared['initial_material_path']),
        ('initial_control.npy',prepared['initial_control_path']),('initial_prior_gradient.npy',prepared['gradient_path'])]:
        (args.out/'arrays'/name).symlink_to(Path(path).resolve())
    started=time.time()
    deadline=started+14400
    run={'status':'initializing_CDF_native_frequency_continuation','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s':started,'deadline_unix_s':deadline,'max_seconds':14400,
        'source_SHA256':hashes,'parent_root':str(parent),'inputs_root':str(args.inputs),
        'sample':SAMPLE,'case':CASE,'initial_material_SHA256':prepared['initial_material_SHA256'],
        'native_anchor_material_SHA256':prepared['native_anchor_material_SHA256'],
        'neural_prior_SHA256':prepared['neural_SHA256'],'CDF_prior_SHA256':prepared['cdf_SHA256'],
        'new_reserved_equivalent_units':66,'previous_cumulative_reserved_units':366,
        'cumulative_reserved_from_closed_predecessor':432,'not_whole_thread_actual_cost':True,
        'all_closed_budgets_retained':True,'closed_parent_created_unix_s':old['created_unix_s'],
        'closed_parent_deadline_unix_s':old['deadline_unix_s'],
        'maximum_outer_updates':8,'low_updates':4,'full_updates':4,
        'maximum_new_adjoint_passes':264,'maximum_new_gradient_background_forwards':264,
        'maximum_new_candidate_forwards':792,'maximum_final_score_forwards':44,
        'maximum_new_physical_forwards':1100,'maximum_new_tangent_passes':0,
        'maximum_candidates_per_point':3,'maximum_low22_candidates':12,'maximum_full44_candidates':12,
        'initial_low_RF':prepared['initial_low_RF'],'initial_full_RF':prepared['initial_full_RF'],
        'initial_regularization':prepared['initial_regularization'],'current_regularization':prepared['initial_regularization'],
        'current_objective':'low_background','current_objective_RF':prepared['initial_low_RF'],
        'current_full_RF_if_known':prepared['initial_full_RF'],'best_full_RF':prepared['initial_full_RF'],
        'best_full_material_SHA256':prepared['initial_material_SHA256'],
        'source_count_in_current_objective':22,'active_point':0,'accepted_updates':0,
        'accepted_low_updates':0,'accepted_full_updates':0,'candidate_checks_issued':0,
        'candidate_checks_completed':0,'final_full_score_checks':0,'accepted_steps':[],
        'patient_truth_read_by_optimizer':False,'intermediate_truth_feedback':False,
        'new_patient_derivative_probes':0,'new_data_or_frequency':False,'reserved_case_unread':'Neg_47_Left',
        'no_material_lowpass':True,'independent_native_fields':True,'one_GPU_experiment':True,
        'devices':args.devices,'per_card_process_tree_limit_GiB':16,'allocator_cap_GiB':10,
        'joint_reconstruction_success_claimed':False}
    atomic(args.out/'run.json',run)
    def stop(number,frame):
        (args.out/'stop_requested.flag').write_text(str(number)+'\n')
    for number in [signal.SIGINT,signal.SIGTERM]:
        signal.signal(number,stop)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    monitor,pool=PressureResources(args.out),None
    def checkpoint():
        if pool is not None:
            run['actual_native_work']=pool.actual_work()
        atomic(args.out/'run.json',run)
    def freeze(final=False,inflight=False):
        if any(digest(source/name)!=sha for name,sha in hashes.items()):
            raise RuntimeError('Source changed during frozen actual trajectory')
        cutoff=60 if final or inflight else 600
        if time.time()>=deadline-cutoff or (args.out/'stop_requested.flag').exists():
            raise RuntimeError('Absolute trajectory deadline or cooperative stop reached')
        if shutil.disk_usage(project).free<reserve or shutil.disk_usage(Path.home()).free<reserve:
            raise RuntimeError('Existing project/HOME disk reserve reached')
    def memory_admission():
        available=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))/2**20
        bank=((36001+127)//128)*7*961*1121*4/2**30
        if available<2*bank*len(args.devices)+24:
            raise RuntimeError('Keep existing checkpoint banks and24GiB host headroom')
    try:
        coordinates=IndependentPhysicalCoordinates(np.load(plan['native_anchor_material']),np.load(edge_path))
        control=np.load(prepared['initial_control_path']).astype(np.float64)
        material=coordinates.material(control)
        if not np.array_equal(material,np.load(prepared['initial_material_path'])):
            raise RuntimeError('Original coordinates do not reproduce the exact initial estimate')
        prior=CDFSpatialRegularization(prepared['neural_path'],prepared['cdf_path'])
        R,gR,parts=prepared['initial_regularization'],np.load(prepared['gradient_path']),prepared['initial_prior_parts']
        z,x=np.ogrid[:961,:1121]
        active=(z>=80)&(z<937)&(x>=24)&(x<1097)
        lateral=abs((np.arange(1121)-560)*.05)<=19.1
        current_material_path=args.out/'arrays/initial_material.npy'
        current_control_path=args.out/'arrays/initial_control.npy'
        current_full_RF=prepared['initial_full_RF']
        current_full_refs=initial_refs
        current_RF=prepared['initial_low_RF']
        previous_objective='low_background'
        best={'loss':prepared['initial_full_RF'],'TV':parts['TV'],
            'material_path':str(current_material_path.resolve()),'control_path':str(current_control_path.resolve()),
            'material_SHA256':prepared['initial_material_SHA256'],'RF_source_records':initial_refs,'prior':R}
        def update_best(value,references):
            nonlocal best
            threshold=8*np.finfo(np.float32).eps*max(abs(best['loss']),1e-12)
            if value<best['loss']-threshold and value+parts['TV']<best['loss']+best['TV']:
                best={'loss':value,'TV':parts['TV'],'material_path':str(current_material_path.resolve()),
                    'control_path':str(current_control_path.resolve()),'material_SHA256':digest(current_material_path),
                    'RF_source_records':references,'prior':R}
                atomic(args.out/'best_full_RF_incumbent.json',best)
                run.update(best_full_RF=value,best_full_material_SHA256=best['material_SHA256'])
        atomic(args.out/'best_full_RF_incumbent.json',best)
        freeze()
        memory_admission()
        pool=CDFContinuationPool(args,monitor,plan,deadline,freeze)
        stop_reason='declared_four_low_and_four_full_updates_finished'
        for point in range(1,9):
            objective='low_background' if point<=4 else 'full_joint'
            count=22 if point<=4 else 44
            if deadline-time.time()<count*55/2+600:
                stop_reason='remaining_time_reserved_for_final_full_RF_score'
                break
            freeze()
            memory_admission()
            folder=args.out/'steps'/f'point{point:02d}'
            folder.mkdir()
            (folder/'start_material.npy').symlink_to(current_material_path.resolve())
            (folder/'start_control.npy').symlink_to(current_control_path.resolve())
            if point>1:
                R,gR,parts=prior.evaluate(coordinates,control)
            array(folder/'prior_gradient.npy',gR)
            run.update(status='actual_native_gradient_for_CDF_continuation',active_point=point,
                current_objective=objective,source_count_in_current_objective=count,
                current_regularization=R,current_material_SHA256=digest(current_material_path),
                current_objective_RF=current_RF if objective==previous_objective else current_full_RF)
            checkpoint()
            gradient=pool.evaluate(material,'gradient',objective,point)
            expected=current_RF if objective==previous_objective else current_full_RF
            allowance=8*np.finfo(np.float32).eps*max(abs(gradient['loss']),1e-12)
            difference=0. if expected is None else gradient['loss']-expected
            atomic(folder/'necessary_gradient_RF_comparison.json',{'expected_RF':expected,
                'required_gradient_RF':gradient['loss'],'difference_if_cached':difference,
                'allowance':float(allowance),'new_objective_needs_new_gradient':objective!=previous_objective,
                'not_a_rigorous_error_bound':True})
            if expected is not None and abs(difference)>allowance:
                raise RuntimeError('Inspect completed required-gradient baseline discrepancy;do not repeat')
            current_RF=gradient['loss']
            if objective=='full_joint':
                current_full_RF=current_RF
                current_full_refs=gradient['RF_source_records']
                update_best(current_full_RF,current_full_refs)
            previous_objective=objective
            run.update(current_objective_RF=current_RF,current_full_RF_if_known=current_full_RF,
                status='CPU_CDF_and_neural_common_direction')
            checkpoint()
            direction,details=common_box_direction(coordinates,control,np.load(gradient['gradient_path']),gR,active,lateral)
            array(folder/'direction.npy',direction)
            atomic(folder/'direction_details.json',{**details,'objective':objective,
                'data_gradient_SHA256':gradient['gradient_SHA256'],'start_material_SHA256':digest(current_material_path)})
            if not details['actual_common_step_available']:
                stop_reason='no_common_descent_with_composite_global_spatial_prior'
                break
            fraction=min(1.,.25*current_RF/(-details['RF_slope']))
            minimum_drop=max(10*abs(difference),float(allowance))
            accepted=False
            for attempt in range(1,4):
                feasible_prior=False
                for backtrack in range(8):
                    freeze()
                    proposal=control+fraction*direction
                    if np.max(abs(coordinates.project(proposal,active)-proposal))>1e-9:
                        raise RuntimeError('Projected direction left original physical domain')
                    candidate_R,_,candidate_parts=prior.evaluate(coordinates,proposal,gradient=False)
                    target_R=R+1e-4*fraction*details['prior_slope']
                    if candidate_R<=target_R:
                        feasible_prior=True
                        break
                    atomic(folder/f'prior_skip_a{attempt:02d}_b{backtrack:02d}.json',{
                        'fraction':fraction,'prior':candidate_R,'prior_Armijo_target':target_R,'no_wave_issued':True})
                    fraction*=.5
                if not feasible_prior:
                    break
                candidate=folder/f'candidate{attempt:02d}'
                candidate.mkdir()
                array(candidate/'control.npy',proposal)
                candidate_material=coordinates.material(proposal)
                array(candidate/'material.npy',candidate_material)
                run.update(status='actual_native_CDF_continuation_candidate',active_attempt=attempt,
                    active_fraction=fraction,active_regularization=candidate_R)
                run['candidate_checks_issued']+=1
                checkpoint()
                trial=pool.evaluate(candidate_material,'forward',objective,point)
                run['candidate_checks_completed']+=1
                accepted=bool(trial['loss']<current_RF-minimum_drop and
                    trial['loss']<=current_RF+1e-4*fraction*details['RF_slope'] and
                    trial['loss']+candidate_parts['TV']<current_RF+parts['TV'] and candidate_R<=target_R)
                curvature=2*(trial['loss']-current_RF-fraction*details['RF_slope'])/(fraction*fraction)
                atomic(candidate/'actual_native_acceptance.json',{'accepted':accepted,'objective':objective,
                    'point':point,'fraction':fraction,'RF_before':current_RF,'RF_after':trial['loss'],
                    'prior_before':R,'prior_after':candidate_R,'parts_after':candidate_parts,
                    'measured_directional_secant_curvature':curvature,'minimum_RF_drop':minimum_drop,
                    'material_SHA256':digest(candidate/'material.npy'),'patient_truth_read':False,
                    'low_acceptance_is_not_full44_improvement':objective=='low_background'})
                print(json.dumps({'point':point,'objective':objective,'attempt':attempt,'fraction':fraction,
                    'RF':trial['loss'],'prior':candidate_R,'accepted':accepted}),flush=True)
                checkpoint()
                if not accepted:
                    next_fraction=.5*fraction
                    if curvature>0:
                        next_fraction=min(next_fraction,.9*(-details['RF_slope'])/curvature)
                    fraction=next_fraction
                    continue
                control,material=proposal,candidate_material
                current_RF,R,parts=trial['loss'],candidate_R,candidate_parts
                current_material_path,current_control_path=candidate/'material.npy',candidate/'control.npy'
                if objective=='full_joint':
                    current_full_RF,current_full_refs=current_RF,trial['RF_source_records']
                    update_best(current_full_RF,current_full_refs)
                else:
                    current_full_RF,current_full_refs=None,None
                run['accepted_updates']+=1
                run['accepted_low_updates' if objective=='low_background' else 'accepted_full_updates']+=1
                run['accepted_steps'].append({'point':point,'objective':objective,'RF':current_RF,
                    'regularization':R,'fraction':fraction,'material_SHA256':digest(current_material_path)})
                run.update(current_objective_RF=current_RF,current_regularization=R,
                    current_full_RF_if_known=current_full_RF,current_material_SHA256=digest(current_material_path))
                atomic(folder/'accepted_source_cache.json',{'RF_source_records':trial['RF_source_records'],
                    'objective':objective,'source_count':count,'material_SHA256':digest(current_material_path),
                    'actual_native_RF':current_RF,'prior':R,'parts':parts})
                checkpoint()
                break
            if not accepted:
                stop_reason='no_current_objective_RF_and_prior_candidate_within_three_real_lines'
                break
        if current_full_RF is None:
            run['status']='mandatory_final_full44_score_of_fixed_low_candidate'
            run['final_full_score_checks']+=1
            checkpoint()
            final_score=pool.evaluate(material,'forward','full_joint',0,final_score=True)
            current_full_RF,current_full_refs=final_score['loss'],final_score['RF_source_records']
            update_best(current_full_RF,current_full_refs)
        pool.close()
        pool=None
        trajectory=args.out/'trajectory_final'
        array(trajectory/'control.npy',control)
        array(trajectory/'material.npy',material)
        atomic(trajectory/'actual_native_acceptance.json',{'accepted_updates':run['accepted_updates'],
            'actual_native_RF':{'loss':current_full_RF,'all44_sources':True},
            'RF_source_records':current_full_refs,'native_TV':parts['TV'],'independent_prior_value':R,
            'material_SHA256':digest(trajectory/'material.npy'),
            'full_RF_improved_from_initial':current_full_RF<prepared['initial_full_RF'],
            'not_a_joint_imaging_success_claim':True})
        final=args.out/'final'
        array(final/'control.npy',np.load(best['control_path']))
        array(final/'material.npy',np.load(best['material_path']))
        atomic(final/'actual_native_acceptance.json',{'actual_native_RF':{'loss':best['loss'],'all44_sources':True},
            'RF_source_records':best['RF_source_records'],'material_SHA256':best['material_SHA256'],
            'native_TV':best['TV'],'independent_prior_value':best['prior'],
            'full_original44_source_acceptance':True,'joint_imaging_success_claimed':False})
        changed=digest(trajectory/'material.npy')!=prepared['initial_material_SHA256']
        if changed:
            (trajectory/'stage0_initial_material.npy').symlink_to(Path(prepared['initial_material_path']).resolve())
            with (trajectory/'maps.log').open('x') as log:
                subprocess.run([sys.executable,'-m','nsrno.evaluate_time_material',str(trajectory/'material.npy'),
                    '--sample',SAMPLE,'--out',str(trajectory/'maps'),
                    '--estimate-label','Independent global CDF + frozen spatial prior'],
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        else:
            for suffix in ['png','metrics.json']:
                shutil.copy2(parent/f'final/maps.{suffix}',trajectory/f'maps.{suffix}')
        best_is_trajectory=digest(final/'material.npy')==digest(trajectory/'material.npy')
        for suffix in ['png','metrics.json']:
            shutil.copy2((trajectory if best_is_trajectory else parent/'final')/f'maps.{suffix}',final/f'maps.{suffix}')
        atomic(args.out/'maps_provenance.json',{'trajectory_new_fixed_material_evaluation':changed,
            'final_same_as_trajectory':best_is_trajectory,'final_else_same_as_original_incumbent':not best_is_trajectory,
            'no_intermediate_truth_feedback':True})
        run.update(status='CDF_native_frequency_continuation_finished',stop_reason=stop_reason,
            trajectory_final_full_RF=current_full_RF,trajectory_final_regularization=R,
            trajectory_material_SHA256=digest(trajectory/'material.npy'),
            final_RF=best['loss'],material_SHA256=digest(final/'material.npy'),
            globally_full_RF_improved=best['material_SHA256']!=prepared['initial_material_SHA256'],
            seconds=time.time()-started,finished_unix_s=time.time())
        checkpoint()
        print(json.dumps({key:run[key] for key in ['status','stop_reason','accepted_updates',
            'accepted_low_updates','accepted_full_updates','trajectory_final_full_RF','final_RF',
            'globally_full_RF_improved','material_SHA256','seconds']}),flush=True)
    except BaseException:
        run.update(status='CDF_native_frequency_continuation_failed_for_inspection',
            error=traceback.format_exc(),seconds=time.time()-started,all_issued_work_retained=True)
        checkpoint()
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__=='__main__':
    main()
