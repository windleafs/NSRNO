"""Two actual all44 independent-field updates with a fixed TRAIN spatial prior."""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
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
from .projected_common_pool import ProjectedCommonPool
from .boxed_common_descent import common_box_direction
from .learned_spatial_energy import LearnedSpatialEnergy
from .independent_physical_coordinates import IndependentPhysicalCoordinates
from .pressure_resources import PressureResources
from .low_frequency_0p5 import atomic, digest, SAMPLE, CASE
from .reconstruct_frequency_multilevel import array
from .reconstruct_native_frequency_incremental import same_process


class TwoUpdatePool(ProjectedCommonPool):
    def actual_work(self):
        totals = super().actual_work()
        if (totals.get('issued_adjoint_direction_passes',0) > 88 or
            totals.get('issued_physical_forward_passes',0) > 440 or
            totals.get('issued_tangent_direction_passes',0) != 0):
            raise RuntimeError('Two-update actual native-work ceiling exceeded')
        return totals

    def evaluate(self, material, kind, step):
        if step not in [1,2] or (kind == 'original44_forward' and self.forward_count >= 8):
            raise RuntimeError('Do not exceed the separately declared2update/8candidate budget')
        return super().evaluate(material,kind,step)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-root',type=Path,required=True)
    p.add_argument('--fit-root',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1])
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--cap-gib',type=float,default=10.)
    p.add_argument('--max-seconds',type=float,default=5400.)
    args = p.parse_args()
    project,source = Path.cwd().resolve(),Path(__file__).parent
    for name in ['parent_root','fit_root','qualification','out','artifact_root']:
        setattr(args,name,getattr(args,name).resolve())
    if (not all(getattr(args,n).is_relative_to(project/'out/v5') for n in ['parent_root','fit_root','out']) or
        not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or
        args.out.exists() or args.artifact_root.exists() or args.devices != [0,1] or
        args.chunk != 128 or args.cap_gib != 10 or args.max_seconds != 5400):
        raise ValueError('Only fresh2update/88adjoint/8candidate/5400s actual trajectory allowed')
    parent = json.loads((args.parent_root/'run.json').read_text())
    old_workers = json.loads((args.parent_root/'worker_processes.json').read_text())
    if (parent['status'] != 'projected_common_descent_native_stage_finished' or
        parent['accepted_updates'] != 3 or parent['original44_checks_completed'] != 3 or same_process(parent) or
        any(Path('/proc',str(pid)).exists() for pid in old_workers['worker_pids'])):
        raise RuntimeError('Previous three-update budget must be finished and all processes gone')
    fitted = json.loads((args.fit_root/'run.json').read_text())
    model_path = (args.fit_root/'products/prior.pt').resolve()
    if (fitted['status'] != 'independent_TRAIN_material_denoisers_training_finished' or
        fitted['completed_steps'] != 3000 or same_process(fitted) or
        fitted['development_arrays_read'] or fitted['reserved_arrays_read'] or
        digest(model_path) != fitted['prior_SHA256']):
        raise RuntimeError('Require the exact finished frozen TRAIN-only model')
    holdout = json.loads((args.fit_root/'holdout_denoising.json').read_text())
    if any(value >= 1 for value in holdout['all_field_scale_average_MSE_ratios'].values()):
        raise RuntimeError('The frozen prior did not learn denoising; do not spend native inverse budget')
    qualification = json.loads(args.qualification.read_text())
    if (qualification['status'] != 'physical_step_box_projection_and_common_descent_synthetic_CPU_passed' or
        qualification['source_SHA256'] != digest(source/'boxed_common_descent.py')):
        raise RuntimeError('New trust-domain projection needs its completed synthetic qualification')
    for name,sha in parent['source_SHA256'].items():
        if digest(source/name) != sha:
            raise RuntimeError('Previously used frozen source changed: '+name)
    if digest(source/'independent_denoising_prior.py') != fitted['source_SHA256']['independent_denoising_prior.py']:
        raise RuntimeError('Fixed prior architecture changed after training')
    initial = (args.parent_root/'final/material.npy').resolve()
    acceptance = json.loads((args.parent_root/'final/actual_native_acceptance.json').read_text())
    if digest(initial) != parent['material_SHA256'] or acceptance['material_SHA256'] != digest(initial):
        raise RuntimeError('Starting accepted material changed')
    RF0 = acceptance['actual_native_RF']['loss']
    if RF0 != parent['final_RF']:
        raise RuntimeError('Wrong physical RF baseline')
    plan = json.loads((args.parent_root/'native_plan.json').read_text())
    entries = [(family,shot) for family in plan['families'] for shot in range(11)]
    references = acceptance['RF_source_records']
    if sorted((r['family'],r['shot']) for r in references) != sorted(entries):
        raise RuntimeError('All44 baseline RF sources required')
    for row in references:
        if digest(row['path']) != row['SHA256']:
            raise RuntimeError('Accepted RF cache changed')
    plan.update(reference_RF=references,reference_material_SHA256=digest(initial),
        learned_spatial_prior=True,cached_old_point_derivatives_not_reused=True)
    edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    for path,sha in [(Path(plan['native_anchor_material']),parent['native_anchor_material_SHA256']),
        (edge_path,plan['known_edge_SHA256']),(Path(plan['data_root'])/'index.json',plan['data_index_SHA256']),
        (Path(plan['high_index']),plan['high_index_SHA256'])]:
        if digest(path) != sha:
            raise RuntimeError('Fixed physical acquisition or coordinate map changed')
    reserve = 2*2**30+400*2**20
    if shutil.disk_usage(project).free < reserve+2*2**20 or shutil.disk_usage(Path.home()).free < reserve+1600*2**20:
        raise RuntimeError('Retain project/HOME reserves for this trajectory')
    names = [Path(__file__).name,'boxed_common_descent.py','learned_spatial_energy.py','independent_denoising_prior.py']
    hashes = {**parent['source_SHA256'],**{n:digest(source/n) for n in names}}
    args.out.mkdir()
    args.artifact_root.mkdir()
    for name in ['arrays','workers','steps','final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name,target_is_directory=True)
    (args.out/'jobs').mkdir()
    for name in hashes:
        shutil.copy2(source/name,args.out/name)
    atomic(args.out/'native_plan.json',plan)
    started = time.time()
    deadline = started+5400
    run = {'status':'initializing_actual_learned_spatial_prior_trajectory','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s':started,'deadline_unix_s':deadline,'max_seconds':5400,
        'source_SHA256':hashes,'parent_root':str(args.parent_root),'fit_root':str(args.fit_root),
        'sample':SAMPLE,'case':CASE,'native_anchor_material_SHA256':parent['native_anchor_material_SHA256'],
        'initial_material_SHA256':digest(initial),'initial_RF':RF0,'current_RF':RF0,
        'learned_prior_SHA256':digest(model_path),'qualification_SHA256':digest(args.qualification),
        'maximum_outer_updates':2,'maximum_new_adjoint_passes':88,
        'maximum_new_gradient_background_forwards':88,'maximum_new_candidate_forwards':352,
        'maximum_new_tangent_passes':0,'maximum_full44_candidates':8,
        'new_reserved_equivalent_units':22,
        'closed_projected_common_budget':{'reserved_units':33,'actual_adjoint_passes':132,
            'actual_full44_checks':3,'created_unix_s':parent['created_unix_s'],
            'deadline_unix_s':parent['deadline_unix_s'],'retained_unchanged':True},
        'closed_spatial_prior_budget':parent['closed_spatial_prior_budget'],
        'closed_value_prior_budget':parent['closed_value_prior_budget'],
        'closed_full_dual_budget':parent['closed_full_dual_budget'],
        'closed_predecessor_budget':parent['closed_predecessor_budget'],
        'cumulative_reserved_units_from_closed_predecessor':89+134+55+33+33+22,
        'reservation_not_actual_thread_total':True,'closed_prior_training_steps':3000,
        'new_data_or_frequency':False,'new_patient_derivative_probes':0,
        'patient_truth_read_by_optimizer':False,'intermediate_truth_feedback':False,
        'reserved_case_unread':'Neg_47_Left','independent_native_material_fields':True,
        'no_material_lowpass':True,'shared_material_labels':False,
        'physical_log_step_box':[.02,.04,.12],'line_fractions':[1.,.5,.25,.125],
        'all11_sources_per_family':True,'source_encoding':False,'one_GPU_experiment':True,
        'devices':args.devices,'per_card_process_tree_limit_GiB':16,'allocator_cap_GiB':10,
        'original44_checks_issued':0,'original44_checks_completed':0,'accepted_updates':0,
        'accepted_steps':[],'active_step':0,'joint_reconstruction_success_claimed':False,
        'new_objective_neural_energy_plus_TV_not_comparable_to_old_prior_values':True}
    atomic(args.out/'run.json',run)
    def stop(number,frame):
        (args.out/'stop_requested.flag').write_text(str(number)+'\n')
    for number in [signal.SIGINT,signal.SIGTERM]:
        signal.signal(number,stop)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    monitor,pool = PressureResources(args.out),None
    def checkpoint():
        if pool is not None:
            run['actual_native_work'] = pool.actual_work()
        atomic(args.out/'run.json',run)
    def freeze():
        if any(digest(source/n) != sha for n,sha in hashes.items()):
            raise RuntimeError('Active frozen source changed')
        if time.time() >= deadline-60 or (args.out/'stop_requested.flag').exists():
            raise RuntimeError('Declared deadline or cooperative stop reached')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Existing disk reserve reached')
    def memory_admission():
        available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
            if s.startswith('MemAvailable:')))/2**20
        bank = ((36001+127)//128)*7*961*1121*4/2**30
        if available < 2*bank*len(args.devices)+24:
            raise RuntimeError('Host memory cannot retain existing checkpoint/headroom requirement')
    try:
        freeze()
        coordinates = IndependentPhysicalCoordinates(np.load(plan['native_anchor_material']),np.load(edge_path))
        control = np.load(args.parent_root/'final/control.npy').astype(np.float64)
        material = coordinates.material(control)
        if not np.array_equal(material,np.load(initial)):
            raise RuntimeError('Coordinate map does not exactly reproduce fixed accepted starting material')
        array(args.out/'arrays/initial_control.npy',control)
        array(args.out/'arrays/initial_material.npy',material)
        z,x = np.ogrid[:961,:1121]
        active = (z >= 80)&(z < 937)&(x >= 24)&(x < 1097)
        lateral = abs((np.arange(1121)-560)*.05) <= 19.1
        prior = LearnedSpatialEnergy(model_path)
        R,gR,parts = prior.evaluate(coordinates,control)
        run.update(initial_regularization=R,current_regularization=R,initial_prior_parts=parts,
            status='admitting_native_workers_for_frozen_learned_prior')
        checkpoint()
        memory_admission()
        pool = TwoUpdatePool(args,monitor,plan,deadline)
        current_RF = RF0
        current_material_path = args.out/'arrays/initial_material.npy'
        current_control_path = args.out/'arrays/initial_control.npy'
        stop_reason = 'declared_two_actual_updates_finished'
        for step in [1,2]:
            freeze()
            memory_admission()
            folder = args.out/'steps'/f'step{step:02d}'
            folder.mkdir()
            (folder/'start_material.npy').symlink_to(current_material_path.resolve())
            (folder/'start_control.npy').symlink_to(current_control_path.resolve())
            if step > 1:
                R,gR,parts = prior.evaluate(coordinates,control)
            array(folder/'prior_gradient.npy',gR)
            run.update(status='actual_all44_gradient_with_frozen_learned_prior',active_step=step,
                current_RF=current_RF,current_regularization=R,current_material_SHA256=digest(current_material_path))
            checkpoint()
            gradient = pool.evaluate(material,'original44_gradient',step)
            difference = gradient['loss']-current_RF
            allowance = 8*float(np.finfo(np.float32).eps)*max(abs(current_RF),1e-12)
            atomic(folder/'same_material_RF_comparison.json',{'cached_RF':current_RF,
                'required_gradient_RF':gradient['loss'],'difference':difference,
                'eight_relative_float32_eps':allowance,'numerical_guard_not_rigorous_error_bound':True})
            if abs(difference) > allowance:
                raise RuntimeError('Inspect existing required-gradient RF discrepancy; do not repeat it')
            run['status'] = 'CPU_frozen_neural_prior_boxed_common_direction'
            checkpoint()
            direction,details = common_box_direction(coordinates,control,np.load(gradient['gradient_path']),gR,active,lateral)
            array(folder/'direction.npy',direction)
            atomic(folder/'direction_details.json',{**details,'data_gradient_SHA256':gradient['gradient_SHA256'],
                'start_material_SHA256':digest(current_material_path)})
            if not details['actual_common_step_available']:
                stop_reason = 'no_common_descent_with_frozen_learned_prior'
                break
            accepted = False
            minimum_drop = max(10*abs(difference),allowance)
            for line,fraction in enumerate([1.,.5,.25,.125],1):
                freeze()
                proposal = control+fraction*direction
                if np.max(abs(coordinates.project(proposal,active)-proposal)) > 1e-9:
                    raise RuntimeError('Trust direction left original physical domain')
                candidate_R,_,candidate_parts = prior.evaluate(coordinates,proposal,gradient=False)
                target_R = R+1e-4*fraction*details['prior_slope']
                if candidate_R > target_R:
                    atomic(folder/f'skipped_line{line:02d}.json',{'fraction':fraction,
                        'reason':'frozen_neural_prior_Armijo_not_met','prior':candidate_R,'required_prior':target_R})
                    continue
                candidate = folder/f'line{line:02d}'
                candidate.mkdir()
                array(candidate/'control.npy',proposal)
                candidate_material = coordinates.material(proposal)
                array(candidate/'material.npy',candidate_material)
                run.update(status='actual_all44_frozen_neural_prior_candidate',active_line=line,
                    active_fraction=fraction,active_regularization=candidate_R)
                run['original44_checks_issued'] += 1
                checkpoint()
                trial = pool.evaluate(candidate_material,'original44_forward',step)
                run['original44_checks_completed'] += 1
                target_RF = current_RF+1e-4*fraction*details['RF_slope']
                accepted = bool(trial['loss'] < current_RF-minimum_drop and trial['loss'] <= target_RF and
                    trial['loss']+candidate_parts['TV'] < current_RF+parts['TV'] and candidate_R <= target_R)
                record = {'accepted':accepted,'outer_step':step,'line':line,'fraction':fraction,
                    'RF_before':current_RF,'RF_after':trial['loss'],'RF_Armijo_target':target_RF,
                    'prior_before':R,'prior_after':candidate_R,'prior_Armijo_target':target_R,
                    'minimum_RF_drop':minimum_drop,'parts_before':parts,'parts_after':candidate_parts,
                    'material_SHA256':digest(candidate/'material.npy'),'all44_true_sources':True,
                    'patient_truth_read_by_optimizer':False,'joint_reconstruction_success_claimed':False}
                atomic(candidate/'actual_native_acceptance.json',record)
                print(json.dumps({'step':step,'line':line,'RF':trial['loss'],'prior':candidate_R,'accepted':accepted}),flush=True)
                checkpoint()
                if not accepted:
                    continue
                references = []
                for index,(family,shot) in enumerate(entries):
                    path = args.out/'workers'/f'rank{index%2:02d}'/'arrays'/f'token{trial["token"]:04d}'/f'{family}_{shot:02d}.npy'
                    references.append({'family':family,'shot':shot,'path':str(path.resolve()),'SHA256':digest(path)})
                current_RF,R,parts = trial['loss'],candidate_R,candidate_parts
                control,material = proposal,candidate_material
                current_material_path,current_control_path = candidate/'material.npy',candidate/'control.npy'
                pool.mark_accepted(step)
                run['accepted_updates'] += 1
                run['accepted_steps'].append({'outer_step':step,'candidate':str(candidate),
                    'material_SHA256':record['material_SHA256'],'RF':current_RF,'regularization':R,'fraction':fraction})
                run.update(current_RF=current_RF,current_regularization=R,current_material_SHA256=record['material_SHA256'])
                atomic(folder/'accepted_RF_cache.json',{'RF_source_records':references,'material_SHA256':record['material_SHA256'],
                    'actual_native_RF':trial,'native_TV':parts['TV']})
                checkpoint()
                break
            if not accepted:
                stop_reason = 'no_actual_RF_and_frozen_prior_candidate_in_declared_four_lines'
                break
        pool.close()
        pool = None
        final = args.out/'final'
        array(final/'control.npy',control)
        array(final/'material.npy',material)
        atomic(final/'actual_native_acceptance.json',{'accepted_updates':run['accepted_updates'],
            'actual_native_RF':{'loss':current_RF,'all44_sources':True,'source_encoding':False},
            'RF_source_records':references,'native_TV':parts['TV'],'independent_prior_value':R,
            'material_SHA256':digest(final/'material.npy'),'full_original44_source_acceptance':True,
            'joint_imaging_success_claimed':False})
        if run['accepted_updates']:
            (final/'stage0_initial_material.npy').symlink_to(initial)
            with (final/'maps.log').open('x') as log:
                subprocess.run([sys.executable,'-m','nsrno.evaluate_time_material',str(final/'material.npy'),
                    '--sample',SAMPLE,'--out',str(final/'maps'),'--estimate-label','Frozen TRAIN neural spatial prior'],
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        else:
            for suffix in ['png','metrics.json']:
                shutil.copy2(args.parent_root/f'final/maps.{suffix}',final/f'maps.{suffix}')
        atomic(final/'maps_provenance.json',{'new_fixed_material_evaluation':run['accepted_updates'] > 0,
            'same_historical_material_if_no_update':str(args.parent_root/'final'),
            'if_no_update_initial_column_is_historical':run['accepted_updates'] == 0,
            'optimizer_used_no_intermediate_truth_feedback':True})
        run.update(status='learned_spatial_prior_native_stage_finished',stop_reason=stop_reason,
            final_RF=current_RF,final_regularization=R,material_SHA256=digest(final/'material.npy'),
            seconds=time.time()-started,finished_unix_s=time.time())
        checkpoint()
        print(json.dumps({k:run[k] for k in ['status','stop_reason','accepted_updates','final_RF',
            'final_regularization','original44_checks_completed','material_SHA256','seconds']}),flush=True)
    except BaseException:
        run.update(status='learned_spatial_prior_native_stage_failed_for_inspection',
            error=traceback.format_exc(),seconds=time.time()-started,all_issued_work_remains_charged=True)
        checkpoint()
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__ == '__main__':
    main()
