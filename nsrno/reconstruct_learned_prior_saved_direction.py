"""One real native material update using an existing full44 gradient direction."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0) != 0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import hashlib
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
from .learned_spatial_energy import LearnedSpatialEnergy
from .independent_physical_coordinates import IndependentPhysicalCoordinates
from .low_frequency_0p5 import atomic,digest,SAMPLE
from .pressure_resources import PressureResources
from .reconstruct_frequency_multilevel import array
from .reconstruct_native_frequency_incremental import same_process


class SavedDirectionPool(ProjectedCommonPool):
    def __init__(self,args,monitor,plan,deadline,base_material):
        super().__init__(args,monitor,plan,deadline)
        # Point1's gradient is already complete in the immutable parent.
        # These are logical point indices; actual new derivative counters stay0.
        self.gradient_step = 1
        self.last_gradient_material = hashlib.sha256(np.ascontiguousarray(base_material,np.float32).tobytes()).hexdigest()

    def actual_work(self):
        totals = super().actual_work()
        if (totals.get('issued_adjoint_direction_passes',0) != 0 or
            totals.get('issued_tangent_direction_passes',0) != 0 or
            totals.get('issued_physical_forward_passes',0) > 88):
            raise RuntimeError('Saved-direction stage allows only88new forwards,0derivatives')
        return totals

    def evaluate(self,material,kind,step=1):
        if kind != 'original44_forward' or step != 1 or self.forward_count >= 2:
            raise RuntimeError('Only two real full44 candidate forwards; no gradient replay')
        return super().evaluate(material,kind,step)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1])
    p.add_argument('--cap-gib',type=float,default=10.)
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--max-seconds',type=float,default=900.)
    args = p.parse_args()
    project,source = Path.cwd().resolve(),Path(__file__).parent
    for key in ['parent_root','out','artifact_root']:
        setattr(args,key,getattr(args,key).resolve())
    if (not args.parent_root.is_relative_to(project/'out/v5') or not args.out.is_relative_to(project/'out/v5') or
        not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or args.out.exists() or
        args.artifact_root.exists() or args.devices != [0,1] or args.cap_gib != 10 or
        args.chunk != 128 or args.max_seconds != 900):
        raise ValueError('Fresh single-update/2candidate/88forward/900s recovery only')
    old = json.loads((args.parent_root/'run.json').read_text())
    workers = json.loads((args.parent_root/'worker_processes.json').read_text())
    work = json.loads((args.parent_root/'actual_native_work.json').read_text())
    if (old['status'] != 'learned_spatial_prior_native_stage_finished' or old['accepted_updates'] != 0 or
        old['original44_checks_completed'] != 3 or work['completed_adjoint_direction_passes'] != 44 or
        work['unfinished_native_phases'] != 0 or same_process(old) or
        any(Path('/proc',str(pid)).exists() for pid in workers['worker_pids'])):
        raise RuntimeError('Only the exact finished failed-interval trajectory can supply this saved direction')
    hashes = dict(old['source_SHA256'])
    for name,sha in hashes.items():
        if digest(source/name) != sha:
            raise RuntimeError('Frozen prior or physical source changed')
    hashes[Path(__file__).name] = digest(source/Path(__file__).name)
    point = args.parent_root/'steps/step01'
    details = json.loads((point/'direction_details.json').read_text())
    if details['start_material_SHA256'] != old['initial_material_SHA256']:
        raise RuntimeError('Saved direction belongs to another material')
    gradient_path = args.parent_root/'arrays/data_control_gradient_token0001.npy'
    if digest(gradient_path) != details['data_gradient_SHA256']:
        raise RuntimeError('Complete existing gradient changed; do not regenerate it')
    initial = (point/'start_material.npy').resolve()
    control_path = (point/'start_control.npy').resolve()
    direction_path = point/'direction.npy'
    if digest(initial) != old['material_SHA256'] or old['final_RF'] != old['initial_RF']:
        raise RuntimeError('Parent changed material; this gradient cannot be reused')
    acceptance = json.loads((args.parent_root/'final/actual_native_acceptance.json').read_text())
    references = acceptance['RF_source_records']
    F0,slope = old['initial_RF'],details['RF_slope']
    if slope >= 0:
        raise RuntimeError('Saved actual RF direction is not descending')
    curvature_rows = []
    for line in [2,3,4]:
        row = json.loads((point/f'line{line:02d}/actual_native_acceptance.json').read_text())
        if row['accepted'] or row['RF_before'] != F0:
            raise RuntimeError('Expected the three existing rejected real candidates')
        t = row['fraction']
        h = 2*(row['RF_after']-F0-t*slope)/(t*t)
        curvature_rows.append({'fraction':t,'RF':row['RF_after'],'secant_curvature':h})
    curvature = max(r['secant_curvature'] for r in curvature_rows)
    if not curvature > 0:
        raise RuntimeError('Existing finite steps did not show the assumed positive curvature')
    fraction = min(.0625,.9*(-slope)/curvature)
    if not 0 < fraction < .125:
        raise RuntimeError('No new smaller candidate from the measured line model')
    comparison = json.loads((point/'same_material_RF_comparison.json').read_text())
    minimum_drop = max(10*abs(comparison['difference']),comparison['eight_relative_float32_eps'])
    plan = json.loads((args.parent_root/'native_plan.json').read_text())
    entries = [(f,s) for f in plan['families'] for s in range(11)]
    if sorted((r['family'],r['shot']) for r in references) != sorted(entries):
        raise RuntimeError('Complete accepted starting RF required')
    for row in references:
        if digest(row['path']) != row['SHA256']:
            raise RuntimeError('Starting RF cache changed')
    model_path = Path(old['fit_root'])/'products/prior.pt'
    if digest(model_path) != old['learned_prior_SHA256']:
        raise RuntimeError('Do not retrain or replace the frozen neural prior')
    reserve = 2*2**30+400*2**20
    if shutil.disk_usage(project).free < reserve+2*2**20 or shutil.disk_usage(Path.home()).free < reserve+600*2**20:
        raise RuntimeError('Retain project/HOME disk reserves')
    args.out.mkdir()
    args.artifact_root.mkdir()
    for name in ['arrays','workers','lines','final']:
        (args.artifact_root/name).mkdir()
        (args.out/name).symlink_to(args.artifact_root/name,target_is_directory=True)
    (args.out/'jobs').mkdir()
    for name in hashes:
        shutil.copy2(source/name,args.out/name)
    for name,path in [('initial_material.npy',initial),('initial_control.npy',control_path),
        ('saved_direction.npy',direction_path),('completed_data_gradient.npy',gradient_path)]:
        (args.out/'arrays'/name).symlink_to(path.resolve())
    plan.update(reference_RF=references,reference_material_SHA256=digest(initial))
    atomic(args.out/'native_plan.json',plan)
    started = time.time()
    deadline = started+900
    run = {'status':'initializing_saved_direction_actual_update','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s':started,'deadline_unix_s':deadline,'max_seconds':900,
        'source_SHA256':hashes,'parent_root':str(args.parent_root),'fit_root':old['fit_root'],
        'learned_prior_SHA256':old['learned_prior_SHA256'],'initial_material_SHA256':digest(initial),
        'saved_direction_SHA256':digest(direction_path),'reused_gradient_SHA256':digest(gradient_path),
        'reused_complete_all44_gradients':44,'initial_RF':F0,'current_RF':F0,
        'initial_regularization':old['initial_regularization'],'current_regularization':old['initial_regularization'],
        'maximum_full44_candidates':2,'maximum_new_forward_passes':88,'maximum_new_adjoint_passes':0,
        'maximum_new_tangent_passes':0,'pure_forward_cost_is_not_zero_compute':True,
        'new_reserved_directional_equivalent_units':0,'closed_parent_reserved_equivalent_units':22,
        'previous_cumulative_reserved_units':366,'not_whole_thread_actual_cost':True,
        'closed_parent_actual_work':work,'closed_parent_created_unix_s':old['created_unix_s'],
        'closed_parent_deadline_unix_s':old['deadline_unix_s'],'all_closed_budgets_retained':True,
        'line_model':{'slope':slope,'rows':curvature_rows,'selected_curvature':curvature,
            'first_fraction':fraction,'second_fraction':fraction/2},
        'original44_checks_issued':0,'original44_checks_completed':0,'accepted_updates':0,
        'patient_truth_read_by_optimizer':False,'new_patient_derivative_probes':0,
        'new_data_or_frequency':False,'reserved_case_unread':'Neg_47_Left',
        'joint_reconstruction_success_claimed':False,'one_GPU_experiment':True,
        'per_card_process_tree_limit_GiB':16,'allocator_cap_GiB':10}
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
            raise RuntimeError('Fixed900second budget or cooperative stop reached')
        if shutil.disk_usage(project).free < reserve or shutil.disk_usage(Path.home()).free < reserve:
            raise RuntimeError('Existing disk reserve reached')
    try:
        freeze()
        edge_path = project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
        coordinates = IndependentPhysicalCoordinates(np.load(plan['native_anchor_material']),np.load(edge_path))
        control = np.load(control_path).astype(np.float64)
        material = coordinates.material(control)
        if not np.array_equal(material,np.load(initial)):
            raise RuntimeError('Saved point control does not reproduce exact material')
        direction = np.load(direction_path)
        if not np.isfinite(direction).all() or direction.shape != control.shape:
            raise RuntimeError('Finite complete saved direction required')
        prior = LearnedSpatialEnergy(model_path)
        z,x = np.ogrid[:961,:1121]
        active = (z >= 80)&(z < 937)&(x >= 24)&(x < 1097)
        parts = old['initial_prior_parts']
        current_RF,R = F0,old['initial_regularization']
        pool = SavedDirectionPool(args,monitor,plan,deadline,material)
        stop_reason = 'no_accepted_smaller_real_candidate'
        for line,t in enumerate([fraction,fraction/2],1):
            freeze()
            proposal = control+t*direction
            if np.max(abs(coordinates.project(proposal,active)-proposal)) > 1e-9:
                raise RuntimeError('Convex缩步 left the original physical domain')
            candidate_R,_,candidate_parts = prior.evaluate(coordinates,proposal,gradient=False)
            target_R = R+1e-4*t*details['prior_slope']
            if candidate_R > target_R:
                atomic(args.out/f'skipped_line{line:02d}.json',{'fraction':t,
                    'reason':'same_frozen_prior_Armijo_failed','prior':candidate_R,'required_prior':target_R})
                continue
            folder = args.out/'lines'/f'line{line:02d}'
            folder.mkdir()
            array(folder/'control.npy',proposal)
            candidate_material = coordinates.material(proposal)
            array(folder/'material.npy',candidate_material)
            run.update(status='actual_all44_saved_direction_small_candidate',active_line=line,
                active_fraction=t,active_regularization=candidate_R)
            run['original44_checks_issued'] += 1
            checkpoint()
            trial = pool.evaluate(candidate_material,'original44_forward')
            run['original44_checks_completed'] += 1
            accepted = bool(trial['loss'] < F0-minimum_drop and
                trial['loss'] <= F0+1e-4*t*slope and
                trial['loss']+candidate_parts['TV'] < F0+parts['TV'] and candidate_R <= target_R)
            record = {'accepted':accepted,'fraction':t,'RF_before':F0,'RF_after':trial['loss'],
                'prior_before':R,'prior_after':candidate_R,'minimum_RF_drop':minimum_drop,
                'parts_after':candidate_parts,'material_SHA256':digest(folder/'material.npy'),
                'new_adjoint_passes':0,'new_JVPs':0,'patient_truth_read_by_optimizer':False}
            atomic(folder/'actual_native_acceptance.json',record)
            checkpoint()
            print(json.dumps({'line':line,'fraction':t,'RF':trial['loss'],'prior':candidate_R,'accepted':accepted}),flush=True)
            if not accepted:
                continue
            references = []
            for index,(family,shot) in enumerate(entries):
                path = args.out/'workers'/f'rank{index%2:02d}'/'arrays'/f'token{trial["token"]:04d}'/f'{family}_{shot:02d}.npy'
                references.append({'family':family,'shot':shot,'path':str(path.resolve()),'SHA256':digest(path)})
            control,material = proposal,candidate_material
            current_RF,R,parts = trial['loss'],candidate_R,candidate_parts
            run.update(accepted_updates=1,current_RF=current_RF,current_regularization=R,
                current_material_SHA256=record['material_SHA256'])
            stop_reason = 'single_saved_direction_actual_update_accepted'
            checkpoint()
            break
        pool.close()
        pool = None
        final = args.out/'final'
        array(final/'control.npy',control)
        array(final/'material.npy',material)
        atomic(final/'actual_native_acceptance.json',{'accepted_updates':run['accepted_updates'],
            'actual_native_RF':{'loss':current_RF,'all44_sources':True},'RF_source_records':references,
            'native_TV':parts['TV'],'independent_prior_value':R,'material_SHA256':digest(final/'material.npy'),
            'full_original44_source_acceptance':True,'joint_imaging_success_claimed':False})
        if run['accepted_updates']:
            (final/'stage0_initial_material.npy').symlink_to(initial)
            with (final/'maps.log').open('x') as log:
                subprocess.run([sys.executable,'-m','nsrno.evaluate_time_material',str(final/'material.npy'),
                    '--sample',SAMPLE,'--out',str(final/'maps'),'--estimate-label','Frozen spatial prior: measured smaller step'],
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        else:
            for suffix in ['png','metrics.json']:
                shutil.copy2(args.parent_root/f'final/maps.{suffix}',final/f'maps.{suffix}')
        atomic(final/'maps_provenance.json',{'new_fixed_material_evaluation':bool(run['accepted_updates']),
            'if_no_update_initial_column_is_historical':not bool(run['accepted_updates']),
            'truth_never_read_during_candidate_selection':True})
        run.update(status='learned_prior_saved_direction_native_stage_finished',stop_reason=stop_reason,
            final_RF=current_RF,final_regularization=R,material_SHA256=digest(final/'material.npy'),
            seconds=time.time()-started,finished_unix_s=time.time())
        checkpoint()
        print(json.dumps({k:run[k] for k in ['status','stop_reason','accepted_updates','final_RF',
            'final_regularization','original44_checks_completed','material_SHA256','seconds']}),flush=True)
    except BaseException:
        run.update(status='learned_prior_saved_direction_failed_for_inspection',error=traceback.format_exc(),
            seconds=time.time()-started,all_issued_work_remains_charged=True)
        checkpoint()
        raise
    finally:
        if pool is not None:
            pool.close()
        monitor.log.close()


if __name__ == '__main__':
    main()
