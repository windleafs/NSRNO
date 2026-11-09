"""Resume the untouched half step after its internal 2.5 GiB cap caused OOM.

The full first candidate was successfully completed and rejected. Its cache
is reused directly. No pressure inverse or receiver basis is recomputed.
All original computation and OOM logs remain immutable.
"""
from pathlib import Path
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0) != 0:
    raise RuntimeError('Cannot disable process THP')
import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import numpy as np
from .wri_material import MaterialVariables
from .coupled_coordinates import dot
from .pressure_resources import PressureResources
from .wri_worker import atomic
from .reconstruct_wri import command
from .reconstruct_pressure_joint import GROUPS,live,sha,init_trial_bank,commit_trial
from .pressure_joint_recovery_worker import run_recovery_worker,original_rows,COST_KEYS

NEW_SOURCES=['resume_joint_candidate.py','pressure_joint_recovery_worker.py',
             'pressure_joint_recovery_kernels.py']

def inspect_source(root):
    run=json.loads((root/'run.json').read_text())
    if (run['status']!='failed' or run.get('solver_revision')!='joint_receiver_v1'
            or run['step']!=2 or run.get('completed_step')!=1 or run.get('candidate')!=1
            or 'Joint worker exit; no automatic partial continuation' not in run.get('error','')):
        raise RuntimeError('Only the original unstarted half-step 2.5 GiB OOM is supported')
    if run['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip() or live(run['pid']):
        raise RuntimeError('Require the same boot and naturally exited original coordinator')
    folder=root/'step002'
    for path in folder.glob('*_processes.json'):
        if any(live(p) for p in json.loads(path.read_text())['pids']):
            raise RuntimeError('An original worker remains live')
    if run['sources']!=list(GROUPS) or run['transmissions']!=11 or run['receivers']!=192:
        raise RuntimeError('Changed acquisition scope')
    for name in ['field_commit.json','material_update.json','complete_sweep.json']:
        if (folder/name).exists():
            raise RuntimeError('Material or field has already been committed')
    if (folder/'candidates/line002').exists():
        raise RuntimeError('A later backtracking proposal already exists')
    for rank in range(3):
        failure=json.loads((folder/f'joint_candidate001_rank{rank}.failed.json').read_text())
        if 'OutOfMemoryError' not in failure['error'] or '2.50 GiB allowed' not in failure['error']:
            raise RuntimeError('Require the diagnosed internal-cap OOM')
        for line in (folder/f'joint_candidate001_rank{rank}.log').read_text().splitlines():
            try:
                record=json.loads(line)
            except (ValueError,TypeError):
                continue
            if 'source' in record and 'position' in record:
                raise RuntimeError('The old half step already committed a frequency')
    first_cost=json.loads((folder/'candidates/line000/cost.json').read_text())
    if first_cost['frequencies']!=450 or first_cost['alpha']!=1. or first_cost['accepted'] is not False:
        raise RuntimeError('Require the original complete rejected first candidate')
    if any(folder.glob('joint_candidate000_rank*.failed.json')):
        raise RuntimeError('A computation failed before the original summary was saved')
    for name,digest in run['source_sha256'].items():
        if sha(Path(__file__).with_name(name))!=digest:
            raise RuntimeError('Frozen source changed: '+name)
    for key in ['field_inverse_checkpoint','stable_field_checkpoint']:
        if key in run and sha(Path(run[key]))!=run[key+'_sha256']:
            raise RuntimeError('Frozen inverse weights changed')
    convergence=json.loads((folder/'field_convergence.json').read_text())
    if convergence['frequencies']!=450 or convergence['transmissions_each']!=11:
        raise RuntimeError('Incomplete source field')
    rows=original_rows(folder)
    expected={(s,i) for s,n in GROUPS.items() for i in range(n)}
    if set(rows)!=expected:
        raise RuntimeError('First-candidate computation is incomplete or repeated')
    summaries=[]
    seen=[]
    for rank in range(3):
        report=json.loads((folder/f'joint_candidate000_rank{rank}.json').read_text())
        if report['status']!='complete' or report['frequencies']!=150 or report['truth_read'] is not False:
            raise RuntimeError('Missing original complete rank summary')
        keys=[tuple(k) for k in report['assignments']]
        if len(keys)!=150:
            raise RuntimeError('Wrong original rank assignment count')
        for name in ['data_halfcost_change','equation_halfcost_change','joint_halfcost_change',
                     'joint_linear_slope','conditional_curvature_along_material',
                     'receiver_reduced_curvature_along_material']:
            total=sum(rows[key][name] for key in keys)
            if not np.isfinite(total) or not np.isclose(total,report[name],rtol=1e-12,atol=1e-14):
                raise RuntimeError('Original rows and complete summary disagree')
        summaries.append(report)
        seen.extend(keys)
    if len(seen)!=450 or set(seen)!=expected:
        raise RuntimeError('Not all450 original frequency blocks are complete')
    scratch=root/'joint_trial_banks'
    owner=json.loads((scratch/'owner.json').read_text())
    ready=json.loads((scratch/'candidate_complete.json').read_text())
    if owner['step']!=2 or owner['status']!='prepared' or ready['status']!='writing' or ready['alpha']!=.5 or ready['suffix']!='line':
        raise RuntimeError('Unexpected first-cache ownership or completion state')
    for source,count in GROUPS.items():
        for kind in ['X','Y','R','Ydata','D','generation','penalty']:
            bank=np.load(root/'banks'/f'{source}_{kind}.npy',mmap_mode='r')
            shape=((count,11,961,1121) if kind in ['X','Y','R'] else
                   (count,11,192) if kind in ['Ydata','D'] else (count,))
            if bank.shape!=shape:
                raise RuntimeError('Wrong immutable bank header')
            if kind in ['X','Y','R','Ydata','D'] and bank.dtype!=np.complex64:
                raise RuntimeError('Wrong immutable bank dtype')
            if kind=='generation' and not np.all(bank==2):
                raise RuntimeError('Incomplete immutable augmented RHS generation')
            if kind=='penalty' and (not np.isfinite(bank).all() or not np.all(bank>0)):
                raise RuntimeError('Invalid unchanged pressure penalty')
        cache=np.load(scratch/f'{source}_first_X.npy',mmap_mode='r')
        if cache.shape!=(count,11,961,1121) or cache.dtype!=np.complex64:
            raise RuntimeError('Wrong full first-cache header')
        if (scratch/f'{source}_first_X.npy').stat().st_size!=cache.offset+cache.nbytes:
            raise RuntimeError('Truncated first cache')
    for parent,names in [(folder,['old_variables.npy','old_material.npy']),
                         (folder/'candidates/line000',['variables.npy','material.npy','parameters.npy']),
                         (folder/'candidates/line001',['variables.npy','material.npy','parameters.npy'])]:
        for name in names:
            value=np.load(parent/name,mmap_mode='r')
            if value.shape!=(3,961,1121) or not np.isfinite(value).all():
                raise RuntimeError('Invalid complete material checkpoint')
    if (root/'source_joint_candidate_recovery_step002').exists():
        raise RuntimeError('No automatic adoption of another partial recovery')
    return run,folder,scratch,summaries


def phase(folder, step, mode, assignments, args, monitor, cap,
          proposal=None, tag=None, line_scale=1.):
    tag = tag or mode
    monitor.wait_budget(args.devices, cap, tag+'_admission')
    if mode in ['field_joint', 'candidate_first']:
        available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                             if s.startswith('MemAvailable:')))/2**20
        if available < 16*len(args.devices):
            raise RuntimeError('Insufficient host memory for native pressure factors')
    context = mp.get_context('spawn')
    processes = []
    try:
        for rank, device in enumerate(args.devices):
            process = context.Process(target=run_recovery_worker, args=(rank, device,
                assignments[rank], folder, step, mode, cap, args.field_cg, .003,
                proposal, tag, line_scale))
            process.start()
            processes.append(process)
        atomic(folder/f'{tag}_processes.json', {'pids': [p.pid for p in processes],
            'devices': args.devices, 'phase': mode, 'parent_pid': os.getpid(),
            'allocator_cap_GiB': cap, 'truth_read': False})
        while any(p.is_alive() for p in processes):
            monitor.sample(tag)
            if monitor.owned_above_limit:
                raise RuntimeError('Joint reconstruction exceeded the per-card model 16 GiB budget')
            if any(p.exitcode is not None and p.exitcode != 0 for p in processes):
                raise RuntimeError('A joint worker failed; immutable RHS and trial state retained')
            time.sleep(20)
        for p in processes:
            p.join()
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError('Joint worker exit; no automatic partial continuation')
        rows = [json.loads((folder/f'{tag}_rank{rank}.json').read_text()) for rank in range(len(processes))]
        observed = [tuple(x) for row in rows for x in row['assignments']]
        expected = {x for assignment in assignments for x in assignment}
        if len(observed) != 450 or set(observed) != expected or any(r['status'] != 'complete' for r in rows):
            raise RuntimeError('Require every one of the 450 complete frequency blocks')
        return rows
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=10)
        atomic(folder/f'{tag}_exit_status.json',
            {'parent_pid':os.getpid(),'phase':mode,
             'workers':[{'pid':p.pid,'exitcode':p.exitcode,'alive':p.is_alive()} for p in processes]})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    root=args.out.resolve()
    source,folder,scratch,original=inspect_source(root)
    if args.devices!=source['devices'] or len(args.devices)!=3:
        raise RuntimeError('Use the same three devices and complete rank assignments as the source')
    if args.check_only:
        print(json.dumps({'source_all_handles_gone':True,'field_and_first_candidate_records_complete':True,
            'first_cache_header_valid':True,'frequencies_to_reuse':450,
            'resume_unstarted_half_step':True,'new_pressure_solves':0,'bank_writes':False,'signals_sent':False}),flush=True)
        return
    lock=open(root/'coordinator.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    monitor=None
    run=None
    changed=False
    try:
        source,folder,scratch,original=inspect_source(root)
        step=2
        previous=root/'step001'
        variables=MaterialVariables(np.load(root/'initial/material.npy'),
                                    np.load(root/'initial/parameters.npy'))
        current=np.load(folder/'old_variables.npy')
        full_proposal=np.load(folder/'candidates/line000/variables.npy')
        direction_record=json.loads((folder/'material_direction.json').read_text())
        if direction_record['zero_material_direction']:
            raise RuntimeError('No computed material direction')
        field_bytes=sum(GROUPS.values())*11*variables.size*8
        needed=field_bytes if not all((scratch/f'{s}_line_X.npy').exists() for s in GROUPS) else 0
        if shutil.disk_usage(root).free<needed+17*2**30:
            raise RuntimeError('Insufficient room for unchanged backtracking cache and reserve')
        archive=root/'source_joint_candidate_recovery_step002'
        archive.mkdir()
        shutil.copy2(root/'run.json',archive/'source_failed_run.json')
        prior_resource=json.loads((root/'resource_summary.json').read_text())
        shutil.copy2(root/'resource_summary.json',archive/'source_resource_summary.json')
        for rank in range(3):
            shutil.copy2(folder/f'joint_candidate000_rank{rank}.json',archive/f'original_candidate_rank{rank}.json')
            for suffix in ['log','failed.json']:
                shutil.copy2(folder/f'joint_candidate001_rank{rank}.{suffix}',archive/f'original_half_rank{rank}.{suffix}')
        shutil.copy2(folder/'candidates/line000/cost.json',archive/'original_first_cost.json')
        names=sorted(set(source['source_sha256'])|set(NEW_SOURCES))
        hashes={name:sha(Path(__file__).with_name(name)) for name in names}
        for name in names:
            shutil.copy2(Path(__file__).with_name(name),archive/name)
        weights=Path(source['field_inverse_checkpoint'])
        frozen=archive/'field_inverse_weights.pt'
        shutil.copy2(weights,frozen)
        if sha(frozen)!=source['field_inverse_checkpoint_sha256']:
            raise RuntimeError('Weights changed while freezing recovery source')
        args.field_cg=64
        args.cap_gib=6.
        args.tv=source['actual_solver_settings']['TV_weight']
        run=dict(source)
        run.setdefault('solver_changes',[]).append({'type':'explicit_complete_cached_candidate_recovery',
            'source_pid':source['pid'],'pid':os.getpid(),'unix_s':time.time(),
            'original_all450_computations_complete':True,
            'original_failure_cause':'half-step internal allocator limit2.5GiB; requested108MiB while2.37GiB allocated',
            'pressure_equation_changed':False,'direction_changed':False,
            'pressure_solve_repeated':False,'complete_first_candidate_reused':True,'half_step_allocator_cap_GiB':6.})
        run.update(pid=os.getpid(),status='joint_candidate_cost',candidate=1,candidate_alpha=.5,candidate_log_tag='joint_candidate001_recovery',error=None,
            solver_revision='joint_cached_backtrack_memory_recovery_v1',
            source_sha256=hashes,devices=args.devices,allocator_cap_GiB=6.,
            field_log_tag='field_joint',resumed_complete_field_count=0,
            prior_peak_owned_MiB=prior_resource.get('sampled_peak_owned_MiB',{}),
            field_inverse_checkpoint=str(frozen),
            field_inverse_checkpoint_sha256=sha(frozen))
        atomic(root/'run.json',run)
        changed=True
        monitor=PressureResources(root)
        tasks=[(s,i) for s,n in GROUPS.items() for i in range(n)]
        assignments=[tasks[r::len(args.devices)] for r in range(len(args.devices))]
        atomic(folder/'cached_candidate_recovery.json',{'status':'complete_first_cache_reused',
            'source_pid':source['pid'],'resume_pid':os.getpid(),'frequencies':450,
            'original_full_step_accepted':False,'half_step_completed_frequencies':0,
            'new_pressure_solves':0,'reason':'internal2.5GiB cap raised to6GiB below model16GiB'})
        old_tv, tv_gradient, _, _ = variables.tv(current, args.tv)
        candidates = [json.loads((folder/'candidates/line000/cost.json').read_text())]
        accepted = None
        if not direction_record['zero_material_direction']:
            for line in range(1,9):
                alpha = .5**line
                candidate=folder/'candidates'/f'line{line:03d}'
                proposed=variables.project(current+alpha*(full_proposal-current))
                if line==1:
                    saved_half=np.load(candidate/'variables.npy')
                    if not np.allclose(saved_half,proposed,rtol=2e-12,atol=2e-12):
                        raise RuntimeError('The saved half-step variables differ from the unchanged direction')
                    proposed=saved_half
                    if not np.array_equal(np.load(candidate/'material.npy'),variables.material(proposed)):
                        raise RuntimeError('The saved half-step physical material differs')
                else:
                    candidate.mkdir()
                    np.save(candidate/'variables.npy',proposed)
                    np.save(candidate/'material.npy',variables.material(proposed))
                    np.save(candidate/'parameters.npy',variables.parameters(proposed))
                suffix='line'
                init_trial_bank(scratch,suffix,variables.shape)
                atomic(scratch/'candidate_complete.json',{'status':'writing','step':step,
                    'alpha':alpha,'suffix':suffix,'candidate':str(candidate)})
                tag=f'joint_candidate{line:03d}_recovery'
                run.update(status='joint_candidate_cost',candidate=line,candidate_alpha=alpha,candidate_log_tag=tag)
                atomic(root/'run.json',run)
                rows=phase(folder,step,'candidate_scaled',assignments,args,monitor,6.,
                    candidate,tag=tag,line_scale=alpha)
                data = sum(r['data_halfcost_change'] for r in rows)
                equation = sum(r['equation_halfcost_change'] for r in rows)
                slope = sum(r['joint_linear_slope'] for r in rows)+dot(tv_gradient, (proposed-current)/.1)
                tv_change = variables.tv(proposed, args.tv)[0]-old_tv
                change = data+equation+tv_change
                actual_rms = np.sqrt(np.mean((proposed-current)[:, variables.unknown]**2, axis=1))
                ok = bool(np.isfinite(change) and slope < 0 and change <= 1e-4*slope
                    and np.all(actual_rms <= np.array([.03, .04, .10])*1.01))
                row = {'alpha': alpha, 'data_cost_change': data, 'equation_cost_change': equation,
                    'TV_change': tv_change, 'joint_objective_change': change,
                    'joint_linear_slope': slope, 'actual_RMS_r_beta_t': actual_rms.tolist(),
                    'accepted': ok, 'frequencies': 450, 'all11_each': True}
                row.update({key:sum(r.get(key,0.) for r in rows) for key in COST_KEYS[4:]})
                row['primitive_projection_deviation']=float(np.linalg.norm(
                    proposed-current-alpha*(full_proposal-current))/max(
                    float(np.linalg.norm(alpha*(full_proposal-current))),1e-300))
                candidates.append(row)
                atomic(candidate/'cost.json', row)
                atomic(scratch/'candidate_complete.json', {'status': 'complete', 'step': step,
                    'alpha': alpha, 'suffix': suffix, 'candidate': str(candidate),
                    'material_sha256': sha(candidate/'material.npy'), 'frequencies': 450})
                if ok:
                    accepted = (candidate, suffix, alpha)
                    break
        run['status'] = 'joint_field_commit'
        atomic(root/'run.json', run)
        if accepted is not None:
            candidate, suffix, alpha = accepted
            commit_trial(root, scratch, suffix, folder, candidate, alpha, monitor)
        else:
            for name in ['variables.npy', 'material.npy', 'parameters.npy']:
                shutil.copy2(previous/name, folder/name)
        atomic(folder/'material_update.json', {'accepted': accepted is not None,
            'line_search': candidates, 'direction': direction_record,
            'complete_original_first_cache_reused':True,'new_pressure_solves':0,
            'same_node_receiver_curvature': True, 'off_node_data_curvature_omitted': True,
            'simultaneous_field_change_in_acceptance': True, 'truth_read': False})
        run['status'] = 'joint_pressure_dual_update'
        atomic(root/'run.json', run)
        phase(folder, step, 'dual_joint', assignments, args, monitor, 6., folder)
        atomic(folder/'complete_sweep.json', {'status': 'complete', 'step': step,
            'frequencies': 450, 'transmissions_each': 11, 'unix_s': time.time(),
            'solver_revision': 'joint_receiver_v1', 'truth_read': False})
        run['status'] = 'full_native_RF_score'
        atomic(root/'run.json', run)
        if sha(folder/'material.npy') == sha(previous/'material.npy'):
            rf = json.loads((previous/'rf_score.json').read_text())
            rf.update(material=str(folder/'material.npy'), reused_unchanged_material_RF=True,
                source_RF=str(previous/'rf_score.json'))
            atomic(folder/'rf_score.json', rf)
        else:
            used = monitor.sample('joint_native_RF_admission')
            device = min(args.devices, key=lambda d: used[d]['whole_MiB'])
            monitor.wait_budget([device], 1.1, 'joint_native_RF_admission')
            command([sys.executable, '-m', 'nsrno.score_wri_material', str(folder/'material.npy'),
                '--sample', run['sample'], '--out', str(folder/'rf_score.json'), '--device', str(device),
                '--cap-gib', '1.1'], folder/'rf_score.log', monitor, 'joint_native_RF_score')
            rf = json.loads((folder/'rf_score.json').read_text())
        best = json.loads((root/'best_RF.json').read_text())
        if rf['primary_loss'] < best['primary_loss']:
            best = {'step': step, 'material': str(folder/'material.npy'), 'primary_loss': rf['primary_loss']}
            atomic(root/'best_RF.json', best)
        run['status'] = 'post_hoc_maps_only'
        atomic(root/'run.json', run)
        shutil.copy2(root/'initial/material.npy', folder/'stage0_initial_material.npy')
        command([sys.executable, '-m', 'nsrno.evaluate_time_material', str(folder/'material.npy'),
            '--sample', run['sample'], '--out', str(folder/'maps'), '--estimate-label',
            'Joint pressure/material update'], folder/'maps.log', monitor, 'post_hoc_joint_maps_only')
        row = {'step': step, 'solver_revision': 'joint_receiver_v1',
            'material_accepted': accepted is not None, 'primary_full_RF_loss': rf['primary_loss'],
            'best_RF': best, 'truth_read_by_optimizer': False}
        with (root/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        atomic(scratch/'owner.json', {'step': step, 'status': 'complete_step',
            'reusable_not_archival': True, 'material_sha256': sha(folder/'material.npy')})
        run.update(status='planned_joint_sweeps_finished', completed_step=step,
            best_RF=best, joint_reconstruction_success_claimed=False)
        atomic(root/'run.json', run)
    except BaseException:
        if changed:
            run.update(status='failed', error=traceback.format_exc())
            atomic(root/'run.json', run)
        raise
    finally:
        if monitor is not None:
            monitor.log.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
