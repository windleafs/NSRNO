"""Explicit takeover of a typed, fully saved material-pending checkpoint.

No partially computed frequency is resumed. Existing field banks are reused,
the coupled material trust region is corrected, and later full field sweeps
use the revised preconditioner and full-RHS convergence measurement.
"""
from pathlib import Path
import ctypes,os
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
import argparse,json,time,signal,hashlib,shutil,traceback,multiprocessing as mp,fcntl
import numpy as np
from .wri_material import MaterialVariables
from .wri_material_v2 import MaterialNormal
from .wri_worker_block import run_worker,atomic
from .reconstruct_wri import Resources,score,evaluate


def workers(out,sweep,mode,assignments,args,monitor,field_iterations,tolerance):
    monitor.wait_budget(args.devices,args.cap_gib,mode+'_v2_admission')
    if mode in ['field','field_continue']:
        available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))/2**20
        if available<48:raise RuntimeError('Require48GiB available host memory before three coarse factorizations')
    context=mp.get_context('spawn');results=context.Queue();processes=[]
    for rank,device in enumerate(args.devices):
        process=context.Process(target=run_worker,args=(rank,device,assignments[rank],out,sweep,mode,args.cap_gib,results,field_iterations,tolerance))
        process.start();processes.append(process)
    atomic(out/(mode+'_processes.json'),{'mode':mode,'pids':[p.pid for p in processes],
        'devices':args.devices,'per_worker_cap_GiB':args.cap_gib,'worker':'wri_worker_v2.py','truth_read':False})
    while any(p.is_alive() for p in processes):monitor.sample(mode+'_v2');time.sleep(20)
    for process in processes:process.join()
    if any(p.exitcode!=0 for p in processes):raise RuntimeError('Worker failure; preserve partial banks without automatic resumption')
    summaries=[json.loads((out/f'{mode}_rank{r}.json').read_text()) for r in range(len(processes))]
    if any(s['status']!='complete' or s['frequencies']!=len(assignments[r]) for r,s in enumerate(summaries)):
        raise RuntimeError('Incomplete frequency sweep')
    results.close();return summaries


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--takeover-held',action='store_true')
    p.add_argument('--cap-gib',type=float,default=11.0);p.add_argument('--field-cg',type=int,default=32)
    p.add_argument('--refine-held-field',action='store_true',help='Refine the saved field problem at its ORIGINAL augmented RHS before its material update')
    p.add_argument('--through-step',type=int,help='Finish this explicit complete outer step and its full RF/maps before scheduling additional expensive sweeps')
    p.add_argument('--check-only',action='store_true',help='Validate the typed checkpoint without signals, archives, or bank writes')
    a=p.parse_args();out=a.out.resolve();checkpoint=json.loads(a.checkpoint.read_text())
    lock=open(out/'v2_coordinator.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not a.takeover_held:raise RuntimeError('Explicit takeover of the reversibly held coordinator is required')
    if checkpoint['type']!='fixed_field_material_pending' or checkpoint['status'] not in ['held_reversibly','held_reversibly_extended','retired_at_material_pending'] or checkpoint['root']!=str(out):
        raise RuntimeError('Invalid typed checkpoint')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if checkpoint['boot_id']!=boot:raise RuntimeError('Checkpoint boot differs')
    step=checkpoint['step'];folder=out/f'step{step:03d}';old_pid=checkpoint['source_pid']
    run=json.loads((out/'run.json').read_text())
    retired=checkpoint['status']=='retired_at_material_pending'
    if run['pid']!=old_pid or run['step']!=step or run['status']!=('solver_upgrade_checkpoint' if retired else 'material_update'):raise RuntimeError('Source phase changed')
    proc=Path(f'/proc/{old_pid}/cmdline')
    if retired:
        if proc.exists() and proc.read_bytes():raise RuntimeError('Retired source handle unexpectedly exists')
    else:
        state=Path(f'/proc/{old_pid}/status').read_text();cmd=proc.read_bytes().replace(b'\0',b' ').decode()
        if '\nState:\tT' not in state or cmd!=checkpoint['source_cmdline']:raise RuntimeError('Source is not the expected held coordinator')
    if (folder/'dual_processes.json').exists() or (out/f'step{step+1:03d}').exists():raise RuntimeError('Later source phase exists')
    for name,sha in run['source_sha256'].items():
        if name!='reconstruct_wri.py' and hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('Original numerical source changed before takeover: '+name)
    for name,value in checkpoint['files'].items():
        path=folder/name
        if path.stat().st_size!=value['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=value['sha256']:
            raise RuntimeError('Checkpoint file changed: '+name)
    for pid in json.loads((folder/'field_processes.json').read_text())['pids']:
        proc=Path(f'/proc/{pid}/cmdline')
        if proc.exists() and proc.read_bytes():raise RuntimeError('Source field worker still live')
    if a.through_step is not None and not step<=a.through_step<=run['iterations']:raise RuntimeError('Invalid explicit final outer step')
    if a.check_only:
        print(json.dumps({'checkpoint_valid':True,'type':checkpoint['type'],'step':step,'source_pid':old_pid,
            'all_field_workers_exited':True,'original_numerical_hashes_unchanged':True,
            'source_held':not retired,'source_retired':retired,'signals_sent':False,'bank_writes':False}),flush=True)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close();return
    # Keep any old CPU material output, including a partial file, for review.
    old_outputs=folder/'v1_uncommitted_material';old_outputs.mkdir(exist_ok=False)
    for name in ['variables.npy','material.npy','parameters.npy','material_update.json','material_CG_progress.json']:
        if (folder/name).exists():shutil.copy2(folder/name,old_outputs/name)
    revision=out/f'source_v2_step{step:03d}';revision.mkdir(exist_ok=False)
    names=['reconstruct_wri_v2.py','wri_worker_block.py','block_wavefield_step.py','batched_first_order_wavefield.py',
        'native_compact_weighted_pressure.py','receiver_normal_preconditioner_batch.py','coarse_pressure_preconditioner.py',
        'reconstruct_wri.py','wri_worker.py','wavefield_step.py',
        'wri_material_v2.py','wri_material.py','first_order_wavefield.py','frequency_group_data.py',
        'conditioned_wave.py','dense_wave.py','score_wri_material.py','native_time.py','native_time.cu']
    hashes={}
    for name in names:
        path=Path(__file__).with_name(name);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest();shutil.copy2(path,revision/name)
    shutil.copy2(out/'run.json',revision/'run_before_takeover.json');shutil.copy2(a.checkpoint,revision/a.checkpoint.name)
    if a.refine_held_field:
        # Retain every original fixed-field moment and normal contribution.
        # This is a small-statistics archive, not another native field bank.
        previous=folder/'field_v1_completed';previous.mkdir(exist_ok=False)
        shutil.copytree(folder/'moments',previous/'moments')
        for rank in range(3):
            name=f'normals_rank{rank}.npz';shutil.copy2(folder/name,previous/name)
            source=folder/f'field_rank{rank}.json';shutil.copy2(source,previous/f'field_rank{rank}_original.json')
            record=json.loads(source.read_text());record['normal_path']=str((previous/name).resolve())
            for moment in record['moments']:moment['path']=str((previous/'moments'/Path(moment['path']).name).resolve())
            atomic(previous/f'field_rank{rank}.json',record)
        for name in ['old_material.npy','old_variables.npy']:shutil.copy2(folder/name,previous/name)
        shutil.copy2(folder/'field_processes.json',previous/'field_processes.json')
    # Recheck the reversible hold immediately before retiring the source.
    if not retired:
        if '\nState:\tT' not in Path(f'/proc/{old_pid}/status').read_text() or json.loads(a.checkpoint.read_text())['status'] not in ['held_reversibly','held_reversibly_extended']:
            raise RuntimeError('Hold was released during preparation; no signal sent')
    # Only the CPU coordinator is retired. All4950 field solves and their
    # multipliers/moments have already been flushed and explicitly checked.
    if not retired:os.kill(old_pid,signal.SIGTERM);os.kill(old_pid,signal.SIGCONT)
    for _ in range(200):
        proc=Path(f'/proc/{old_pid}/cmdline')
        if not proc.exists() or not proc.read_bytes():break
        time.sleep(.05)
    else:raise RuntimeError('Original coordinator did not exit; no bank writes performed')
    checkpoint.update(status='taken_over',takeover_pid=os.getpid(),takeover_unix_s=time.time(),
        signals_sent=checkpoint['signals_sent']+([] if retired else ['SIGTERM','SIGCONT']),GPU_iterations_lost=0)
    atomic(a.checkpoint,checkpoint)
    a.devices=run['devices'];a.sample=run['sample'];a.iterations=run['iterations']
    a.material_cg=run['material_CG'];a.tv=run['TV_weight'];a.damping=run['damping_relative']
    last_step=a.iterations if a.through_step is None else a.through_step
    if not step<=last_step<=a.iterations:raise RuntimeError('Invalid explicit final outer step')
    monitor=Resources(out,resume=True);start=time.time()
    run.setdefault('solver_changes',[]).append({'step':step,'pid':os.getpid(),'source_pid':old_pid,
        'checkpoint_type':checkpoint['type'],'source_sha256':hashes,'unix_s':time.time(),
        'material_fix':'RMS constraints inside the coupled normal system',
        'field_fix':'native50um compact pressure inverse WITH compact RHS Q; all192 receiver Woodbury block; all11 transmission blockCG; true full-RHS normal residual',
        'refine_saved_field_at_original_augmented_RHS':a.refine_held_field,
        'native_operator_changed':False,'saved_field_sweep_reused':True})
    run.update(pid=os.getpid(),cap_GiB=a.cap_gib,status='material_update',solver_revision=2)
    atomic(out/'run.json',run)
    initial=out/'initial';variable=MaterialVariables(np.load(initial/'material.npy'),np.load(initial/'parameters.npy'))
    current=np.load(folder/'old_variables.npy');best=json.loads((out/'best_RF.json').read_text())
    groups={'lf':180,'point3':270};tasks=[(s,i) for s,n in groups.items() for i in range(n)]
    assignments=[tasks[r::len(a.devices)] for r in range(len(a.devices))]
    try:
        for number in range(step,last_step+1):
            folder=out/f'step{number:03d}'
            if number==step:
                if a.refine_held_field:
                    run.update(status='field_continue',step=number,phase_started_unix_s=time.time());atomic(out/'run.json',run)
                    records=workers(folder,number,'field_continue',assignments,a,monitor,a.field_cg,.003)
                else:records=[json.loads((folder/f'field_rank{r}.json').read_text()) for r in range(3)]
            else:
                folder.mkdir();(folder/'moments').mkdir()
                np.save(folder/'old_variables.npy',current);np.save(folder/'old_material.npy',variable.material(current))
                run.update(status='field_sweep',step=number,phase_started_unix_s=time.time());atomic(out/'run.json',run)
                records=workers(folder,number,'field',assignments,a,monitor,a.field_cg,.003)
            normal={};moments=[]
            for record in records:
                with np.load(record['normal_path']) as data:
                    for key in data.files:normal[key]=normal.get(key,0)+data[key]
                moments.extend(record['moments'])
            if len(moments)!=450:raise RuntimeError('Missing full-frequency moments')
            atomic(folder/'field_convergence_summary_v2.json',{'frequencies':450,'transmissions_per_frequency':11,
                'maximum_physical_global_normal_residual':max(r.get('max_physical_global_normal_residual',0) for r in records),
                'maximum_preconditioned_global_normal_residual':max(r.get('max_preconditioned_global_normal_residual',0) for r in records),
                'unconverged_shots':sum(r.get('unconverged_shots',0) for r in records),'requested_tolerance':.003,
                'scope':'field normal equations only, not physical RF reconstruction accuracy'})
            run.update(status='material_update',step=number);atomic(out/'run.json',run)
            normal_system=MaterialNormal(variable,records[0]['geometry'],normal)
            def progress(row):
                atomic(folder/'material_CG_progress_v2.json',row)
                if row['iteration']%8==0:print(json.dumps({'step':number,'material_CG_v2':row}),flush=True)
            proposed,record=normal_system.solve(current,moments,a.tv,a.damping,maximum=a.material_cg,callback=progress)
            np.save(folder/'variables.npy',proposed);np.save(folder/'material.npy',variable.material(proposed))
            np.save(folder/'parameters.npy',variable.parameters(proposed));atomic(folder/'material_update.json',record)
            del normal_system,normal,records,moments
            run['status']='dual_update';atomic(out/'run.json',run)
            workers(folder,number,'dual',assignments,a,monitor,a.field_cg,.003)
            current=proposed;atomic(folder/'complete_sweep.json',{'step':number,'status':'complete','unix_s':time.time(),'truth_read':False,'solver_revision':2})
            run['status']='full_native_RF_score';atomic(out/'run.json',run)
            rf=score(folder/'material.npy',a.sample,folder/'rf_score.json',a.devices,1.1,monitor)
            if rf['primary_loss']<best['primary_loss']:
                best={'step':number,'material':str(folder/'material.npy'),'primary_loss':rf['primary_loss']};atomic(out/'best_RF.json',best)
            evaluate(folder/'material.npy',a.sample,initial/'material.npy',monitor)
            row={'step':number,'material_accepted':record['accepted'],'primary_full_RF_loss':rf['primary_loss'],
                 'best_RF':best,'solver_revision':2,'elapsed_since_takeover_seconds':time.time()-start,'truth_read_by_optimizer':False}
            with open(out/'history.jsonl','a') as stream:stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        run.update(status='planned_sweeps_finished' if last_step==a.iterations else 'revised_step_finished',
            completed_step=last_step,remaining_original_planned_steps=a.iterations-last_step,
            best_RF=best,joint_reconstruction_success_claimed=False)
        atomic(out/'run.json',run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc());atomic(out/'run.json',run);raise
    finally:monitor.log.close();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
