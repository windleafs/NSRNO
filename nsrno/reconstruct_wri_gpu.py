"""Resume the SAME fixed-model WRI problem from immutable augmented RHS.

All450 frequencies and all11 transmissions are recomputed for valid complete
material statistics. X is warm, Y_base is rewritten consistently, and the
material update/dual/fullRF/maps retain the original order and definitions.
CPU fallback uses the same ledger without changing the scientific problem.
"""
from pathlib import Path
import ctypes,os
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
import argparse,json,time,hashlib,shutil,traceback,multiprocessing as mp,fcntl
import numpy as np
from .wri_material import MaterialVariables
from .wri_material_v2 import MaterialNormal
from .wri_worker_block_gpu import run_worker,atomic
from .reconstruct_wri import Resources,score,evaluate


def workers(folder,step,mode,assignments,args,monitor):
    monitor.wait_budget(args.devices,args.cap_gib,mode+'_backend_admission')
    info=Path('/proc/meminfo').read_text().splitlines()
    available=int(next(s.split()[1] for s in info if s.startswith('MemAvailable:')))/2**20
    if available<48:raise RuntimeError('Require48GiB host memory for three concurrent native pressure factors')
    context=mp.get_context('spawn');queue=context.Queue();processes=[]
    for rank,device in enumerate(args.devices):
        process=context.Process(target=run_worker,args=(rank,device,assignments[rank],folder,step,mode,
            args.cap_gib,queue,args.field_cg,.003));process.start();processes.append(process)
    atomic(folder/(mode+'_processes.json'),{'pids':[p.pid for p in processes],'mode':mode,'devices':args.devices,
        'worker':'wri_worker_block_gpu.py','backend':args.backend,'per_worker_cap_GiB':args.cap_gib,'truth_read':False})
    while any(p.is_alive() for p in processes):monitor.sample(mode+'_'+args.backend);time.sleep(20)
    for p in processes:p.join()
    if any(p.exitcode!=0 for p in processes):raise RuntimeError('Backend worker failed; preserve immutable RHS and warm banks, no automatic duplicate continuation')
    summaries=[json.loads((folder/f'{mode}_rank{r}.json').read_text()) for r in range(3)]
    if any(s['status']!='complete' or s['frequencies']!=len(assignments[r]) for r,s in enumerate(summaries)):
        raise RuntimeError('Incomplete full-frequency statistics')
    queue.close();return summaries


def validate_checkpoint(path,allow_held=False):
    cp=json.loads(path.read_text());root=Path(cp['root']);folder=root/f"step{cp['step']:03d}"
    if cp['type']!='fixed_model_field_continue_partial_boundary' or cp['status'] not in (['fixed_rhs_ready','retired_with_fixed_rhs'] if allow_held else ['retired_with_fixed_rhs']):
        raise RuntimeError('Require a complete immutable RHS partial-continuation checkpoint')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=cp['boot_id']:raise RuntimeError('Checkpoint boot changed')
    for name,sha in cp['material_sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen model changed')
    for name,sha in cp['active_source_sha256'].items():
        if hashlib.sha256((Path('nsrno')/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Original CPU numerical source changed')
    ledger=Path(cp['fixed_rhs_directory']);index=json.loads((ledger/'index.json').read_text())
    if hashlib.sha256((ledger/'index.json').read_bytes()).hexdigest()!=cp['fixed_rhs_index_sha256']:raise RuntimeError('Immutable RHS manifest changed')
    if index['status']!='complete' or len(index['frequencies'])!=450 or index['groups']!={'lf':180,'point3':270} or index['transmissions']!=11:
        raise RuntimeError('Require complete450 frequencies/all11 original augmented right-hand sides')
    if index['material_sha256']!=cp['material_sha256']:raise RuntimeError('RHS belongs to a different frozen model')
    for value in index['files'].values():
        p=Path(value['path'])
        if not p.exists() or p.stat().st_size!=value['bytes']:raise RuntimeError('Immutable RHS file missing or changed')
    identities=[cp['source_main_identity'],*cp['source_worker_identities'].values()]
    for identity in identities:
        proc=Path(f"/proc/{identity['pid']}/cmdline")
        if cp['status']=='retired_with_fixed_rhs':
            if proc.exists() and proc.read_bytes():raise RuntimeError('Retired source process handle still exists')
        elif not proc.exists() or proc.read_bytes().replace(b'\0',b' ').decode()!=identity['cmdline'] or '\nState:\tT' not in proc.with_name('status').read_text():
            raise RuntimeError('Source process is not the exact held identity')
    return cp,index


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--backend',choices=['gpu','cpu'],default='gpu');p.add_argument('--cap-gib',type=float)
    p.add_argument('--field-cg',type=int,default=64);p.add_argument('--check-only',action='store_true')
    p.add_argument('--gpu-qualification',type=Path);a=p.parse_args();cp,index=validate_checkpoint(a.checkpoint,allow_held=a.check_only)
    root=Path(cp['root']);step=cp['step'];folder=root/f'step{step:03d}';run=json.loads((root/'run.json').read_text())
    a.cap_gib=a.cap_gib if a.cap_gib is not None else (14.5 if a.backend=='gpu' else 11.)
    if a.cap_gib+.75>16:raise RuntimeError('Allocator cap leaves insufficient whole-card context allowance')
    if a.check_only:
        print(json.dumps({'checkpoint_valid':True,'immutable_frequencies':450,'transmissions':11,'source_status':cp['status'],
            'backend':a.backend,'cap_GiB':a.cap_gib,'signals_sent':False,'bank_writes':False}),flush=True);return
    if a.backend=='gpu':
        if a.gpu_qualification is None:raise RuntimeError('GPU backend requires the completed integration report')
        q=json.loads(a.gpu_qualification.read_text())
        if not q.get('passed') or q.get('all11_shots') is not True or q['sampled_maximum_whole_card_MiB'][str(q['device'])]>16384:
            raise RuntimeError('GPU integration/memory qualification did not pass')
        for name,sha in q['code_sha256'].items():
            if hashlib.sha256((Path('nsrno')/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('GPU qualified source changed: '+name)
    lock=open(root/'gpu_coordinator.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if run['status']!='field_continue_checkpoint' or run['pid']!=cp['source_pid']:raise RuntimeError('Root is not the retired CPU checkpoint')
    version=root/f'source_v3_{a.backend}_step{step:03d}';version.mkdir(exist_ok=False)
    names=['reconstruct_wri_gpu.py','wri_worker_block_gpu.py','native_compact_weighted_pressure_gpu.py','gpu_sparse_lu.py',
        'block_wavefield_step.py','batched_first_order_wavefield.py','native_compact_weighted_pressure.py',
        'receiver_normal_preconditioner_batch.py','wri_material.py','wri_material_v2.py','first_order_wavefield.py',
        'conditioned_wave.py','dense_wave.py','frequency_group_data.py','reconstruct_wri.py','wri_worker.py',
        'score_wri_material.py','native_time.py','native_time.cu']
    hashes={}
    for name in names:
        path=Path(__file__).with_name(name);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest();shutil.copy2(path,version/name)
    shutil.copy2(root/'run.json',version/'run_before_backend_resume.json');shutil.copy2(a.checkpoint,version/a.checkpoint.name)
    cp.update(status='resumed_fixed_rhs',resume_pid=os.getpid(),backend=a.backend,resumed_unix_s=time.time())
    atomic(a.checkpoint,cp)
    a.devices=run['devices'];a.sample=run['sample'];a.tv=run['TV_weight'];a.damping=run['damping_relative'];a.material_cg=run['material_CG']
    monitor=Resources(root,resume=True);start=time.time();run.setdefault('solver_changes',[]).append({
        'step':step,'pid':os.getpid(),'source_pid':cp['source_pid'],'backend':a.backend,'unix_s':time.time(),
        'source_sha256':hashes,'fixed_rhs_index_sha256':cp['fixed_rhs_index_sha256'],'all450_statistics_recomputed':True,
        'physical_equation_changed':False,'observations_changed':False,'CPU_field_progress_reused_as_warm_X':True})
    run.update(pid=os.getpid(),status='field_continue',solver_revision=3,pressure_backend=a.backend,cap_GiB=a.cap_gib,
        fixed_rhs_directory=cp['fixed_rhs_directory'],step=step,phase_started_unix_s=time.time(),GPU_qualification=str(a.gpu_qualification) if a.gpu_qualification else None)
    atomic(root/'run.json',run)
    tasks=[(s,i) for s,n in [('lf',180),('point3',270)] for i in range(n)]
    assignments=[tasks[r::3] for r in range(3)]
    initial=root/'initial';variables=MaterialVariables(np.load(initial/'material.npy'),np.load(initial/'parameters.npy'))
    current=np.load(folder/'old_variables.npy');best=json.loads((root/'best_RF.json').read_text())
    try:
        records=workers(folder,step,'field_continue',assignments,a,monitor);normal={};moments=[]
        for record in records:
            with np.load(record['normal_path']) as data:
                for key in data.files:normal[key]=normal.get(key,0)+data[key]
            moments.extend(record['moments'])
        if len(moments)!=450:raise RuntimeError('Incomplete material moments')
        atomic(folder/'field_convergence_summary_v2.json',{'frequencies':450,'transmissions_per_frequency':11,'backend':a.backend,
            'maximum_physical_global_normal_residual':max(r['max_physical_global_normal_residual'] for r in records),
            'maximum_preconditioned_global_normal_residual':max(r['max_preconditioned_global_normal_residual'] for r in records),
            'unconverged_shots':sum(r['unconverged_shots'] for r in records),'requested_tolerance':.003,
            'scope':'fixed-field normal equations, not RF reconstruction accuracy'})
        run['status']='material_update';atomic(root/'run.json',run)
        normal_system=MaterialNormal(variables,records[0]['geometry'],normal)
        def progress(row):
            atomic(folder/'material_CG_progress_v3.json',row)
            if row['iteration']%8==0:print(json.dumps({'step':step,'material_CG':row}),flush=True)
        proposed,record=normal_system.solve(current,moments,a.tv,a.damping,maximum=a.material_cg,callback=progress)
        np.save(folder/'variables.npy',proposed);np.save(folder/'material.npy',variables.material(proposed))
        np.save(folder/'parameters.npy',variables.parameters(proposed));atomic(folder/'material_update.json',record)
        del normal_system,normal,records,moments
        run['status']='dual_update';atomic(root/'run.json',run);workers(folder,step,'dual',assignments,a,monitor)
        atomic(folder/'complete_sweep.json',{'step':step,'status':'complete','unix_s':time.time(),'solver_revision':3,'truth_read':False})
        run['status']='full_native_RF_score';atomic(root/'run.json',run)
        rf=score(folder/'material.npy',a.sample,folder/'rf_score.json',a.devices,1.1,monitor)
        if rf['primary_loss']<best['primary_loss']:
            best={'step':step,'material':str(folder/'material.npy'),'primary_loss':rf['primary_loss']};atomic(root/'best_RF.json',best)
        evaluate(folder/'material.npy',a.sample,initial/'material.npy',monitor)
        row={'step':step,'material_accepted':record['accepted'],'primary_full_RF_loss':rf['primary_loss'],
            'best_RF':best,'solver_revision':3,'backend':a.backend,'elapsed_since_resume_seconds':time.time()-start,'truth_read_by_optimizer':False}
        with (root/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        run.update(status='revised_step_finished',completed_step=step,remaining_original_planned_steps=run['iterations']-step,
            best_RF=best,joint_reconstruction_success_claimed=False);atomic(root/'run.json',run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc());atomic(root/'run.json',run);raise
    finally:monitor.log.close();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
