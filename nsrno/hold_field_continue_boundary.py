"""Hold CPU field-continuation workers at flushed frequency boundaries.

The continuation solves a FIXED-material, FIXED-augmented-RHS problem.
Every committed frequency preserves b_shift=Y_base+B_old X. No material or
dual update has occurred. Incomplete CPU aggregate normals are deliberately
not treated as a material checkpoint: the new worker must recompute ALL450.
"""
from pathlib import Path
import argparse,json,os,signal,time,hashlib,shutil,traceback


def atomic(path,obj):
    tmp=path.with_suffix('.boundary.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def identity(pid):
    base=Path(f'/proc/{pid}');cmd=(base/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    start=(base/'stat').read_text().rsplit(')',1)[1].split()[19]
    return {'pid':pid,'cmdline':cmd,'start_ticks':start}


def stopped(pid):return '\nState:\tT' in Path(f'/proc/{pid}/status').read_text()


def wait_stopped(pid):
    for _ in range(100):
        if stopped(pid):return
        time.sleep(.02)
    raise RuntimeError('Process did not enter the reversible stopped state')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--wait-seconds',type=float,default=900);p.add_argument('--lease-seconds',type=float,default=1800)
    a=p.parse_args();root=a.root.resolve();run=json.loads((root/'run.json').read_text());folder=root/f"step{run['step']:03d}"
    if a.checkpoint.exists():raise FileExistsError('Preserve the existing partial-field checkpoint')
    if run['pid']!=a.pid or run['status']!='field_continue' or run.get('solver_revision')!=2:
        raise RuntimeError('Only the known fixed-model field continuation can be held here')
    main_id=identity(a.pid)
    if '-m nsrno.reconstruct_wri_v2 ' not in main_id['cmdline']:raise RuntimeError('Coordinator identity differs')
    revision=root/f"source_v2_step{run['step']:03d}"
    sources=['wri_worker_block.py','block_wavefield_step.py','batched_first_order_wavefield.py',
        'native_compact_weighted_pressure.py','receiver_normal_preconditioner_batch.py','first_order_wavefield.py']
    source_hashes={}
    for name in sources:
        current=hashlib.sha256((Path('nsrno')/name).read_bytes()).hexdigest()
        if current!=hashlib.sha256((revision/name).read_bytes()).hexdigest():raise RuntimeError('Active numerical source differs: '+name)
        source_hashes[name]=current
    processes=json.loads((folder/'field_continue_processes.json').read_text());pids=processes['pids']
    if len(pids)!=3:raise RuntimeError('Expected exactly the three known workers')
    ids={pid:identity(pid) for pid in pids}
    if any('multiprocessing.spawn' not in row['cmdline'] for row in ids.values()):raise RuntimeError('Worker identities differ')
    held=[];boundaries={};start=time.time();original=run.copy();saved=False
    archive=root/f"cpu_field_partial_step{run['step']:03d}_before_gpu";archive.mkdir(exist_ok=False)
    try:
        os.kill(a.pid,signal.SIGSTOP);held.append(a.pid);wait_stopped(a.pid)
        now=json.loads((root/'run.json').read_text())
        if now['status']!='field_continue' or now['pid']!=a.pid:raise RuntimeError('Coordinator phase changed before hold')
        while len(boundaries)<3 and time.time()-start<a.wait_seconds:
            for rank,pid in enumerate(pids):
                if rank in boundaries:continue
                log=folder/f'field_continue_rank{rank}.log';stamp=log.stat().st_mtime
                # In the archived worker, bank flush + moments + CPU stats
                # precede this log. Next bank writes follow a fresh native LU,
                # 192-receiver setup and a block solve. Stop within2s of log.
                if time.time()-stamp>1.:continue
                lines=log.read_text().splitlines()
                if not lines:continue
                try:last=json.loads(lines[-1])
                except json.JSONDecodeError:continue
                if len(last.get('angle_records',[]))!=11 or 'block_record' not in last:continue
                if identity(pid)!=ids[pid]:raise RuntimeError('Worker handle changed')
                os.kill(pid,signal.SIGSTOP);held.append(pid);wait_stopped(pid)
                if time.time()-stamp>2.:raise RuntimeError('Frequency boundary hold was too late; resume without using checkpoint')
                progress=json.loads((folder/f'progress_rank{rank}.json').read_text())
                if progress['mode']!='field_continue' or progress['pid']!=pid:raise RuntimeError('Worker phase changed')
                boundaries[rank]={'pid':pid,'source':last['source'],'last_committed_position':last['frequency_position'],
                    'log_mtime':stamp,'held_unix_s':time.time(),'current_progress':progress,
                    'completed_frequencies':len([line for line in lines if '"angle_records"' in line]),
                    'bank_writes_completed_before_log':True,'next_field_solve_not_interrupted':True}
                print(json.dumps({'worker_boundary_held':boundaries[rank]}),flush=True)
            time.sleep(.05)
        if len(boundaries)!=3:raise TimeoutError('Did not catch all three flushed frequency boundaries')
        if any(identity(pid)!=ids[pid] or not stopped(pid) for pid in pids):raise RuntimeError('Worker hold changed')
        for name in ['run.json','resource_summary.json']:
            if (root/name).exists():shutil.copy2(root/name,archive/name)
        for rank in range(3):
            for name in [f'field_continue_rank{rank}.log',f'progress_rank{rank}.json']:
                shutil.copy2(folder/name,archive/name)
        (archive/'moments').mkdir()
        committed=[]
        for rank in range(3):
            for line in (archive/f'field_continue_rank{rank}.log').read_text().splitlines():
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                if 'angle_records' not in row:continue
                source,position=row['source'],row['frequency_position']
                if source not in ['lf','point3'] or not 0<=position<({'lf':180,'point3':270}[source]):raise RuntimeError('Invalid committed frequency')
                name=f'{source}_{position:03d}.npz';shutil.copy2(folder/'moments'/name,archive/'moments'/name)
                committed.append({'source':source,'position':position,'rank':rank})
        shutil.copy2(folder/'field_continue_processes.json',archive/'field_continue_processes.json')
        shutil.copy2(Path(__file__),archive/Path(__file__).name)
        hashes={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['old_material.npy','old_variables.npy']}
        record={'type':'fixed_model_field_continue_partial_boundary','status':'held_for_fixed_rhs',
            'root':str(root),'step':run['step'],'sample':run['sample'],'source_pid':a.pid,
            'source_main_identity':main_id,'source_worker_identities':ids,'boundaries':boundaries,
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'unix_s':time.time(),
            'lease_deadline_unix_s':time.time()+a.lease_seconds,'archive':str(archive.resolve()),
            'material_sha256':hashes,'active_source_sha256':source_hashes,
            'all_banks_state':'X warm fields; Y is Y_base; Ydata is data_base, all flushed at committed frequency',
            'original_augmented_rhs_invariant':'Y_base+B_old X and Ydata_base+C X, same frozen material for all450 frequencies',
            'material_updated':False,'dual_updated':False,'GPU_field_iterations_lost':0,
            'CPU_aggregate_normals_complete':False,'must_recompute_all450_material_statistics':True,
            'committed_CPU_frequencies':committed,'committed_pressure_moments_archived':len(committed),
            'signals_sent':{'parent':'SIGSTOP','workers':'SIGSTOP at flushed frequency boundary'},
            'fixed_rhs_directory':None,'goal_complete':False}
        atomic(a.checkpoint,record);saved=True
        run.update(status='field_continue_held_for_gpu_upgrade',upgrade_checkpoint=str(a.checkpoint.resolve()))
        atomic(root/'run.json',run);print(json.dumps({'all_held':True,'checkpoint':str(a.checkpoint),'lease_deadline':record['lease_deadline_unix_s']}),flush=True)
        while time.time()<record['lease_deadline_unix_s']:
            latest=json.loads(a.checkpoint.read_text())
            if latest['status'] not in ['held_for_fixed_rhs','fixed_rhs_ready']:return
            time.sleep(1)
        latest=json.loads(a.checkpoint.read_text())
        if latest['status'] in ['held_for_fixed_rhs','fixed_rhs_ready']:
            atomic(root/'run.json',original)
            for pid in reversed(held):
                expected=main_id if pid==a.pid else ids[pid]
                if identity(pid)==expected:os.kill(pid,signal.SIGCONT)
            latest.update(status='released_without_gpu_takeover',release_unix_s=time.time());atomic(a.checkpoint,latest)
            print(json.dumps({'resumed_original_CPU':True,'reason':'bounded hold lease expired'}),flush=True)
    except BaseException:
        release=not saved
        if saved:
            try:release=json.loads(a.checkpoint.read_text())['status'] in ['held_for_fixed_rhs','fixed_rhs_ready']
            except Exception:release=False
        if release:
            atomic(root/'run.json',original)
            for pid in reversed(held):
                try:
                    expected=main_id if pid==a.pid else ids[pid]
                    if identity(pid)==expected:os.kill(pid,signal.SIGCONT)
                except Exception:pass
            if saved:
                latest=json.loads(a.checkpoint.read_text());latest.update(status='released_after_hold_error',release_unix_s=time.time());atomic(a.checkpoint,latest)
        raise


if __name__=='__main__':main()
