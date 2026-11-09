"""Finish saved material -> dual -> RF/maps on available low-memory devices.

Only an ACTUAL whole-card-budget wait after complete material output can be
held. The hold is bounded and reversible; no GPU field solve is stopped.
Numerical worker/material/source files remain unchanged.
"""
from pathlib import Path
import argparse, ctypes, hashlib, json, os, signal, time, shutil, traceback


def atomic(path, value):
    tmp = path.with_suffix('.dual_resume.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def identity(pid):
    p = Path('/proc') / str(pid)
    if not p.exists() or not (cmd := (p/'cmdline').read_bytes()):
        return None
    return {'pid': pid, 'cmdline': cmd.replace(b'\0', b' ').decode(),
        'start_ticks': (p/'stat').read_text().rsplit(')', 1)[1].split()[19],
        'uid': p.stat().st_uid}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def boundary(root, pid, step):
    run = json.loads((root/'run.json').read_text())
    if run.get('boot_id') != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise RuntimeError('Source run belongs to another boot')
    if run['pid'] != pid or run['step'] != step or run['status'] not in ['dual_update', 'dual_pending_held_for_low_memory', 'dual_pending_checkpoint']:
        raise RuntimeError('Require the saved-material, unstarted-dual boundary')
    folder = root/f'step{step:03d}'
    if (folder/'dual_processes.json').exists() or (folder/'complete_sweep.json').exists() or (root/f'step{step+1:03d}').exists():
        raise RuntimeError('A later dual or outer phase already exists')
    for worker in json.loads((folder/'field_continue_processes.json').read_text())['pids']:
        if identity(worker) is not None:
            raise RuntimeError('A field worker is still live')
    records = [json.loads((folder/f'field_continue_rank{r}.json').read_text()) for r in range(3)]
    if any(r['status'] != 'complete' or r['frequencies'] != 150 for r in records):
        raise RuntimeError('Field statistics are incomplete')
    keys = [(m['source'], m['position']) for r in records for m in r['moments']]
    expected = {(s, i) for s, n in [('lf', 180), ('point3', 270)] for i in range(n)}
    if len(keys) != 450 or set(keys) != expected:
        raise RuntimeError('Require all450 distinct frequencies before finishing material output')
    if (folder/'rf_score.json').exists():
        raise RuntimeError('RF score already exists; inspect instead of duplicating work')
    names = ['old_material.npy', 'old_variables.npy', 'material.npy', 'variables.npy', 'parameters.npy',
        'material_update.json', 'field_convergence_summary_v2.json'] + [f'field_continue_rank{r}.json' for r in range(3)]
    for name in names:
        if not (folder/name).is_file():
            raise RuntimeError('Saved material boundary file missing: '+name)
    return run, folder, names


def hold(a):
    root = a.root.resolve();cp_path = root/f'step{a.step:03d}'/'dual_pending_low_memory.json'
    if cp_path.exists():
        raise FileExistsError('Preserve the existing handoff record')
    run, folder, names = boundary(root, a.pid, a.step)
    ident = identity(a.pid)
    if ident is None or ident['uid'] != os.getuid() or '-m nsrno.reconstruct_wri_gpu ' not in ident['cmdline']:
        raise RuntimeError('Not our expected current coordinator')
    resource = json.loads((root/'resource_summary.json').read_text())['latest']
    if resource['phase'] != 'dual_backend_admission' or time.time()-resource['unix_s'] > 45:
        raise RuntimeError('No fresh actual dual budget wait; no signal sent')
    # Python3.10 time.sleep uses select on this server; newer runtimes may
    # use clock_nanosleep. The phase and child-process guards still apply.
    wait_channel = Path(f'/proc/{a.pid}/wchan').read_text().strip()
    if not any(name in wait_channel for name in ['hrtimer_nanosleep','do_select']):
        raise RuntimeError('Source is not in the budget sleep; retry observation without stopping it')
    if a.check_only:
        print(json.dumps({'dual_budget_wait_valid':True,'all450_fields_saved':True,
            'material_saved':True,'signals_sent':False,'bank_writes':False}),flush=True);return
    held = False;record = None
    try:
        os.kill(a.pid, signal.SIGSTOP);held = True
        for _ in range(100):
            if '\nState:\tT' in Path(f'/proc/{a.pid}/status').read_text():break
            time.sleep(.02)
        else:raise RuntimeError('CPU coordinator did not enter the stopped state')
        if identity(a.pid) != ident:raise RuntimeError('Source identity changed')
        frozen, folder, names = boundary(root, a.pid, a.step)
        for value in Path(f'/proc/{a.pid}/task/{a.pid}/children').read_text().split():
            child = identity(int(value))
            if child and 'multiprocessing.resource_tracker' not in child['cmdline']:
                raise RuntimeError('Unexpected active child; resume original coordinator')
        hashes = frozen['solver_changes'][-1]['source_sha256']
        for name, sha in hashes.items():
            if digest(Path(__file__).with_name(name)) != sha:
                raise RuntimeError('Numerical source changed: '+name)
        files = {name: {'bytes': (folder/name).stat().st_size, 'sha256': digest(folder/name)} for name in names}
        record = {'type': 'complete_material_dual_pending', 'status': 'held_reversibly',
            'root': str(root), 'step': a.step, 'source_identity': ident,
            'source_run': json.loads(json.dumps(frozen)),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'held_unix_s': time.time(), 'lease_deadline': time.time()+a.lease_seconds,
            'files': files, 'source_sha256': hashes, 'all450_fields_complete': True,
            'material_saved': True, 'Y_state': 'Y_base before local material-change dual correction',
            'GPU_field_iterations_lost': 0, 'reason': 'actual shared-GPU admission wait for oversized dual allocator cap'}
        atomic(cp_path, record)
        frozen['status'] = 'dual_pending_held_for_low_memory';atomic(root/'run.json', frozen)
        print(json.dumps({'held': True, 'checkpoint': str(cp_path), 'lease_deadline': record['lease_deadline'],
            'source_pid': a.pid, 'GPU_field_iterations_lost': 0}), flush=True)
        while time.time() < record['lease_deadline']:
            current = json.loads(cp_path.read_text())
            if current['status'] != 'held_reversibly':return
            time.sleep(1)
        current = json.loads(cp_path.read_text())
        if current['status'] == 'held_reversibly':
            atomic(root/'run.json', record['source_run'])
            if identity(a.pid) == ident:os.kill(a.pid, signal.SIGCONT)
            current.update(status='released_without_takeover', released_unix_s=time.time());atomic(cp_path, current)
            print(json.dumps({'released_original': True, 'reason': 'bounded hold expired'}), flush=True)
    except BaseException:
        if held:
            if record is not None:atomic(root/'run.json', record['source_run'])
            if identity(a.pid) == ident:os.kill(a.pid, signal.SIGCONT)
            if record is not None:
                record['status'] = 'released_after_hold_error';atomic(cp_path, record)
        raise


def validate(cp_path):
    cp = json.loads(cp_path.read_text());root = Path(cp['root'])
    if cp['type'] != 'complete_material_dual_pending' or cp['status'] not in ['held_reversibly','retired_with_saved_material']:
        raise RuntimeError('Require the reversible complete-material hold')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != cp['boot_id']:
        raise RuntimeError('Boot changed')
    ident = cp['source_identity']
    if cp['status']=='held_reversibly':
        if identity(ident['pid']) != ident or '\nState:\tT' not in Path(f"/proc/{ident['pid']}/status").read_text():
            raise RuntimeError('Source is not the exact held CPU coordinator')
    elif identity(ident['pid']) is not None:
        raise RuntimeError('Retired source unexpectedly exists')
    run, folder, _ = boundary(root, ident['pid'], cp['step'])
    for name, value in cp['files'].items():
        p = folder/name
        if p.stat().st_size != value['bytes'] or digest(p) != value['sha256']:
            raise RuntimeError('Saved material or complete statistics changed: '+name)
    for name, sha in cp['source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != sha:raise RuntimeError('Source changed: '+name)
    if cp['status']=='held_reversibly' and time.time()+30 > cp['lease_deadline']:
        raise RuntimeError('Insufficient safe hold time remaining')
    return cp, root, folder, run


def finish(a):
    if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:raise RuntimeError('Cannot disable process THP')
    cp, root, folder, run = validate(a.checkpoint)
    if a.check_only:
        print(json.dumps({'checkpoint_valid': True, 'material_saved': True, 'all450_fields': True,
            'signals_sent': False, 'bank_writes': False}), flush=True);return
    if len(set(a.devices)) != len(a.devices) or not a.devices or a.cap_gib <= 0 or a.cap_gib+.75 > 16:
        raise RuntimeError('Invalid low-memory dual device/cap request')
    import fcntl,multiprocessing as mp
    from .wri_worker_block_gpu import run_worker
    from .reconstruct_wri import Resources,score,evaluate
    takeover = open(root/'low_memory_phase_takeover.lock','a')
    fcntl.flock(takeover,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock = None;monitor = None;adopted = False
    try:
        cp, root, folder, run = validate(a.checkpoint)
        # Preserve the first failed handoff and every distinct source version.
        archive = root/f"source_low_memory_dual_step{cp['step']:03d}_{digest(Path(__file__))[:10]}"
        archive.mkdir(exist_ok=True)
        hashes = {}
        for name in sorted(set(cp['source_sha256']) | {Path(__file__).name}):
            p = Path(__file__).with_name(name);hashes[name]=digest(p)
            if (archive/name).exists() and digest(archive/name)!=hashes[name]:
                raise RuntimeError('Preserve the previous handoff source archive: '+name)
            if not (archive/name).exists():shutil.copy2(p,archive/name)
        if not (archive/a.checkpoint.name).exists():shutil.copy2(a.checkpoint,archive/a.checkpoint.name)
        if not (archive/'run_before_dual_resume.json').exists():
            shutil.copy2(root/'run.json',archive/'run_before_dual_resume.json')
        cp, root, folder, run = validate(a.checkpoint)
        if cp['status']=='held_reversibly':
            ident=cp['source_identity']
            atomic(archive/'retirement_intent.json',{'pid':ident['pid'],'unix_s':time.time(),
                'all450_fields_saved':True,'material_saved':True,'GPU_field_iterations_lost':0,
                'only_CPU_coordinator_in_budget_sleep':True})
            os.kill(ident['pid'],signal.SIGTERM)
            os.kill(ident['pid'],signal.SIGCONT)
            for _ in range(200):
                if identity(ident['pid']) is None:break
                time.sleep(.05)
            else:raise RuntimeError('Saved CPU source did not retire; no dual banks written')
            cp.update(status='retired_with_saved_material',retired_unix_s=time.time())
            atomic(a.checkpoint,cp)
            run['status']='dual_pending_checkpoint';atomic(root/'run.json',run)
        # Only now is the old coordinator's kernel-released lock available.
        lock=open(root/'gpu_coordinator.lock','a')
        deadline=time.time()+10
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.time()>=deadline:raise
                # /proc/cmdline can disappear before exit cleanup has fully
                # released kernel file descriptions. Never remove the lock.
                time.sleep(.1)
        run.setdefault('solver_changes',[]).append({'step':cp['step'],'pid':os.getpid(),
            'source_pid':cp['source_identity']['pid'],'unix_s':time.time(),'source_sha256':hashes,
            'checkpoint_type':cp['type'],'resource_change':{'dual_devices':a.devices,'dual_cap_GiB':a.cap_gib},
            'material_changed':False,'physical_equation_changed':False,'observations_changed':False})
        run.update(pid=os.getpid(),status='dual_update',active_phase_cap_GiB=a.cap_gib,
            dual_phase_devices=a.devices,phase_resume='complete_material_low_memory_dual')
        cp.update(status='adopted',resume_pid=os.getpid(),adopted_unix_s=time.time())
        atomic(a.checkpoint,cp);atomic(root/'run.json',run);adopted=True
        monitor=Resources(root,resume=True);start=time.time()
        tasks=[(s,i) for s,n in [('lf',180),('point3',270)] for i in range(n)]
        assignments=[tasks[r::len(a.devices)] for r in range(len(a.devices))]
        monitor.wait_budget(a.devices,a.cap_gib,'low_memory_dual_admission')
        context=mp.get_context('spawn');queue=context.Queue();processes=[]
        for rank,device in enumerate(a.devices):
            process=context.Process(target=run_worker,args=(rank,device,assignments[rank],folder,
                cp['step'],'dual',a.cap_gib,queue,64,.003))
            process.start();processes.append(process)
        atomic(folder/'dual_processes.json',{'mode':'dual','pids':[p.pid for p in processes],
            'devices':a.devices,'per_worker_cap_GiB':a.cap_gib,'worker':'wri_worker_block_gpu.py',
            'scope':'same450/all11 local material-change dual correction, no field iterations','truth_read':False})
        while any(p.is_alive() for p in processes):monitor.sample('low_memory_dual');time.sleep(20)
        for p in processes:p.join()
        if any(p.exitcode!=0 for p in processes):
            raise RuntimeError('Dual worker failed; preserve partial banks and original fixed RHS, no automatic repeat')
        for rank in range(len(processes)):
            summary=json.loads((folder/f'dual_rank{rank}.json').read_text())
            if summary['status']!='complete' or summary['frequencies']!=len(assignments[rank]):
                raise RuntimeError('Incomplete full-frequency dual phase')
        queue.close()
        atomic(folder/'complete_sweep.json',{'status':'complete','step':cp['step'],'solver_revision':3,
            'phase_resume':'low_memory_dual','unix_s':time.time(),'truth_read':False})
        run['status']='full_native_RF_score';atomic(root/'run.json',run)
        rf=score(folder/'material.npy',run['sample'],folder/'rf_score.json',a.devices,1.1,monitor)
        best=json.loads((root/'best_RF.json').read_text())
        if rf['primary_loss']<best['primary_loss']:
            best={'step':cp['step'],'material':str(folder/'material.npy'),'primary_loss':rf['primary_loss']}
            atomic(root/'best_RF.json',best)
        evaluate(folder/'material.npy',run['sample'],root/'initial'/'material.npy',monitor)
        material_record=json.loads((folder/'material_update.json').read_text())
        row={'step':cp['step'],'material_accepted':material_record['accepted'],
            'primary_full_RF_loss':rf['primary_loss'],'best_RF':best,'solver_revision':3,
            'phase_resume':'low_memory_dual','elapsed_since_resume_seconds':time.time()-start,
            'truth_read_by_optimizer':False}
        with (root/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        run.update(status='revised_step_finished',completed_step=cp['step'],
            remaining_original_planned_steps=run['iterations']-cp['step'],best_RF=best,
            joint_reconstruction_success_claimed=False,active_phase_cap_GiB=None)
        atomic(root/'run.json',run)
        cp.update(status='complete',complete_unix_s=time.time());atomic(a.checkpoint,cp)
    except BaseException:
        if adopted:
            run.update(status='failed',error=traceback.format_exc());atomic(root/'run.json',run)
        raise
    finally:
        if monitor is not None:monitor.log.close()
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        fcntl.flock(takeover,fcntl.LOCK_UN);takeover.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['hold','finish'])
    p.add_argument('--root',type=Path);p.add_argument('--pid',type=int);p.add_argument('--step',type=int,default=3)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--lease-seconds',type=float,default=300)
    p.add_argument('--devices',type=int,nargs='+',default=[1,2]);p.add_argument('--cap-gib',type=float,default=2.)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.action=='hold':
        if a.root is None or a.pid is None:raise ValueError('Hold requires root and exact source PID')
        hold(a)
    else:
        if a.checkpoint is None:raise ValueError('Finish requires a typed held checkpoint')
        finish(a)


if __name__=='__main__':main()
