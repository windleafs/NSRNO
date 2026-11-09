"""CPU-only dependency and memory wait; never restarts an existing inversion."""
from argparse import ArgumentParser
from pathlib import Path
import fcntl, hashlib, json, os, shutil, subprocess, time, traceback


def atomic_json(path, value):
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid':pid,'start_tick':stat[19],
                'cmdline':Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')}
    except FileNotFoundError:
        return None


def main():
    p = ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('out/v5/reconstruction_coupled_v1_train_200'))
    p.add_argument('--phase', type=Path, default=Path('out/v5/reconstruction_kinematic_v1_train_200'))
    p.add_argument('--initializer', choices=['rf','phase'], default='rf')
    p.add_argument('--rf-seed', type=Path, default=Path('out/v5/reconstruction_geometry_v1_train_200/snapshots/stage0_step064'))
    p.add_argument('--allow-concurrent-controls', action='store_true')
    p.add_argument('--memory-limit-gib', type=float, default=5.75)
    p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--minimum-host-available-gib', type=float, default=32.)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lock = (a.out/'launcher.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.out/'workflow.json').exists() or (a.out/'run.json').exists():
        raise FileExistsError('An existing run or workflow must be inspected, never restarted here')
    # Register current handles once. A later PID reuse does not prolong this wait.
    old = []
    for main_pid,worker_pids,post_pid,name in [
        (1003839,[1004186,1004187],1003840,'reconstruction_geometry_v1_train_200'),
        (1075517,[1075566,1075567],1075518,'reconstruction_travel_geometry_v1_train_200')]:
        main_handle = identity(main_pid)
        if main_handle and ('nsrno.reconstruct_encoded' not in main_handle['cmdline'] or name not in main_handle['cmdline']):
            raise RuntimeError('A registered control PID was reused; inspect its real handle')
        for pid in [main_pid,*worker_pids,post_pid]:
            item = identity(pid)
            if item is None:
                continue
            if pid in worker_pids:
                stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
                if int(stat[1]) != main_pid or 'multiprocessing.spawn' not in item['cmdline']:
                    raise RuntimeError('A control worker identity changed')
            elif name not in item['cmdline']:
                raise RuntimeError('A control process identity changed')
            old.append(item)
    record = {'status':'waiting','pid':os.getpid(),'registered_control_handles':old,
              'initializer':a.initializer,'phase_dependency':str(a.phase) if a.initializer=='phase' else None,
              'rf_seed':str(a.rf_seed),'action':'No control is stopped by this launcher',
              'memory_limit_GiB_per_worker':a.memory_limit_gib,'checkpoint_chunk':a.chunk,
              'allow_concurrent_controls':a.allow_concurrent_controls,
              'minimum_host_available_GiB':a.minimum_host_available_gib,'whole_card_limit_GiB':16}
    atomic_json(a.out/'launch.json', record)
    try:
        while True:
            alive = []
            for handle in old:
                now = identity(handle['pid'])
                if now and now['start_tick'] == handle['start_tick']:
                    alive.append(handle['pid'])
            complete = a.phase/'stage0_result.json' if a.initializer=='phase' else a.rf_seed/'rf_selection.json'
            if not complete.exists() or (alive and not a.allow_concurrent_controls):
                record.update(status='waiting_for_initializer' if not complete.exists() else 'waiting_for_control_resources',
                              live_control_pids=alive,unix_s=time.time())
                atomic_json(a.out/'launch.json',record)
                time.sleep(20)
                continue
            usage = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                              '--format=csv,noheader,nounits'],text=True)
            used = {int(row.split(',')[0]):int(row.split(',')[1]) for row in usage.splitlines()}
            meminfo = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            available = int(meminfo['MemAvailable'].split()[0])/1024**2
            if any(used[d]+a.memory_limit_gib*1024+768 > 16*1024 for d in [0,1,2]) or available < a.minimum_host_available_gib:
                record.update(status='waiting_for_memory',GPU_used_MiB=used,host_available_GiB=available,unix_s=time.time())
                atomic_json(a.out/'launch.json',record)
                time.sleep(20)
                continue
            source_parameters = a.phase/'stage0_best_parameters.npy' if a.initializer=='phase' else a.rf_seed/'parameters.npy'
            source_material = a.phase/'stage0_best_material.npy' if a.initializer=='phase' else a.rf_seed/'material.npy'
            source_selection = a.phase/'stage0_best.json' if a.initializer=='phase' else a.rf_seed/'rf_selection.json'
            before = source_selection.read_bytes()
            stamps = {f:(f.stat().st_mtime_ns,f.stat().st_size) for f in [source_parameters,source_material]}
            shutil.copy2(source_parameters,a.out/'seed_parameters.npy')
            shutil.copy2(source_material,a.out/'seed_material.npy')
            if before != source_selection.read_bytes() or any((f.stat().st_mtime_ns,f.stat().st_size)!=s for f,s in stamps.items()):
                raise RuntimeError('Initializer changed during freezing; inspect before launching')
            atomic_json(a.out/'source_seed.json', {'parameters':str(source_parameters.resolve()),
                'material':str(source_material.resolve()),'selection':json.loads(before),
                'selection_scope':('Complete phase-stage candidate' if a.initializer=='phase' else
                                   'Complete RF-selected fixed 3 MHz checkpoint; same starting field as the physical control HF stage'),
                'truth_used_for_seed_selection':False,
                'parameters_sha256':hashlib.sha256((a.out/'seed_parameters.npy').read_bytes()).hexdigest(),
                'material_sha256':hashlib.sha256((a.out/'seed_material.npy').read_bytes()).hexdigest(),
                'source_completion':json.loads(complete.read_text())})
            env = os.environ.copy()
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1',NUMPY_MADVISE_HUGEPAGE='0',
                       PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:256',PYTHONPATH='.')
            python = '/home/zhuangyang/miniconda3/envs/py310/bin/python'
            source_metadata = json.loads(((a.phase if a.initializer=='phase' else a.rf_seed.parents[1])/'run.json').read_text())
            argv = [python,'-u','-m','nsrno.reconstruct_coupled','--out',str(a.out.resolve()),
                    '--initial-parameters',str((a.out/'seed_parameters.npy').resolve()),
                    '--sample',source_metadata['sample'],
                    '--devices','0','1','2','--iterations','8','--cg-iterations','8',
                    '--memory-limit-gib',str(a.memory_limit_gib),'--chunk',str(a.chunk)]
            with (a.out/'run.log').open('w') as log:
                main_proc = subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
            with (a.out/'postprocess.log').open('w') as log:
                post = subprocess.Popen([python,'-u','-m','nsrno.run_coupled_postprocess','--out',str(a.out.resolve()),
                    '--pid',str(main_proc.pid)],env=env,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
            with (a.out/'snapshot_watcher.log').open('w') as log:
                watch = subprocess.Popen([python,'-u','-m','nsrno.watch_material_snapshots','--out',str(a.out.resolve()),
                    '--pid',str(main_proc.pid),'--min-step','1','--label','Coupled curvature / full RF',
                    '--process-module','nsrno.reconstruct_coupled'],env=env,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
            atomic_json(a.out/'workflow.json', {'optimization_pid':main_proc.pid,'postprocess_pid':post.pid,
                'snapshot_watcher_pid':watch.pid,'argv':argv,'unix_s':time.time()})
            record.update(status='launched',optimization_pid=main_proc.pid,
                GPU_used_MiB_before=used,host_available_GiB_before=available,unix_s=time.time())
            atomic_json(a.out/'launch.json',record)
            break
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),unix_s=time.time())
        atomic_json(a.out/'launch.json',record)
        raise


if __name__ == '__main__':
    main()
