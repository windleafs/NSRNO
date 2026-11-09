"""Launch one authorized experiment after its data and GPU dependencies finish.

This process does not import torch, inspect truth for initialization, restart a
running optimizer, or treat a quiet log as failure. Actual PID signatures and
GPU usage determine whether the resources are available.
"""
from argparse import ArgumentParser
from pathlib import Path
import ctypes, fcntl, hashlib, json, os, shutil, subprocess, time, traceback


PYTHON = '/home/zhuangyang/miniconda3/envs/py310/bin/python'


def atomic_json(path, value):
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def main():
    p = ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--original-pid', type=int, required=True)
    p.add_argument('--original-post-pid', type=int, required=True)
    p.add_argument('--devices', type=int, nargs='+', default=[1, 2])
    p.add_argument('--memory-limit-gib', type=float, default=5.75)
    a = p.parse_args()
    ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)
    a.out = a.out.resolve()
    a.geometry = a.geometry.resolve()
    a.original = a.original.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    lock = (a.out/'launch.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = a.out/'launch.json'
    root = Path.cwd()
    last_state = None

    def live(pid, module, out):
        path = Path(f'/proc/{pid}/cmdline')
        return path.exists() and module.encode() in path.read_bytes() and out.name.encode() in path.read_bytes()

    def update(state, **details):
        nonlocal last_state
        item = {'status': state, 'launcher_pid': os.getpid(), 'unix_s': time.time(), **details}
        atomic_json(status, item)
        if state != last_state:
            print(json.dumps(item), flush=True)
            last_state = state

    if (a.out/'run.json').exists():
        raise FileExistsError('Preserve the existing optimization')
    prior = a.out/'workflow.json'
    if prior.exists():
        old = json.loads(prior.read_text())
        if live(old.get('optimization_pid', -1), 'nsrno.reconstruct_kinematic', a.out):
            raise RuntimeError('This experiment already has a live optimization handle')
        raise FileExistsError('Existing workflow requires explicit diagnosis; do not duplicate it')
    try:
        while True:
            if live(a.original_pid, 'nsrno.reconstruct_time', a.original):
                update('waiting_for_original_optimization', optimization_pid=a.original_pid)
                time.sleep(20)
                continue
            if live(a.original_post_pid, 'nsrno.run_time_postprocess', a.original):
                update('waiting_for_original_replay', postprocess_pid=a.original_post_pid)
                time.sleep(20)
                continue
            # The original handles ended. Record their actual terminal state.
            original = json.loads((a.original/'run.json').read_text())
            post = json.loads((a.original/'postprocess.json').read_text())
            if original['status'] not in ['optimization_finished', 'failed']:
                raise RuntimeError('Original handle ended without a terminal optimization record')
            if post['status'] not in ['finished', 'failed']:
                raise RuntimeError('Original replay handle ended without a terminal record')
            metrics = a.original/'final_material.metrics.json'
            if metrics.exists():
                maps = json.loads(metrics.read_text())['maps']
                promising = all(v['reconstruction']['correlation'] is not None and
                    v['reconstruction']['correlation'] >= .6 and
                    v['reconstruction']['rmse'] <= .8*v['initial']['rmse'] for v in maps.values())
                if promising:
                    update('baseline_maps_need_research_assessment', metrics=str(metrics),
                           reason='Do not consume more GPU time before assessing this potentially useful baseline',
                           goal_is_not_complete=True)
                    return
            result = a.geometry/'stage0_result.json'
            if not result.exists():
                run = json.loads((a.geometry/'run.json').read_text())
                if run['status'] == 'failed':
                    raise RuntimeError('Geometry initialization ended without completing stage0')
                update('waiting_for_geometry_stage0')
                time.sleep(20)
                continue
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                            '--format=csv,noheader,nounits'], text=True)
            used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in rows.splitlines()}
            # The PyTorch cap excludes part of the CUDA context overhead.
            budget = a.memory_limit_gib*1024+768
            if any(used[d]+budget > 16*1024 for d in a.devices):
                update('waiting_for_GPU_budget', memory_MiB=used, requested_devices=a.devices,
                       additional_budget_MiB_per_device=budget)
                time.sleep(20)
                continue
            mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
            if mem['MemAvailable'] < 22*1024**2:
                update('waiting_for_host_checkpoint_memory', available_GiB=mem['MemAvailable']/1024**2)
                time.sleep(20)
                continue
            break

        selection_path = a.geometry/'stage0_best.json'
        selection = json.loads(selection_path.read_text())
        # stage0_result exists, so the stage0 checkpoint is immutable while stage1 runs.
        seed = {}
        for source_name, name in [('stage0_best_parameters.npy', 'initial_parameters.npy'),
                                  ('stage0_best_material.npy', 'seed_material.npy')]:
            destination = a.out/name
            if destination.exists():
                raise FileExistsError('Preserve an existing frozen seed')
            shutil.copy2(a.geometry/source_name, destination)
            seed[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
        atomic_json(a.out/'source_seed.json', {'source_run': str(a.geometry),
            'RF_selection': selection, 'stage0_result': json.loads(result.read_text()),
            'files_sha256': seed, 'truth_used_for_initialization': False})
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1', NUMPY_MADVISE_HUGEPAGE='0',
                   PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:256', PYTHONPATH='.')
        argv = [PYTHON, '-u', '-m', 'nsrno.reconstruct_kinematic', '--out', str(a.out),
                '--resume-parameters', str(a.out/'initial_parameters.npy'),
                '--devices', *map(str, a.devices), '--steps', '64', '128',
                '--validate-every', '32', '--memory-limit-gib', str(a.memory_limit_gib),
                '--initial-objective', 'differential', '--initial-kinematic-weight', '1',
                '--initial-update', 'c_only']
        log = (a.out/'run.log').open('w')
        proc = subprocess.Popen(argv, cwd=root, env=env, stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        workflow = {'optimization_pid': proc.pid, 'argv': argv, 'source_seed': str(a.out/'source_seed.json'),
                    'original_terminal_status': original['status'],
                    'original_postprocess_terminal_status': post['status'],
                    'memory_MiB_before_launch': used, 'host_available_GiB': mem['MemAvailable']/1024**2}
        atomic_json(a.out/'workflow.json', workflow)
        post_log = (a.out/'postprocess.log').open('w')
        post_proc = subprocess.Popen([PYTHON, '-u', '-m', 'nsrno.run_kinematic_postprocess',
            '--out', str(a.out), '--pid', str(proc.pid)], cwd=root, env=env, stdout=post_log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        watcher_log = (a.out/'snapshot_watcher.log').open('w')
        watcher = subprocess.Popen([PYTHON, '-u', '-m', 'nsrno.watch_material_snapshots',
            '--out', str(a.out), '--pid', str(proc.pid), '--min-step', '32',
            '--label', 'Kinematic initialization / joint RF',
            '--process-module', 'nsrno.reconstruct_kinematic'], cwd=root, env=env, stdout=watcher_log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        workflow.update(postprocess_pid=post_proc.pid, snapshot_watcher_pid=watcher.pid)
        atomic_json(a.out/'workflow.json', workflow)
        update('launched', **workflow, goal_completion_requires_actual_three_maps=True)
    except BaseException:
        update('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
