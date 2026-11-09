"""Foreground SSH supervisor: profile, recover within the old budget, then map outputs.

Own children have separate process groups. Exit codes/signals are retained.
No automatic training restart occurs after an abnormal child exit.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback


def atomic(path, value):
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(value):
    try:
        print(json.dumps(value), flush=True)
    except BrokenPipeError:
        sys.stdout = open(os.devnull, 'w')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--profile-out', type=Path, required=True)
    p.add_argument('--train-out', type=Path, required=True)
    p.add_argument('--finalizer-out', type=Path, required=True)
    p.add_argument('--summary-out', type=Path, required=True)
    a = p.parse_args()
    allowed = (Path.cwd() / 'out/v5').resolve()
    for path in [a.out,a.profile_out,a.train_out,a.finalizer_out,a.summary_out]:
        path.resolve().relative_to(allowed)
        if path.exists():
            raise RuntimeError('Inspect existing supervised outputs; never replay implicitly')
    receipt = json.loads(a.receipt.read_text())
    if not receipt['PID_verified_gone'] or Path('/proc',str(receipt['pid'])).exists():
        raise RuntimeError('Previous trainer is not confirmed gone')
    deadline = receipt['original_shared_deadline_unix_s']
    if time.time() >= deadline - 30:
        raise RuntimeError('Original remaining budget is exhausted')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Another GPU job is still live')
    a.out.mkdir()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    source_names = ['run_amplitude_recovery.py','train_amplitude_recovery.py',
                    'evaluate_amplitude_angular_inverse.py','finalize_amplitude_angular_inverse.py']
    hashes = {n:digest(Path(__file__).parent/n) for n in source_names}
    run = {
        'status':'starting_supervised_recovery','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        'boot_id':boot,'created_unix_s':time.time(),'source_SHA256':hashes,
        'receipt_SHA256':digest(a.receipt),'training_out':str(a.train_out.resolve()),
        'profile_out':str(a.profile_out.resolve()),'summary_out':str(a.summary_out.resolve()),
        'finalizer_out':str(a.finalizer_out.resolve()),
        'original_shared_deadline_unix_s':deadline,
        'optimizer_work_already_spent':receipt['total_completed_optimizer_work'],
        'maximum_remaining_updates':2400-receipt['total_completed_optimizer_work'],
        'available_parent_EMA_step':600,'optimizer_work_lost_without_weights':459,
        'per_card_limit_GiB':16,'new_wave_solves':0,'new_material_derivatives':0,
        'automatic_training_restart_enabled':False,'coordinator_GPU_work':False,
        'foreground_SSH_owns_supervisor':True,'stages':[],
        'joint_imaging_success_claimed':False,'reserved_case_unread':'Neg_47_Left'}
    atomic(a.out/'run.json',run)
    active=None; stopping=0
    signal.signal(signal.SIGHUP,signal.SIG_IGN)

    def stop_group(signum,frame):
        nonlocal stopping
        stopping=int(signum)
        if active is not None and active.poll() is None:
            try:os.killpg(active.pid,signum)
            except ProcessLookupError:pass
    signal.signal(signal.SIGINT,stop_group)
    signal.signal(signal.SIGTERM,stop_group)

    def stage(name,command,maximum_seconds):
        nonlocal active
        if stopping:
            raise InterruptedError('Supervisor stop requested before next stage')
        for n,h in hashes.items():
            if digest(Path(__file__).parent/n)!=h:
                raise RuntimeError('Supervised source changed: '+n)
        env=os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   NUMPY_MADVISE_HUGEPAGE='0',PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH='.',PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128',
                   CUDA_VISIBLE_DEVICES='0' if name in ['profile','training'] else '')
        logpath=a.out/(name+'.log')
        with logpath.open('x') as log:
            active=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,env=env,start_new_session=True)
        ticks=int(Path('/proc',str(active.pid),'stat').read_text().rsplit(')',1)[1].split()[19])
        info={'name':name,'pid':active.pid,'start_ticks':ticks,'boot_id':boot,
              'command':command,'started_unix_s':time.time(),'log':str(logpath.resolve())}
        run.update(status=name,child=info);atomic(a.out/'run.json',run);emit(info)
        started=time.time();sent_timeout=False;last_step=None
        while active.poll() is None:
            now=time.time()
            if now-started>maximum_seconds and not sent_timeout:
                sent_timeout=True
                run['supervisor_timeout_signal']={'signal':'SIGTERM','unix_s':now,'stage':name}
                os.killpg(active.pid,signal.SIGTERM)
            if sent_timeout and now-started>maximum_seconds+60:
                os.killpg(active.pid,signal.SIGKILL)
                active.wait(timeout=30)
                break
            if name=='training':
                progress=a.train_out/'progress.json'
                if progress.exists():
                    item=json.loads(progress.read_text())
                    run['last_training_progress']=item
                    if item['step']!=last_step:
                        last_step=item['step']
                        emit({'stage':name,'step':item['step'],
                              'total_optimizer_work':item['global_optimization_updates'],
                              'latest_checkpoint_step':json.loads((a.train_out/'run.json').read_text()).get('latest_checkpoint_step')})
            run['observed_unix_s']=now;atomic(a.out/'run.json',run)
            time.sleep(5)
        code=active.wait()
        info.update(returncode=code,finished_unix_s=time.time(),
                    signal_name=signal.Signals(-code).name if code<0 else None)
        run['stages'].append(info);run['child']=None
        atomic(a.out/'run.json',run);emit(info);active=None
        if code or sent_timeout or stopping:
            raise RuntimeError('Child stopped; exact return code retained, no automatic restart')
        return info

    try:
        common=['--index','out/v5/conditional_secant_data_v1/index.json',
                '--parent-root',receipt['latest_available_parent_root'],
                '--receipt',str(a.receipt),'--deadline-unix-s',str(deadline),
                '--steps',str(run['maximum_remaining_updates'])]
        stage('profile',[sys.executable,'-m','nsrno.train_amplitude_recovery',
                         *common,'--out',str(a.profile_out),'--profile-only'],120)
        prof=json.loads((a.profile_out/'run.json').read_text())
        if (prof['status']!='conditional_secant_full_step_profiled' or
            not prof.get('profile_full_checkpoint_save_and_restore_passed') or
            prof['peak_reserved_GiB']>=8):
            raise RuntimeError('Full training/checkpoint profile did not pass')
        if Path('/proc',str(prof['pid'])).exists():
            raise RuntimeError('Profile process still exists')
        needed=max(2*prof['latest_checkpoint_bytes']+4*2**20,
                   prof['latest_checkpoint_bytes']+80*2**20)
        import shutil
        if shutil.disk_usage('.').free<2*2**30+400*2**20+needed:
            raise RuntimeError('Not enough reserve for atomic full checkpoints and final predictions')
        if time.time()>=deadline-30:
            raise RuntimeError('Original budget expired after profiling')
        release_started=time.time()
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            if time.time()-release_started>30 or time.time()>=deadline-30:
                raise RuntimeError('GPU has not been released after the profile')
            time.sleep(1)
        stage('training',[sys.executable,'-m','nsrno.train_amplitude_recovery',
                          *common,'--out',str(a.train_out),'--profile',str(a.profile_out)],
              max(1,deadline-time.time())+30)
        trained=json.loads((a.train_out/'run.json').read_text())
        if trained['status']!='conditional_secant_training_finished':
            raise RuntimeError('Training stopped with checkpoint; no evaluation started')
        stage('evaluation_and_maps',[sys.executable,'-m','nsrno.finalize_amplitude_angular_inverse',
              '--train-root',str(a.train_out),
              '--comparison-root','out/v5/current_ray_conditional_inverse_uno_v1',
              '--summary-out',str(a.summary_out),'--out',str(a.finalizer_out)],2400)
        final=json.loads((a.finalizer_out/'run.json').read_text())
        if final['status']!='fixed_predictions_and_CPU_maps_ready':
            raise RuntimeError('Final maps are not all ready')
        run.update(status='supervised_recovery_maps_ready',finished_unix_s=time.time(),
                   final_comparison=final['comparison_file'],
                   actual_visual_review_still_required=True,joint_imaging_success_claimed=False)
        atomic(a.out/'run.json',run);emit(run)
    except BaseException:
        run.update(status='supervised_recovery_needs_inspection',error=traceback.format_exc(),
                   observed_unix_s=time.time(),stop_requested_signal=stopping)
        atomic(a.out/'run.json',run);emit(run)
        raise


if __name__=='__main__':
    main()
