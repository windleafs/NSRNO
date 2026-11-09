"""Sequential single-GPU feature/training/evaluation experiment; no side jobs."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback


def atomic(path,value):
    temporary=path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('out/v5/angular_spectral_experiment_v1'))
    p.add_argument('--after-root',type=Path,required=True)
    p.add_argument('--source-index',type=Path,default=Path('out/v5/inverse_focus_v1/index.json'))
    p.add_argument('--features',type=Path,default=Path('out/v5/inverse_angular_spectral_v1'))
    p.add_argument('--model',type=Path,default=Path('out/v5/inverse_angular_spectral_uno_v1'))
    p.add_argument('--device',type=int,default=0)
    args=p.parse_args()
    prior=json.loads((args.after_root/'run.json').read_text())
    if prior['status']!='planned_native_LBFGS_budget_finished':raise RuntimeError('Prior native experiment is not terminal')
    oldp=json.loads((args.after_root/'worker_processes.json').read_text())
    for pid in [prior['pid']]+oldp['worker_pids']:
        if Path(f'/proc/{pid}').exists():raise RuntimeError(f'Prior GPU tree still exists: {pid}')
    for name,sha in prior['source_sha256'].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f'Prior frozen runtime changed: {name}')
    if not prior['branch_improves_original_incumbent']:
        old=Path(prior['incumbent_root'])/'step001/material.npy'
        if hashlib.sha256((args.after_root/'current_material.npy').read_bytes()).digest()!=hashlib.sha256(old.read_bytes()).digest():
            raise RuntimeError('An unaccepted branch replaced the physical incumbent')
    for root in [args.out,args.features,args.model]:
        if root.exists():raise RuntimeError(f'Inspect existing {root}; never restart on an observation timeout')
    files=['angular_spectral_features.py','prepare_angular_spectral_inverse.py',
        'angular_spectral_inverse_uno.py','train_angular_spectral_inverse.py',
        'evaluate_angular_spectral_inverse.py','run_angular_spectral_experiment.py',
        'focus_features.py','inverse_uno.py','uno_rf.py','evaluate_inverse_uno.py']
    hashes={}
    for name in files:
        path=Path(__file__).parent/name;ast.parse(path.read_text())
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    args.out.mkdir(parents=True)
    run={'status':'starting','pid':os.getpid(),'device':args.device,
        'after_root':str(args.after_root.resolve()),'features':str(args.features.resolve()),
        'model':str(args.model.resolve()),'source_sha256':hashes,
        'source_index':str(args.source_index.resolve()),'stage_order':['features','training','evaluation'],
        'one_GPU_stage_at_a_time':True,'allocator_cap_GiB':8,'model_per_card_limit_GiB':16,
        'patient_wave_solves':0,'new_data_acquisition':False,
        'joint_reconstruction_success_claimed':False,'created_unix_s':time.time()}
    atomic(args.out/'run.json',run)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMPY_MADVISE_HUGEPAGE='0',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.',
        PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    index=args.features/'index.json'
    stages=[
        ('features',args.features,[sys.executable,'-m','nsrno.prepare_angular_spectral_inverse',
            '--index',str(args.source_index),'--out',str(args.features),'--after-root',str(args.after_root),
            '--device',str(args.device),'--cap-gib','8']),
        ('training',args.model,[sys.executable,'-m','nsrno.train_angular_spectral_inverse',
            '--index',str(index),'--out',str(args.model),'--steps','4000','--batch','4',
            '--device',str(args.device),'--cap-gib','8']),
        ('evaluation',args.model/'evaluation',[sys.executable,'-m','nsrno.evaluate_angular_spectral_inverse',
            '--index',str(index),'--run',str(args.model),'--device',str(args.device),'--cap-gib','8'])]
    child=None
    try:
        for stage,output,command in stages:
            for name,sha in hashes.items():
                if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=sha:
                    raise RuntimeError(f'Experiment runtime changed: {name}')
            with open(args.out/f'{stage}.log','x') as log:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
                run.update(status='running_'+stage,stage=stage,child_pid=child.pid,
                    stage_output=str(output.resolve()),stage_started_unix_s=time.time())
                atomic(args.out/'run.json',run)
                atomic(args.out/'worker_processes.json',{'coordinator_pid':os.getpid(),
                    'worker_pids':[child.pid],'devices':[args.device],'boot_id':boot})
                while child.poll() is None:
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:pass
                if child.returncode:raise RuntimeError(f'{stage} failed with exit {child.returncode}; inspect {stage}.log')
                if Path(f'/proc/{child.pid}').exists():raise RuntimeError('Completed GPU child was not reaped')
                atomic(args.out/f'{stage}.complete.json',{'pid':child.pid,'returncode':0,
                    'process_missing':True,'output':str(output.resolve()),'finished_unix_s':time.time()})
                child=None
        run.update(status='feature_training_and_evaluation_finished',finished_unix_s=time.time())
        atomic(args.out/'run.json',run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',run);raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
