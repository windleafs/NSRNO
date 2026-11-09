"""One GPU workflow for paired-window training, field evaluation and CPU maps."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from .native_gn_json import atomic


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('out/v5/raw_rf_window_balanced_experiment_v1'))
    p.add_argument('--model',type=Path,default=Path('out/v5/inverse_raw_rf_window_balanced_v1'))
    p.add_argument('--index',type=Path,default=Path('out/v5/raw_rf_window_balanced_data_v1/index.json'))
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--max-seconds',type=float,default=5400)
    p.add_argument('--device',type=int,default=0)
    a=p.parse_args();preparation=json.loads((a.index.parent/'run.json').read_text())
    previous=json.loads(Path('out/v5/raw_rf_inverse_experiment_v1/run.json').read_text())
    diagnosis=json.loads(Path('out/v5/raw_rf_record_window_diagnosis_v1/run.json').read_text())
    if preparation['status']!='paired_RF_windows_prepared' or previous['status']!='raw_RF_inverse_training_and_evaluations_finished' or diagnosis['status']!='fixed_window_diagnosis_finished':
        raise RuntimeError('Complete original experiment, diagnosis and CPU cache first')
    for pid in {preparation['pid'],previous['pid'],previous.get('child_pid'),diagnosis['pid']}:
        if pid and Path(f'/proc/{pid}').exists():raise RuntimeError('Existing workflow has not exited')
    if a.out.exists() or a.model.exists():raise RuntimeError('Preserve existing experiment')
    if digest(a.index)!=preparation['index_SHA256']:raise RuntimeError('Prepared input index changed')
    names=['run_raw_rf_window_balanced_experiment.py','train_raw_rf_window_balanced.py','train_raw_rf_inverse.py',
        'raw_rf_inverse_uno.py','uno_rf.py','inverse_uno.py','pressure_resources.py','evaluate_raw_rf_inverse.py',
        'evaluate_raw_rf_window_views.py','summarize_raw_rf_inverse_results.py','evaluate_inverse_uno.py',
        'evaluate_time_material.py','v5_materials.py','native_gn_json.py']
    hashes={}
    for name in names:
        source=Path(__file__).parent/name;ast.parse(source.read_text());hashes[name]=digest(source)
    old_training=json.loads(Path('out/v5/inverse_raw_rf_uno_v1/run.json').read_text())
    if hashes['raw_rf_inverse_uno.py']!=old_training['source_sha256']['raw_rf_inverse_uno.py']:
        raise RuntimeError('Keep the measured architecture unchanged')
    a.out.mkdir()
    for name in names:(a.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    state={'status':'starting','pid':os.getpid(),'device':a.device,'model':str(a.model.resolve()),
        'experiment_type':'paired_record_window_finetune','index':str(a.index.resolve()),'index_SHA256':digest(a.index),
        'source_sha256':hashes,'planned_steps':a.steps,'training_walltime_budget_seconds':a.max_seconds,
        'one_optimizer_update_per_step':True,'alternate_independent_and_window_tasks':True,
        'only_one_GPU_stage_at_a_time':True,'allocator_cap_GiB':8,'model_per_card_limit_GiB':16,
        'warmstart':str(Path('out/v5/inverse_raw_rf_uno_v1/weights.pt').resolve()),
        'warmstart_SHA256':digest('out/v5/inverse_raw_rf_uno_v1/weights.pt'),'new_wave_solves':0,
        'development_used_for_method_design':True,'reserved_test_case_unread':'Neg_47_Left',
        'joint_imaging_success_claimed':False,'created_unix_s':time.time()}
    atomic(a.out/'run.json',state);child=None
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMPY_MADVISE_HUGEPAGE='0',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.',
        PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    def stage(name,module,arguments,output,cpu=False):
        nonlocal child
        for filename,expected in hashes.items():
            if digest(Path(__file__).parent/filename)!=expected:raise RuntimeError('Frozen source changed: '+filename)
        command=[sys.executable,'-m','nsrno.'+module,*arguments]
        with (a.out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env={**env,'CUDA_VISIBLE_DEVICES':''} if cpu else env)
            state.update(status='running_'+name,stage=name,child_pid=child.pid,
                stage_output=str(output.resolve()),stage_started_unix_s=time.time())
            atomic(a.out/'run.json',state);code=child.wait()
        if code:raise RuntimeError(f'{name} exited with {code}; inspect preserved stage log')
        if Path(f'/proc/{child.pid}').exists():raise RuntimeError('Stage child was not reaped')
    try:
        stage('training','train_raw_rf_window_balanced',['--index',str(a.index),'--out',str(a.model),
            '--steps',str(a.steps),'--max-seconds',str(a.max_seconds),'--device',str(a.device)],a.model)
        stage('evaluation','evaluate_raw_rf_inverse',['--run',str(a.model),'--device',str(a.device)],a.model/'evaluation')
        stage('window_evaluation','evaluate_raw_rf_window_views',['--run',str(a.model),'--device',str(a.device)],a.model/'window_evaluation')
        shutil.copyfile('out/v5/inverse_uno_v1/evaluation/uniform_initial_material.npy',a.model/'evaluation/uniform_initial_material.npy')
        for sample in ['train_200','val_059']:
            prefix=a.model/'evaluation'/f'{sample}_native_full'
            stage('native_maps_'+sample,'evaluate_time_material',
                [str(a.model/'evaluation'/f'{sample}_initial.material.npy'),'--sample',sample,'--out',str(prefix),
                 '--estimate-label','Paired-window RF inverse UNO'],prefix,cpu=True)
        state.update(status='raw_RF_inverse_training_and_evaluations_finished',finished_unix_s=time.time())
        atomic(a.out/'run.json',state);print(json.dumps(state),flush=True)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time());atomic(a.out/'run.json',state);raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
