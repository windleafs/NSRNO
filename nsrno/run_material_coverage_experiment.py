"""Wait for acquisition, then compare matched 60/90 us UNO fits sequentially."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback


def atomic(path,value):
    temporary=path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def certificate(pair_file):
    metric=json.loads(pair_file.read_text());result={}
    scales=[5.,5.,.05]
    for changed,kind in enumerate(['pure_c','pure_rho','alpha']):
        chosen=[r for r in metric['pairs'] if r['kind']==kind]
        other=[j for j in range(3) if j!=changed]
        values=[sum(r['maps_c_rho_alpha'][changed]['relative_difference_error']**2 for r in chosen)/len(chosen)]
        values += [sum((r['maps_c_rho_alpha'][j]['predicted_difference_rms']/scales[j])**2
                       for r in chosen)/len(chosen) for j in other]
        result[kind]={'pairs':len(chosen),'changed_parameter':changed,'invariant_parameters':other,
                      'values':values,'targets':[.16,.04,.04],
                      'satisfied':[v<=t for v,t in zip(values,[.16,.04,.04])],
                      'changed_relative_RMS_error':math.sqrt(values[0]),
                      'invariant_RMS_in_physical_units':[scales[j]*math.sqrt(v) for j,v in zip(other,values[1:])]}
    return {'fixed_final_EMA_evaluation':True,'training_pairs':result,
            'joint_imaging_success_claimed':False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('out/v5/independent_material_3mhz_v1'))
    p.add_argument('--out',type=Path,default=Path('out/v5/material_coverage_experiment_v1'))
    p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    if a.out.exists():raise RuntimeError('Inspect the already scheduled coverage experiment')
    generation_root=a.root/'generation'
    if not (generation_root/'run.json').exists():raise RuntimeError('Start the explicit acquisition first')
    names=['run_material_coverage_experiment.py','prepare_material_coverage_features.py','record_window_features.py',
           'angular_spectral_features.py','focus_features.py','rf_representation.py',
           'train_material_coverage_inverse.py','angular_spectral_inverse_uno.py','inverse_uno.py','uno_rf.py',
           'evaluate_training_parameter_pairs.py','compare_angular_spectral_inverse.py',
           'evaluate_angular_spectral_inverse.py','evaluate_inverse_uno.py','evaluate_time_material.py','v5_materials.py']
    hashes={}
    for name in names:
        file=Path(__file__).parent/name;ast.parse(file.read_text());hashes[name]=sha(file)
    plan_sha=sha(a.root/'material_plan.json')
    warmstart=Path('out/v5/inverse_response_constrained_uno_v1/weights.pt')
    warm_sha=sha(warmstart)
    a.out.mkdir()
    for name in names:(a.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    state={'status':'waiting_for_generation','pid':os.getpid(),'root':str(a.root.resolve()),
           'device':a.device,'source_sha256':hashes,'material_plan_SHA256':plan_sha,
           'warmstart':str(warmstart.resolve()),'warmstart_SHA256':warm_sha,
           'record_window_controls_us':[60,90],'steps_per_control':4000,
           'identical_initial_weights_optimizer_reset_seed_and_training_policy':True,
           'allocator_cap_GiB':8,'model_per_card_limit_GiB':16,
           'only_one_GPU_stage_at_a_time':True,'development_used_for_method_design':True,
           'reserved_test_case_unread':'Neg_47_Left','joint_imaging_success_claimed':False,
           'created_unix_s':time.time()}
    atomic(a.out/'run.json',state)
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               NUMPY_MADVISE_HUGEPAGE='0',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.',
               PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    child=None
    try:
        while True:
            generation=json.loads((generation_root/'run.json').read_text())
            if generation['status']=='failed':raise RuntimeError('Acquisition failed; preserve its exact shot ledger')
            if generation['material_plan_SHA256']!=plan_sha:raise RuntimeError('Acquisition material plan changed')
            if generation['status']=='independent_material_generation_finished':
                if all(not Path(f'/proc/{pid}').exists() for pid in [generation['pid'],generation['child_pid']]):break
            elif not Path(f"/proc/{generation['pid']}").exists():
                raise RuntimeError('Acquisition process disappeared without a completed status; inspect before continuing')
            time.sleep(20)
        stages=[('features',a.root/'focus',[sys.executable,'-m','nsrno.prepare_material_coverage_features',
                  '--root',str(a.root),'--device',str(a.device)],False)]
        controls={}
        native_score=Path('out/v5/native_inverse_angular_spectral_score_v1_train_200')
        for window in [60,90]:
            model=Path(f'out/v5/inverse_material_coverage_{window}_v1')
            pairs=Path(f'out/v5/material_coverage_training_pairs_{window}_v1')
            controls[str(window)]={'model':str(model.resolve()),'training_pairs':str(pairs.resolve())}
            index=a.root/'focus'/f'index{window}.json'
            stages += [
                (f'training_{window}',model,[sys.executable,'-m','nsrno.train_material_coverage_inverse',
                 '--index',str(index),'--out',str(model),'--warmstart',str(warmstart),'--root',str(a.root),
                 '--device',str(a.device),'--steps','4000'],False),
                (f'training_pairs_{window}',pairs,[sys.executable,'-m','nsrno.evaluate_training_parameter_pairs',
                 '--index',str(index),'--model',str(model),'--after-root',str(native_score),
                 '--out',str(pairs),'--device',str(a.device)],False),
                (f'evaluation_{window}',model/'evaluation',[sys.executable,'-m','nsrno.evaluate_angular_spectral_inverse',
                 '--index',str(index),'--run',str(model),'--device',str(a.device),'--cap-gib','8'],False)]
        state['controls']=controls;atomic(a.out/'run.json',state)
        for name,output,command,cpu in stages:
            for source,checksum in hashes.items():
                if sha(Path(__file__).parent/source)!=checksum:raise RuntimeError(f'Frozen source changed: {source}')
            if sha(warmstart)!=warm_sha:raise RuntimeError('Warm start changed; controls must start identically')
            if sha(a.root/'material_plan.json')!=plan_sha:raise RuntimeError('Fixed material plan changed')
            with (a.out/f'{name}.log').open('x') as log:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
                state.update(status='running_'+name,stage=name,child_pid=child.pid,
                             stage_output=str(output.resolve()),stage_started_unix_s=time.time())
                atomic(a.out/'run.json',state)
                atomic(a.out/'worker_processes.json',{'coordinator_pid':os.getpid(),'worker_pids':[child.pid],
                       'devices':[a.device],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
                while child.poll() is None:
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:pass
                if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}; inspect its log')
                if Path(f'/proc/{child.pid}').exists():raise RuntimeError('Previous GPU child must actually be reaped')
                atomic(a.out/f'{name}.complete.json',{'pid':child.pid,'returncode':0,'process_missing':True,'finished_unix_s':time.time()})
                child=None
            if name.startswith('evaluation_'):
                window=name.split('_')[-1];model=Path(controls[window]['model'])
                cert=certificate(Path(controls[window]['training_pairs'])/'metrics.json')
                atomic(a.out/f'final_training_constraints_{window}.json',cert)
                uniform=model/'evaluation/uniform_initial_material.npy'
                if uniform.exists():raise RuntimeError('Preserve existing baseline')
                shutil.copyfile('out/v5/inverse_uno_v1/evaluation/uniform_initial_material.npy',uniform)
                cpu_env={**env,'CUDA_VISIBLE_DEVICES':''}
                for sample in ['train_200','val_059']:
                    command=[sys.executable,'-m','nsrno.evaluate_time_material',
                             str(model/'evaluation'/f'{sample}_initial.material.npy'),'--sample',sample,
                             '--out',str(model/'evaluation'/f'{sample}_native_full'),
                             '--estimate-label',f'Independent-material UNO, {window} us']
                    with (a.out/f'native_maps_{window}_{sample}.log').open('x') as log:
                        subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=cpu_env,check=True)
        state.update(status='material_coverage_training_and_evaluations_finished',finished_unix_s=time.time())
        atomic(a.out/'run.json',state)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(a.out/'run.json',state);raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
