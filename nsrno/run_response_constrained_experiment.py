"""Sequential constrained training and fixed-model response evaluations."""
from pathlib import Path
import argparse,ast,hashlib,json,math,os,subprocess,sys,time,traceback
from .run_angular_spectral_experiment import atomic


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('out/v5/response_constrained_experiment_v1'))
    p.add_argument('--model',type=Path,default=Path('out/v5/inverse_response_constrained_uno_v1'))
    p.add_argument('--after-root',type=Path,default=Path('out/v5/angular_spectral_training_pair_response_v1'))
    p.add_argument('--native-score-root',type=Path,default=Path('out/v5/native_inverse_angular_spectral_score_v1_train_200'))
    p.add_argument('--device',type=int,default=0)
    args=p.parse_args();prior=json.loads((args.after_root/'run.json').read_text())
    if prior['status']!='fixed_training_pairs_evaluated':raise RuntimeError('Require completed training-pair diagnosis')
    if Path(f"/proc/{prior['pid']}").exists():raise RuntimeError('Previous GPU diagnosis still exists')
    score=json.loads((args.native_score_root/'run.json').read_text())
    workers=json.loads((args.native_score_root/'worker_processes.json').read_text())
    for pid in [score['pid']]+workers['worker_pids']:
        if Path(f'/proc/{pid}').exists():raise RuntimeError('Previous native score tree still exists')
    pair_out=Path('out/v5/response_constrained_training_pairs_v1')
    for path in [args.out,args.model,pair_out]:
        if path.exists():raise RuntimeError(f'Inspect existing {path}; no implicit restart')
    names=['run_response_constrained_experiment.py','run_angular_spectral_experiment.py','train_response_constrained_inverse.py',
        'angular_spectral_inverse_uno.py','inverse_uno.py','uno_rf.py','angular_spectral_features.py',
        'evaluate_angular_spectral_inverse.py','evaluate_inverse_uno.py',
        'evaluate_training_parameter_pairs.py','compare_angular_spectral_inverse.py']
    hashes={}
    for name in names:
        path=Path(__file__).parent/name;ast.parse(path.read_text())
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    args.out.mkdir(parents=True)
    record={'status':'starting','pid':os.getpid(),'device':args.device,'model':str(args.model.resolve()),
        'pair_evaluation':str(pair_out.resolve()),'source_sha256':hashes,
        'allocator_cap_GiB':8,'model_per_card_limit_GiB':16,'one_GPU_stage_at_a_time':True,
        'patient_wave_solves':0,'joint_imaging_success_claimed':False,'created_unix_s':time.time()}
    atomic(args.out/'run.json',record)
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMPY_MADVISE_HUGEPAGE='0',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.',
        PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    index=Path('out/v5/inverse_angular_spectral_v1/index.json')
    stages=[
        ('training',args.model,[sys.executable,'-m','nsrno.train_response_constrained_inverse',
            '--index',str(index),'--after-root',str(args.after_root),'--out',str(args.model),
            '--device',str(args.device),'--steps','4000','--batch','4']),
        ('training_pairs',pair_out,[sys.executable,'-m','nsrno.evaluate_training_parameter_pairs',
            '--index',str(index),'--model',str(args.model),'--after-root',str(args.native_score_root),
            '--out',str(pair_out),'--device',str(args.device)]),
        ('evaluation',args.model/'evaluation',[sys.executable,'-m','nsrno.evaluate_angular_spectral_inverse',
            '--index',str(index),'--run',str(args.model),'--device',str(args.device),'--cap-gib','8'])]
    child=None
    try:
        for name,output,command in stages:
            for source,sha in hashes.items():
                if hashlib.sha256((Path(__file__).parent/source).read_bytes()).hexdigest()!=sha:
                    raise RuntimeError(f'Frozen experiment source changed: {source}')
            with open(args.out/f'{name}.log','x') as log:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
                record.update(status='running_'+name,stage=name,child_pid=child.pid,
                    stage_output=str(output.resolve()),stage_started_unix_s=time.time())
                atomic(args.out/'run.json',record)
                atomic(args.out/'worker_processes.json',{'coordinator_pid':os.getpid(),
                    'worker_pids':[child.pid],'devices':[args.device],'boot_id':boot})
                while child.poll() is None:
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:pass
                if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}; inspect its log')
                if Path(f'/proc/{child.pid}').exists():raise RuntimeError('Completed GPU child was not reaped')
                atomic(args.out/f'{name}.complete.json',{'pid':child.pid,'returncode':0,
                    'process_missing':True,'finished_unix_s':time.time()})
                child=None
        metric=json.loads((pair_out/'metrics.json').read_text())
        alpha=[r for r in metric['pairs'] if r['kind']=='alpha']
        constraints=[sum(r['maps_c_rho_alpha'][2]['relative_difference_error']**2 for r in alpha)/len(alpha)]
        constraints.extend(sum((r['maps_c_rho_alpha'][j]['predicted_difference_rms']/5)**2 for r in alpha)/len(alpha) for j in [0,1])
        targets=[.16,.04,.04]
        certificate={'constraint_values':constraints,'targets':targets,
            'satisfied':[value<=target for value,target in zip(constraints,targets)],
            'alpha_relative_RMS_error':math.sqrt(constraints[0]),
            'c_invariance_RMS_m_s':5*math.sqrt(constraints[1]),
            'rho_invariance_RMS_kg_m3':5*math.sqrt(constraints[2]),
            'fixed_final_EMA_evaluation':True,'joint_imaging_success_claimed':False}
        atomic(args.out/'final_response_constraints.json',certificate)
        record.update(status='constrained_training_and_evaluations_finished',
            training_response_constraints=certificate,finished_unix_s=time.time())
        atomic(args.out/'run.json',record)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(args.out/'run.json',record);raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
