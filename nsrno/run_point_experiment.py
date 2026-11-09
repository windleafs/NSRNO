"""Continue from acquisition generation to a reviewable physical inversion.

Generation and numerical agreement must finish before any reconstruction is
started. Truth is used only by the separate operator-agreement check.
"""
from pathlib import Path
from argparse import ArgumentParser
import json,os,subprocess,time,traceback

PYTHON='/home/zhuangyang/miniconda3/envs/py310/bin/python'


def main():
    p=ArgumentParser();p.add_argument('--generation-pid',type=int,required=True)
    p.add_argument('--response-pid',type=int,required=True)
    p.add_argument('--out',type=Path,default=Path('out/v5/reconstruction_point_env_v1_train_200'))
    a=p.parse_args();root=Path('out/v5/point_tx_3mhz_v1');status=a.out/'workflow.json'
    def save(**v):
        v['updated_unix_s']=time.time();temp=status.with_suffix('.partial.json');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(status)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',NUMPY_MADVISE_HUGEPAGE='0',PYTHONPATH='.')
    try:
        for pid,path,label in [(a.generation_pid,root/'index.json','generation'),(a.response_pid,Path('out/v5/time_triparam_v1/response_point3.npy'),'response')]:
            while not path.exists():
                save(status='waiting',dependency=label,pid=pid)
                proc=Path(f'/proc/{pid}/stat')
                if not proc.exists() or proc.read_text().split()[2]=='Z':raise RuntimeError(f'{label} ended before recording completion')
                time.sleep(15)
        meta=json.loads((root/'index.json').read_text())
        if meta['status']!='complete':raise RuntimeError('Incomplete acquisition index')
        save(status='checking_new_acquisition_operator')
        checkenv=env.copy();checkenv['CUDA_VISIBLE_DEVICES']='0'
        with (a.out/'check_point3.log').open('w') as log:
            subprocess.run([PYTHON,'-u','-m','nsrno.check_time_data','--source','point3','--sample','train_200','--angle','5'],env=checkenv,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads(Path('out/v5/time_triparam_v1/check_point3_train_200_5.json').read_text())
        if report['relative_rms']>1e-3:raise RuntimeError(f'Operator agreement failed: {report}')
        (a.out/'new_acquisition_operator_check.json').write_text(json.dumps(report,indent=2)+'\n')
        command=[PYTHON,'-u','-m','nsrno.reconstruct_time','--out',str(a.out),'--devices','0','1',
            '--iterations','12','12','20','24','--resume-parameters',str(a.out/'plane_fwi_initial_stage2_best_parameters.npy'),
            '--parameterization','multiscale','--optimizer','filtered_lbfgs','--memory-limit-gib','5.75',
            '--acquisition-mode','plane_point','--objective-schedule','envelope_to_waveform']
        with (a.out/'run.log').open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        with (a.out/'postprocess.log').open('w') as log:
            post=subprocess.Popen([PYTHON,'-u','-m','nsrno.run_time_postprocess','--out',str(a.out),'--pid',str(process.pid)],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(status='optimization_running',optimization_pid=process.pid,postprocess_pid=post.pid,command=command,
            goal_completion_requires_reconstruction_and_independent_alpha_counterexample=True)
        print(json.dumps({'optimization_pid':process.pid,'postprocess_pid':post.pid}),flush=True)
    except BaseException:
        save(status='failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
