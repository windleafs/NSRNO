from pathlib import Path
import os,sys,json,subprocess,shutil
root=Path('/home_nvme/zhuangyang/nsrno'); os.chdir(root)
out=root/'out/v5/reconstruction_round1_train_200_gel1mm'; out.mkdir(parents=True,exist_ok=True)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1',
     'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(root)}
command=[sys.executable,'-m','nsrno.reconstruct_material','--steps','200','--out',str(out),'--fixed-gel-mm','1']
with (out/'reconstruction.log').open('w') as log:
    p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(out/'process.json').write_text(json.dumps({'pid':p.pid,'gpu':2,'command':command},indent=2))
with (out/'postprocess.log').open('w') as log:
    post=subprocess.Popen([sys.executable,'scripts/postprocess_reconstruction.py','--out',str(out)],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env={**env,'CUDA_VISIBLE_DEVICES':'0'},start_new_session=True)
(out/'postprocess_process.json').write_text(json.dumps({'pid':post.pid,'gpu':0},indent=2))
print(json.dumps({'pid':p.pid,'postprocess_pid':post.pid,'out':str(out)}))
