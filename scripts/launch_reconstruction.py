from pathlib import Path
import os,sys,json,subprocess
root=Path('/home_nvme/zhuangyang/nsrno'); os.chdir(root)
out=root/'out/v5/reconstruction_round1_train_200'; out.mkdir(parents=True,exist_ok=True)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1',
     'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(root)}
command=[sys.executable,'-m','nsrno.reconstruct_material','--steps','200']
with (out/'reconstruction.log').open('w') as log:
    p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(out/'process.json').write_text(json.dumps({'pid':p.pid,'gpu':1,'command':command},indent=2))
print(json.dumps({'pid':p.pid,'log':str(out/'reconstruction.log')}))
