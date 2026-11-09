from pathlib import Path
import os,sys,json,subprocess
root=Path('/home_nvme/zhuangyang/nsrno'); os.chdir(root); out=root/'out/v5/reconstruction_round2_dense_train_200'; out.mkdir(parents=True,exist_ok=True)
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(root),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
with (out/'reconstruction.log').open('w') as log:
    p=subprocess.Popen([sys.executable,'scripts/run_dense_reconstruction.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(out/'process.json').write_text(json.dumps({'pid':p.pid},indent=2)); print(json.dumps({'pid':p.pid}))
