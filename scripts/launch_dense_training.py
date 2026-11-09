from pathlib import Path
import os,sys,json,subprocess
root=Path('/home_nvme/zhuangyang/nsrno'); os.chdir(root); out=root/'out/v5/forward_round8_dense'; out.mkdir(parents=True,exist_ok=True)
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(root),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
command=[sys.executable,'scripts/wait_dense_references.py']
with (out/'training.log').open('w') as log:
    p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(out/'process.json').write_text(json.dumps({'pid':p.pid,'command':command},indent=2)); print(json.dumps({'pid':p.pid}))
