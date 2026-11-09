from pathlib import Path
import os,sys,subprocess,json
root=Path('/home_nvme/zhuangyang/nsrno'); os.chdir(root); out=root/'out/v5/forward_round8_dense_full'
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(root),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
with (out/'validation_driver.log').open('w') as log:
    p=subprocess.Popen([sys.executable,'scripts/run_dense_validation.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(out/'validation_driver_process.json').write_text(json.dumps({'pid':p.pid},indent=2)); print(json.dumps({'pid':p.pid}))
