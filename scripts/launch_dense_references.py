from pathlib import Path
import os,json,sys,subprocess
root=Path('/home_nvme/zhuangyang/nsrno'); os.chdir(root); out=root/'out/v5/conditioned_dense_v1'
records=[]
for part in range(3):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(part),'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(root),
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    command=[sys.executable,'scripts/dense_reference_worker.py','--part',str(part)]
    with (out/f'capture_worker{part}.log').open('w') as log:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    records.append({'pid':p.pid,'gpu':part,'command':command})
(out/'reference_processes.json').write_text(json.dumps(records,indent=2)); print(json.dumps(records))
