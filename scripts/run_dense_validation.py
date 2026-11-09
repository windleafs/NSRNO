"""Run the complete, reviewable forward comparison after training finishes."""
from pathlib import Path
import json,os,subprocess,sys,time
root=Path('/home_nvme/zhuangyang/nsrno'); os.chdir(root); out=root/'out/v5/forward_round8_dense_full'
start=time.monotonic()
while not (out/'training_complete.json').exists():
    if time.monotonic()-start>7200: raise TimeoutError('Extended training did not complete')
    time.sleep(10)
workers=[]; processes=[]
for part in range(3):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(part),'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(root),
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    command=[sys.executable,'scripts/evaluate_dense_forward.py','--part',str(part),'--out',str(out),
        '--root','out/v5/conditioned_dense_full_v1']
    with (out/f'validation_{part}.log').open('w') as log:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    workers.append(p); processes.append({'pid':p.pid,'gpu':part,'command':command})
(out/'validation_processes.json').write_text(json.dumps(processes,indent=2))
codes=[p.wait() for p in workers]
if any(codes): raise RuntimeError('Dense forward evaluation failed: '+str(codes))
subprocess.run([sys.executable,'scripts/summarize_dense_forward.py'],check=True)
print('Dense forward accuracy comparison complete.',flush=True)
