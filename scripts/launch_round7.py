"""Launch bounded, logged round-7 jobs without retaining an SSH connection."""
from pathlib import Path
import json,os,subprocess,sys
root=Path('/home_nvme/zhuangyang/nsrno'); os.chdir(root)
out=root/'out/v5/conditioned_wave_v1'; processes=[]
for part in range(3):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(part),'PYTHONDONTWRITEBYTECODE':'1',
         'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(root)}
    command=[sys.executable,'scripts/prepare_wave_teachers.py','--part',str(part),'--parts','3']
    with (out/f'teachers_{part}.log').open('w') as log:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    processes.append({'pid':p.pid,'gpu':part,'command':command})
(out/'teacher_processes.json').write_text(json.dumps(processes,indent=2)); print(json.dumps(processes))
