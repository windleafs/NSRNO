from pathlib import Path
import subprocess,os,json
root=Path('/home_nvme/zhuangyang/nsrno'); out=root/'out/v5/wave_diagnosis_v1'; out.mkdir(parents=True,exist_ok=True)
if (out/'processes.json').exists(): raise FileExistsError('Diagnosis has already been started')
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(root),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
jobs=[('capture','0',['-m','nsrno.capture_wavefield']),('convergence','2',['scripts/solve_wave_reference.py'])]
records=[]
for name,gpu,args in jobs:
    with (out/f'{name}.log').open('ab') as handle:
        p=subprocess.Popen(['/home/zhuangyang/miniconda3/envs/py310/bin/python',*args],cwd=root,env={**env,'CUDA_VISIBLE_DEVICES':gpu},stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    records.append(dict(name=name,pid=p.pid,gpu=gpu))
(out/'processes.json').write_text(json.dumps(records)); print(json.dumps(records))
