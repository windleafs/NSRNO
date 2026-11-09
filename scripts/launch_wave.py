from pathlib import Path
from argparse import ArgumentParser
import subprocess,os,json
p=ArgumentParser(); p.add_argument('--preconditioner',default='shifted'); p.add_argument('--name',default='forward_round5_wave_v1'); p.add_argument('--attenuation',default='imaginary'); args=p.parse_args()
root=Path('/home_nvme/zhuangyang/nsrno'); out=root/'out/v5'/args.name; out.mkdir(parents=True,exist_ok=True)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
if (out/'run.json').exists(): raise FileExistsError('Training already started')
with (out/'train.log').open('ab') as f:
    process=subprocess.Popen(['/home/zhuangyang/miniconda3/envs/py310/bin/python','-m','nsrno.train_wave','--out',str(out),'--steps','240','--iterations','8','--preconditioner',args.preconditioner,'--attenuation',args.attenuation],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(out/'pid.json').write_text(json.dumps({'pid':process.pid}))
print(process.pid)
