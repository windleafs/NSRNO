from pathlib import Path
import subprocess,os,json
root=Path('/home_nvme/zhuangyang/nsrno'); out=root/'out/v5/forward_cache_3mhz_v1'; out.mkdir(parents=True,exist_ok=True)
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
with (out/'prepare.log').open('ab') as f:
    process=subprocess.Popen(['/home/zhuangyang/miniconda3/envs/py310/bin/python','-m','nsrno.prepare_low_frequency','--wait'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(out/'pid.json').write_text(json.dumps({'pid':process.pid})); print(process.pid)
