"""Start the authorized training pair after both feature caches are complete."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home_nvme/zhuangyang/nsrno')
PY='/home/zhuangyang/miniconda3/envs/py310/bin/python'
OUT=ROOT/'out/v5/forward_round4_checks'
index=json.loads((ROOT/'out/v5/forward_cache_v1/index.json').read_text())
shared=ROOT/'out/v5/forward_ray_shared_fp32_v1'

def write(name,value):
    temp=OUT/(name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(OUT/name)

while True:
    ready=all((ROOT/('out/v5/forward_ray_'+mode+'_v1/completion.json')).exists() for mode in ('material','reference'))
    if ready:
        ready=all((shared/'baseline'/(r['id']+'.npy')).exists() and (shared/'background'/(r['id']+'.npy')).exists() for r in index['records'])
    if ready: break
    processes=json.loads((OUT/'cache_pids.json').read_text())
    running=[]
    for row in processes:
        p=Path('/proc')/str(row['pid'])/'cmdline'
        if p.exists() and b'nsrno.prepare_ray' in p.read_bytes(): running.append(row['pid'])
    write('pipeline.json',{'stage':'waiting_for_features','running_cache_pids':running,'updated_unix':time.time()})
    if not running: raise RuntimeError('Caches incomplete and no cache workers remain')
    time.sleep(10)

processes=[]
for mode,gpu in [('material','0'),('reference','1')]:
    folder=ROOT/('out/v5/forward_round4_'+mode); folder.mkdir(exist_ok=True)
    if (folder/'completion.json').exists(): continue
    argv=[PY,'-u','-m','nsrno.train_ray','--delay',mode,'--features',str(ROOT/('out/v5/forward_ray_'+mode+'_v1')),'--out',str(folder)]
    if (folder/'last.pt').exists(): argv.append('--resume')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=gpu)
    with (folder/'train.log').open('ab') as log:
        p=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    processes.append((mode,p))
write('training_pids.json',[{'mode':mode,'pid':p.pid} for mode,p in processes])
write('pipeline.json',{'stage':'training','pids':[p.pid for _,p in processes],'updated_unix':time.time()})
for mode,p in processes:
    code=p.wait()
    if code: raise RuntimeError('%s training exited %d'%(mode,code))
write('pipeline.json',{'stage':'complete','updated_unix':time.time()})
print('Both training runs and their final evaluations completed.',flush=True)
