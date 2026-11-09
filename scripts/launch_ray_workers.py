"""Resume disjoint cache shards and the authorized training pipeline."""
from pathlib import Path
import json
import os
import subprocess

root=Path('/home_nvme/zhuangyang/nsrno')
folder=root/'out/v5/forward_round4_checks'
py='/home/zhuangyang/miniconda3/envs/py310/bin/python'
if (folder/'cache_pids.json').exists():
    for row in json.loads((folder/'cache_pids.json').read_text()):
        process=Path('/proc')/str(row['pid'])/'cmdline'
        if process.exists() and b'nsrno.prepare_ray' in process.read_bytes():
            raise RuntimeError('A previously launched cache worker is still running')
processes=[]
for mode in ('material','reference'):
    out=root/('out/v5/forward_ray_'+mode+'_v1')
    if (out/'completion.json').exists(): continue
    for shard in range(3):
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=str(shard))
        argv=[py,'-u','-m','nsrno.prepare_ray','--delay',mode,'--out',str(out),'--shards','3','--shard-id',str(shard)]
        with (out/('cache_resume_%d.log'%shard)).open('ab') as log:
            p=subprocess.Popen(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append({'mode':mode,'gpu':shard,'shard':shard,'pid':p.pid})
(folder/'cache_pids.json').write_text(json.dumps(processes,indent=2)+'\n')
pipeline_running=False
if (folder/'pipeline_pid.json').exists():
    pid=json.loads((folder/'pipeline_pid.json').read_text())['pid']
    process=Path('/proc')/str(pid)/'cmdline'
    pipeline_running=process.exists() and b'scripts/run_ray_pipeline.py' in process.read_bytes()
if not pipeline_running:
    with (folder/'pipeline_resume.log').open('ab') as log:
        p=subprocess.Popen([py,'-u','scripts/run_ray_pipeline.py'],cwd=root,stdout=log,stderr=subprocess.STDOUT,
                            start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    (folder/'pipeline_pid.json').write_text(json.dumps({'pid':p.pid})+'\n')
print(json.dumps(processes),flush=True)
