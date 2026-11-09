"""Launch exactly once after the prior physical result is complete."""
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

root=Path('/home_nvme/zhuangyang/nsrno')
os.chdir(root)
source=root/'out/v5/reconstruction_native_gn_enriched_v1_train_200'
run=json.loads((source/'run.json').read_text())
assert run['status']=='planned_native_GN_steps_finished',run['status']
workers=json.loads((source/'worker_processes.json').read_text())
for pid in [run['pid']]+workers['worker_pids']:
    assert not Path(f'/proc/{pid}').exists(),f'Prior process {pid} still exists'
seed=source/'step001/material.npy'
update=json.loads((source/'step001/material_update.json').read_text())
assert update['accepted'] and abs(update['native_data_after']-.2260736902854553)<1e-12
for name in ['native_gn_block_worker.py','native_gn_block_pool.py','native_gn_block_metric.py',
             'native_gn_fresh_state.py','reconstruct_native_gn_nonlinear.py']:
    ast.parse((root/'nsrno'/name).read_text())
out=root/'out/v5/reconstruction_native_gn_nonlinear_v1_train_200'
launch=out.with_suffix('.launch.json')
logfile=out.with_suffix('.log')
assert not out.exists() and not launch.exists() and not logfile.exists(),'Inspect existing run; never relaunch blindly'
assert shutil.disk_usage(root).free>10*2**30,'Insufficient space for full-grid checkpoints'
args=[sys.executable,'-m','nsrno.reconstruct_native_gn_nonlinear',
    '--seed',str(seed),'--out',str(out),
    '--symbol-snapshot','out/v5/native_gn_spectral_v1/symbol.npy',
    '--preconditioner-statistics','out/v5/reconstruction_pressure_v1_train_200/step002/joint_normal.npz',
    '--calibration-record','out/v5/reconstruction_native_gn_secant_v1_train_200/step001/hvp00/progress.json',
    '--sample','train_200','--outer-steps','2','--inner-hvps','2',
    '--CPU-maximum','1200','--CPU-tolerance','.01','--true-QP-forcing','.3',
    '--trust','.02','.10','.35','--devices','0','1','2','--chunk','128','--cap-gib','10']
env=os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES',None)
env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    NUMPY_MADVISE_HUGEPAGE='0',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.',
    PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
with open(logfile,'x') as log:
    proc=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        env=env,start_new_session=True)
record={'pid':proc.pid,'out':str(out),'log':str(logfile),'args':args,
    'seed_SHA256':hashlib.sha256(seed.read_bytes()).hexdigest(),
    'previous_RF':update['native_data_after'],'created_unix_s':time.time(),
    'actual_native_job_started':True,'goal_achieved':False}
with open(launch,'x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record))
