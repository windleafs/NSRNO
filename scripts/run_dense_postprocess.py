from pathlib import Path
import os,sys,json,time,subprocess
root=Path('/home_nvme/zhuangyang/nsrno'); os.chdir(root); out=root/'out/v5/reconstruction_round2_dense_train_200'
start=time.monotonic()
while not (out/'completion.json').exists():
    if time.monotonic()-start>43200: raise TimeoutError('Dense reconstruction incomplete after12 hours')
    time.sleep(15)
# Original time-domain simulation is both independent of the frequency solver
# and much cheaper for complete broadband replay than hundreds of new solves.
processes=[]; workers=[]
for part in range(3):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(part),'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(root),
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    command=[sys.executable,'scripts/replay_dense_reconstruction.py','--part',str(part)]
    with (out/f'physical_{part}.log').open('w') as log:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    processes.append({'pid':p.pid,'gpu':part,'command':command}); workers.append(p)
(out/'physical_processes.json').write_text(json.dumps(processes,indent=2))
codes=[p.wait() for p in workers]
if any(codes): raise RuntimeError('Final complete-band RF evaluation failed: '+str(codes))
subprocess.run([sys.executable,'scripts/summarize_time_replay.py'],check=True)
# Material truth is first accessed here, after RF-only checkpoint selection.
subprocess.run([sys.executable,'scripts/plot_dense_reconstruction.py'],check=True)
(out/'postprocess_complete.json').write_text(json.dumps({'completed':True}))
print('Dense reconstruction images and complete RF evaluation finished.',flush=True)
