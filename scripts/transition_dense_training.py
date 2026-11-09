"""Finish two nominal-band passes, then train the actual RF-energy band."""
from pathlib import Path
import os,sys,json,time,signal
import torch
root=Path('/home_nvme/zhuangyang/nsrno'); os.chdir(root)
nominal=root/'out/v5/forward_round8_dense'; full=root/'out/v5/forward_round8_dense_full'
full.mkdir(parents=True,exist_ok=True); start=time.monotonic()
while not (nominal/'cycle_2.pt').exists():
    if time.monotonic()-start>3600: raise TimeoutError('Nominal-band training did not finish two cycles')
    time.sleep(10)
time.sleep(2)
pid=json.load(open(nominal/'process.json'))['pid']
try: os.killpg(pid,signal.SIGTERM)
except ProcessLookupError: pass
initial=nominal/'last.pt'; saved=torch.load(initial,map_location='cpu',weights_only=True)
(nominal/'transition.json').write_text(json.dumps({'completed_cycles':2,'minimum_complete_coverage_observations':4620,
    'saved_step':saved['step'],'saved_observations':saved['step']*3,
    'reason':'extend nominal passband to actual99.99-percent RF energy band','initial_for_next_stage':'last.pt'},indent=2))
while not all((root/f'out/v5/conditioned_dense_full_v1/reference_complete_{j}.json').exists() for j in range(3)):
    if time.monotonic()-start>4200: raise TimeoutError('Extended homogeneous references incomplete')
    time.sleep(10)
time.sleep(3)
command=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--node_rank=0','--master_addr=127.0.0.1','--master_port=29978',
    '--nproc_per_node=3','-m','nsrno.train_dense_wave','--cycles','2','--iterations','32',
    '--root','out/v5/conditioned_dense_full_v1','--out','out/v5/forward_round8_dense_full',
    '--initial',str(initial),'--lr','.0005','--energy-weighted']
print(json.dumps({'starting_extended_band_training':command}),flush=True)
os.execv(sys.executable,command)
