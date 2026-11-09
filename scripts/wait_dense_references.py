from pathlib import Path
import sys,os,time,json
root=Path('out/v5/conditioned_dense_v1'); start=time.monotonic()
while not all((root/f'reference_complete_{j}.json').exists() for j in range(3)):
    if time.monotonic()-start>1200: raise TimeoutError('Dense references incomplete')
    time.sleep(10)
if not json.load(open(root/'physics_checks.json'))['passed']: raise RuntimeError('Physics checks failed')
os.execv(sys.executable,[sys.executable,'-m','torch.distributed.run','--nnodes=1','--node_rank=0',
    '--master_addr=127.0.0.1','--master_port=29977','--nproc_per_node=3','-m','nsrno.train_dense_wave',
    '--cycles','3','--iterations','32'])
