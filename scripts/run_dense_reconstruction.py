from pathlib import Path
import os,json,sys,time
root=Path('/home_nvme/zhuangyang/nsrno'); os.chdir(root); forward=root/'out/v5/forward_round8_dense_full'
start=time.monotonic()
while not (forward/'forward_summary.json').exists():
    if time.monotonic()-start>21600: raise TimeoutError('Complete forward validation missing')
    time.sleep(15)
if not json.load(open(forward/'forward_summary.json'))['passed_for_reconstruction']:
    raise RuntimeError('Forward accuracy criterion failed; reconstruction is intentionally not started')
command=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--node_rank=0',
    '--master_addr=127.0.0.1','--master_port=29979','--nproc_per_node=3','-m','nsrno.reconstruct_dense','--cycles','2']
if (root/'out/v5/reconstruction_round2_dense_train_200/resume.pt').exists(): command.append('--resume')
os.execv(sys.executable,command)
