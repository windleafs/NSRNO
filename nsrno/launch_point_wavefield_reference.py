"""Wait for natural LF-reference completion; never signal an existing task."""
from pathlib import Path
import fcntl,json,os,subprocess,time


def main():
    root=Path('out/v5/conditioned_point_full_v1');root.mkdir(exist_ok=True)
    lock=(root/'launcher.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=Path('out/v5/conditioned_lf_full_v1')
    while True:
        run=json.loads((prior/'run.json').read_text())
        cmd=Path(f"/proc/{run['pid']}/cmdline")
        try:command=cmd.read_bytes()
        except FileNotFoundError:command=b''
        live=b'nsrno.generate_lf_wavefield_reference' in command
        if run['status']=='complete' and (prior/'index.json').exists() and not live:
            break
        if run['status']!='complete' and not live:
            raise RuntimeError('LF generator ended without the required full reference; no duplicate launch')
        (root/'waiting.json').write_text(json.dumps({'status':'waiting_for_natural_LF_completion',
            'pid':os.getpid(),'observed_generator_pid':run['pid'],'observed_generator_live':live,
            'no_signals_or_stops':True,'unix_s':time.time()},indent=2))
        time.sleep(20)
    if (root/'run.json').exists() or (root/'launch.json').exists():raise RuntimeError('Preserve existing point job')
    while True:
        s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        cards={int(row.split(',')[0]):int(row.split(',')[1]) for row in s.splitlines()}
        if cards[2]+1.1*1024+768<=16*1024:break
        (root/'waiting.json').write_text(json.dumps({'status':'waiting_for_16GiB_card_budget',
            'pid':os.getpid(),'cards_MiB':cards,'no_signals_or_stops':True,'unix_s':time.time()},indent=2))
        time.sleep(20)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                   NUMPY_MADVISE_HUGEPAGE='0',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.',
                                   PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    argv=['/home/zhuangyang/miniconda3/envs/py310/bin/python','-u','-m','nsrno.generate_point_wavefield_reference',
          '--out',str(root),'--device','2','--memory-limit-gib','1.1','--frequency-block','32']
    with open(root/'generation.log','w') as log:
        child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (root/'launch.json').write_text(json.dumps({'pid':child.pid,'argv':argv,'unix_s':time.time(),
         'no_old_processes_stopped':True,'known_reference_only':True},indent=2))
    (root/'waiting.json').write_text(json.dumps({'status':'launched','pid':os.getpid(),
         'child_pid':child.pid,'no_signals_or_stops':True},indent=2))


if __name__=='__main__':main()
