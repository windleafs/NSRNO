"""Supervise the bounded training and publish complete augmentation coverage."""
from pathlib import Path
import json,time,os,sys,subprocess,traceback
from nsrno.dataset import write_json

def live(pid):
    try: return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError: return False

def main():
    os.chdir('/home_nvme/zhuangyang/nsrno'); root=Path('out/v5/uno_triparam_v1'); out=Path('out/v5/uno_round1_triparam'); out.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'.',
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    started=time.monotonic()
    while not (root/'augmentation_index.json').exists():
        workers=json.load(open(root/'generation_processes.json')); parts=[p.get('part',p['gpu']) for p in workers]
        if all((root/f'generation_complete_{i}.json').exists() for i in parts):
            records=[r for i in parts for r in json.load(open(root/f'augmentation_part{i}.json'))['records']]
            if len(records)!=96 or len({r['id'] for r in records})!=96: raise RuntimeError('Incomplete/duplicate paired data')
            if any(len(r['angles_deg'])!=11 for r in records): raise RuntimeError('Missing angles')
            write_json(root/'augmentation_index.json',{'records':sorted(records,key=lambda r:r['id']),
                'completed':True,'independent_material_states':48,'training_states':32,'validation_states':16,
                'source_records':96,'sources_mhz':[3.,7.5],'angles_per_record':11})
            print('All96 paired source records and all11 angles published.',flush=True); break
        if time.monotonic()-started>10800: raise TimeoutError('Pair generation exceeded3 hours')
        for proc in workers:
            part=proc.get('part',proc['gpu'])
            if not live(proc['pid']) and not (root/f'generation_complete_{part}.json').exists():
                raise RuntimeError('Generation process exited without completion: '+str(proc))
        time.sleep(10)
    # Release the idle training CUDA context while physics targets are generated.
    args=[sys.executable,'-m','nsrno.train_uno_rf','--warmup-steps','2000']
    if (out/'last.pt').exists(): args.append('--resume')
    with (out/'training.log').open('a') as stream:
        trainer=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    write_json(out/'training_process.json',{'pid':trainer.pid,'command':args,'gpu':0})
    code=trainer.wait()
    if code: raise RuntimeError('Training/evaluation exit code '+str(code))
    result=json.load(open(out/'evaluation_complete.json'))
    write_json(out/'controller_complete.json',{'completed':True,'forward_complete':True,'passed_for_inverse':result['passed_for_inverse']})
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        Path('out/v5/uno_round1_triparam').mkdir(parents=True,exist_ok=True)
        write_json(Path('out/v5/uno_round1_triparam/controller_error.json'),{'error':traceback.format_exc()})
        raise
