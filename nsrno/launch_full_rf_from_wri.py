"""Explicit full-band refinement of a completed RF-selected WRI candidate.

This entry point does not wait for or automatically launch another experiment.
The research decision to use it follows assessment of actual WRI materials.
"""
from pathlib import Path
import argparse,json,os,subprocess,time,shutil,hashlib


def monitor(out,pid):
    maximum={};boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    with (out/'full_rf_resources.jsonl').open('a',buffering=1) as log:
        while True:
            path=Path('/proc')/str(pid)/'cmdline'
            if not path.exists():break
            command=path.read_bytes()
            if b'nsrno.reconstruct_coupled_v2' not in command or out.name.encode() not in command:break
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            used={int(s.split(',')[0]):int(s.split(',')[1]) for s in rows.splitlines()}
            for device,value in used.items():maximum[device]=max(maximum.get(device,0),value)
            row={'unix_s':time.time(),'card_MiB':used,'boot_id':boot};log.write(json.dumps(row)+'\n')
            summary={'sampled_max_card_MiB':maximum,'observed_above_16_GiB':any(v>16384 for v in maximum.values()),
                'latest':row,'sample_interval_seconds':15,'sampled_not_continuous':True}
            temporary=out/'full_rf_resource_summary.tmp.json';temporary.write_text(json.dumps(summary,indent=2))
            temporary.replace(out/'full_rf_resource_summary.json');time.sleep(15)


def main():
    p=argparse.ArgumentParser();p.add_argument('--wri',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--step',type=int)
    p.add_argument('--monitor-pid',type=int)
    p.add_argument('--devices',type=int,nargs=3,default=[0,1,2])
    p.add_argument('--iterations',type=int,default=8);p.add_argument('--cap-gib',type=float,default=4.25)
    args=p.parse_args();out=args.out.resolve()
    if args.monitor_pid is not None:monitor(out,args.monitor_pid);return
    if args.wri is None:p.error('--wri is required to start a refinement')
    wri=args.wri.resolve()
    run=json.loads((wri/'run.json').read_text());best=json.loads((wri/'best_RF.json').read_text())
    step=best['step'] if args.step is None else args.step
    if step<1:raise RuntimeError('No updated WRI material has been selected')
    source=wri/f'step{step:03d}'
    if not (source/'complete_sweep.json').exists():raise RuntimeError('The selected WRI sweep is incomplete')
    rf=json.loads((source/'rf_score.json').read_text());initial=json.loads((wri/'initial'/'rf_score.json').read_text())
    if not rf['primary_loss']<initial['primary_loss']:
        raise RuntimeError('The selected WRI material did not lower the complete 3-MHz RF objective')
    if out.exists():raise FileExistsError('Preserve previous refinement outputs')
    if len(set(args.devices))!=3:raise ValueError('Three distinct GPU devices required')
    memory={s.split(':',1)[0]:s.split(':',1)[1] for s in Path('/proc/meminfo').read_text().splitlines()}
    available=int(memory['MemAvailable'].split()[0])/2**20
    if available<100:raise RuntimeError(f'Insufficient host memory for two-code curvature: {available:.2f} GiB')
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(s.split(',')[0]):int(s.split(',')[1]) for s in lines.splitlines()}
    if any(used[d]+args.cap_gib*1024+768>16384 for d in args.devices):
        raise RuntimeError('Whole-card 16-GiB admission failed; no process launched')
    out.mkdir(parents=True)
    parameters=out/'wri_initial_parameters.npy';shutil.copy2(source/'parameters.npy',parameters)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMPY_MADVISE_HUGEPAGE='0',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.',
        PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    python='/home/zhuangyang/miniconda3/envs/py310/bin/python'
    command=[python,'-u','-m','nsrno.reconstruct_coupled_v2','--out',str(out),
        '--initial-parameters',str(parameters),'--sample',run['sample'],'--devices',*[str(d) for d in args.devices],
        '--sources','point3','lf','hf','--iterations',str(args.iterations),'--cg-iterations','8',
        '--memory-limit-gib',str(args.cap_gib),'--minimum-host-available-gib','100','--chunk','96']
    with (out/'run.log').open('a',buffering=1) as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    post=[python,'-u','-m','nsrno.run_coupled_postprocess','--out',str(out),'--pid',str(process.pid)]
    with (out/'postprocess.log').open('a',buffering=1) as log:
        evaluation=subprocess.Popen(post,env=env,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    with (out/'resource_monitor.log').open('a',buffering=1) as log:
        resources=subprocess.Popen([python,'-u','-m','nsrno.launch_full_rf_from_wri','--out',str(out),
            '--monitor-pid',str(process.pid)],env=env,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    record={'pid':process.pid,'postprocess_pid':evaluation.pid,'resource_monitor_pid':resources.pid,
        'command':command,'created_unix_s':time.time(),
        'WRI':str(wri),'selected_step':step,'selection':'completed candidate with lower full 3-MHz RF objective; not truth',
        'coarse_primary_loss':rf['primary_loss'],'initial_primary_loss':initial['primary_loss'],
        'source_parameters_sha256':hashlib.sha256(parameters.read_bytes()).hexdigest(),
        'all_sources':['point3','lf','hf'],'all_shots_per_source':11,'receivers':192,
        'retained_RF':'full dense time-domain records, all retained bins; no frequency subsampling',
        'per_worker_cap_GiB':args.cap_gib,'whole_card_limit_GiB':16,'card_before_MiB':used,
        'host_available_GiB':available,'truth_used_by_launcher':False,'signals_sent':False,
        'map_recovery_claimed':False,'original_solver_replay':'requires subsequent research assessment'}
    (out/'wri_handoff.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
