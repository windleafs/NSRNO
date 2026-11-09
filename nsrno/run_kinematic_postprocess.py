"""Finish map evaluation and independent RF replay after this exact run ends."""
from pathlib import Path
from argparse import ArgumentParser
import json,os,subprocess,time,traceback

def main():
    parser=ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--process-module',default='nsrno.reconstruct_kinematic')
    args=parser.parse_args();status=args.out/'postprocess.json';python='/home/zhuangyang/miniconda3/envs/py310/bin/python'
    def save(value):
        temp=status.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(status)
    save({'status':'waiting_for_optimization','optimization_pid':args.pid})
    try:
        while True:
            # The optimizer may still be importing torch before its first
            # atomic metadata write. Its live process, not file timing, is the
            # authoritative indication that initialization is still running.
            metadata=args.out/'run.json'
            cmd=Path(f'/proc/{args.pid}/cmdline')
            if not metadata.exists():
                if not cmd.exists() or args.process_module.encode() not in cmd.read_bytes() or args.out.name.encode() not in cmd.read_bytes():raise RuntimeError('Optimizer ended before recording initialization')
                time.sleep(3);continue
            run=json.loads(metadata.read_text())
            if run['status']=='optimization_finished':break
            if run['status']=='failed':raise RuntimeError(run.get('error','Optimization failed'))
            if not cmd.exists() or args.process_module.encode() not in cmd.read_bytes() or args.out.name.encode() not in cmd.read_bytes():raise RuntimeError('Optimization process ended before recording completion')
            time.sleep(30)
        save({'status':'evaluating_material_maps'})
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        for path in [*(args.out.glob('stage*_best_material.npy')),args.out/'final_material.npy']:
            initial=path.name.startswith('stage0_') and run['stages'][0].get('coordinate_mask')==[1,0,0]
            label='Kinematic initialization / fixed Z and alpha' if initial else 'Joint c, density, attenuation / full RF'
            subprocess.run([python,'-m','nsrno.evaluate_time_material',str(path),'--sample',run['sample'],'--estimate-label',label],env=env,check=True)
        # Optimization workers have been asked to exit; wait until they release their context.
        time.sleep(30);save({'status':'independent_original_solver_replay'})
        sources=list(dict.fromkeys(source for stage in run['stages'] if stage['maxiter']>0 for source in stage['sources']))
        # At most one independent forward process per GPU, including the third source group.
        devices=list(dict.fromkeys(run['devices']))
        for begin in range(0,len(sources),len(devices)):
            processes=[];logs=[]
            for device,source in zip(devices,sources[begin:begin+len(devices)]):
                source_env=env.copy();source_env.update(CUDA_VISIBLE_DEVICES=str(device))
                log=(args.out/f'replay_{source}.log').open('w');logs.append(log)
                processes.append(subprocess.Popen([python,'-u','-m','nsrno.replay_time_material',str(args.out/'final_material.npy'),'--source',source,'--sample',run['sample']],env=source_env,stdout=log,stderr=subprocess.STDOUT))
            codes=[p.wait() for p in processes]
            for log in logs:log.close()
            if any(codes):raise RuntimeError(f'Replay exit codes {codes}; inspect replay logs')
        save({'status':'finished','maps':str(args.out/'final_material.metrics.json'),'replays':{source:str(args.out/f'final_material_original_{source}.json') for source in sources},'goal_completion_requires_research_assessment':True})
    except BaseException:
        save({'status':'failed','error':traceback.format_exc()});raise

if __name__=='__main__':main()
