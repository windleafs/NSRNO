"""Save and plot data-selected checkpoints while a specific run is live.

Evaluation reads truth in a separate process. These values never select a
checkpoint or feed back into the optimizer.
"""
from argparse import ArgumentParser
from pathlib import Path
import json,os,shutil,subprocess,time,traceback


def atomic_json(path,value):
    temporary=path.with_suffix('.partial.json');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    p=ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--pid',type=int,required=True)
    p.add_argument('--min-step',type=int,default=16);p.add_argument('--label',required=True)
    p.add_argument('--process-module',default='nsrno.reconstruct_encoded');a=p.parse_args()
    root=a.out/'snapshots';root.mkdir(exist_ok=True);status=root/'status.json'
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMPY_MADVISE_HUGEPAGE='0',PYTHONDONTWRITEBYTECODE='1')
    python='/home/zhuangyang/miniconda3/envs/py310/bin/python';completed=[]
    def parent_live():
        cmd=Path(f'/proc/{a.pid}/cmdline')
        return cmd.exists() and a.process_module.encode() in cmd.read_bytes() and a.out.name.encode() in cmd.read_bytes()
    def snapshot(selection,run):
        before=selection.read_bytes();chosen=json.loads(before);step=chosen.get('step',0)
        if step<a.min_step:return
        stage=chosen['stage'];destination=root/f'stage{stage}_step{step:03d}'
        if (destination/'material.metrics.json').exists():return
        temporary=root/(destination.name+'.partial');temporary.mkdir(exist_ok=True)
        files={a.out/f'stage{stage}_best_material.npy':'material.npy',a.out/f'stage{stage}_best_parameters.npy':'parameters.npy'}
        stamp=selection.stat().st_mtime_ns;versions={path:(path.stat().st_mtime_ns,path.stat().st_size) for path in files}
        if any(version[0]>stamp for version in versions.values()):return
        for source,name in files.items():shutil.copy2(source,temporary/name)
        if selection.read_bytes()!=before or any((path.stat().st_mtime_ns,path.stat().st_size)!=version for path,version in versions.items()):return
        shutil.copy2(a.out/'uniform_initial_material.npy',temporary/'uniform_initial_material.npy')
        (temporary/'rf_selection.json').write_bytes(before)
        if not destination.exists():temporary.rename(destination)
        with (destination/'evaluation.log').open('w') as log:
            subprocess.run([python,'-m','nsrno.evaluate_time_material',str(destination/'material.npy'),'--sample',run['sample'],'--estimate-label',f'{a.label} / stage {stage}, step {step}'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        rule=chosen.get('selection_description','complete RF objective only')
        completed.append(str(destination));print(json.dumps({'snapshot':str(destination),'selected_by':rule,'kinematic_weight':chosen.get('kinematic_weight',0.),'evaluation_only':True}),flush=True)
    try:
        while True:
            metadata=a.out/'run.json'
            if not metadata.exists():
                if not parent_live():raise RuntimeError('Parent ended before recording metadata')
                time.sleep(5);continue
            run=json.loads(metadata.read_text())
            for path in sorted(a.out.glob('stage*_best.json')):snapshot(path,run)
            if run['status']=='optimization_finished':
                atomic_json(status,{'status':'finished','snapshots':completed,'goal_completion_requires_research_assessment':True});break
            if run['status']=='failed':raise RuntimeError('Optimizer recorded a terminal failure or deliberate stop')
            if not parent_live():raise RuntimeError('Specific optimization handle is no longer live')
            atomic_json(status,{'status':'waiting_for_next_data_selection','optimization_pid':a.pid,'snapshots':completed,'truth_evaluation_does_not_affect_selection':True})
            time.sleep(20)
    except BaseException:
        atomic_json(status,{'status':'failed','error':traceback.format_exc(),'snapshots':completed});raise


if __name__=='__main__':main()
