"""CPU final-map assessment; original RF replay is reserved for useful maps."""
from argparse import ArgumentParser
from pathlib import Path
import json, os, subprocess, time, traceback
from .reconstruct_time import atomic_json


def main():
    p = ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pid', type=int, required=True)
    a = p.parse_args()
    status = a.out/'postprocess.json'
    atomic_json(status, {'status':'waiting_for_optimization','optimization_pid':a.pid})
    try:
        while True:
            cmd = Path(f'/proc/{a.pid}/cmdline')
            live = cmd.exists() and b'nsrno.reconstruct_coupled' in cmd.read_bytes() and a.out.name.encode() in cmd.read_bytes()
            meta = a.out/'run.json'
            if meta.exists():
                run = json.loads(meta.read_text())
                if run['status'] in ['optimization_finished','interrupted_at_safe_boundary']:
                    break
                if run['status'] == 'failed':
                    raise RuntimeError(run.get('error','Coupled optimizer failed'))
            if not live:
                raise RuntimeError('The specific coupled optimizer ended without its terminal record')
            time.sleep(20)
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        atomic_json(status, {'status':'evaluating_final_material_maps'})
        with (a.out/'material_evaluation.log').open('w') as log:
            subprocess.run(['/home/zhuangyang/miniconda3/envs/py310/bin/python','-B','-m',
                'nsrno.evaluate_time_material',str(a.out/'final_material.npy'),'--sample',run['sample'],
                '--estimate-label','Coupled native c, density, attenuation / full unencoded RF'],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        atomic_json(status, {'status':'finished_maps_only','metrics':str(a.out/'final_material.metrics.json'),
            'figure':str(a.out/'final_material.png'),
            'original_solver_replay':'Not run automatically for an unsuccessful map; research assessment decides next action',
            'goal_completion_requires':'useful three maps, original-solver replay, independent-alpha pair, another development material'})
    except BaseException:
        atomic_json(status, {'status':'failed','error':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
