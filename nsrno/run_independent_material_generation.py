"""Single-GPU acquisition with process-tree memory observation and fixed inputs."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from .pressure_resources import PressureResources


def atomic(path,value):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('out/v5/independent_material_3mhz_v1'))
    p.add_argument('--after-root',type=Path,default=Path('out/v5/response_constrained_experiment_v1'))
    a=p.parse_args()
    prior=json.loads((a.after_root/'run.json').read_text())
    if prior['status']!='constrained_training_and_evaluations_finished':
        raise RuntimeError('Do not overlap the current training workflow')
    for pid in [prior['pid'],prior.get('child_pid')]:
        if pid and Path(f'/proc/{pid}').exists():raise RuntimeError('Previous GPU process still exists')
    plan_path=a.root/'material_plan.json';plan=json.loads(plan_path.read_text())
    if plan['status']!='materials_prepared' or len(plan['records'])!=66:
        raise ValueError('Require the declared 56 training and 10 development materials')
    if shutil.disk_usage(a.root).free<8*2**30:raise RuntimeError('Require 8 GiB disk reserve before generation')
    out=a.root/'generation'
    if out.exists():raise RuntimeError('Inspect the existing generation; no implicit restart')
    names=['run_independent_material_generation.py','generate_independent_material_3mhz.py',
           'fullwave.py','rf_representation.py']
    source={}
    for name in names:
        file=Path(__file__).parent/name;ast.parse(file.read_text());source[name]=digest(file)
    out.mkdir()
    record={'status':'starting','pid':os.getpid(),'root':str(a.root.resolve()),'device':0,
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'source_sha256':source,'material_plan_SHA256':digest(plan_path),
            'planned_materials':66,'training_materials':56,'development_materials':10,
            'new_centre_frequencies_hz':[3e6],'shots_per_material':11,
            'native_duration_us':90,'matched_short_control_us':60,
            'only_one_GPU_workflow':True,'per_card_observed_limit_GiB':16,
            'native_solver_not_a_PyTorch_allocator':True,'joint_imaging_success_claimed':False,
            'created_unix_s':time.time()}
    atomic(out/'run.json',record)
    monitor=PressureResources(out);child=None
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',ACC_DEVICE_NUM='0',OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMPY_MADVISE_HUGEPAGE='0',
               PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.')
    try:
        monitor.wait_budget([0],8,'before_independent_material_generation')
        for name,sha in source.items():
            if digest(Path(__file__).parent/name)!=sha:raise RuntimeError('Source changed before launch')
            (out/name).write_bytes((Path(__file__).parent/name).read_bytes())
        with (out/'worker.log').open('x') as log:
            command=[sys.executable,'-m','nsrno.generate_independent_material_3mhz',
                     '--root',str(a.root),'--out',str(out/'worker')]
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            record.update(status='generating',child_pid=child.pid,child_output=str((out/'worker').resolve()))
            atomic(out/'run.json',record)
            atomic(out/'worker_processes.json',{'coordinator_pid':os.getpid(),'worker_pids':[child.pid],
                   'devices':[0],'boot_id':record['boot_id']})
            while child.poll() is None:
                devices=monitor.sample('independent_material_3MHz_generation')
                if any(v['owned_MiB']>16384 for v in devices.values()):
                    raise RuntimeError('Observed process-tree memory exceeded 16 GiB per card')
                if shutil.disk_usage(a.root).free<3*2**30:
                    raise RuntimeError('Disk free space fell below the 3 GiB hard reserve')
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
            if child.returncode:raise RuntimeError(f'Generation worker exited {child.returncode}; inspect exact shot ledger')
            if Path(f'/proc/{child.pid}').exists():raise RuntimeError('Worker was not reaped')
            child=None
        result=json.loads((a.root/'generated_index.json').read_text())
        if result['status']!='complete' or result['completed']!=66:
            raise RuntimeError('Incomplete acquisition; do not claim completion')
        record.update(status='independent_material_generation_finished',completed=66,finished_unix_s=time.time())
        atomic(out/'run.json',record)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(out/'run.json',record);raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        monitor.log.close()


if __name__=='__main__':main()
