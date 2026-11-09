"""Versioned all-record feature preparation after the active inverse job ends."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
import traceback
import numpy as np
import torch
from .angular_spectral_features import AngularSpectralFocus,CHANNELS,BANDS
from .focus_features import FOCUS_SPEEDS,HEIGHT,WIDTH
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=Path,default=Path('out/v5/inverse_focus_v1/index.json'))
    p.add_argument('--out',type=Path,default=Path('out/v5/inverse_angular_spectral_v1'))
    p.add_argument('--after-root',type=Path,required=True)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--cap-gib',type=float,default=8.)
    args=p.parse_args();args.out=args.out.resolve()
    prior=json.loads((args.after_root/'run.json').read_text())
    if prior['status']!='planned_native_LBFGS_budget_finished':
        raise RuntimeError('Inspect the active/failed native job; no concurrent GPU feature preparation')
    processes=json.loads((args.after_root/'worker_processes.json').read_text())
    for pid in [prior['pid']]+processes['worker_pids']:
        if Path(f'/proc/{pid}').exists():raise RuntimeError(f'Prior process {pid} still exists')
    index=json.loads(args.index.read_text())
    if index['completed']!=index['planned_records']:raise RuntimeError('Require the complete old observation/label index')
    rows=index['records']
    if {r['case'] for r in rows if r['role']=='train'}!={'Neg_07_Left'}:
        raise ValueError('Existing training partition changed')
    if not 0<args.cap_gib<=10:raise ValueError('Keep full context memory below 16 GiB')
    code={name:digest(Path(__file__).parent/name) for name in
        ['prepare_angular_spectral_inverse.py','angular_spectral_features.py',
         'focus_features.py','rf_representation.py']}
    provenance={'source_index':str(args.index.resolve()),'source_index_SHA256':digest(args.index),
        'code_sha256':code,'bands_hz':{str(k):v for k,v in BANDS.items()},'channels_per_source':CHANNELS}
    args.out.mkdir(exist_ok=True)
    lock=(args.out/'preparation.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    previous=args.out/'provenance.json'
    if previous.exists():
        if json.loads(previous.read_text())!=json.loads(json.dumps(provenance)):
            raise RuntimeError('Do not mix features from different code or observations')
    else:atomic(previous,provenance)
    oldrun=args.out/'run.json'
    if oldrun.exists():
        previous_run=json.loads(oldrun.read_text())
        if Path(f"/proc/{previous_run['pid']}").exists():
            raise RuntimeError('Existing preparer still exists; inspect its process')
        if previous_run['status']=='features_prepared':
            print(json.dumps({'already_complete':True,'out':str(args.out)}));return
    feature_dir=args.out/'features';feature_dir.mkdir(exist_ok=True)
    remaining=sum(not (feature_dir/f"{r['id']}_{s}.npy").exists() for r in rows for s in r['sources'])
    required=remaining*CHANNELS*HEIGHT*WIDTH*2
    if shutil.disk_usage(args.out).free<required+3*2**30:
        raise RuntimeError('Insufficient space for complete float16 features plus 3 GiB reserve')
    run={'status':'preparing_angular_spectral_features','pid':os.getpid(),
        'source_index':str(args.index.resolve()),'source_index_SHA256':provenance['source_index_SHA256'],
        'source_sha256':code,'planned_records':len(rows),'completed_records':0,
        'feature_shape_per_source':[CHANNELS,HEIGHT,WIDTH],'device':args.device,
        'allocator_cap_GiB':args.cap_gib,'model_per_card_limit_GiB':16,
        'no_new_patient_wave_propagation':True,'truth_read_for_features':False,
        'labels':'reuse existing labels without consulting development truth for feature construction',
        'point_acquisitions_used':False,'created_unix_s':time.time()}
    atomic(oldrun,run)
    monitor=PressureResources(args.out)
    try:
        monitor.wait_budget([args.device],args.cap_gib,'before_angular_spectral_features')
        torch.set_num_threads(1);torch.cuda.set_device(args.device)
        torch.cuda.set_per_process_memory_fraction(args.cap_gib*2**30/
            torch.cuda.get_device_properties(args.device).total_memory,args.device)
        torch.backends.cuda.matmul.allow_tf32=False
        model=AngularSpectralFocus(f'cuda:{args.device}')
        monitor.sample('angular_spectral_geometry_prepared')
        completed=[]
        for row in rows:
            features={}
            for source,path in row['sources'].items():
                if source not in ['hf','lf']:raise ValueError('Preserve the declared plane-wave source set')
                target=feature_dir/f"{row['id']}_{source}.npy"
                if not target.exists():
                    feature=model(np.load(path),3e6 if source=='lf' else 7.5e6)
                    temporary=target.with_suffix('.partial.npy');np.save(temporary,feature);temporary.replace(target)
                header=np.load(target,mmap_mode='r')
                if header.shape!=(CHANNELS,HEIGHT,WIDTH) or header.dtype!=np.float16:
                    raise RuntimeError('Existing feature header differs from this representation')
                features[source]=str(target)
            if not Path(row['label']).exists():raise RuntimeError('Existing label is missing')
            completed.append({**row,'features':features})
            run.update(completed_records=len(completed),last_id=row['id'],
                peak_Torch_GiB=torch.cuda.max_memory_allocated()/2**30)
            atomic(oldrun,run)
            monitor.sample('angular_spectral_features_'+row['id'])
            if len(completed)%20==0:
                print(json.dumps({'completed':len(completed),'planned':len(rows),
                    'peak_Torch_GiB':run['peak_Torch_GiB']}),flush=True)
        meta={k:v for k,v in index.items() if k!='records'}
        meta.update(records=completed,completed=len(completed),planned_records=len(rows),
            feature_shape_per_source=[CHANNELS,HEIGHT,WIDTH],channels_per_source=CHANNELS,
            representation='complex subbands plus centered unit-RMS bistatic angular mode',
            focus_speeds_m_s=FOCUS_SPEEDS,code_sha256=code,
            feature_provenance=str(previous),point_acquisitions_used=False)
        atomic(args.out/'index.json',meta)
        run.update(status='features_prepared',finished_unix_s=time.time())
        atomic(oldrun,run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(oldrun,run);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
