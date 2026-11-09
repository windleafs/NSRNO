"""Fit independent global material distributions from existing TRAIN only."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import shutil
import signal
import time
import traceback
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    tmp=path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True)
    a=p.parse_args()
    project=Path.cwd().resolve()
    out,products=a.out.resolve(),a.artifact_root.resolve()
    if (out.exists() or products.exists() or not out.is_relative_to(project/'out/v5') or
        not products.is_relative_to(Path.home()/'nsrno_artifacts')):
        raise ValueError('Require fresh bounded TRAIN-only distribution fit')
    reserve=2*2**30+400*2**20
    if shutil.disk_usage(project).free<reserve+2*2**20 or shutil.disk_usage(Path.home()).free<reserve+64*2**20:
        raise RuntimeError('Keep existing disk reserves')
    resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30))
    base_index=project/'out/v5/forward_cache_v1/index.json'
    aug_index=project/'out/v5/uno_triparam_v1/augmentation_index.json'
    base=[r for r in json.loads(base_index.read_text())['records'] if r['role']=='train']
    aug={r['material3']:r for r in json.loads(aug_index.read_text())['records'] if r['role']=='train'}
    if (len(base)!=510 or len(aug)!=32 or
        {r['case'] for r in base+list(aug.values())}!={'Neg_07_Left'}):
        raise RuntimeError('TRAIN isolation failed before patient array reads')
    tasks=[('base',r,[base_index.parent/r['medium'],project/f'out/v5/uno_triparam_v1/alpha/{r["id"]}.npy']) for r in base]
    tasks += [('augmentation',r,[Path(path)]) for path,r in sorted(aug.items())]
    out.mkdir()
    products.mkdir()
    (out/'products').symlink_to(products,target_is_directory=True)
    started=time.time()
    deadline=started+600
    source=Path(__file__)
    sha=digest(source)
    shutil.copy2(source,products/source.name)
    run={'status':'fitting_independent_TRAIN_global_quantiles','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s':started,'deadline_unix_s':deadline,'max_seconds':600,
        'source_SHA256':sha,'completed_TRAIN_materials':0,'source_case':'Neg_07_Left',
        'base_records':510,'augmentation_unique_materials':32,'quantiles':2049,
        'CUDA_used':False,'RF_arrays_read':False,'development_arrays_read':False,
        'reserved_arrays_read':False,'new_acoustic_solves':0,'new_material_derivative_probes':0,
        'CPU_address_space_limit_GiB':4,'not_a_reconstruction':True}
    atomic(out/'run.json',run)
    def stop(number,frame):
        (out/'stop_requested.flag').write_text(str(number)+'\n')
    for number in [signal.SIGINT,signal.SIGTERM]:
        signal.signal(number,stop)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    grid=np.linspace(0.,1.,2049)
    quantiles=np.empty((len(tasks),3,len(grid)),np.float64)
    weights=np.r_[np.full(510,.75/510),np.full(32,.25/32)]
    edge=np.broadcast_to(np.load(project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'),(961,1121))[80:921,178:943]
    manifest=[]
    try:
        for index,(kind,row,paths) in enumerate(tasks):
            if time.time()>=deadline or (out/'stop_requested.flag').exists():
                raise InterruptedError('Fixed600second TRAIN fit budget or cooperative stop')
            if digest(source)!=sha:
                raise RuntimeError('Frozen fit source changed')
            medium=np.load(paths[0],mmap_mode='r')
            if kind=='base':
                alpha=np.load(paths[1],mmap_mode='r')
                assert medium.shape==(2,961,1121) and alpha.shape==(961,1121)
                fields=[medium[0],medium[1],alpha]
            else:
                assert medium.shape==(3,961,1121)
                fields=list(medium)
            for f in range(3):
                values=np.asarray(fields[f][80:921,178:943],np.float64).copy()
                if f==0:
                    values=(values-1540.)/80.
                elif f==1:
                    values=(values-1000.)/100.
                else:
                    values-=edge
                    if np.any(values<=0):
                        raise ValueError('Positive TRAIN tissuealpha required')
                    values=np.log(values/.5)/.7
                if not np.isfinite(values).all():
                    raise ValueError('Finite TRAIN features required')
                quantiles[index,f]=np.quantile(values,grid,method='linear')
            manifest.append({'id':row['id'],'case':row['case'],'role':row['role'],
                'kind':kind,'paths':[str(p) for p in paths],'weight':float(weights[index])})
            if kind=='base':
                del alpha
            del medium,fields
            if (index+1)%40==0:
                run.update(completed_TRAIN_materials=index+1,seconds=time.time()-started,
                    peak_CPU_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
                atomic(out/'run.json',run)
        mean=np.tensordot(weights,quantiles,axes=(0,0))
        variance=np.tensordot(weights,(quantiles-mean[None])**2,axes=(0,0))
        np.savez(products/'prior.npz',probability=grid,mean_quantile=mean,
            standard_deviation_quantile=np.sqrt(variance),training_quantiles=quantiles,training_weights=weights)
        atomic(products/'training_sources.json',{'sources':manifest,
            'index_SHA256':{str(p):digest(p) for p in [base_index,aug_index]},
            'three_independent_distributions':True,'one_training_patient_not542independent_patients':True})
        run.update(status='independent_TRAIN_global_quantile_fit_finished',completed_TRAIN_materials=len(tasks),
            prior_SHA256=digest(products/'prior.npz'),seconds=time.time()-started,finished_unix_s=time.time(),
            peak_CPU_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            median_and_interquartile_features=[mean[:,i].tolist() for i in [512,1024,1536]])
        atomic(out/'run.json',run)
        print(json.dumps(run),flush=True)
    except BaseException:
        run.update(status='global_quantile_fit_stopped_for_inspection',error=traceback.format_exc(),
            seconds=time.time()-started,partial_products_retained=True)
        atomic(out/'run.json',run)
        raise


if __name__=='__main__':
    main()
