"""Prepare actual starting prior gradient and cached low RF value, no waves."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable THP')
from pathlib import Path
import argparse
import json
import os
import resource
import shutil
import time
import numpy as np
from .cdf_spatial_regularization import CDFSpatialRegularization,objectives
from .independent_physical_coordinates import IndependentPhysicalCoordinates
from .low_frequency_0p5 import digest,atomic


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent-root',type=Path,required=True)
    p.add_argument('--template-root',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True)
    a=p.parse_args()
    project,source=Path.cwd().resolve(),Path(__file__).parent
    a.parent_root,a.template_root,a.out,a.artifact_root=[v.resolve() for v in [a.parent_root,a.template_root,a.out,a.artifact_root]]
    if (a.out.exists() or a.artifact_root.exists() or not a.out.is_relative_to(project/'out/v5') or
        not a.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts')):
        raise ValueError('Dedicated fresh actual-input preparation only')
    old=json.loads((a.parent_root/'run.json').read_text())
    workers=json.loads((a.parent_root/'worker_processes.json').read_text())
    if old['status']!='learned_prior_saved_direction_native_stage_finished' or any(
        Path('/proc',str(i)).exists() for i in [old['pid'],*workers['worker_pids']]):
        raise RuntimeError('Finish old fixed inverse before preparation')
    metadata=json.loads((a.template_root/'metadata.json').read_text())
    qualification=json.loads(a.qualification.read_text())
    cdf_path=a.template_root/'prior.npz'
    if (digest(cdf_path)!=metadata['prior_SHA256'] or
        qualification['status']!='independent_CDF_envelope_full_physical_chain_synthetic_CPU_passed' or
        qualification['source_SHA256']!=digest(source/'global_cdf_prior.py') or
        metadata['global_cdf_source_SHA256']!=qualification['source_SHA256']):
        raise RuntimeError('Frozen and qualified new CDF energy required')
    for name,sha in old['source_SHA256'].items():
        if digest(source/name)!=sha:
            raise RuntimeError('Previously used frozen source changed: '+name)
    model_path=Path(old['fit_root'])/'products/prior.pt'
    if digest(model_path)!=old['learned_prior_SHA256']:
        raise RuntimeError('Neural prior changed')
    material_path=a.parent_root/'final/material.npy'
    control_path=a.parent_root/'final/control.npy'
    if digest(material_path)!=old['material_SHA256']:
        raise RuntimeError('Actual starting estimate changed')
    plan=json.loads((a.parent_root/'native_plan.json').read_text())
    acceptance=json.loads((a.parent_root/'final/actual_native_acceptance.json').read_text())
    refs={(r['family'],r['shot']):r for r in acceptance['RF_source_records']}
    spec=objectives(plan)
    data_index_path=Path(plan['data_root'])/'index.json'
    index=json.loads(data_index_path.read_text())
    if digest(data_index_path)!=plan['data_index_SHA256']:
        raise RuntimeError('Low observations changed')
    f=np.asarray(index['acquisition']['retained_frequency_hz'])
    low_RF,details=0.,{}
    for family,definition in spec['low_background']['families'].items():
        source_name=definition['source']
        records=sorted([r for r in index['shots'] if r['family']==source_name and r['kind']=='train_200'],key=lambda r:r['shot'])
        if [r['shot'] for r in records]!=list(range(11)):
            raise RuntimeError('All11 existing scattered observations required')
        observed=np.stack([np.load(r['path']) for r in records])
        predicted=[]
        for shot in range(11):
            row=refs[family,shot]
            if digest(row['path'])!=row['SHA256']:
                raise RuntimeError('Same-material cached RF changed')
            predicted.append(np.load(row['path']))
        predicted=np.stack(predicted)
        mask=(f>=.20e6)&(f<=.325e6)
        energy=float(np.sum(abs(observed[:,:,mask])**2))
        error=float(np.sum(abs(predicted[:,:,mask]-observed[:,:,mask])**2))
        value=.25*error/energy
        low_RF+=value
        details[family]={'loss':value,'observed_energy':energy,'source_count':11,'frequencies':int(mask.sum())}
    reserve=2*2**30+400*2**20
    if shutil.disk_usage(project).free<reserve+2*2**20 or shutil.disk_usage(Path.home()).free<reserve+64*2**20:
        raise RuntimeError('Retain existing disk reserves')
    a.out.mkdir()
    a.artifact_root.mkdir()
    (a.out/'products').symlink_to(a.artifact_root,target_is_directory=True)
    started=time.time()
    run={'status':'preparing_actual_CDF_neural_prior_initial_gradient','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s':started,'deadline_unix_s':started+600,'max_seconds':600,
        'parent_root':str(a.parent_root),'initial_material_path':str(material_path.resolve()),
        'initial_control_path':str(control_path.resolve()),'initial_material_SHA256':old['material_SHA256'],
        'native_anchor_material_SHA256':digest(plan['native_anchor_material']),
        'neural_path':str(model_path.resolve()),'neural_SHA256':old['learned_prior_SHA256'],
        'cdf_path':str(cdf_path.resolve()),'cdf_SHA256':metadata['prior_SHA256'],
        'initial_low_RF':low_RF,'low_RF_details':details,'initial_full_RF':old['final_RF'],
        'new_wave_solves':0,'new_patient_derivative_probes':0,'patient_truth_read':False,
        'reserved_arrays_read':False,'CUDA_used':False,'gradient_for_actual_inverse':True}
    names=['global_cdf_prior.py','cdf_spatial_regularization.py','learned_spatial_energy.py',
           'independent_denoising_prior.py','independent_physical_coordinates.py',Path(__file__).name]
    run['source_SHA256']={name:digest(source/name) for name in names}
    atomic(a.out/'run.json',run)
    edge_path=project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'
    coordinates=IndependentPhysicalCoordinates(np.load(plan['native_anchor_material']),np.load(edge_path))
    control=np.load(control_path).astype(np.float64)
    if not np.array_equal(coordinates.material(control),np.load(material_path)):
        raise RuntimeError('Coordinates do not reproduce exact starting estimate')
    prior=CDFSpatialRegularization(model_path,cdf_path)
    R,g,parts=prior.evaluate(coordinates,control)
    if parts['global_CDF']<=1e-12:
        raise RuntimeError('CDF hypothesis has no effect here; do not start needless native work')
    if time.time()>=run['deadline_unix_s'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20>8:
        raise RuntimeError('CPU preparation time/RSS budget reached')
    gradient_path=a.artifact_root/'initial_prior_gradient.npy'
    np.save(gradient_path,g)
    run.update(status='actual_CDF_spatial_inverse_inputs_prepared',seconds=time.time()-started,
        finished_unix_s=time.time(),initial_regularization=R,initial_prior_parts=parts,
        gradient_path=str(gradient_path),gradient_SHA256=digest(gradient_path),
        peak_CPU_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    atomic(a.out/'run.json',run)
    atomic(a.out/'objective_specifications.json',spec)
    print(json.dumps({k:run[k] for k in ['status','pid','seconds','initial_low_RF','initial_full_RF',
        'initial_regularization','initial_prior_parts','gradient_SHA256','peak_CPU_RSS_GiB']}),flush=True)


if __name__=='__main__':
    main()
