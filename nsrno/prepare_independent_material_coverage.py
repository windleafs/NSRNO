"""CPU preparation of independent c/rho/alpha fields on unused V5 contexts."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import time
import numpy as np
from scipy.ndimage import zoom, gaussian_filter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def save(path, value):
    temporary = path.with_suffix('.partial.npy')
    with temporary.open('wb') as f:
        np.save(f, value)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def coarse(material):
    return np.stack([gaussian_filter(v, 1.)[np.ix_(60+4*np.arange(216),
                                                178+4*np.arange(192))]
                     for v in material]).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--after-root', type=Path, default=Path('out/v5/response_constrained_experiment_v1'))
    p.add_argument('--index', type=Path, default=Path('out/v5/inverse_angular_spectral_v1/index.json'))
    p.add_argument('--out', type=Path, default=Path('out/v5/independent_material_3mhz_v1'))
    a = p.parse_args()
    previous = json.loads((a.after_root/'run.json').read_text())
    if previous['status'] != 'constrained_training_and_evaluations_finished':
        raise RuntimeError('Finish the active response experiment first')
    for pid in [previous['pid'], previous.get('child_pid')]:
        if pid and Path(f'/proc/{pid}').exists():
            raise RuntimeError('Previous GPU process still exists')
    if a.out.exists():
        raise RuntimeError('Inspect the existing material plan; no implicit restart')
    if shutil.disk_usage(a.out.parent).free < 8*2**30:
        raise RuntimeError('Require 8 GiB free before this bounded data extension')
    index = json.loads(a.index.read_text())
    rows = index['records']
    old_contexts = {r['parent_id'] for r in rows if r['role']=='train' and r['kind']=='alpha'}
    manifest = json.loads(Path('out/v5/manifest.json').read_text())
    by_id = {r['id']:r for r in manifest['records']}
    candidates = [r for r in rows if r['role']=='train' and r['kind']=='original'
                  and r['id'] not in old_contexts and 'lf' in r['sources']
                  and by_id[r['id']]['variant']=='oa_breast_original']
    candidates.sort(key=lambda r:(by_id[r['id']]['z_index'],r['id']))
    if len(candidates) < 8:
        raise ValueError('Insufficient new training contexts')
    selected = [candidates[i] for i in np.linspace(0,len(candidates)-1,8,dtype=int)]
    if {r['case'] for r in selected} != {'Neg_07_Left'}:
        raise ValueError('Do not train on the development or reserved test case')
    a.out.mkdir()
    for name in ['materials','labels','fields']:
        (a.out/name).mkdir()
    edge_path = Path('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    shape = (961,1121)
    edge = np.broadcast_to(np.load(edge_path),shape).copy()
    z = (np.arange(shape[0])-60)*50e-6
    x = (np.arange(shape[1])-560)*50e-6
    zz,xx = np.meshgrid(z,x,indexing='ij')
    taper = np.clip((zz-.0012)/.0004,0,1)*np.clip((.043-zz)/.0005,0,1)
    taper *= np.clip((.0191-abs(xx))/.0005,0,1)
    inside = (taper>.999)
    planned = []

    def add(material, identifier, source, kind, **extra):
        if material.shape != (3,*shape) or not np.isfinite(material).all():
            raise ValueError('Invalid native material')
        material = material.astype(np.float32)
        path = a.out/'materials'/(identifier+'.npy')
        label = a.out/'labels'/(identifier+'.npy')
        save(path,material)
        save(label,coarse(material-np.stack([np.zeros_like(edge),np.zeros_like(edge),edge])))
        record = {'id':identifier,'source_record_id':source['id'],'role':source['role'],
                  'case':source['case'],'kind':kind,'parent_id':source['parent_id'],
                  'material3':str(path.resolve()),'material_file_SHA256':digest(path),
                  'material_array_SHA256':hashlib.sha256(material.tobytes()).hexdigest(),
                  'label':str(label.resolve()),'label_SHA256':digest(label),**extra}
        planned.append(record)
        return record

    for i,source in enumerate(selected):
        base = np.concatenate([np.load(source['medium']),np.load(source['alpha'])[None]],axis=0)
        if base.shape != (3,*shape):
            raise ValueError('Different V5 grid')
        seed = 202611070100+i
        rng = np.random.default_rng(seed)
        scale_mm = [.6,.9,1.5,2.4,3.5,5.,7.,9.][i]
        nz = max(5,int(round(48/scale_mm)))
        nx = max(5,int(round(56/scale_mm)))
        raw = rng.standard_normal((nz,nx))
        field = zoom(raw,(shape[0]/nz,shape[1]/nx),order=3)
        field = (field-field[inside].mean())/field[inside].std()
        # Include sharp local boundaries; the background c is never smoothed.
        sharp = np.zeros(shape)
        for j in range(3):
            cx,cz = rng.uniform(-.015,.015),rng.uniform(.004,.039)
            rx,rz = rng.uniform(.0004,.003),rng.uniform(.0004,.003)
            r = np.sqrt(((xx-cx)/rx)**2+((zz-cz)/rz)**2)
            sharp += rng.choice([-1.,1.])*np.clip((1.05-r)/.1,0,1)
        field = (.8*field+.6*sharp)
        field /= np.sqrt(np.mean(field[inside]**2))
        field = (np.clip(field,-2,2)*taper).astype(np.float32)
        field_path = a.out/'fields'/f'context_{i:02d}.npy'
        save(field_path,field)
        common = {'context':i,'field':str(field_path.resolve()),'field_seed':seed,
                  'field_scale_mm':scale_mm,'field_SHA256':digest(field_path),
                  'independent_of_tissue_classes':True}
        add(base,f'coverage_train_{i:02d}_base',source,'coverage_base',**common)
        tissue = np.maximum(base[2]-edge,.002)
        for kind,j in [('pure_c',0),('pure_rho',1),('alpha',2)]:
            group = f'coverage_train_{i:02d}_{kind}'
            pair = []
            for sign in [-1,1]:
                value = base.copy()
                if j==0:value[j] = base[j]+sign*25*field
                elif j==1:value[j] = base[j]+sign*30*field
                else:value[j] = np.clip(tissue*np.exp(sign*.25*field),.002,2.)+edge
                unchanged = [k for k in range(3) if k!=j]
                if not np.array_equal(value[unchanged],base[unchanged]):
                    raise ValueError('The independent parameter edit changed another field')
                if np.min(value[0])<1350 or np.max(value[0])>1750 or np.min(value[1])<700:
                    raise ValueError('Material outside the declared physical range')
                pair.append(add(value,group+('_minus' if sign<0 else '_plus'),source,kind,
                                pair_group=group,pair_sign=sign,changed_parameter=j,**common))
        print(json.dumps({'prepared_training_context':i,'parent':source['id'],
                          'scale_mm':scale_mm,'materials':len(planned)}),flush=True)
    # Existing development states are repeated only to evaluate the longer clock.
    development = [r for r in rows if r['role']!='train' and
                   (r['id'] in ['train_200','val_059'] or r['kind']=='alpha')]
    for source in development:
        if source['case']!='Neg_35_Left':raise ValueError('Unexpected development case')
        if 'material3' in source:material=np.load(source['material3'])
        else:material=np.concatenate([np.load(source['medium']),np.load(source['alpha'])[None]],axis=0)
        extras = {k:source[k] for k in ['pair_group','pair_sign'] if k in source}
        add(material,'coverage_dev_'+source['id'],source,source['kind'],**extras)
    acquisition = {'source_f0_hz':3e6,'native_nt':36001,'native_dt_s':2.5e-9,
                   'duration_s':90e-6,'rf_samples':3601,'fs_hz':40e6,
                   'angles_deg':manifest['acquisition']['angles_deg'],
                   'elements':192,'band_hz':[1.5e6,4.5e6],
                   'short_control_native_nt':24001,'short_control_rf_samples':2401}
    plan = {'status':'materials_prepared','records':planned,'training_materials':56,
            'development_materials':len(development),'training_contexts':[r['id'] for r in selected],
            'new_training_pairs_per_parameter':8,'acquisition':acquisition,
            'training_case':'Neg_07_Left','development_case':'Neg_35_Left',
            'reserved_test_case_unread':manifest['test_case'],
            'development_used_for_method_design':True,'new_centre_frequencies_hz':[3e6],
            'known_edge_alpha':str(edge_path.resolve()),'known_edge_SHA256':digest(edge_path),
            'source_index':str(a.index.resolve()),'source_index_SHA256':digest(a.index),
            'preparation_source_SHA256':digest(Path(__file__)),
            'one_field_shared_by_c_rho_alpha_edits_per_context':True,
            'sharp_and_multiscale_material_changes':True,
            'controls':'Process the same native propagation at 60 and 90 us independently',
            'patient_wave_solves':0,'created_unix_s':time.time()}
    atomic(a.out/'material_plan.json',plan)
    print(json.dumps({k:v for k,v in plan.items() if k!='records'}),flush=True)


if __name__=='__main__':
    main()
