"""Prepare existing matched 3 MHz RF for the single authorized low-frequency case.

Only existing observed pressure and known-water reference metadata are read.
No material truth array, new patient or wave solve is involved.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy import fft
from .low_frequency_0p5 import atomic,digest,SAMPLE,CASE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--low-root',type=Path,default=Path('out/v5/single_case_0p5mhz_v1'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise RuntimeError('Preserve the fixed existing-data adapter')
    started=time.time()
    low=json.loads((a.low_root/'index.json').read_text())
    if low['sample']!=SAMPLE or low['case']!=CASE or low['completed_wave_solves']!=44:
        raise ValueError('Only the authorized original development case is admitted')
    root=Path('out/v5/independent_material_3mhz_v1')
    generated=json.loads((root/'generated_index.json').read_text())
    row=next(r for r in generated['records'] if r['id']=='coverage_dev_train_200')
    point_path=Path('out/v5/point_tx_3mhz_v1/index.json')
    point=json.loads(point_path.read_text())
    point_row=next(r for r in point['records'] if r['id']==SAMPLE)
    original_low_path=Path('out/v5/low_frequency_3mhz_v1/index.json')
    original=json.loads(original_low_path.read_text())
    if not (point['absorption']==original['absorption']==low['absorption']):
        raise ValueError('The actual data must share the same material constitutive law')
    if row['case']!=CASE or point_row['material_sha256']!=low['material_input_SHA256']:
        raise ValueError('The 3 MHz and 0.5 MHz patient inputs do not match')
    acquisition=generated['acquisition']
    if (acquisition['native_nt']!=36001 or acquisition['native_dt_s']!=2.5e-9 or
            acquisition['rf_samples']!=3601 or len(acquisition['angles_deg'])!=11):
        raise ValueError('All 11 original 90 us plane acquisitions are required')
    frequency=fft.rfftfreq(3601,1/40e6)
    mask=(frequency>=1.5e6)&(frequency<=4.5e6)
    if int(mask.sum())!=270:
        raise ValueError('Dense valid-band DFT clock changed')
    a.out.mkdir()
    families={}
    for family,path,expected,reference in [
        ('lf',Path(row['rf90']),row['RF90_file_SHA256'],root/'reference_native.npy'),
        ('point3',Path(point_row['rf']),None,Path(point['reference_native']))]:
        observed_hash=digest(path)
        if expected is not None and observed_hash!=expected:
            raise ValueError('Fixed existing RF changed')
        observed=np.load(path,mmap_mode='r')
        if observed.shape!=(11,192,3601) or observed.dtype!=np.float32:
            raise ValueError('Require full recorded receiver/time arrays')
        known_reference=np.load(reference,mmap_mode='r')
        if known_reference.shape!=(11,36001,192):
            raise ValueError('The original full-duration native water reference is required')
        reference_hash=digest(reference)
        if family=='lf' and reference_hash!=row['reference_SHA256']:
            raise ValueError('Protected original reference changed')
        coefficients=fft.rfft(observed.astype(np.float64),axis=-1,norm='ortho')[...,mask]
        if not np.isfinite(coefficients).all():
            raise ValueError('Invalid observed DFT coefficients')
        out=a.out/(family+'_3mhz_observed_coefficients.npy')
        np.save(out,coefficients)
        families[family]={
            'observed_coefficients':str(out.resolve()),
            'observed_coefficients_SHA256':digest(out),
            'original_observed_RF':str(path.resolve()),
            'original_observed_RF_SHA256':observed_hash,
            'raw_reference':str(reference.resolve()),
            'raw_reference_SHA256':reference_hash,
            'shape':list(coefficients.shape)}
    result={
        'status':'matched_existing_3MHz_full90_ready',
        'sample':SAMPLE,'case':CASE,'source_f0_hz':3e6,
        'native_nt':36001,'native_dt_s':2.5e-9,
        'RF_nt':3601,'RF_dt_s':2.5e-8,'record_us':90,
        'source_peak_pressure_Pa':1e5,'source_sigma_s':1/3e6,
        'angles_deg':acquisition['angles_deg'],'receivers':192,
        'source_layout':str(Path(point['source_layout']).resolve()),
        'source_layout_SHA256':digest(point['source_layout']),
        'absorption':low['absorption'],
        'frequency_hz':frequency[mask].tolist(),'DFT_mask':mask.tolist(),
        'dense_frequency_count':int(mask.sum()),
        'format':'complex128 orthonormal positive-frequency DFT, shot/receiver/frequency',
        'families':families,
        'low_0p5MHz_index':str((a.low_root/'index.json').resolve()),
        'low_0p5MHz_index_SHA256':digest(a.low_root/'index.json'),
        'LF_material_id':row['id'],'LF_case':row['case'],
        'LF_material_SHA256_from_closed_generator_metadata':row['material_file_SHA256'],
        'point_and_low_material_input_hashes_match':True,
        'low_generator_previously_checked_exact_LF_material_array_match':True,
        'new_wave_solves':0,'new_material_derivatives':0,'GPU_work':False,
        'patient_material_arrays_read':False,'new_patient_cases':0,
        'known_water_reference_remains_native_for_float32_pre_filter_subtraction':True,
        'no_dense_receiver_matrix_created':True,
        'reserved_case_unread':'Neg_47_Left','seconds':time.time()-started,
        'source_SHA256':digest(__file__)}
    atomic(a.out/'index.json',result)
    print(json.dumps({k:result[k] for k in ['status','sample','record_us',
          'dense_frequency_count','seconds','new_wave_solves','patient_material_arrays_read']}),flush=True)


if __name__=='__main__':
    main()
