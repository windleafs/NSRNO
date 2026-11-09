"""CPU-only, phase-preserving LF RF input for an independent inverse operator."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time
import traceback
import numpy as np
from scipy import fft
from .native_gn_json import atomic


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def envelope(row):
    value=np.load(row['sources']['lf'])
    if value.shape==(11,2,192,640):
        signal=value[:,0]+1j*value[:,1]
        positive=np.zeros((11,192,1201),np.complex64)
        positive[...,40:680]=fft.fftshift(fft.fft(signal,axis=-1,norm='forward'),axes=-1)/math.sqrt(2.)
        n=2401;form='original_complex_IQ'
    elif value.shape==(11,192,3601):
        positive=fft.rfft(value,axis=-1,norm='forward');n=3601;form='RF_90us'
    else:
        raise ValueError(f"Unsupported complete LF acquisition: {value.shape}")
    period=n/40e6
    frequencies=fft.rfftfreq(n,1/40e6)
    bins=np.flatnonzero((frequencies>=1.5e6)&(frequencies<=4.5e6))
    centre=round(3e6*period);samples=math.ceil(6e6*period)
    offsets=bins-centre
    if abs(offsets).max()>=samples//2:raise ValueError('Envelope sampling aliases selected frequencies')
    spectrum=np.zeros((11,192,samples),np.complex64)
    spectrum[...,offsets%samples]=math.sqrt(2.)*positive[...,bins]
    t=np.arange(samples,dtype=np.float64)*period/samples
    rotation=np.exp(2j*np.pi*(centre/period-3e6)*t).astype(np.complex64)
    result=fft.ifft(spectrum,axis=-1,norm='forward')*rotation
    # Algebraic round trip checks this new representation, with no wave solves.
    restored=fft.fft(result*rotation.conj(),axis=-1,norm='forward')[...,offsets%samples]/math.sqrt(2.)
    exact=positive[...,bins]
    error=float(np.sqrt(np.sum(abs(restored-exact)**2,dtype=np.float64)/
                        max(np.sum(abs(exact)**2,dtype=np.float64),1e-30)))
    if not np.isfinite(result).all() or error>5e-6:raise ValueError('Complex RF representation failed its round trip')
    packed=np.stack((result.real,result.imag),axis=1).astype(np.float32)
    return packed,{'source_format':form,'rf_samples':n,'rf_period_s':period,
        'valid_samples':samples,'sample_dt_s':period/samples,'carrier_hz':3e6,
        'selected_frequency_count':len(bins),'first_last_frequency_hz':frequencies[bins[[0,-1]]].tolist(),
        'effective_envelope_fs_hz':samples/period,'round_trip_relative_RMS_error':error}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=Path,default=Path('out/v5/independent_material_3mhz_v1/focus/index90.json'))
    p.add_argument('--out',type=Path,default=Path('out/v5/raw_rf_inverse_data_v1'))
    a=p.parse_args()
    previous=json.loads(Path('out/v5/material_coverage_experiment_v1/run.json').read_text())
    if previous['status']!='material_coverage_training_and_evaluations_finished' or Path(f"/proc/{previous['pid']}").exists():
        raise RuntimeError('Finish the existing coverage experiment first')
    if a.out.exists():raise RuntimeError('Inspect existing data; no implicit overwrite or resume')
    source=json.loads(a.index.read_text());rows=[r for r in source['records'] if 'lf' in r['sources']]
    train=[r for r in rows if r['role']=='train'];dev=[r for r in rows if r['role']!='train']
    if len(train)!=112 or len(dev)!=32 or {r['case'] for r in train}!={'Neg_07_Left'}:
        raise ValueError('Unexpected LF partition')
    a.out.mkdir();(a.out/'rf').mkdir()
    state={'status':'preparing','pid':os.getpid(),'training_records':len(train),'development_records':len(dev),
        'source_index':str(a.index.resolve()),'source_index_SHA256':digest(a.index),
        'source_SHA256':digest(__file__),'all_angles':11,'receivers':192,'band_hz':[1.5e6,4.5e6],
        'only_centre_frequency_hz':3e6,'no_wave_solves':True,'no_GPU_work':True,
        'created_unix_s':time.time(),'reserved_test_case_unread':'Neg_47_Left'}
    atomic(a.out/'run.json',state)
    energy=0.;count=0;records=[];started=time.time()
    try:
        for i,row in enumerate(rows):
            data,meta=envelope(row)
            target=a.out/'rf'/f"{row['id']}.npy";np.save(target,data)
            records.append({**row,'raw_rf':str(target.resolve()),'raw_rf_SHA256':digest(target),**meta})
            if row['role']=='train':
                energy+=float(np.sum(data.astype(np.float64)**2));count+=data.size
            state.update(completed_records=i+1,last_id=row['id'])
            atomic(a.out/'run.json',state)
            if (i+1)%16==0:print(json.dumps({'completed':i+1,'planned':len(rows)}),flush=True)
        scale=math.sqrt(energy/count)
        if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid training-only global RF scale')
        originals=[r for r in train if r['kind']=='original']
        mean=np.mean([np.load(r['label']).astype(np.float64) for r in originals],axis=0).astype(np.float32)
        np.save(a.out/'training_mean_material.npy',mean)
        result={'records':records,'training_RF_component_RMS_scale':scale,'scale_uses_development':False,
            'training_mean_material':str((a.out/'training_mean_material.npy').resolve()),
            'padded_samples':544,'all_angles_deg':np.linspace(-8,8,11).tolist(),
            'rx_first_last_m':[-.0191,.0191],'source_reference_s':
                (4/3e6+.0191*np.abs(np.sin(np.linspace(-8,8,11)*np.pi/180))/1540).tolist(),
            'input':'full signed real/imaginary LF envelope, validity and actual sample time',
            'record_windows':'90 us where generated; existing 60 us otherwise; explicit validity mask',
            'native_sampling_preserved_by_exact_DFT_reindexing':True,
            'development_used_for_method_design':True,'reserved_test_case_unread':'Neg_47_Left'}
        atomic(a.out/'index.json',result)
        state.update(status='raw_RF_prepared',seconds=time.time()-started,finished_unix_s=time.time(),
            worst_round_trip_error=max(r['round_trip_relative_RMS_error'] for r in records),
            global_RF_scale=scale,index_SHA256=digest(a.out/'index.json'))
        atomic(a.out/'run.json',state);print(json.dumps(state),flush=True)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());atomic(a.out/'run.json',state);raise


if __name__=='__main__':main()
