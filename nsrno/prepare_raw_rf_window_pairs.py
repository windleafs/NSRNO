"""CPU cache of actual paired record lengths, without new propagation."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import traceback
from .prepare_raw_rf_inverse import envelope
from .native_gn_json import atomic
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('out/v5/raw_rf_inverse_data_v1/index.json'))
    p.add_argument('--out',type=Path,default=Path('out/v5/raw_rf_window_balanced_data_v1'))
    a=p.parse_args()
    run=json.loads(Path('out/v5/raw_rf_inverse_experiment_v1/run.json').read_text())
    diagnosis=json.loads(Path('out/v5/raw_rf_record_window_diagnosis_v1/run.json').read_text())
    if run['status']!='raw_RF_inverse_training_and_evaluations_finished' or diagnosis['status']!='fixed_window_diagnosis_finished':
        raise RuntimeError('Complete current training and fixed-window diagnosis')
    for pid in {run['pid'],run.get('child_pid'),diagnosis['pid']}:
        if pid and Path(f'/proc/{pid}').exists():raise RuntimeError('Existing GPU workflow still exists')
    if a.out.exists():raise RuntimeError('Inspect existing paired-view cache; no implicit overwrite')
    source=json.loads(a.source.read_text())
    generated=json.loads(Path('out/v5/independent_material_3mhz_v1/generated_index.json').read_text())
    receipts={r['id']:r for r in generated['records']}
    a.out.mkdir();(a.out/'RF60').mkdir()
    state={'status':'preparing','pid':os.getpid(),'created_unix_s':time.time(),'no_GPU_work':True,
        'new_wave_solves':0,'source_index':str(a.source.resolve()),'source_index_SHA256':digest(a.source),
        'source_SHA256':digest(__file__),'global_scale_unchanged':True,'label_files_unchanged':True,
        'reserved_test_case_unread':'Neg_47_Left'}
    atomic(a.out/'run.json',state);rows=[];counts={'train':0,'validation':0};started=time.time()
    try:
        for row in source['records']:
            copy=dict(row)
            if row['source_format']=='RF_90us':
                receipt=receipts[row.get('coverage_generation_id',row['id'])]
                wave,meta=envelope({**row,'sources':{'lf':receipt['iq60']}})
                path=a.out/'RF60'/f"{row['id']}.npy";np.save(path,wave)
                copy['short_view']={**row,**meta,'raw_rf':str(path.resolve()),'raw_rf_SHA256':digest(path),
                    'sources':{**row['sources'],'lf':receipt['iq60']},
                    'source_formats':{**row.get('source_formats',{}),'lf':'original_complex_IQ'},
                    'low_frequency_record_us':60,'original_IQ60':receipt['iq60']}
                counts[row['role']]+=1
            rows.append(copy)
        if counts!={'train':56,'validation':10}:raise ValueError('Unexpected actual paired-window counts')
        result={**source,'records':rows,'source_index_SHA256':state['source_index_SHA256'],
            'paired_record_counts':counts,'training_view_policy':'existing full record or its actual same-material prefix',
            'full_record_remains_default_evaluation_input':True,'new_wave_solves':0}
        atomic(a.out/'index.json',result)
        state.update(status='paired_RF_windows_prepared',paired_record_counts=counts,
            training_records=sum(r['role']=='train' for r in rows),
            development_records=sum(r['role']!='train' for r in rows),
            index_SHA256=digest(a.out/'index.json'),seconds=time.time()-started,finished_unix_s=time.time())
        atomic(a.out/'run.json',state);print(json.dumps(state),flush=True)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());atomic(a.out/'run.json',state);raise


if __name__=='__main__':main()
