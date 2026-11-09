"""Repair copied short-view provenance before any paired-window training."""
from pathlib import Path
import hashlib
import json
import os
import time
from .native_gn_json import atomic


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path('out/v5/raw_rf_window_balanced_data_v1')
    if any(Path('out/v5',name).exists() for name in
           ['raw_rf_window_balanced_experiment_v1','inverse_raw_rf_window_balanced_v1']):
        raise RuntimeError('Do not modify inputs after training starts')
    state=json.loads((root/'run.json').read_text())
    if state['status']!='paired_RF_windows_prepared' or Path(f"/proc/{state['pid']}").exists():
        raise RuntimeError('Cache preparation must have exited')
    if digest(root/'index.json')!=state['index_SHA256']:raise RuntimeError('Unexpected input revision')
    if (root/'provenance_correction.json').exists():raise RuntimeError('Already corrected')
    old_source=Path(__file__).with_name('prepare_raw_rf_window_pairs.py')
    if digest(old_source)!=state['source_SHA256']:raise RuntimeError('Preserve the executed preparation source first')
    (root/'prepare_raw_rf_window_pairs_at_creation.py').write_bytes(old_source.read_bytes())
    for name in ['run.json','index.json']:
        (root/(name+'.before_provenance_correction')).write_bytes((root/name).read_bytes())
    index=json.loads((root/'index.json').read_text());changes=[]
    for row in index['records']:
        if 'short_view' not in row:continue
        short=row['short_view'];actual=digest(short['raw_rf'])
        changes.append({'id':row['id'],'copied_RF90_SHA256':short['raw_rf_SHA256'],'actual_RF60_SHA256':actual})
        short.update(raw_rf_SHA256=actual,sources={**short['sources'],'lf':short['original_IQ60']},
            source_formats={**short.get('source_formats',{}),'lf':'original_complex_IQ'},low_frequency_record_us=60)
    if len(changes)!=66:raise ValueError('Expected sixty-six metadata corrections')
    atomic(root/'index.json',index)
    correction={'pid':os.getpid(),'unix_s':time.time(),'source_SHA256':digest(__file__),
        'previous_index_SHA256':state['index_SHA256'],'corrected_index_SHA256':digest(root/'index.json'),
        'RF_arrays_and_material_labels_unchanged':True,'training_had_not_started':True,
        'corrections':changes,'reason':'Nested short views inherited long-record hashes and source descriptions.'}
    atomic(root/'provenance_correction.json',correction)
    state.update(index_SHA256=correction['corrected_index_SHA256'],
        provenance_correction=str((root/'provenance_correction.json').resolve()))
    atomic(root/'run.json',state)
    print(json.dumps({k:v for k,v in correction.items() if k!='corrections'}),flush=True)


if __name__=='__main__':main()
