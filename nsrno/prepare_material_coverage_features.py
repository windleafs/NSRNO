"""Sequential, fixed-data feature extraction after the generation process exits."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import shutil
import time
import traceback
import numpy as np
import torch
from .record_window_features import RecordWindowFocus
from .angular_spectral_features import CHANNELS,BANDS
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block=stream.read(8*2**20)
            if not block:break
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('out/v5/independent_material_3mhz_v1'))
    p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    generation=json.loads((a.root/'generation/run.json').read_text())
    if generation['status']!='independent_material_generation_finished':
        raise RuntimeError('Finish and inspect the existing acquisition; no concurrent GPU feature work')
    for pid in [generation['pid'],generation.get('child_pid')]:
        if pid and Path(f'/proc/{pid}').exists():raise RuntimeError('Generation GPU process still exists')
    generated=json.loads((a.root/'generated_index.json').read_text())
    plan=json.loads((a.root/'material_plan.json').read_text())
    if generated['status']!='complete' or generated['completed']!=66:
        raise ValueError('The declared paired acquisition is incomplete')
    source=Path(plan['source_index'])
    if digest(source)!=plan['source_index_SHA256']:raise ValueError('Original observation index changed')
    original=json.loads(source.read_text())
    out=a.root/'focus'
    if out.exists():raise RuntimeError('Inspect the existing feature process; no implicit restart')
    if shutil.disk_usage(a.root).free<5*2**30:raise RuntimeError('Require 5 GiB free before paired features')
    out.mkdir()
    for window in [60,90]:(out/f'features{window}').mkdir()
    sources={name:digest(Path(__file__).parent/name) for name in
             ['prepare_material_coverage_features.py','record_window_features.py',
              'angular_spectral_features.py','focus_features.py','rf_representation.py']}
    state={'status':'preparing','pid':os.getpid(),'device':a.device,
           'source_sha256':sources,'generated_index_SHA256':digest(a.root/'generated_index.json'),
           'completed_source_records':0,'planned_source_records':132,
           'allocator_cap_GiB':8,'per_card_limit_GiB':16,'no_patient_wave_solves':True,
           'all_angles':11,'receivers':192,'all_DFT_frequencies_inside_subbands':True,
           'bands_hz':BANDS[3e6],'record_windows_us':[60,90],'created_unix_s':time.time()}
    atomic(out/'run.json',state);monitor=PressureResources(out)
    try:
        monitor.wait_budget([a.device],8,'before_material_coverage_features')
        torch.set_num_threads(1);torch.cuda.set_device(a.device)
        torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
        torch.backends.cuda.matmul.allow_tf32=False
        model=RecordWindowFocus(f'cuda:{a.device}')
        for row in generated['records']:
            for window,key,sha in [(60,'iq60','IQ60_file_SHA256'),(90,'rf90','RF90_file_SHA256')]:
                if digest(row[key])!=row[sha]:raise ValueError('Generated RF was modified')
                values=np.load(row[key])
                feature=model.from_iq60(values) if window==60 else model.from_rf(values)
                path=out/f'features{window}'/(row['id']+'.npy')
                temporary=path.with_suffix('.partial.npy');np.save(temporary,feature);temporary.replace(path)
                state.update(completed_source_records=state['completed_source_records']+1,
                             last_id=row['id'],last_window_us=window,
                             peak_Torch_GiB=torch.cuda.max_memory_allocated()/2**30)
                atomic(out/'run.json',state);monitor.sample('paired_record_features')
                print(json.dumps({k:state[k] for k in ['completed_source_records','planned_source_records','last_id','last_window_us','peak_Torch_GiB']}),flush=True)
        for window,key in [(60,'iq60'),(90,'rf90')]:
            rows=copy.deepcopy(original['records']);by_id={r['id']:r for r in rows}
            for row in generated['records']:
                feature=str((out/f'features{window}'/(row['id']+'.npy')).resolve())
                if row['role']!='train' or row['kind']=='coverage_base':
                    target=by_id[row['source_record_id']]
                    if not np.array_equal(np.load(target['label']),np.load(row['label'])):
                        if not np.allclose(np.load(target['label']),np.load(row['label']),rtol=0,atol=1e-4):
                            raise ValueError('Replacement RF and existing label represent different material')
                    target['features']['lf']=feature
                    target['sources']['lf']=row[key]
                    target['source_formats']={**target.get('source_formats',{}),'lf':f'RF_{window}us' if window==90 else 'V5_IQ640'}
                    target['low_frequency_record_us']=window
                    target['coverage_generation_id']=row['id']
                else:
                    item={**row,'sources':{'lf':row[key]},'features':{'lf':feature},
                          'source_formats':{'lf':f'RF_{window}us' if window==90 else 'V5_IQ640'},
                          'low_frequency_record_us':window}
                    if item['id'] in by_id:raise ValueError('Duplicate new training ID')
                    rows.append(item);by_id[item['id']]=item
            if {r['case'] for r in rows if r['role']=='train'}!={'Neg_07_Left'}:
                raise ValueError('Development contamination')
            if len(rows)!=876:raise ValueError('Unexpected merged data size')
            meta={k:v for k,v in original.items() if k!='records'}
            meta.update(records=rows,completed=len(rows),planned_records=len(rows),
                        record_window_control_us=window,new_independent_pairs_per_parameter=8,
                        features_source_SHA256=sources,material_coverage_root=str(a.root.resolve()),
                        development_used_for_method_design=True,reserved_test_case_unread='Neg_47_Left')
            atomic(out/f'index{window}.json',meta)
        state.update(status='paired_features_prepared',finished_unix_s=time.time())
        atomic(out/'run.json',state)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(out/'run.json',state);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
