"""Fixed-weight window diagnosis using existing matched RF; no wave solves."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import traceback
import numpy as np
import torch
from .prepare_raw_rf_inverse import envelope
from .raw_rf_inverse_uno import RawRFInverseUNO,load_batch
from .evaluate_inverse_uno import metric
from .summarize_raw_rf_inverse_results import response
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('out/v5/raw_rf_record_window_diagnosis_v1'))
    p.add_argument('--device',type=int,default=0)
    a=p.parse_args();root=Path('out/v5/raw_rf_inverse_experiment_v1');model_root=Path('out/v5/inverse_raw_rf_uno_v1')
    run=json.loads((root/'run.json').read_text());training=json.loads((model_root/'run.json').read_text())
    if run['status']!='raw_RF_inverse_training_and_evaluations_finished':raise RuntimeError('Complete the existing experiment')
    for pid in {run['pid'],run.get('child_pid'),training['pid']}:
        if pid and Path(f'/proc/{pid}').exists():raise RuntimeError('Existing experiment still exists')
    if a.out.exists():raise RuntimeError('Preserve existing diagnosis')
    if digest(Path(__file__).with_name('raw_rf_inverse_uno.py'))!=training['source_sha256']['raw_rf_inverse_uno.py']:
        raise RuntimeError('Fixed model implementation changed')
    previous_evaluation=json.loads((model_root/'evaluation/run.json').read_text())
    if previous_evaluation['weights_SHA256']!=digest(model_root/'weights.pt'):
        raise RuntimeError('Saved RF90 predictions belong to different weights')
    if digest(training['index'])!=training['index_sha256']:raise RuntimeError('Fixed input index changed')
    index=json.loads(Path(training['index']).read_text())
    rows=[r for r in index['records'] if r['role']!='train' and r['source_format']=='RF_90us']
    if len(rows)!=10:raise ValueError('Expected two original materials and four independent alpha pairs')
    generated=json.loads(Path('out/v5/independent_material_3mhz_v1/generated_index.json').read_text())
    receipts={r['id']:r for r in generated['records']}
    a.out.mkdir();(a.out/'RF60').mkdir()
    state={'status':'preparing_existing_prefixes','pid':os.getpid(),'created_unix_s':time.time(),
        'weights_SHA256':digest(model_root/'weights.pt'),'source_SHA256':digest(__file__),
        'fixed_step':training['completed_steps'],'training_or_wave_solves':0,'new_forward_inferences_planned':14,
        'records':len(rows),'all_angles':11,'receivers':192,'joint_imaging_success_claimed':False}
    atomic(a.out/'run.json',state)
    short=[]
    for row in rows:
        receipt=receipts[row['coverage_generation_id']]
        wave,meta=envelope({**row,'sources':{'lf':receipt['iq60']}})
        path=a.out/'RF60'/f"{row['id']}.npy";np.save(path,wave)
        short.append({**row,**meta,'raw_rf':str(path.resolve()),'original_IQ60':receipt['iq60']})
    monitor=PressureResources(a.out)
    try:
        monitor.wait_budget([a.device],8,'before_fixed_window_diagnosis')
        torch.set_num_threads(1);torch.cuda.set_device(a.device);device=f'cuda:{a.device}'
        torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
        torch.backends.cuda.matmul.allow_tf32=False
        model=RawRFInverseUNO().to(device)
        checkpoint=torch.load(model_root/'weights.pt',map_location='cpu',weights_only=False)
        if checkpoint['step']!=6000:raise RuntimeError('Require the same fixed final weights')
        model.load_state_dict(checkpoint['ema']);model.eval();del checkpoint
        original={r['id']:r for r in rows};scale=index['training_RF_component_RMS_scale']
        predictions={};targets={};cases=[];inferences=0;started=time.time()
        with torch.no_grad():
            for row in short:
                x,y=load_batch([row],device,scale);value=model(x)[0].cpu().numpy();inferences+=1
                key=row['id'];target=y[0].cpu().numpy();targets[key]=target
                previous=np.load(model_root/'evaluation'/f'{key}.prediction.npy')
                predictions[key]=value;np.save(a.out/f'{key}.RF60.prediction.npy',value)
                variants={'existing_RF90':metric(previous,target),'matched_RF60':metric(value,target)}
                differences={'matched_RF60':np.sqrt(np.mean((value[:,5:]-previous[:,5:])**2,axis=(1,2))).tolist()}
                if row['kind']=='original':
                    full,_=load_batch([original[key]],device,scale)
                    late=full[:,:,3:4]>(60./90.)
                    zero_values=full.clone();zero_values[:,:,:2]=torch.where(late,0,zero_values[:,:,:2])
                    masked=zero_values.clone();masked[:,:,2:]=torch.where(late,0,masked[:,:,2:])
                    for name,signal in [('RF90_zero_tail_keep_metadata',zero_values),('RF90_zero_tail_mask_to_60us',masked)]:
                        predicted=model(signal)[0].cpu().numpy();inferences+=1
                        np.save(a.out/f'{key}.{name}.prediction.npy',predicted)
                        variants[name]=metric(predicted,target)
                        differences[name]=np.sqrt(np.mean((predicted[:,5:]-previous[:,5:])**2,axis=(1,2))).tolist()
                cases.append({'id':key,'kind':row['kind'],'case':row['case'],'variants':variants,
                    'prediction_difference_RMS_from_RF90_c_rho_alpha':differences,
                    'actual_60us_record_from_same_propagation':True})
            torch.cuda.synchronize()
        groups={}
        for row in short:
            if row['kind']=='alpha':groups.setdefault(row['pair_group'],[]).append(row)
        pairs=[]
        for group,members in sorted(groups.items()):
            minus,plus=sorted(members,key=lambda r:r['pair_sign'])
            exact=targets[plus['id']]-targets[minus['id']]
            delta=predictions[plus['id']]-predictions[minus['id']]
            previous=np.load(model_root/'evaluation'/f"{plus['id']}.prediction.npy")-np.load(model_root/'evaluation'/f"{minus['id']}.prediction.npy")
            pairs.append({'pair_group':group,'RF60':response(delta,exact),'RF90':response(previous,exact)})
        report={'cases':cases,'independent_alpha_pairs':pairs,'actual_new_inferences':inferences,
            'no_training_or_wave_solves':True,'weights_fixed_before_comparison':True,
            'control_definition':'Two RF90 controls are synthetic input ablations: zero late RF with metadata preserved, then additionally mask late samples. They are not new physical measurements.',
            'clock_filter_difference':'Matched RF60 uses the actual filtered native prefix; each record retains its own exact Fourier period and sampling time.',
            'causal_interpretation_limit':'Window sensitivity alone does not prove a pure length shortcut: genuine late signal and representation effects must be distinguished by the input controls.',
            'seconds':time.time()-started,'peak_Torch_GiB':torch.cuda.max_memory_allocated()/2**30,
            'development_used_for_method_design':True,'reserved_test_case_unread':'Neg_47_Left',
            'joint_imaging_success_claimed':False}
        atomic(a.out/'metrics.json',report);monitor.sample('fixed_window_diagnosis_finished')
        state.update(status='fixed_window_diagnosis_finished',actual_new_inferences=inferences,finished_unix_s=time.time())
        atomic(a.out/'run.json',state);print(json.dumps(report),flush=True)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());atomic(a.out/'run.json',state);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
