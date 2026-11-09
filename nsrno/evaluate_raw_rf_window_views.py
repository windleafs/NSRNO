"""Matched-view evaluation of a fixed paired-window model, no optimization."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import traceback
import numpy as np
import torch
from .raw_rf_inverse_uno import RawRFInverseUNO,load_batch
from .evaluate_inverse_uno import metric
from .summarize_raw_rf_inverse_results import response
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=Path('out/v5/inverse_raw_rf_window_balanced_v1'))
    p.add_argument('--device',type=int,default=0)
    a=p.parse_args();training=json.loads((a.run/'run.json').read_text())
    evaluation=json.loads((a.run/'evaluation/run.json').read_text())
    if training['status']!='training_finished' or evaluation['status']!='evaluation_finished':
        raise RuntimeError('Complete training and standard fixed evaluation first')
    for pid in [training['pid'],evaluation['pid']]:
        if Path(f'/proc/{pid}').exists():raise RuntimeError('Previous GPU process still exists')
    if digest(a.run/'weights.pt')!=evaluation['weights_SHA256']:raise RuntimeError('Different final weights')
    index_path=Path(training['index'])
    if digest(index_path)!=training['index_sha256']:raise RuntimeError('Input index changed')
    index=json.loads(index_path.read_text());rows=[r for r in index['records'] if r['role']!='train' and 'short_view' in r]
    if len(rows)!=10:raise ValueError('Expected ten paired development states')
    out=a.run/'window_evaluation'
    if out.exists():raise RuntimeError('Preserve existing paired-window evaluation')
    out.mkdir();state={'status':'starting','pid':os.getpid(),'created_unix_s':time.time(),
        'weights_SHA256':evaluation['weights_SHA256'],'source_SHA256':digest(__file__),
        'fixed_step':training['completed_steps'],'new_forward_inferences_planned':14,
        'training_or_wave_solves':0,'joint_imaging_success_claimed':False}
    atomic(out/'run.json',state);monitor=PressureResources(out)
    try:
        monitor.wait_budget([a.device],8,'before_paired_window_evaluation')
        torch.set_num_threads(1);torch.cuda.set_device(a.device);device=f'cuda:{a.device}'
        torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
        torch.backends.cuda.matmul.allow_tf32=False
        model=RawRFInverseUNO().to(device)
        checkpoint=torch.load(a.run/'weights.pt',map_location='cpu',weights_only=False)
        if checkpoint['step']!=training['completed_steps']:raise RuntimeError('Wrong fixed final step')
        model.load_state_dict(checkpoint['ema']);model.eval();del checkpoint
        scale=index['training_RF_component_RMS_scale'];predictions={};targets={};cases=[];inferences=0;started=time.time()
        with torch.no_grad():
            for row in rows:
                key=row['id'];x,y=load_batch([row['short_view']],device,scale)
                value=model(x)[0].cpu().numpy();inferences+=1;target=y[0].cpu().numpy();targets[key]=target
                previous=np.load(a.run/'evaluation'/f'{key}.prediction.npy')
                predictions[key]=value;np.save(out/f'{key}.RF60.prediction.npy',value)
                variants={'existing_RF90':metric(previous,target),'matched_RF60':metric(value,target)}
                differences={'matched_RF60':np.sqrt(np.mean((value[:,5:]-previous[:,5:])**2,axis=(1,2))).tolist()}
                if row['kind']=='original':
                    full,_=load_batch([row],device,scale);late=full[:,:,3:4]>(60./90.)
                    zero_values=full.clone();zero_values[:,:,:2]=torch.where(late,0,zero_values[:,:,:2])
                    masked=zero_values.clone();masked[:,:,2:]=torch.where(late,0,masked[:,:,2:])
                    for name,signal in [('RF90_zero_tail_keep_metadata',zero_values),('RF90_zero_tail_mask_to_60us',masked)]:
                        predicted=model(signal)[0].cpu().numpy();inferences+=1
                        np.save(out/f'{key}.{name}.prediction.npy',predicted)
                        variants[name]=metric(predicted,target)
                        differences[name]=np.sqrt(np.mean((predicted[:,5:]-previous[:,5:])**2,axis=(1,2))).tolist()
                cases.append({'id':key,'kind':row['kind'],'case':row['case'],'variants':variants,
                    'prediction_difference_RMS_from_RF90_c_rho_alpha':differences})
        groups={}
        for row in rows:
            if row['kind']=='alpha':groups.setdefault(row['pair_group'],[]).append(row)
        pairs=[]
        for name,members in sorted(groups.items()):
            minus,plus=sorted(members,key=lambda r:r['pair_sign']);exact=targets[plus['id']]-targets[minus['id']]
            delta=predictions[plus['id']]-predictions[minus['id']]
            old=np.load(a.run/'evaluation'/f"{plus['id']}.prediction.npy")-np.load(a.run/'evaluation'/f"{minus['id']}.prediction.npy")
            pairs.append({'pair_group':name,'RF60':response(delta,exact),'RF90':response(old,exact)})
        torch.cuda.synchronize()
        result={'cases':cases,'independent_alpha_pairs':pairs,'actual_new_inferences':inferences,
            'no_training_or_wave_solves':True,'weights_fixed_before_comparison':True,
            'synthetic_controls':'Zero-tail variants are input ablations; matched RF60 is an actual same-material prefix.',
            'same_material_consistency_is_not_joint_reconstruction_success':True,
            'seconds':time.time()-started,'peak_Torch_GiB':torch.cuda.max_memory_allocated()/2**30,
            'development_used_for_method_design':True,'reserved_test_case_unread':'Neg_47_Left',
            'joint_imaging_success_claimed':False}
        atomic(out/'metrics.json',result);monitor.sample('paired_window_evaluation_finished')
        state.update(status='window_evaluation_finished',actual_new_inferences=inferences,finished_unix_s=time.time())
        atomic(out/'run.json',state);print(json.dumps(result),flush=True)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());atomic(out/'run.json',state);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
