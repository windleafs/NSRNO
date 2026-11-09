"""Fixed-final raw RF operator maps and independent material responses."""
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
from .evaluate_inverse_uno import metric,native_initial
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics_without_crop(prediction,target):
    p=prediction.reshape(3,-1).astype(np.float64);t=target.reshape(3,-1).astype(np.float64)
    return {'rmse':np.sqrt(np.mean((p-t)**2,axis=1)).tolist(),
        'mae':np.mean(abs(p-t),axis=1).tolist(),
        'correlation':[float(np.corrcoef(p[j],t[j])[0,1]) if p[j].std()>1e-10 and t[j].std()>1e-10 else None for j in range(3)]}


def pair_response(members,predictions,targets):
    minus,plus=sorted(members,key=lambda r:r['pair_sign'])
    value=predictions[plus['id']][:,5:].astype(np.float64)-predictions[minus['id']][:,5:]
    exact=targets[plus['id']][:,5:].astype(np.float64)-targets[minus['id']][:,5:]
    details=[]
    for v,t in zip(value,exact):
        vn,tn=np.linalg.norm(v),np.linalg.norm(t)
        details.append({'difference_rmse':float(np.sqrt(np.mean((v-t)**2))),
            'predicted_difference_rms':float(np.sqrt(np.mean(v*v))),
            'target_difference_rms':float(np.sqrt(np.mean(t*t))),
            'relative_difference_error':float(np.linalg.norm(v-t)/tn) if tn>1e-5 else None,
            'difference_cosine':float(np.sum(v*t)/(vn*tn)) if vn>1e-10 and tn>1e-5 else None,
            'RMS_gain':float(vn/tn) if tn>1e-5 else None})
    return {'pair_group':minus['pair_group'],'kind':minus['kind'],'role':minus['role'],
        'parent_id':minus['parent_id'],'maps_c_rho_alpha':details}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=Path('out/v5/inverse_raw_rf_uno_v1'))
    p.add_argument('--device',type=int,default=0)
    a=p.parse_args();training=json.loads((a.run/'run.json').read_text())
    if training['status']!='training_finished' or Path(f"/proc/{training['pid']}").exists():
        raise RuntimeError('Finish and reap training before evaluation')
    index_path=Path(training['index'])
    if digest(index_path)!=training['index_sha256']:raise RuntimeError('Input index changed')
    out=a.run/'evaluation'
    if out.exists():raise RuntimeError('Preserve existing evaluation; no implicit overwrite')
    out.mkdir();state={'status':'starting','pid':os.getpid(),'created_unix_s':time.time(),
        'fixed_final_checkpoint':True,'checkpoint_selection_uses_development':False,
        'weights_SHA256':digest(a.run/'weights.pt'),'index_SHA256':digest(index_path),
        'source_SHA256':digest(__file__),'joint_imaging_success_claimed':False}
    atomic(out/'run.json',state);monitor=PressureResources(out)
    try:
        monitor.wait_budget([a.device],8,'before_raw_RF_evaluation')
        torch.set_num_threads(1);torch.cuda.set_device(a.device);device=f'cuda:{a.device}'
        torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
        torch.backends.cuda.matmul.allow_tf32=False
        index=json.loads(index_path.read_text());rows=index['records'];scale=index['training_RF_component_RMS_scale']
        model=RawRFInverseUNO().to(device)
        checkpoint=torch.load(a.run/'weights.pt',map_location='cpu',weights_only=False)
        if checkpoint['step']!=training['completed_steps']:raise RuntimeError('Not the fixed final checkpoint')
        model.load_state_dict(checkpoint['ema']);model.eval();del checkpoint
        mean=np.load(a.run/'training_mean_material.npy');cases=[];predictions={};targets={};z=np.arange(216)*.0002
        started=time.time()
        with torch.no_grad():
            for i,row in enumerate(rows):
                x,y=load_batch([row],device,scale);value=model(x)[0].cpu().numpy();target=y[0].cpu().numpy()
                np.save(out/f"{row['id']}.prediction.npy",value);predictions[row['id']]=value;targets[row['id']]=target
                depths={}
                for low,high in [(1,5),(5,12),(12,25),(25,43.01)]:
                    selected=(z>=low/1000)&(z<high/1000)
                    depths[f'{low}-{min(high,43):g}']=metrics_without_crop(value[:,selected],target[:,selected])
                cases.append({'id':row['id'],'kind':row['kind'],'role':row['role'],'case':row['case'],
                    'record_window_us':90 if row['source_format']=='RF_90us' else 60,
                    'model':metric(value,target),'training_mean':metric(mean,target),'by_depth_mm':depths})
                if row['id'] in ['train_200','val_059']:native_initial(value,out/f"{row['id']}_initial")
                if (i+1)%8==0:
                    state.update(status='evaluating',completed_records=i+1)
                    atomic(out/'run.json',state);monitor.sample('raw_RF_inverse_evaluation')
        grouped={}
        for row in rows:
            if 'pair_group' in row:grouped.setdefault(row['pair_group'],[]).append(row)
        pairs=[pair_response(v,predictions,targets) for v in grouped.values() if len(v)==2]
        constraints={}
        for changed,kind in enumerate(['pure_c','pure_rho','alpha']):
            chosen=[v for v in pairs if v['role']=='train' and v['kind']==kind];others=[j for j in range(3) if j!=changed]
            values=[]
            for pair in chosen:
                q=pair['maps_c_rho_alpha'];s=[5.,5.,.05]
                values.append([q[changed]['relative_difference_error']**2,
                    *[(q[j]['predicted_difference_rms']/s[j])**2 for j in others]])
            average=np.mean(values,axis=0)
            constraints[kind]={'pairs':len(chosen),'values':average.tolist(),'targets':[.16,.04,.04],
                'satisfied':(average<=np.array([.16,.04,.04])).tolist(),
                'changed_relative_RMS_error':float(np.sqrt(average[0])),
                'invariant_parameters':others,'invariant_RMS_in_physical_units':
                    [float(np.sqrt(average[i+1])*[5.,5.,.05][j]) for i,j in enumerate(others)]}
        summary={}
        for role in ['train','validation']:
            summary[role]={}
            for kind in sorted({r['kind'] for r in cases if r['role']==role}):
                chosen=[r for r in cases if r['role']==role and r['kind']==kind]
                summary[role][kind]={'records':len(chosen),
                    'mean_case_RMSE':np.mean([r['model']['rmse'] for r in chosen],axis=0).tolist(),
                    'training_mean_RMSE':np.mean([r['training_mean']['rmse'] for r in chosen],axis=0).tolist()}
        previous={}
        for name in ['60','90']:
            old=json.loads(Path(f'out/v5/inverse_material_coverage_{name}_v1/evaluation/metrics.json').read_text())
            selected=[r for r in old['cases'] if r['id'] in {r['id'] for r in rows if r['role']!='train'}]
            previous[name]={'records':len(selected),'mean_case_RMSE':np.mean([r['model']['rmse'] for r in selected],axis=0).tolist(),
                'input_policy':'previous model uses HF and LF; raw operator uses LF only; not a single-factor ablation'}
        result={'fixed_final_step':training['completed_steps'],'cases':cases,'independent_pairs':pairs,
            'training_constraints':constraints,'summary':summary,'previous_models_on_same_development_ids':previous,
            'seconds':time.time()-started,'peak_Torch_GiB':torch.cuda.max_memory_allocated()/2**30,
            'development_used_for_method_design':True,'development_independent_c_rho_pairs_present':False,
            'development_simultaneously_independent_three_fields_present':False,'reserved_test_case_unread':'Neg_47_Left',
            'native_RF_score_not_computed':True,'joint_imaging_success_claimed':False}
        atomic(out/'metrics.json',result)
        state.update(status='evaluation_finished',finished_unix_s=time.time());atomic(out/'run.json',state)
        print(json.dumps({k:v for k,v in result.items() if k not in ['cases','independent_pairs']}),flush=True)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());atomic(out/'run.json',state);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
