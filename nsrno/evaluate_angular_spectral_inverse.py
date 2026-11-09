"""Fixed-final-checkpoint images and independent material-pair responses."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import traceback
import numpy as np
import torch
from .angular_spectral_inverse_uno import AngularSpectralInverseUNO,load_batch
from .evaluate_inverse_uno import metric,native_initial
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def metrics_without_crop(prediction,target):
    p=prediction.reshape(3,-1).astype(np.float64);t=target.reshape(3,-1).astype(np.float64)
    return {'rmse':np.sqrt(np.mean((p-t)**2,axis=1)).tolist(),
        'mae':np.mean(abs(p-t),axis=1).tolist(),
        'correlation':[float(np.corrcoef(p[j],t[j])[0,1]) if p[j].std()>1e-10 and t[j].std()>1e-10 else None for j in range(3)]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=Path,default=Path('out/v5/inverse_angular_spectral_v1/index.json'))
    p.add_argument('--run',type=Path,default=Path('out/v5/inverse_angular_spectral_uno_v1'))
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--native-id',nargs='+',default=['train_200','val_059'])
    p.add_argument('--cap-gib',type=float,default=8.)
    args=p.parse_args();record=json.loads((args.run/'run.json').read_text())
    if record['status']!='training_finished':raise RuntimeError('Require the fixed completed training checkpoint')
    if Path(f"/proc/{record['pid']}").exists():raise RuntimeError('Training GPU process still exists')
    if hashlib.sha256(args.index.read_bytes()).hexdigest()!=record['index_sha256']:
        raise RuntimeError('Evaluation observations differ from the declared index')
    if not 0<args.cap_gib<=10:raise ValueError('Keep context plus allocation below 16 GiB')
    out=args.run/'evaluation';out.mkdir(exist_ok=True)
    provenance={'weights_sha256':hashlib.sha256((args.run/'weights.pt').read_bytes()).hexdigest(),
        'index_sha256':record['index_sha256'],'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_exporter_sha256':hashlib.sha256(Path(__file__).with_name('evaluate_inverse_uno.py').read_bytes()).hexdigest()}
    path=out/'provenance.json'
    if path.exists() and json.loads(path.read_text())!=provenance:
        raise RuntimeError('Do not overwrite results from another checkpoint/evaluator')
    if (out/'metrics.json').exists():
        print(json.dumps({'already_complete':True,'out':str(out)}));return
    atomic(path,provenance)
    state={'status':'starting','pid':os.getpid(),'checkpoint_fixed_before_evaluation':True,
        'training_gradients_or_checkpoint_selection':False,'created_unix_s':time.time()}
    atomic(out/'run.json',state);monitor=PressureResources(out)
    try:
        monitor.wait_budget([args.device],args.cap_gib,'before_angular_spectral_evaluation')
        torch.set_num_threads(1);torch.cuda.set_device(args.device);device=f'cuda:{args.device}'
        torch.cuda.set_per_process_memory_fraction(args.cap_gib*2**30/
            torch.cuda.get_device_properties(args.device).total_memory,args.device)
        torch.backends.cuda.matmul.allow_tf32=False
        index=json.loads(args.index.read_text());rows=[r for r in index['records'] if r['role']!='train']
        model=AngularSpectralInverseUNO().to(device)
        checkpoint=torch.load(args.run/'weights.pt',map_location=device,weights_only=False)
        if checkpoint['step']!=record['steps']:raise RuntimeError('Not the predeclared final checkpoint')
        model.load_state_dict(checkpoint['ema']);model.eval()
        mean=np.load(args.run/'training_mean_material.npy')
        predictions={};targets={};cases=[];started=time.time()
        z=np.arange(216)*.0002
        with torch.no_grad():
            for begin in range(0,len(rows),4):
                batch=rows[begin:begin+4];x,y=load_batch(batch,device)
                predicted=model(x).cpu().numpy();truth=y.cpu().numpy()
                for row,value,target in zip(batch,predicted,truth):
                    predictions[row['id']]=value;targets[row['id']]=target
                    np.save(out/f"{row['id']}.prediction.npy",value)
                    depths={}
                    for low,high in [(1,5),(5,12),(12,25),(25,43.01)]:
                        selected=(z>=low/1000)&(z<high/1000)
                        depths[f'{low}-{min(high,43):g}']=metrics_without_crop(value[:,selected],target[:,selected])
                    cases.append({'id':row['id'],'kind':row['kind'],'case':row['case'],
                        'has_low_frequency':'lf' in row['features'],'model':metric(value,target),
                        'training_mean':metric(mean,target),'by_depth_mm':depths})
                    if row['id'] in args.native_id:native_initial(value,out/f"{row['id']}_initial")
                state.update(status='evaluating',completed_records=min(begin+4,len(rows)))
                atomic(out/'run.json',state);monitor.sample('angular_spectral_evaluation')
        groups={}
        for row in rows:
            if 'pair_group' in row:groups.setdefault(row['pair_group'],[]).append(row)
        pairs=[]
        for name,members in groups.items():
            if len(members)!=2:continue
            minus,plus=sorted(members,key=lambda r:r['pair_sign'])
            delta=predictions[plus['id']][:,5:]-predictions[minus['id']][:,5:]
            exact=targets[plus['id']][:,5:]-targets[minus['id']][:,5:]
            detail=[]
            for j in range(3):
                v=delta[j].astype(np.float64).ravel();t=exact[j].astype(np.float64).ravel()
                norm=np.linalg.norm(t);pnorm=np.linalg.norm(v)
                detail.append({'difference_rmse':float(np.sqrt(np.mean((v-t)**2))),
                    'target_difference_rms':float(np.sqrt(np.mean(t*t))),
                    'relative_difference_error':float(np.linalg.norm(v-t)/norm) if norm>1e-5 else None,
                    'difference_cosine':float(np.dot(v,t)/(pnorm*norm)) if norm>1e-5 and pnorm>1e-10 else None})
            pairs.append({'pair_group':name,'kind':minus['kind'],'maps_c_rho_alpha':detail})
        summary={}
        for kind in sorted({r['kind'] for r in cases}):
            selected=[r for r in cases if r['kind']==kind]
            summary[kind]={'records':len(selected),
                'mean_case_rmse':np.mean([r['model']['rmse'] for r in selected],axis=0).tolist(),
                'training_mean_rmse':np.mean([r['training_mean']['rmse'] for r in selected],axis=0).tolist()}
        result={'step':checkpoint['step'],'coarse_initialization_only':True,
            'native_physics_reconstruction_required':True,'point_acquisitions_used':False,
            'development_used_for_method_design':True,'cases':cases,'independent_pairs':pairs,
            'summary':summary,'seconds':time.time()-started,
            'peak_GiB':torch.cuda.max_memory_allocated()/2**30,**provenance}
        atomic(out/'metrics.json',result)
        state.update(status='evaluation_finished',finished_unix_s=time.time())
        atomic(out/'run.json',state)
        print(json.dumps({k:v for k,v in result.items() if k not in ['cases','independent_pairs']}),flush=True)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(out/'run.json',state);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
