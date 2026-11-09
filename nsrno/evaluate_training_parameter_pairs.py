"""Fixed-model accuracy on the existing training parameter pairs; no updates."""
from pathlib import Path
import argparse,json,os,time,traceback,hashlib
import numpy as np
import torch
from .angular_spectral_inverse_uno import AngularSpectralInverseUNO,load_batch
from .compare_angular_spectral_inverse import response
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=Path,default=Path('out/v5/inverse_angular_spectral_v1/index.json'))
    p.add_argument('--model',type=Path,default=Path('out/v5/inverse_angular_spectral_uno_v1'))
    p.add_argument('--after-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',type=int,default=0)
    args=p.parse_args()
    prior=json.loads((args.after_root/'run.json').read_text())
    if prior['status']!='native_RF_scored':raise RuntimeError('Complete native RF score first')
    workers=json.loads((args.after_root/'worker_processes.json').read_text())
    for pid in [prior['pid']]+workers['worker_pids']:
        if Path(f'/proc/{pid}').exists():raise RuntimeError(f'Previous process {pid} still exists')
    if args.out.exists():raise RuntimeError('Read the existing pair evaluation; do not overwrite it')
    index=json.loads(args.index.read_text());groups={}
    for row in index['records']:
        if row['role']=='train' and 'pair_group' in row:groups.setdefault(row['pair_group'],[]).append(row)
    groups={k:sorted(v,key=lambda r:r['pair_sign']) for k,v in groups.items() if len(v)==2}
    rows=[row for pair in groups.values() for row in pair]
    args.out.mkdir();record={'status':'starting','pid':os.getpid(),'training':False,
        'existing_training_pairs':len(groups),'patient_wave_solves':0,
        'weights_SHA256':hashlib.sha256((args.model/'weights.pt').read_bytes()).hexdigest(),
        'device':args.device,'allocator_cap_GiB':8,'model_per_card_limit_GiB':16,
        'created_unix_s':time.time()}
    atomic(args.out/'run.json',record);monitor=PressureResources(args.out)
    try:
        monitor.wait_budget([args.device],8,'before_fixed_training_pair_evaluation')
        torch.set_num_threads(1);torch.cuda.set_device(args.device);device=f'cuda:{args.device}'
        torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(args.device).total_memory,args.device)
        torch.backends.cuda.matmul.allow_tf32=False
        model=AngularSpectralInverseUNO().to(device)
        state=torch.load(args.model/'weights.pt',map_location=device,weights_only=False)
        model.load_state_dict(state['ema']);model.eval();pred={};target={}
        with torch.no_grad():
            for begin in range(0,len(rows),4):
                batch=rows[begin:begin+4];x,y=load_batch(batch,device)
                values=model(x).cpu().numpy();truth=y.cpu().numpy()
                for row,value,t in zip(batch,values,truth):
                    pred[row['id']]=value;target[row['id']]=t
                    np.save(args.out/f"{row['id']}.prediction.npy",value)
                monitor.sample('fixed_training_pair_evaluation')
        details=[]
        for name,(minus,plus) in groups.items():
            details.append({'pair_group':name,'kind':minus['kind'],
                'maps_c_rho_alpha':response(pred[minus['id']],pred[plus['id']],target[minus['id']],target[plus['id']])})
        summary={}
        for kind in sorted({r['kind'] for r in details}):
            chosen=[r for r in details if r['kind']==kind];fields=[]
            for j in range(3):
                q=[r['maps_c_rho_alpha'][j] for r in chosen]
                err=[r['relative_difference_error'] for r in q if r['relative_difference_error'] is not None]
                cos=[r['difference_cosine'] for r in q if r['difference_cosine'] is not None]
                gain=[r['predicted_difference_rms']/r['target_difference_rms'] for r in q if r['target_difference_rms']>1e-8]
                fields.append({'relative_error':float(np.mean(err)) if err else None,
                    'cosine':float(np.mean(cos)) if cos else None,'RMS_gain':float(np.mean(gain)) if gain else None})
            summary[kind]={'pairs':len(chosen),'maps_c_rho_alpha':fields}
        atomic(args.out/'metrics.json',{'summary':summary,'pairs':details,'fixed_step':state['step']})
        record.update(status='fixed_training_pairs_evaluated',summary=summary,finished_unix_s=time.time())
        atomic(args.out/'run.json',record);print(json.dumps(record),flush=True)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc());atomic(args.out/'run.json',record);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
