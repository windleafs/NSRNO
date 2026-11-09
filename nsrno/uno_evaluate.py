"""Complete-angle RF assessment and paired attenuation-response assessment."""
from pathlib import Path
import json,math,time
import numpy as np
import torch
from nsrno.uno_data import rf_statistics,atomic_json
from nsrno.rf_representation import decode

@torch.no_grad()
def evaluate(net,data,rows,device='cuda',save=None):
    net.eval(); total=np.zeros((5,2)); by_source={}; records=[]; pairs={}; pair_results=[]; start=time.perf_counter()
    if save is not None: Path(save).mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows):
        material=data.material(row)[None].to(device); latent=net.encode(material); stats=np.zeros((5,2)); values=[]
        source=torch.tensor([row['source_mhz']],device=device)
        for a,angle in enumerate(np.linspace(-8,8,11)):
            prediction=net.observe(latent,torch.tensor([angle],device=device,dtype=torch.float32),source)*data.scale(row)
            target=data.target(row,a)[None].to(device); stats+=rf_statistics(prediction,target)
            if row.get('kind')=='alpha' or (save is not None and row['id'] in ('hf_train_200','lf_train_200','hf_train_000')):
                values.append(prediction[0].cpu().numpy())
        total+=stats; key=str(row['source_mhz']); by_source.setdefault(key,np.zeros((5,2))); by_source[key]+=stats
        record={'id':row['id'],'source_mhz':row['source_mhz'],'kind':row['kind'],'error_sq':stats[:,0].tolist(),
                'target_sq':stats[:,1].tolist(),'relative_rms':np.sqrt(stats[:,0]/np.maximum(stats[:,1],1e-30)).tolist()}
        records.append(record)
        if values:
            values=np.stack(values)
            if save is not None: np.save(Path(save)/(row['id']+'.prediction.npy'),values)
            if row.get('kind')=='alpha':
                group=(row['pair_group'],row['source_mhz']); pairs.setdefault(group,[]).append((row,values))
                if len(pairs[group])==2:
                    sides=sorted(pairs.pop(group),key=lambda x:x[0]['pair_sign'])
                    pred=torch.from_numpy(sides[1][1]-sides[0][1]).to(device)
                    obs=torch.from_numpy(np.load(sides[1][0]['iq'])-np.load(sides[0][0]['iq'])).to(device)
                    ps=rf_statistics(pred,obs); p=decode(pred).flatten(); t=decode(obs).flatten()
                    pair_results.append({'group':group[0],'source_mhz':group[1],'error_sq':float(ps[0,0]),
                        'target_sq':float(ps[0,1]),'relative_rms':math.sqrt(ps[0,0]/max(ps[0,1],1e-30)),
                        'response_cosine':float(torch.dot(p,t)/(p.norm()*t.norm()).clamp_min(1e-30))})
        if save is not None and (i+1)%12==0: print(json.dumps({'evaluation_records':i+1,'total':len(rows)}),flush=True)
        del material,latent,prediction,target
    errors=np.sqrt(total[:,0]/np.maximum(total[:,1],1e-30))
    response=math.sqrt(sum(v['error_sq'] for v in pair_results)/max(sum(v['target_sq'] for v in pair_results),1e-30)) if pair_results else None
    result={'records':len(rows),'angles_per_record':11,'rf_windows_us':[[0,60.025],[0,8],[8,18],[18,36],[36,60.025]],
        'relative_rms':errors.tolist(),'error_sq':total[:,0].tolist(),'target_sq':total[:,1].tolist(),
        'by_source':{k:np.sqrt(v[:,0]/np.maximum(v[:,1],1e-30)).tolist() for k,v in by_source.items()},
        'paired_alpha_relative_rms':response,'alpha_pairs':pair_results,'per_record':records,'seconds':time.perf_counter()-start}
    result['passed_for_inverse']=bool(errors[0]<=.05 and errors[3]<=.20 and response is not None and response<=.20)
    return result
