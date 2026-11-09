"""Post-training tests include actual independent attenuation perturbations."""
from argparse import ArgumentParser
from pathlib import Path
import json,time
import numpy as np
import torch
from torch.nn import functional as F
from .inverse_uno import InverseUNO,load_batch

def metric(prediction,target):
    x=prediction[:,5:].reshape(3,-1).astype(np.float64);y=target[:,5:].reshape(3,-1).astype(np.float64)
    return {'rmse':np.sqrt(np.mean((x-y)**2,1)).tolist(),'mae':np.mean(abs(x-y),1).tolist(),
        'correlation':[float(np.corrcoef(x[i],y[i])[0,1]) if x[i].std()>1e-10 else None for i in range(3)]}

def native_initial(prediction,out):
    # No case-specific truth is used to extend or interpolate the prediction.
    image=torch.from_numpy(prediction)[None]
    up=F.interpolate(image,size=(861,765),mode='bilinear',align_corners=True)
    full=F.pad(up,(178,178,60,40),mode='replicate')[0]
    x=(torch.arange(1121)-560)*50e-6;w=torch.cos(torch.clamp((x.abs()-.0191)/.006,0,1)*torch.pi/2).square()[None,None,:]
    base=torch.tensor([1540.,1000.,.5])[:,None,None]
    material=w*full+(1-w)*base
    material[:,:80]=torch.tensor([1540.,1000.,.002])[:,None,None]
    tissue=material[2].clone();edge=torch.from_numpy(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'))
    material[2]+=edge
    q=torch.stack([((material[0]-1540)/80).clamp(-1.75,1.75),
        (torch.log(material[1]*material[0]/1540000)/.1).clamp(-2.5,2.5),(torch.log(tissue.clamp_min(.001)/.5)/.7).clamp(-8,2)])
    q[:,:80]=0
    np.save(out.with_suffix('.material.npy'),material.numpy());np.save(out.with_suffix('.parameters.npy'),q.numpy())

def main():
    p=ArgumentParser();p.add_argument('--index',type=Path,default=Path('out/v5/inverse_focus_v1/index.json'));p.add_argument('--run',type=Path,default=Path('out/v5/inverse_uno_v1'));p.add_argument('--device',type=int,default=0)
    a=p.parse_args();torch.set_num_threads(1);torch.cuda.set_device(a.device);device=f'cuda:{a.device}'
    torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device);torch.backends.cuda.matmul.allow_tf32=False
    index=json.loads(a.index.read_text());rows=[r for r in index['records'] if r['role']!='train'];model=InverseUNO().to(device)
    checkpoint=torch.load(a.run/'weights.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['ema']);model.eval()
    mean=np.load(a.run/'training_mean_material.npy');out=a.run/'evaluation';out.mkdir(exist_ok=True);predictions={};targets={};per_case=[];start=time.time()
    uniform=np.empty((3,961,1121),np.float32);uniform[0]=1540.;uniform[1]=1000.;uniform[2]=.5;uniform[2,:80]=.002
    uniform[2]+=np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy');np.save(out/'uniform_initial_material.npy',uniform)
    with torch.no_grad():
        for begin in range(0,len(rows),4):
            batch=rows[begin:begin+4];x,y=load_batch(batch,device);pred=model(x).cpu().numpy();truth=y.cpu().numpy()
            for row,value,target in zip(batch,pred,truth):
                predictions[row['id']]=value;targets[row['id']]=target;np.save(out/f"{row['id']}.prediction.npy",value)
                per_case.append({'id':row['id'],'kind':row['kind'],'has_low_frequency':'lf' in row['features'],'model':metric(value,target),'training_mean':metric(mean,target)})
                if row['id'] in ['train_200','val_059']:native_initial(value,out/f"{row['id']}_initial")
    groups={}
    for row in rows:
        if 'pair_group' in row:groups.setdefault(row['pair_group'],[]).append(row)
    pairs=[]
    for group,members in groups.items():
        if len(members)!=2:continue
        minus,plus=sorted(members,key=lambda r:r['pair_sign']);delta=predictions[plus['id']][:,5:]-predictions[minus['id']][:,5:];true=targets[plus['id']][:,5:]-targets[minus['id']][:,5:]
        detail=[]
        for j in range(3):
            v=delta[j].astype(np.float64).ravel();t=true[j].astype(np.float64).ravel();norm=np.linalg.norm(t)
            detail.append({'difference_rmse':float(np.sqrt(np.mean((v-t)**2))),'target_difference_rms':float(np.sqrt(np.mean(t*t))),
                'relative_difference_error':float(np.linalg.norm(v-t)/norm) if norm>1e-5 else None,'difference_cosine':float(np.dot(v,t)/(np.linalg.norm(v)*norm)) if norm>1e-5 and np.linalg.norm(v)>1e-10 else None})
        pairs.append({'pair_group':group,'kind':minus['kind'],'maps_c_rho_alpha':detail})
    summary={}
    for kind in ['original','alpha','trajectory']:
        selected=[r for r in per_case if r['kind']==kind]
        summary[kind]={'records':len(selected),'mean_case_rmse':np.mean([r['model']['rmse'] for r in selected],0).tolist(),'training_mean_rmse':np.mean([r['training_mean']['rmse'] for r in selected],0).tolist()}
    result={'step':checkpoint['step'],'coarse_initialization_only':True,'native_physics_reconstruction_required':True,'training_case':'Neg_07_Left','development_case':'Neg_35_Left','summary':summary,'cases':per_case,'independent_pairs':pairs,'seconds':time.time()-start,'peak_GiB':torch.cuda.max_memory_allocated()/2**30}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['cases','independent_pairs']}),flush=True)

if __name__=='__main__':main()
