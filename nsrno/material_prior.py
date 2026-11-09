"""A training-case-only continuous joint material prior, separate from RF physics.

Small explicit regularized EM avoids adding a machine-learning dependency to
the simulation environment. It supplies material statistics, not an RF map.
"""
from pathlib import Path
from argparse import ArgumentParser
import hashlib,json,time
import numpy as np
from scipy.special import logsumexp

def features(material):
    c,rho,alpha=material
    return np.stack([(c-1540)/80,(rho-1000)/100,np.log(np.maximum(alpha,.001)/.5)/.7],axis=-1)

def sample_material(material,n,rng):
    # Probe face is row 60; 1 mm known gel is 20 native cells.
    z=rng.integers(80,921,n);x=rng.integers(178,943,n)
    return features(material[:,z,x]).astype(np.float64)

def log_components(x,weight,mean,covariance):
    out=np.empty((len(x),len(weight)),np.float64)
    for k in range(len(weight)):
        diff=x-mean[k];sign,logdet=np.linalg.slogdet(covariance[k])
        if sign<=0:raise ValueError('Non-positive covariance')
        inv=np.linalg.inv(covariance[k])
        out[:,k]=np.log(weight[k])-.5*(3*np.log(2*np.pi)+logdet+np.einsum('ni,ij,nj->n',diff,inv,diff,optimize=True))
    return out

def fit(x,k=12,iterations=100,seed=3116):
    rng=np.random.default_rng(seed);subset=x[rng.choice(len(x),min(30000,len(x)),replace=False)]
    means=[subset[rng.integers(len(subset))]];distance=np.sum((subset-means[0])**2,1)
    for _ in range(1,k):
        means.append(subset[rng.choice(len(subset),p=distance/distance.sum())])
        distance=np.minimum(distance,np.sum((subset-means[-1])**2,1))
    means=np.asarray(means)
    for _ in range(30):
        d=np.sum((subset[:,None]-means[None])**2,-1);labels=d.argmin(1)
        for j in range(k):
            if np.any(labels==j):means[j]=subset[labels==j].mean(0)
    floor=np.diag(np.array([.06,.06,.15])**2)
    cov=np.repeat((np.cov(x.T)+floor)[None],k,0);weight=np.ones(k)/k
    history=[];previous=-np.inf
    for iteration in range(iterations):
        logp=log_components(x,weight,means,cov);normalizer=logsumexp(logp,axis=1)
        score=float(normalizer.mean());history.append(score)
        response=np.exp(logp-normalizer[:,None]);count=response.sum(0).clip(1e-9)
        means=response.T@x/count[:,None]
        for j in range(k):
            diff=x-means[j];cov[j]=(diff.T*response[:,j])@diff/count[j]+floor
        weight=count/count.sum()
        if iteration%10==0:print(json.dumps({'iteration':iteration,'mean_log_density':score,'smallest_weight':float(weight.min())}),flush=True)
        if iteration>=15 and abs(score-previous)<1e-6:break
        previous=score
    return weight,means,cov,history

class MaterialPrior:
    def __init__(self,path):
        import torch
        state=np.load(path);self.torch=torch
        self.weight=torch.from_numpy(state['weight'].astype(np.float32))
        self.mean=torch.from_numpy(state['mean'].astype(np.float32))
        self.precision=torch.from_numpy(np.linalg.inv(state['covariance']).astype(np.float32))
        self.log_normalizer=torch.from_numpy((np.log(state['weight'])-.5*(3*np.log(2*np.pi)+np.linalg.slogdet(state['covariance'])[1])).astype(np.float32))

    def __call__(self,material):
        torch=self.torch
        # Deterministic 0.1 mm lattice, FOV only; numerical absorption is zero here.
        c,rho,alpha=material[:,80:921:2,178:943:2]
        x=torch.stack([(c-1540)/80,(rho-1000)/100,torch.log(alpha.clamp_min(.001)/.5)/.7],-1).reshape(-1,3)
        logp=[]
        for k in range(len(self.weight)):
            diff=x-self.mean[k]
            logp.append(self.log_normalizer[k]-.5*((diff@self.precision[k])*diff).sum(-1))
        return -torch.logsumexp(torch.stack(logp,-1),-1).mean()

def main():
    p=ArgumentParser();p.add_argument('--out',type=Path,default=Path('out/v5/material_prior_v1'));p.add_argument('--components',type=int,default=12);p.add_argument('--iterations',type=int,default=100)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'prior.npz').exists():raise FileExistsError('Preserve the prior; use a new output folder')
    rng=np.random.default_rng(3116);index=Path('out/v5/forward_cache_v1/index.json')
    raw=json.loads(index.read_text());rows=[r for r in raw['records'] if r['role']=='train']
    if {r['case'] for r in rows}!={'Neg_07_Left'}:raise ValueError('Training-case isolation failed')
    samples=[];start=time.time();ids=[]
    for row in rows:
        c_rho=np.load(index.parent/row['medium'],mmap_mode='r');alpha=np.load(f"out/v5/uno_triparam_v1/alpha/{row['id']}.npy",mmap_mode='r')
        z=rng.integers(80,921,512);x=rng.integers(178,943,512)
        samples.append(features(np.stack([c_rho[0,z,x],c_rho[1,z,x],alpha[z,x]])).astype(np.float64));ids.append(row['id'])
    aug_index=Path('out/v5/uno_triparam_v1/augmentation_index.json');augment=json.loads(aug_index.read_text())['records']
    aug={r['material3']:r for r in augment if r['role']=='train'}
    if {r['case'] for r in aug.values()}!={'Neg_07_Left'}:raise ValueError('Augmentation split isolation failed')
    per_aug=round(len(rows)*512/3/len(aug))
    for path,row in sorted(aug.items()):samples.append(sample_material(np.load(path,mmap_mode='r'),per_aug,rng))
    x=np.concatenate(samples);rng.shuffle(x);np.save(a.out/'training_features.npy',x.astype(np.float32))
    metadata={'source_case':'Neg_07_Left','development_case_excluded':'Neg_35_Left','base_record_ids':ids,'augmented_material_paths':sorted(aug),'base_pixels':len(rows)*512,'augmented_pixels':per_aug*len(aug),'sample_count':len(x),
        'features':['(c-1540)/80','(rho-1000)/100','log(alpha_tissue/.5)/.7'],'independent_feature_standard_deviation_floor':[.06,.06,.15],
        'index_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [index,aug_index]},'feature_sha256':hashlib.sha256(x.astype(np.float32).tobytes()).hexdigest(),'uses_rf':False,'uses_development_materials':False}
    weight,mean,cov,history=fit(x,a.components,a.iterations)
    np.savez(a.out/'prior.npz',weight=weight,mean=mean,covariance=cov)
    metadata.update(seconds=time.time()-start,mean_log_density_history=history,status='fitted_not_reconstruction')
    (a.out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({k:v for k,v in metadata.items() if k not in ['base_record_ids','augmented_material_paths','mean_log_density_history']}),flush=True)

if __name__=='__main__':main()
