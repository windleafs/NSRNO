"""All-angle, fixed-objective c/impedance/attenuation FWI, with CPU L-BFGS.

Two persistent GPU workers process disjoint angles. Each GPU holds one shot
and one checkpoint-recompute segment, never a space-time autograd graph.
Truth is deliberately absent from this optimization module.
"""
from pathlib import Path
from argparse import ArgumentParser
import hashlib,json,multiprocessing as mp,os,time,traceback
import numpy as np
from scipy.optimize import minimize
import torch
from torch.nn import functional as F
from .time_acquisition import TimeAcquisition,bandlimit,ROOT
from .waveform_objective import smooth_envelope

WINDOWS=[(0,8),(8,18),(18,36),(36,60.025)]
WINDOW_WEIGHTS=[.25,.35,.35,.05]

def worker(rank,device,jobs,results,sample,chunk,memory_limit=16):
    try:
        torch.set_num_threads(1);torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False
        capacity=torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,memory_limit*2**30/capacity),device)
        acquisition={};cached_targets={}
        while True:
            job=jobs.get()
            if job is None:break
            token,m,stage,need_gradient=job
            material=torch.from_numpy(m).to(f'cuda:{device}').requires_grad_(need_gradient)
            total=0.;numerators={};start=time.time();torch.cuda.reset_peak_memory_stats()
            for source in stage['sources']:
                if source not in acquisition:
                    torch.cuda.empty_cache()
                    acquisition[source]=TimeAcquisition(source,sample,device=f'cuda:{device}',chunk=chunk)
                a=acquisition[source];lo,hi=stage['bands'][source]
                windows=WINDOWS if source!='point3' else WINDOWS[:-1]+[(36,90.025)]
                key=(source,lo,hi)
                if key not in cached_targets:
                    obs=bandlimit(a.observed,lo,hi)
                    denominators=[max(float(obs[:,:,round(b*40):round(e*40)].double().square().sum()),1e-20) for b,e in windows]
                    cached_targets[key]=(obs,denominators)
                target,den=cached_targets[key];num=np.zeros(4,np.float64)
                sigma=stage.get('envelope_sigma_us',0.);envelope_weight=stage.get('envelope_weight',0.)
                if envelope_weight:
                    envelope_key=(source,lo,hi,'envelope',sigma)
                    if envelope_key not in cached_targets:
                        env=smooth_envelope(target,sigma)
                        eden=[max(float(env[:,:,round(b*40):round(e*40)].double().square().sum()),1e-20) for b,e in windows]
                        cached_targets[envelope_key]=(env,eden)
                    target_env,eden=cached_targets[envelope_key];enum=np.zeros(4,np.float64)
                for angle in range(rank,len(a.angles),stage['workers']):
                    with torch.set_grad_enabled(need_gradient):
                        pred=bandlimit(a.predict(material,angle),lo,hi);diff=pred-target[angle]
                        terms=[diff[:,round(b*40):round(e*40)].square().sum()/den[j] for j,(b,e) in enumerate(windows)]
                        phase_loss=sum(w*t for w,t in zip(WINDOW_WEIGHTS,terms))
                        if envelope_weight:
                            ediff=smooth_envelope(pred,sigma)-target_env[angle]
                            eterms=[ediff[:,round(b*40):round(e*40)].square().sum()/eden[j] for j,(b,e) in enumerate(windows)]
                            envelope_loss=sum(w*t for w,t in zip(WINDOW_WEIGHTS,eterms))
                            combined=(1-envelope_weight)*phase_loss+envelope_weight*envelope_loss
                        else:combined=phase_loss
                        loss=.5*combined/len(stage['sources'])
                        if need_gradient:loss.backward()
                    total+=float(loss.detach())
                    num+=np.asarray([float(t.detach()) for t in terms])
                    if envelope_weight:
                        enum+=np.asarray([float(t.detach()) for t in eterms]);del ediff,eterms,envelope_loss
                    del pred,diff,terms,loss
                numerators[source]=num.tolist()
                if envelope_weight:numerators[source+':envelope']=enum.tolist()
            grad=material.grad.cpu().numpy() if need_gradient else None
            results.put((rank,token,total,grad,numerators,time.time()-start,torch.cuda.max_memory_allocated()/2**30,None))
            del material
    except BaseException:
        results.put((rank,-1,None,None,None,None,None,traceback.format_exc()))

class Parameters:
    def __init__(self,shape,step):
        self.native=shape;self.small=((shape[0]-1)//step+1,(shape[1]-1)//step+1)
        self.mask=(torch.arange(shape[0])[:,None]>=80).expand(shape)
        self.edge=torch.from_numpy(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')).float()
        if self.edge.ndim==1:self.edge=self.edge[None].expand(shape)

    def material(self,q):
        q=F.interpolate(q[None],size=self.native,mode='bilinear',align_corners=True)[0]
        c=1540+80*q[0];impedance=1540e3*torch.exp(.1*q[1])
        rho=impedance/c;alpha=.5*torch.exp(.7*q[2])
        m=torch.stack([torch.where(self.mask,c,1540.),torch.where(self.mask,rho,1000.),torch.where(self.mask,alpha,.002)+self.edge])
        return m

    def fields(self,q):return q
    def checkpoint_fields(self,q):return self.fields(q)

    def initialize(self,previous=None):
        return torch.zeros((3,*self.small)) if previous is None else F.interpolate(previous[None],size=self.small,mode='bilinear',align_corners=True)[0]

    def coordinate_norms(self,gradient):return gradient.reshape(3,*self.small).square().sum((-2,-1)).sqrt().tolist()

    def regularization(self,q,weight):
        # Weak isotropic TV on each independent coordinate; no imposed co-edges.
        dz=q[:,1:,:-1]-q[:,:-1,:-1];dx=q[:,:-1,1:]-q[:,:-1,:-1]
        return weight*((dz*dz+dx*dx+1e-6).sqrt()-.001).mean()

    def bounds(self):
        n=np.prod(self.small)
        return [(-1.75,1.75)]*n+[(-2.5,2.5)]*n+[(-8,2)]*n

def atomic_json(path,value):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def main():
    parser=ArgumentParser();parser.add_argument('--out',type=Path,default=Path('out/v5/reconstruction_time_v1_train_200'))
    parser.add_argument('--sample',default='train_200');parser.add_argument('--devices',type=int,nargs='+',default=[1,2])
    parser.add_argument('--iterations',type=int,nargs='+',default=[12,12,18,18]);parser.add_argument('--chunk',type=int,default=128)
    parser.add_argument('--resume-parameters',type=Path)
    parser.add_argument('--material-prior',type=Path)
    parser.add_argument('--prior-weights',type=float,nargs=4,default=[.05,.03,.01,.003])
    parser.add_argument('--parameterization',choices=['pixel','multiscale','native_residual','travel_time'],default='pixel')
    parser.add_argument('--optimizer',choices=['scipy_lbfgsb','filtered_lbfgs'],default='scipy_lbfgsb')
    parser.add_argument('--memory-limit-gib',type=float,default=16)
    parser.add_argument('--acquisition-mode',choices=['plane','plane_point'],default='plane')
    parser.add_argument('--initial-band-hz',type=float,nargs=2,default=[1.5e6,2.6e6])
    parser.add_argument('--objective-schedule',choices=['waveform','envelope_to_waveform'],default='waveform')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.optimizer=='filtered_lbfgs' and args.parameterization=='pixel':
        parser.error('Filtered L-BFGS requires bounds enforced by the multiscale parameterization')
    parameter_class=Parameters
    if args.parameterization=='multiscale':
        from .multiscale_parameters import MultiScaleParameters
        parameter_class=MultiScaleParameters
    elif args.parameterization=='native_residual':
        from .native_residual_parameters import NativeResidualParameters
        parameter_class=NativeResidualParameters
    elif args.parameterization=='travel_time':
        from .travel_time_parameters import TravelTimeParameters
        parameter_class=TravelTimeParameters
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'run.json').exists():raise FileExistsError('Use a new output directory; do not overwrite a run')
    stages=[
        {'step':4,'sources':['lf'],'bands':{'lf':args.initial_band_hz}},
        {'step':2,'sources':['lf'],'bands':{'lf':[1.5e6,3.3e6]}},
        {'step':1,'sources':['lf'],'bands':{'lf':[0.,20e6]}},
        {'step':1,'sources':['lf','hf'],'bands':{'lf':[0.,20e6],'hf':[0.,20e6]}}
    ]
    if args.acquisition_mode=='plane_point':
        for stage in stages:
            stage['sources'].insert(0,'point3');stage['bands']['point3']=stage['bands']['lf'].copy()
    if args.objective_schedule=='envelope_to_waveform':
        for stage,sigma,weight in zip(stages[:2],[1.,.35],[.999,.99]):
            stage.update(envelope_sigma_us=sigma,envelope_weight=weight)
            for source in stage['sources']:stage['bands'][source]=[0.,20e6]
    if len(args.iterations)!=4:parser.error('Four iteration limits are required')
    for j,(s,n) in enumerate(zip(stages,args.iterations)):
        s.update(workers=len(args.devices),maxiter=n,material_prior_weight=args.prior_weights[j] if args.material_prior else 0.)
    record={'sample':args.sample,'stages':stages,'parameterization':args.parameterization,'optimizer':args.optimizer,'objective_schedule':args.objective_schedule,'initial':{'c':1540.,'rho':1000.,'alpha_tissue':.5},'coordinates':['(c-1540)/80','log(Z/1540000)/.1','log(alpha_tissue/.5)/.7'],
        'window_weights':dict(zip([str(w) for w in WINDOWS],WINDOW_WEIGHTS)),'devices':args.devices,'memory_limit_GiB_per_device':args.memory_limit_gib,'truth_used_by_optimizer':False,'status':'running',
        'sources_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('native_time.cu'),Path(__file__).with_name('native_time.py'),Path(__file__).with_name('time_acquisition.py'),Path(__file__).with_name('waveform_objective.py')]}}
    if args.resume_parameters:
        record['resume_parameters']={'path':str(args.resume_parameters),'sha256':hashlib.sha256(args.resume_parameters.read_bytes()).hexdigest()}
    if args.acquisition_mode=='plane_point':
        point_index=Path('out/v5/point_tx_3mhz_v1/index.json')
        point=json.loads(point_index.read_text())
        if not any(r['id']==args.sample for r in point['records']):raise ValueError('Missing additional RF sample')
        record['additional_acquisition']={'index':str(point_index),'sha256':hashlib.sha256(point_index.read_bytes()).hexdigest(),'acquisition':point['acquisition'],
            'windows_us':WINDOWS[:-1]+[(36,90.025)],'scope':'Changed acquisition; original 11-plane-wave sufficiency is not implied'}
    prior=None
    if args.material_prior:
        from .material_prior import MaterialPrior
        prior_path=args.out/'material_prior.npz'
        prior_path.write_bytes(args.material_prior.read_bytes())
        record['material_prior']={'path':str(args.material_prior),'sha256':hashlib.sha256(prior_path.read_bytes()).hexdigest(),'weights':args.prior_weights}
        prior=MaterialPrior(prior_path)
        record['sources_sha256']['material_prior.py']=hashlib.sha256(Path(__file__).with_name('material_prior.py').read_bytes()).hexdigest()
    record['actual_initialization']='resumed_parameters' if args.resume_parameters else 'uniform'
    if args.parameterization in ['multiscale','native_residual','travel_time']:
        record['sources_sha256']['multiscale_parameters.py']=hashlib.sha256(Path(__file__).with_name('multiscale_parameters.py').read_bytes()).hexdigest()
    if args.parameterization=='native_residual':
        record['sources_sha256']['native_residual_parameters.py']=hashlib.sha256(Path(__file__).with_name('native_residual_parameters.py').read_bytes()).hexdigest()
        record['coarse_parameterization']='native fine initialization is preserved as a fixed raw-field residual during coarse updates; final native field remains free'
    if args.parameterization=='travel_time':
        record['sources_sha256']['travel_time_parameters.py']=hashlib.sha256(Path(__file__).with_name('travel_time_parameters.py').read_bytes()).hexdigest()
        record['coordinates']=['(c(xi)-1540)/80','log(Z(xi)/1540000)/.1','log(beta(xi)/.5)/.7']
        record['coordinate_map']={'xi_spacing_m':40e-6,'xi_start_m':-.003,'xi_end_m':.05,'physical_native_spacing_m':50e-6,'rho':'Z/c','alpha_tissue':'beta*1540/c','normal_1d_invariance':'continuous bulk equations only; 2D forward remains unchanged'}
    if args.optimizer=='filtered_lbfgs':
        record['sources_sha256']['armijo_lbfgs.py']=hashlib.sha256(Path(__file__).with_name('armijo_lbfgs.py').read_bytes()).hexdigest()
    atomic_json(args.out/'run.json',record)
    uniform=Parameters((961,1121),4)
    np.save(args.out/'uniform_initial_material.npy',uniform.material(torch.zeros(3,*uniform.small)).numpy())
    for name in record['sources_sha256']:(args.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    context=mp.get_context('spawn');results=context.Queue();queues=[context.Queue() for _ in args.devices]
    processes=[context.Process(target=worker,args=(r,d,queues[r],results,args.sample,args.chunk,args.memory_limit_gib)) for r,d in enumerate(args.devices)]
    for process in processes:process.start()
    counter=0;accepted=0;start=time.time()
    q_previous=torch.from_numpy(np.load(args.resume_parameters).astype(np.float32)) if args.resume_parameters else None
    log=args.out/'evaluations.jsonl';history=args.out/'accepted.jsonl'
    try:
        for stage_id,stage in enumerate(stages):
            if stage['maxiter']<=0:continue
            model=parameter_class((961,1121),stage['step'])
            q0=model.initialize(q_previous)
            np.save(args.out/f'stage{stage_id}_initial_material.npy',model.material(q0).numpy())
            best={'value':float('inf')};cache={}
            def evaluate(x,need_gradient=True):
                nonlocal counter
                q=torch.tensor(x.reshape(q0.shape),dtype=torch.float32,requires_grad=need_gradient)
                material=model.material(q);m=material.detach().numpy();counter+=1;token=counter
                for queue in queues:queue.put((token,m,stage,need_gradient))
                value=0.;gradient=np.zeros_like(m);windows={};peak=[];seconds=[]
                for _ in processes:
                    rank,returned,v,g,parts,elapsed,memory,error=results.get()
                    if error:raise RuntimeError(error)
                    if returned!=token:raise RuntimeError('Out-of-order worker result')
                    value+=v
                    if need_gradient:gradient+=g
                    peak.append(memory);seconds.append(elapsed)
                    for source,part in parts.items():windows[source]=np.asarray(windows.get(source,np.zeros(4)))+part
                if need_gradient:material.backward(torch.from_numpy(gradient))
                tv=model.regularization(q,1e-3)
                prior_nll=prior(model.material(q)) if prior is not None else None
                regularizer=tv+(stage['material_prior_weight']*prior_nll if prior_nll is not None else 0.)
                if need_gradient:regularizer.backward()
                total=value+float(regularizer.detach());grad=q.grad.detach().numpy().astype(np.float64).ravel() if need_gradient else None
                item={'evaluation':counter,'stage':stage_id,'gradient_computed':need_gradient,'data_loss':value,'regularization':float(regularizer.detach()),'tv_regularization':float(tv.detach()),'material_prior_nll':float(prior_nll.detach()) if prior_nll is not None else None,'material_prior_weight':stage['material_prior_weight'],'objective':total,'window_relative_rms':{k:np.sqrt(v).tolist() for k,v in windows.items()},
                    'gradient_norm':float(np.linalg.norm(grad)) if need_gradient else None,'gradient_norm_by_coordinate':model.coordinate_norms(q.grad.detach()) if need_gradient else None,'material_gradient_norm':[float(np.linalg.norm(v)) for v in gradient] if need_gradient else None,
                    'peak_GiB_per_worker':peak,'worker_seconds':seconds,'elapsed_seconds':time.time()-start,'material_minmax':[[float(v.min()),float(v.max())] for v in m]}
                with log.open('a') as f:f.write(json.dumps(item)+'\n')
                print(json.dumps(item),flush=True)
                cache.update(x=x.copy(),item=item)
                if total<best['value']:
                    best.update(value=total,x=x.copy(),item=item)
                    np.save(args.out/f'stage{stage_id}_best_material.npy',m)
                    np.save(args.out/f'stage{stage_id}_best_parameters.npy',model.checkpoint_fields(q.detach()).numpy())
                    atomic_json(args.out/f'stage{stage_id}_best.json',item)
                return total,grad
            def callback(x):
                nonlocal accepted
                accepted+=1;item={'accepted_step':accepted,'stage':stage_id,'evaluation':counter,'elapsed_seconds':time.time()-start}
                if np.array_equal(cache.get('x'),x):item.update(objective=cache['item']['objective'],data_loss=cache['item']['data_loss'])
                with history.open('a') as f:f.write(json.dumps(item)+'\n')
            # Full fixed-band all-angle objective on every line-search evaluation.
            if args.optimizer=='filtered_lbfgs':
                from .armijo_lbfgs import minimize as armijo_minimize
                result=armijo_minimize(evaluate,q0.numpy().astype(np.float64).ravel(),callback=callback,maxiter=stage['maxiter'])
            else:
                result=minimize(evaluate,q0.numpy().astype(np.float64).ravel(),method='L-BFGS-B',jac=True,bounds=model.bounds(),callback=callback,
                    options={'maxiter':stage['maxiter'],'maxcor':8,'maxls':12,'ftol':1e-9,'gtol':1e-7,'maxfun':stage['maxiter']*6+12})
            q_previous=model.checkpoint_fields(torch.from_numpy(best['x'].reshape(q0.shape).astype(np.float32))).detach()
            atomic_json(args.out/f'stage{stage_id}_result.json',{'message':str(result.message),'success':bool(result.success),'iterations':int(result.nit),'evaluations':int(result.nfev),'best_objective':best['value']})
        np.save(args.out/'final_material.npy',Parameters((961,1121),1).material(q_previous).numpy())
        record.update(status='optimization_finished',elapsed_seconds=time.time()-start,accepted_steps=accepted)
        atomic_json(args.out/'run.json',record)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc());atomic_json(args.out/'run.json',record);raise
    finally:
        for queue in queues:queue.put(None)
        for process in processes:process.join(timeout=20)
        for process in processes:
            if process.is_alive():process.terminate();process.join()

if __name__=='__main__':main()
