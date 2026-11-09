"""Native three-material FWI with all-shot stochastic source encoding.

CPU Adam updates unrestricted multiscale coordinates. GPU workers hold one
encoded propagation at a time. Only full unencoded RF selects checkpoints.
Truth is absent from this optimizer.
"""
from argparse import ArgumentParser
from pathlib import Path
import hashlib,json,multiprocessing as mp,os,time,traceback
import numpy as np
import torch
from .time_acquisition import TimeAcquisition
from .encoded_time import EncodedAcquisition,prepare_bank
from .multiscale_parameters import MultiScaleParameters
from .reconstruct_time import Parameters,atomic_json,WINDOW_WEIGHTS
from .rf_depth_weights import DepthWeights,NAMES as DEPTH_NAMES,WEIGHTS as DEPTH_WEIGHTS


def save_numpy(path,value):
    partial=path.with_name(path.stem+'.partial.npy');np.save(partial,value);partial.replace(path)


def worker(rank,device,jobs,results,sample,banks,out,cap):
    try:
        torch.set_num_threads(1);torch.cuda.set_device(device);torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(cap*2**30/torch.cuda.get_device_properties(device).total_memory,device)
        acquisition={};normalizers={};depth_weights={}
        while True:
            job=jobs.get()
            if job is None:break
            token,m,stage,codes,mode=job;gradient=mode=='encoded';start=time.time();parts={};value=0.
            material=torch.from_numpy(m).to(f'cuda:{device}').requires_grad_(gradient)
            torch.cuda.reset_peak_memory_stats()
            for source in stage['assignment'][rank]:
                if source not in acquisition:
                    torch.cuda.empty_cache();a=TimeAcquisition(source,sample,device=f'cuda:{device}')
                    acquisition[source]=EncodedAcquisition(a,banks[source]);normalizers[source]=acquisition[source].denominators()
                    if stage['loss_windows']=='geometry':depth_weights[source]=DepthWeights(a)
                encoded=acquisition[source];a=encoded.acquisition;windows,den=normalizers[source];num=np.zeros(4)
                geometric=stage['loss_windows']=='geometry'
                if geometric:num=np.zeros(5);legacy=np.zeros(4)
                count=len(codes[source]) if gradient else len(a.angles)
                for shot in range(count):
                    atomic_json(out/f'worker{rank}_progress.json',{'token':token,'mode':mode,'source':source,'shot':shot,'count':count,'state':'computing','unix_s':time.time()})
                    with torch.set_grad_enabled(gradient):
                        sampled=gradient and stage['sampling']=='shot';index=int(codes[source][shot]) if sampled else shot
                        pred=encoded.predict(material,codes[source][shot]) if gradient and not sampled else a.predict(material,index)
                        target=encoded.observed(codes[source][shot]) if gradient and not sampled else a.observed[index]
                        difference=pred-target
                        time_terms=[difference[:,round(b*40):round(e*40)].square().sum()/den[j] for j,(b,e) in enumerate(windows)]
                        terms=depth_weights[source].terms(difference,index) if geometric else time_terms
                        weights=DEPTH_WEIGHTS if geometric else WINDOW_WEIGHTS
                        multiplier=len(a.angles) if sampled else 1
                        loss=.5*sum(w*t for w,t in zip(weights,terms))*multiplier/len(stage['sources'])/(count if gradient else 1)
                        if gradient:loss.backward()
                    value+=float(loss.detach());num+=np.asarray([float(t.detach()) for t in terms])*multiplier/(count if gradient else 1)
                    if geometric:legacy+=np.asarray([float(t.detach()) for t in time_terms])*multiplier/(count if gradient else 1)
                    del pred,target,difference,terms,time_terms,loss
                parts[source]=num.tolist()
                if geometric:parts[source+':original_time_windows']=legacy.tolist()
            g=material.grad.cpu().numpy() if gradient else None
            peak=torch.cuda.max_memory_allocated()/2**30;elapsed=time.time()-start
            atomic_json(out/f'worker{rank}_progress.json',{'token':token,'mode':mode,'state':'returned','seconds':elapsed,'peak_GiB':peak,'unix_s':time.time()})
            results.put((rank,token,value,g,parts,elapsed,peak,None));del material
    except BaseException:
        results.put((rank,-1,None,None,None,None,None,traceback.format_exc()))


def main():
    p=ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--sample',default='train_200')
    p.add_argument('--resume-parameters',type=Path,required=True);p.add_argument('--devices',type=int,nargs='+',default=[0,2])
    p.add_argument('--steps',type=int,nargs=2,default=[64,128]);p.add_argument('--validate-every',type=int,default=16)
    p.add_argument('--codes',type=int,default=1);p.add_argument('--memory-limit-gib',type=float,default=5.75)
    p.add_argument('--learning-rates',type=float,nargs=2,default=[.006,.003]);p.add_argument('--seed',type=int,default=20261106)
    p.add_argument('--sampling',choices=['encoded','shot'],default='encoded');p.add_argument('--loss-windows',choices=['time','geometry'],default='time')
    p.add_argument('--parameterization',choices=['physical','travel_exact'],default='physical')
    args=p.parse_args();torch.set_num_threads(1)
    if args.loss_windows=='geometry' and args.sampling!='shot':p.error('Source-dependent geometry masks cannot be applied after all-source encoding')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'run.json').exists():raise FileExistsError('Preserve previous experiments')
    stages=[{'step':1,'sources':['point3','lf'],'maxiter':args.steps[0]},{'step':1,'sources':['point3','lf','hf'],'maxiter':args.steps[1]}]
    for stage in stages:
        stage.update(sampling=args.sampling,loss_windows=args.loss_windows)
        loads=[0.]*len(args.devices);assignment=[[] for _ in loads]
        for source in sorted(stage['sources'],key=lambda s:1.5 if s=='point3' else 1.,reverse=True):
            rank=int(np.argmin(loads));assignment[rank].append(source);loads[rank]+=1.5 if source=='point3' else 1.
        stage['assignment']=assignment
    sources=['reconstruct_encoded.py','encoded_time.py','time_acquisition.py','native_time.py','native_time.cu','multiscale_parameters.py','reconstruct_time.py','rf_depth_weights.py']
    if args.parameterization=='travel_exact':sources+=['travel_time_parameters.py','exact_travel_parameters.py']
    record={'sample':args.sample,'stages':stages,'devices':args.devices,'memory_limit_GiB_per_worker':args.memory_limit_gib,'optimizer':'CPU Adam; independent per-level learning rates',
        'learning_rates':args.learning_rates,'level_multipliers':[1.,1.,.5,.25],'codes_per_source_group':args.codes,'seed':args.seed,
        'sampling':args.sampling,'parameterization':args.parameterization,'shots_per_encoded_source':11 if args.sampling=='encoded' else 1,'target':'full RF difference with fixed observed-energy normalization; all retained frequencies',
        'window_definition':{'kind':args.loss_windows,'geometry_names':DEPTH_NAMES,'geometry_weights':DEPTH_WEIGHTS,'interpretation':'homogeneous midpoint-equivalent arrival classes, not true depth localization'},
        'truth_used_by_optimizer':False,'selection':'only full unencoded RF plus existing weak TV','status':'initializing_source_banks',
        'resume_parameters':{'path':str(args.resume_parameters),'sha256':hashlib.sha256(args.resume_parameters.read_bytes()).hexdigest()},
        'sources_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}}
    atomic_json(args.out/'run.json',record)
    for name in sources:(args.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    banks={}
    for source in ['point3','lf','hf']:
        if args.sampling=='encoded':
            cpu=TimeAcquisition(source,args.sample,device='cpu');banks[source]=str(prepare_bank(cpu));del cpu
        else:banks[source]=None
    record.update(status='running',source_banks=banks);atomic_json(args.out/'run.json',record)
    parameter_class=MultiScaleParameters
    if args.parameterization=='travel_exact':
        from .exact_travel_parameters import ExactTravelTimeParameters
        parameter_class=ExactTravelTimeParameters
    model=parameter_class((961,1121),1);initial=torch.from_numpy(np.load(args.resume_parameters).astype(np.float32))
    packed=model.initialize(initial);parts=[torch.nn.Parameter(v.clone()) for v in packed.split(model.counts)]
    if args.parameterization=='travel_exact':
        record['initialization_preservation']=model.initialization_report
        record['coordinates']=['c(xi,x)','Z(xi,x)','beta(xi,x)=alpha*c/1540']
        atomic_json(args.out/'run.json',record)
    uniform=Parameters((961,1121),1);save_numpy(args.out/'uniform_initial_material.npy',uniform.material(torch.zeros(3,*uniform.small)).numpy())
    context=mp.get_context('spawn');results=context.Queue();queues=[context.Queue() for _ in args.devices]
    processes=[context.Process(target=worker,args=(rank,device,queues[rank],results,args.sample,banks,args.out,args.memory_limit_gib)) for rank,device in enumerate(args.devices)]
    for process in processes:process.start()
    rng=np.random.default_rng(args.seed);token=0;start=time.time();log=args.out/'evaluations.jsonl';shot_pools={}
    def evaluate(material,stage,mode):
        nonlocal token
        token+=1;codes=None
        if mode=='encoded':
            codes={}
            for source in stage['sources']:
                if args.sampling=='encoded':codes[source]=rng.choice([-1.,1.],(args.codes,11)).astype(np.float32)
                else:
                    selected=[]
                    for _ in range(args.codes):
                        if not shot_pools.get(source):shot_pools[source]=rng.permutation(11).tolist()
                        selected.append(shot_pools[source].pop())
                    codes[source]=selected
        m=material.detach().numpy()
        for queue in queues:queue.put((token,m,stage,codes,mode))
        value=0.;g=np.zeros_like(m);windows={};peak=[];seconds=[]
        for _ in processes:
            rank,returned,v,grad,numerators,elapsed,memory,error=results.get()
            if error:raise RuntimeError(error)
            if returned!=token:raise RuntimeError('Out-of-order encoded worker response')
            value+=v
            if grad is not None:g+=grad
            windows.update({source:np.sqrt(v).tolist() for source,v in numerators.items()});peak.append(memory);seconds.append(elapsed)
        return value,g,windows,peak,seconds
    try:
        for stage_id,stage in enumerate(stages):
            if stage['maxiter']<=0:continue
            rate=args.learning_rates[stage_id];multipliers=[1.,1.,.5,.25]
            optimizer=torch.optim.Adam([{'params':[part],'lr':rate*scale} for part,scale in zip(parts,multipliers)])
            best={'objective':float('inf')}
            def full_score(step):
                fields=torch.cat(parts);material=model.material(fields)
                value,_,windows,peak,seconds=evaluate(material,stage,'full')
                tv=float(model.regularization(fields,1e-3).detach());total=value+tv
                item={'kind':'full_unencoded','stage':stage_id,'step':step,'data_loss':value,'regularization':tv,'objective':total,
                    'window_relative_rms':windows,'peak_GiB_per_worker':peak,'worker_seconds':seconds,'elapsed_seconds':time.time()-start}
                with log.open('a') as f:f.write(json.dumps(item)+'\n')
                print(json.dumps(item),flush=True)
                if total<best['objective']:
                    best.update(objective=total,packed=fields.detach().clone(),item=item)
                    save_numpy(args.out/f'stage{stage_id}_best_material.npy',material.detach().numpy())
                    save_numpy(args.out/f'stage{stage_id}_best_parameters.npy',model.checkpoint_fields(fields.detach()).numpy())
                    atomic_json(args.out/f'stage{stage_id}_best.json',item)
            save_numpy(args.out/f'stage{stage_id}_initial_material.npy',model.material(torch.cat(parts)).detach().numpy())
            full_score(0)
            for step in range(1,stage['maxiter']+1):
                optimizer.zero_grad();fields=torch.cat(parts);material=model.material(fields)
                value,g,windows,peak,seconds=evaluate(material,stage,'encoded')
                material.backward(torch.from_numpy(g));tv=model.regularization(torch.cat(parts),1e-3);tv.backward()
                norms=model.coordinate_norms(torch.cat([part.grad for part in parts]));decay=.2+.8*.5*(1+np.cos(np.pi*(step-1)/stage['maxiter']))
                for group,scale in zip(optimizer.param_groups,multipliers):group['lr']=rate*scale*decay
                optimizer.step()
                item={'kind':'stochastic_'+args.sampling+'_update','stage':stage_id,'step':step,'estimated_data_loss':value,'regularization':float(tv.detach()),
                    'gradient_norm_by_coordinate':norms,'peak_GiB_per_worker':peak,'worker_seconds':seconds,'elapsed_seconds':time.time()-start,
                    'not_used_for_checkpoint_selection':True}
                with log.open('a') as f:f.write(json.dumps(item)+'\n')
                print(json.dumps(item),flush=True)
                if step%args.validate_every==0 or step==stage['maxiter']:full_score(step)
            for part,value in zip(parts,best['packed'].split(model.counts)):part.data.copy_(value)
            atomic_json(args.out/f'stage{stage_id}_result.json',{'updates':stage['maxiter'],'best_full_unencoded_objective':best['objective'],'selected_step':best['item']['step']})
        save_numpy(args.out/'final_material.npy',model.material(torch.cat(parts)).detach().numpy())
        record.update(status='optimization_finished',elapsed_seconds=time.time()-start);atomic_json(args.out/'run.json',record)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc());atomic_json(args.out/'run.json',record);raise
    finally:
        for queue in queues:queue.put(None)
        for process in processes:process.join(timeout=20)
        for process in processes:
            if process.is_alive():process.terminate();process.join()


if __name__=='__main__':main()
