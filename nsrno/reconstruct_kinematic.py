"""Kinematic initialization followed by unrestricted native three-material FWI.

CPU Adam updates unrestricted multiscale coordinates. GPU workers hold one
shot propagation at a time. Initial selection uses the stated kinematic target;
final selection uses only full unencoded RF. Neither selection reads truth.
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
from .adaptive_rf_loss import AdaptiveMatchingLoss
from .differential_phase_loss import DifferentialPhaseLoss


def save_numpy(path,value):
    partial=path.with_name(path.stem+'.partial.npy');np.save(partial,value);partial.replace(path)


def worker(rank,device,jobs,results,sample,banks,out,cap):
    try:
        torch.set_num_threads(1);torch.cuda.set_device(device);torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(cap*2**30/torch.cuda.get_device_properties(device).total_memory,device)
        acquisition={};normalizers={};depth_weights={};kinematic={}
        while True:
            job=jobs.get()
            if job is None:break
            token,m,stage,codes,mode=job;gradient=mode=='encoded';start=time.time();parts={};value=0.
            if not stage['kinematic_weight']:
                kinematic.clear()
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
                blend=stage['kinematic_weight'];knum=np.zeros(5)
                if blend and source not in kinematic:
                    provider=AdaptiveMatchingLoss(a,bins=depth_weights[source].bins,
                        sigma_us=stage['sigma_us'],filter_energy_floor=stage['filter_energy_floor'])
                    kinematic[source]=(DifferentialPhaseLoss(a,provider,lag_bins=stage['phase_lag_bins'])
                                       if stage['kinematic_objective']=='differential' else provider)
                    torch.cuda.empty_cache()
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
                        rf_loss=sum(w*t for w,t in zip(weights,terms))
                        if blend:
                            kterms=kinematic[source].terms(pred,index)
                            kloss=sum(w*t for w,t in zip(weights,kterms))
                            combined=(1-blend)*rf_loss+blend*kloss
                        else:combined=rf_loss
                        loss=.5*combined*multiplier/len(stage['sources'])/(count if gradient else 1)
                        if gradient:loss.backward()
                    value+=float(loss.detach());num+=np.asarray([float(t.detach()) for t in terms])*multiplier/(count if gradient else 1)
                    if geometric:legacy+=np.asarray([float(t.detach()) for t in time_terms])*multiplier/(count if gradient else 1)
                    if blend:
                        knum+=np.asarray([float(t.detach()) for t in kterms])*multiplier/(count if gradient else 1)
                        del kterms,kloss
                    del pred,target,difference,terms,time_terms,loss,rf_loss,combined
                parts[source]=num.tolist()
                if geometric:parts[source+':original_time_windows']=legacy.tolist()
                if blend:parts[source+':sqrt_kinematic_penalty']=knum.tolist()
            g=material.grad.cpu().numpy() if gradient else None
            peak=torch.cuda.max_memory_allocated()/2**30;elapsed=time.time()-start
            atomic_json(out/f'worker{rank}_progress.json',{'token':token,'mode':mode,'state':'returned','seconds':elapsed,'peak_GiB':peak,'unix_s':time.time()})
            results.put((rank,token,value,g,parts,elapsed,peak,None));del material
    except BaseException:
        results.put((rank,-1,None,None,None,None,None,traceback.format_exc()))


def main():
    p=ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--sample',default='train_200')
    p.add_argument('--resume-parameters',type=Path,required=True);p.add_argument('--devices',type=int,nargs='+',default=[0,2])
    p.add_argument('--steps',type=int,nargs=2,default=[64,128]);p.add_argument('--validate-every',type=int,default=32)
    p.add_argument('--codes',type=int,default=1);p.add_argument('--memory-limit-gib',type=float,default=5.75)
    p.add_argument('--learning-rates',type=float,nargs=2,default=[.015,.003]);p.add_argument('--seed',type=int,default=20261106)
    p.add_argument('--joint-material-rates',type=float,nargs=3)
    p.add_argument('--level-multipliers',type=float,nargs=4,default=[1.,1.,.5,1.])
    p.add_argument('--sampling',choices=['shot'],default='shot');p.add_argument('--loss-windows',choices=['geometry'],default='geometry')
    p.add_argument('--parameterization',choices=['physical','travel_exact'],default='physical')
    p.add_argument('--initial-objective',choices=['adaptive','differential'],default='differential')
    p.add_argument('--initial-kinematic-weight',type=float,default=1.)
    p.add_argument('--initial-update',choices=['all','c_only'],default='c_only')
    p.add_argument('--sigma-us',type=float,default=1.5);p.add_argument('--filter-energy-floor',type=float,default=1e-3)
    p.add_argument('--phase-lag-bins',type=int,default=16)
    args=p.parse_args();torch.set_num_threads(1)
    if args.joint_material_rates is None:args.joint_material_rates=[args.learning_rates[1],.008,.008]
    if args.loss_windows=='geometry' and args.sampling!='shot':p.error('Source-dependent geometry masks cannot be applied after all-source encoding')
    if not 0<args.initial_kinematic_weight<=1:p.error('Initial kinematic weight must be in (0,1]')
    if args.initial_update=='all' and args.initial_kinematic_weight>=1:p.error('Updating all materials requires a nonzero RF amplitude weight')
    if args.initial_update=='c_only' and args.parameterization!='physical':p.error('This c-only comparison holds Z and alpha in physical coordinates')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'run.json').exists():raise FileExistsError('Preserve previous experiments')
    stages=[{'step':1,'sources':['point3','lf'],'maxiter':args.steps[0]},{'step':1,'sources':['point3','lf','hf'],'maxiter':args.steps[1]}]
    for stage_id,stage in enumerate(stages):
        stage.update(sampling=args.sampling,loss_windows=args.loss_windows,
            kinematic_weight=args.initial_kinematic_weight if stage_id==0 else 0.,
            kinematic_objective=args.initial_objective,sigma_us=args.sigma_us,
            filter_energy_floor=args.filter_energy_floor,phase_lag_bins=args.phase_lag_bins,
            coordinate_mask=[1,0,0] if stage_id==0 and args.initial_update=='c_only' else [1,1,1])
        loads=[0.]*len(args.devices);assignment=[[] for _ in loads]
        for source in sorted(stage['sources'],key=lambda s:1.5 if s=='point3' else 1.,reverse=True):
            rank=int(np.argmin(loads));assignment[rank].append(source);loads[rank]+=1.5 if source=='point3' else 1.
        stage['assignment']=assignment
    sources=['reconstruct_kinematic.py','adaptive_rf_loss.py','differential_phase_loss.py','encoded_time.py','time_acquisition.py','v5_observations.py','native_time.py','native_time.cu','multiscale_parameters.py','reconstruct_time.py','rf_depth_weights.py']
    if args.parameterization=='travel_exact':sources+=['travel_time_parameters.py','exact_travel_parameters.py']
    record={'sample':args.sample,'stages':stages,'devices':args.devices,'memory_limit_GiB_per_worker':args.memory_limit_gib,'optimizer':'CPU Adam; independent per-level learning rates',
        'learning_rates':args.learning_rates,'level_multipliers':args.level_multipliers,
        'coordinate_learning_rates':[[args.learning_rates[0]]*3,args.joint_material_rates],
        'codes_per_source_group':args.codes,'seed':args.seed,
        'sampling':args.sampling,'parameterization':args.parameterization,'shots_per_encoded_source':1,
        'target':'self-consistent kinematic initialization, then full RF difference with fixed observed-energy normalization; all retained frequencies',
        'window_definition':{'kind':args.loss_windows,'geometry_names':DEPTH_NAMES,'geometry_weights':DEPTH_WEIGHTS,'interpretation':'homogeneous midpoint-equivalent arrival classes, not true depth localization'},
        'truth_used_by_optimizer':False,
        'selection':'stage0: full unencoded stated kinematic/RF mixture plus weak TV; stage1: full unencoded RF plus weak TV',
        'initialization_scope':'c coordinates may update first while physical Z and tissue alpha stay at RF-derived seed; final stage jointly frees all native fields',
        'differential_phase_hz':args.phase_lag_bins*40e6/8192,
        'status':'initializing_source_banks',
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
    packed=model.initialize(initial)
    # Separate c, Z and alpha parameters at each level so Adam's normalization
    # cannot cancel an intended coordinate-specific learning-rate multiplier.
    parts=[];part_coordinates=[];part_levels=[]
    for level,v in enumerate(packed.split(model.counts)):
        for coordinate,plane in enumerate(v.reshape(3,-1)):
            parts.append(torch.nn.Parameter(plane.clone()))
            part_coordinates.append(coordinate);part_levels.append(level)
    part_sizes=[part.numel() for part in parts]
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
    def rf_only_value(windows,stage):
        return .5*sum(sum(w*r*r for w,r in zip(DEPTH_WEIGHTS,windows[source]))
                      for source in stage['sources'])/len(stage['sources'])
    try:
        for stage_id,stage in enumerate(stages):
            if stage['maxiter']<=0:continue
            coordinate_rates=record['coordinate_learning_rates'][stage_id]
            rates=[coordinate_rates[c]*args.level_multipliers[level]
                   for c,level in zip(part_coordinates,part_levels)]
            optimizer=torch.optim.Adam([{'params':[part],'lr':rate} for part,rate in zip(parts,rates)])
            best={'objective':float('inf')}
            def full_score(step):
                fields=torch.cat(parts);material=model.material(fields)
                value,_,windows,peak,seconds=evaluate(material,stage,'full')
                tv=float(model.regularization(fields,1e-3).detach());total=value+tv
                item={'kind':'full_unencoded','stage':stage_id,'step':step,'data_loss':value,'regularization':tv,'objective':total,
                    'RF_only_data_loss':rf_only_value(windows,stage),'kinematic_weight':stage['kinematic_weight'],
                    'selection_description':record['selection'],'coordinate_mask':stage['coordinate_mask'],
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
                for part,coordinate in zip(parts,part_coordinates):
                    if not stage['coordinate_mask'][coordinate]:part.grad.zero_()
                if not np.isfinite(value) or not all(bool(torch.isfinite(part.grad).all()) for part in parts):
                    raise FloatingPointError('Nonfinite data value or parameter gradient; preserve the run for diagnosis')
                norms=model.coordinate_norms(torch.cat([part.grad for part in parts]));decay=.2+.8*.5*(1+np.cos(np.pi*(step-1)/stage['maxiter']))
                for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate*decay
                optimizer.step()
                item={'kind':'stochastic_'+args.sampling+'_update','stage':stage_id,'step':step,'estimated_data_loss':value,'regularization':float(tv.detach()),
                    'estimated_RF_only_data_loss':rf_only_value(windows,stage),'kinematic_weight':stage['kinematic_weight'],'coordinate_mask':stage['coordinate_mask'],
                    'gradient_norm_by_coordinate':norms,'peak_GiB_per_worker':peak,'worker_seconds':seconds,'elapsed_seconds':time.time()-start,
                    'not_used_for_checkpoint_selection':True}
                with log.open('a') as f:f.write(json.dumps(item)+'\n')
                print(json.dumps(item),flush=True)
                if step%args.validate_every==0 or step==stage['maxiter']:full_score(step)
            for part,value in zip(parts,best['packed'].split(part_sizes)):part.data.copy_(value)
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
