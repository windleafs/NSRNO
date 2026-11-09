"""One bounded pure-UNO training run, original and independent triparameter data."""
from pathlib import Path
from argparse import ArgumentParser
import json,time,random,math,os
import numpy as np
import torch
from nsrno.uno_rf import ChannelUNO
from nsrno.uno_data import UNOData,ROOT,rf_objective,atomic_json,atomic_checkpoint
from nsrno.uno_evaluate import evaluate

def wait_for(path,timeout=10800):
    start=time.monotonic()
    while not path.exists():
        if time.monotonic()-start>timeout: raise TimeoutError(str(path))
        time.sleep(10)

def main():
    p=ArgumentParser(); p.add_argument('--steps',type=int,default=6000); p.add_argument('--warmup-steps',type=int,default=1500)
    p.add_argument('--resume',action='store_true'); args=p.parse_args()
    out=Path('out/v5/uno_round1_triparam'); out.mkdir(parents=True,exist_ok=True)
    wait_for(ROOT/'preparation_complete.json'); wait_for(ROOT/'memory_profile.json')
    if not json.load(open(ROOT/'memory_profile.json'))['passed']: raise RuntimeError('Memory/gradient check did not pass')
    torch.set_num_threads(2); torch.manual_seed(20261105); rng=random.Random(20261105)
    torch.cuda.set_per_process_memory_fraction(14*2**30/torch.cuda.get_device_properties(0).total_memory)
    data=UNOData(); net=ChannelUNO().cuda(); optimizer=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=3e-5)
    settings={'steps':args.steps,'warmup_steps':args.warmup_steps,'loss':'complex signed RF energy +0.1*8-18us +0.1*18-36us, independent alpha-pair response every4 updates',
        'network':net.config,'trainable_real_parameters':sum(p.numel()*(2 if p.is_complex() else 1) for p in net.parameters()),
        'initial_lr':3e-4,'final_lr':3e-5,'batch_angles':1,'seed':20261105,'full_rf_per_angle':True,
        'alpha_pair_normalization_floor_relative_to_source_rms_squared':1e-6,
        'pure_neural_inference':True,'physical_solver_in_training_graph':False,'alpha_unknown_in_inverse':True,
        'checkpoint_selection':'fixed development forward RF error; stage1 and mixed stage2 separated'}
    atomic_json(out/'settings.json',settings)
    history=[]; validations=[]; coverage={}; step0=0; best=float('inf'); stage=0
    if args.resume:
        saved=torch.load(out/'last.pt',map_location='cuda',weights_only=True); net.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        step0=saved['step']; history=saved['history']; validations=saved['validations']; coverage=saved['coverage']; best=saved['best']; stage=saved['stage']
        if 'rng_state' in saved: rng.setstate(saved['rng_state'])
        elif stage==0:
            # Exact replay of the original-stage sampler from the old receipt.
            for old in history:
                a=rng.randrange(11); group=rng.random()
                if group<.7: pool=[r for r in data.groups['train'] if r['kind']=='original' and r['source_mhz']==7.5]
                elif group<.85: pool=[r for r in data.groups['train'] if r['kind']=='original' and r['source_mhz']==3.]
                else: pool=data.groups['train']
                row=rng.choice(pool)
                if old['angle_index']!=a or old['ids']!=[row['id']]: raise RuntimeError('Sampler replay mismatch')
        else: raise RuntimeError('Mixed-stage checkpoint lacks reproducible RNG state')
    def save(step):
        snapshot={'model':net.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'step':step,'settings':settings,'history':history,'validations':validations,'coverage':coverage,'best':best,'stage':stage,
            'rng_state':rng.getstate()}
        atomic_checkpoint(out/'last.pt',snapshot); atomic_json(out/'history.json',history); atomic_json(out/'validations.json',validations)
        atomic_json(out/'coverage.json',{'counts':coverage,'observations':sum(coverage.values()),'unique_material_angle_source':len(coverage)})
        return snapshot
    start=time.perf_counter(); pairs=[]
    for step in range(step0,args.steps):
        if step>=args.warmup_steps and stage==0:
            save(step); atomic_json(out/'waiting_for_pairs.json',{'step':step,'waiting':True})
            wait_for(ROOT/'augmentation_index.json'); data.refresh(); stage=1; best=float('inf')
            atomic_json(out/'waiting_for_pairs.json',{'step':step,'waiting':False})
        if stage==1 and not pairs:
            groups={}
            for r in data.groups['train']:
                if r['kind']=='alpha': groups.setdefault((r['pair_group'],r['source_mhz']),[]).append(r)
            pairs=[sorted(v,key=lambda r:r['pair_sign']) for v in groups.values() if len(v)==2]
            if not pairs: raise RuntimeError('Independent attenuation pairs missing')
        net.train(); optimizer.zero_grad(set_to_none=True); begin=time.perf_counter(); a=rng.randrange(11)
        angle=torch.tensor([-8+1.6*a],device='cuda'); pair_step=bool(stage and step%4==3)
        if pair_step: rows=rng.choice(pairs)
        else:
            group=rng.random()
            if group<.7: pool=[r for r in data.groups['train'] if r['kind']=='original' and r['source_mhz']==7.5]
            elif group<.85: pool=[r for r in data.groups['train'] if r['kind']=='original' and r['source_mhz']==3.]
            else: pool=[r for r in data.groups['train'] if r['kind']!='original'] if stage else data.groups['train']
            rows=[rng.choice(pool)]
        predicted=[]; observed=[]; losses=[]; stats=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for row in rows:
                m=data.material(row)[None].cuda(); source=torch.tensor([row['source_mhz']],device='cuda')
                target=data.target(row,a)[None].cuda()/data.scale(row); value=net(m,angle,source)
                loss,parts=rf_objective(value,target); losses.append(loss); stats.append(parts)
                predicted.append(value); observed.append(target)
                key=row['id']+':a'+str(a); coverage[key]=coverage.get(key,0)+1
            loss=sum(losses)/len(losses)
            if pair_step:
                change,_=rf_objective(predicted[1]-predicted[0],observed[1]-observed[0],floor=1e-6); loss=loss+.2*change
        if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite UNO loss')
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not bool(torch.isfinite(norm)): raise RuntimeError('Nonfinite UNO gradient')
        optimizer.step(); scheduler.step()
        record={'step':step+1,'stage':'mixed' if stage else 'original','loss':float(loss.detach()),
            'batch_rf_relative_rms':math.sqrt(sum(float(v['full']) for v in stats)/len(stats)),
            'batch_late_relative_rms':math.sqrt(sum(float(v['late']) for v in stats)/len(stats)),
            'pair_step':pair_step,'ids':[r['id'] for r in rows],'angle_index':a,'seconds':time.perf_counter()-begin,
            'elapsed_s':time.perf_counter()-start,'lr':optimizer.param_groups[0]['lr'],
            'allocated_peak_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_peak_gib':torch.cuda.max_memory_reserved()/2**30}
        history.append(record)
        if (step+1)%25==0 or step==0: print(json.dumps(record),flush=True)
        del m,source,target,value,loss,losses,predicted,observed,stats
        if (step+1)%500==0 or step==args.steps-1:
            fixed=[data.lookup[k] for k in ['hf_train_200','hf_train_250','hf_train_300','lf_train_200'] if k in data.lookup]
            if stage:
                fixed+=[r for r in data.groups['validation'] if r['kind']=='alpha' and r['source_mhz']==7.5][:4]
            result=evaluate(net,data,fixed); result['step']=step+1; result['stage']=stage; validations.append(result)
            improved=result['relative_rms'][0]<best; best=min(best,result['relative_rms'][0]); snapshot=save(step+1)
            if improved: atomic_checkpoint(out/('best.pt' if stage else 'best_original.pt'),snapshot)
            print(json.dumps({'validation_step':step+1,'stage':stage,'rf':result['relative_rms'],'alpha_response':result['paired_alpha_relative_rms']}),flush=True)
        elif (step+1)%100==0: save(step+1)
    atomic_json(out/'training_complete.json',{'completed':True,'steps':args.steps,'seconds':time.perf_counter()-start,
        'allocated_peak_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_peak_gib':torch.cuda.max_memory_reserved()/2**30})
    # Full development acquisition, never restricted to a few favorable frequencies.
    saved=torch.load(out/'best.pt',map_location='cuda',weights_only=True); net.load_state_dict(saved['model'])
    result=evaluate(net,data,data.groups['validation'],save=out/'predictions'); result['selected_step']=saved['step']
    atomic_json(out/'forward_evaluation.json',result)
    train_rows=[data.lookup[k] for k in ['hf_train_000','hf_train_050','lf_train_000'] if k in data.lookup]
    train_rows += [r for r in data.groups['train'] if r['kind']=='alpha' and r['source_mhz']==7.5][:2]
    atomic_json(out/'training_examples_evaluation.json',evaluate(net,data,train_rows,save=out/'predictions'))
    atomic_json(out/'evaluation_complete.json',{'completed':True,'passed_for_inverse':result['passed_for_inverse']})
    print(json.dumps({'full_development_rf':result['relative_rms'],'alpha_response':result['paired_alpha_relative_rms'],
                      'passed_for_inverse':result['passed_for_inverse']}),flush=True)

if __name__=='__main__': main()
