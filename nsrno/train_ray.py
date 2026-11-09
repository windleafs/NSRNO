"""Fit a late-RF correction using existing labels and cached ray features."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from zlib import crc32
import json
import math
import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .dataset import read_json,write_json as atomic_write_json
from .forward_model import make_forward_model,load_forward_model
from .train_forward import save_checkpoint as atomic_save_checkpoint,VAL_ANGLES
from .rf_representation import IQ_SAMPLES,RF_SAMPLES,RF_FS,WINDOW_EDGES_US,time_windows,decode

DURATION_US=1e6*RF_SAMPLES/RF_FS
LOSS_WEIGHTS=(.05,.05,.45,.45)


def durable(path):
    with path.open('rb') as handle: os.fsync(handle.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def write_json(path,value):
    atomic_write_json(path,value); durable(path)


def save_checkpoint(path,value):
    atomic_save_checkpoint(path,value); durable(path)


def late_score(metrics):
    return float(np.mean(metrics['model']['window_relative_rf_rms'][2:]))


class RayDataset(Dataset):
    def __init__(self,rows,args,epoch=0,fixed=None):
        self.rows,self.args,self.epoch,self.fixed=rows,args,epoch,fixed
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]; args=self.args
        angles=self.fixed
        if angles is None:
            start=(crc32(row['id'].encode())+self.epoch*args.angles_per_record)%11
            angles=(start+np.arange(args.angles_per_record))%11
        angles=np.asarray(angles,dtype=np.int64)
        def selected(path):
            return torch.from_numpy(np.load(path,mmap_mode='c',allow_pickle=False)[angles])
        background=torch.from_numpy(np.load(args.shared/'background'/(row['id']+'.npy'),allow_pickle=False))
        return (background,torch.from_numpy(angles),
                selected(args.features/'features'/(row['id']+'.npy')),
                selected(args.shared/'baseline'/(row['id']+'.npy')),
                selected(args.cache/row['iq']))


def loader(rows,args,epoch=0,fixed=None):
    return DataLoader(RayDataset(rows,args,epoch,fixed),batch_size=args.batch if fixed is None else args.eval_batch,
                      shuffle=fixed is None,num_workers=args.workers,pin_memory=True,
                      generator=torch.Generator().manual_seed(args.seed+epoch))


def loss_fn(prediction,target,scales):
    squared=(prediction-target).square().sum(2)
    terms=[]
    for window,scale,weight in zip(time_windows(IQ_SAMPLES,DURATION_US),scales,LOSS_WEIGHTS):
        error=squared[...,window].mean((-2,-1))
        energy=target[...,window].square().sum(2).mean((-2,-1))
        terms.append(weight*(error/energy.clamp_min(scale**2)).mean())
    return torch.stack(terms).sum()


@torch.no_grad()
def evaluate(model,rows,args,angles,grouped=False):
    model.eval()
    energy=torch.zeros((4,len(angles),4),device='cuda',dtype=torch.float64)
    windows=time_windows(RF_SAMPLES,DURATION_US)
    template=decode(model.template)[list(angles)]
    groups={v:torch.zeros_like(energy) for v in {r['variant'] for r in rows}} if grouped else {}
    position=0
    for batch in loader(rows,args,fixed=angles):
        background,indices,features,baseline,target=(v.cuda(non_blocking=True) for v in batch)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            predicted=model.predict_cached(background,indices,features,baseline)
        truth,prediction,frozen=decode(target),decode(predicted),decode(baseline)
        membership={v:[i for i,r in enumerate(rows[position:position+len(background)]) if r['variant']==v] for v in groups}
        for j,window in enumerate(windows):
            ref=truth[...,window]
            contributions=torch.stack((ref.square().sum((2,3),dtype=torch.float64),
                 (prediction[...,window]-ref).square().sum((2,3),dtype=torch.float64),
                 (frozen[...,window]-ref).square().sum((2,3),dtype=torch.float64),
                 (template[None,...,window]-ref).square().sum((2,3),dtype=torch.float64)))
            energy[:,:,j]+=contributions.sum(1)
            for variant,selection in membership.items():
                if selection: groups[variant][:,:,j]+=contributions[:,selection].sum(1)
        position+=len(background)
    target,error,base,mean=energy.cpu().numpy()
    def summary(values,truth=target):
        w=np.sqrt(values.sum(0)/truth.sum(0))
        return {'relative_rf_rms':float(np.sqrt(values.sum()/truth.sum())),
                'window_relative_rf_rms':w.tolist(),
                'late_window_mean_rf_rms':float(w[2:].mean()),
                'angle_relative_rf_rms':np.sqrt(values.sum(1)/truth.sum(1)).tolist()}
    result={'records':len(rows),'angles_deg':[model.spec['angles_deg'][i] for i in angles],
            'time_window_edges_us':WINDOW_EDGES_US,'model':summary(error),
            'frozen_baseline':summary(base),'train_mean':summary(mean),
            'baseline_error_energy_reduction':float(1-error.sum()/base.sum()),
            'window_baseline_error_energy_reduction':(1-error.sum(0)/base.sum(0)).tolist()}
    if grouped:
        result['by_variant']={}
        for variant,value in groups.items():
            truth,error,base,mean=value.cpu().numpy()
            result['by_variant'][variant]={'records':sum(r['variant']==variant for r in rows),
                'model':summary(error,truth),'frozen_baseline':summary(base,truth),
                'window_baseline_error_energy_reduction':(1-error.sum(0)/base.sum(0)).tolist()}
    return result


def main():
    p=ArgumentParser()
    p.add_argument('--delay',choices=('material','reference'),required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('out/v5/forward_cache_v1'))
    p.add_argument('--shared',type=Path,default=Path('out/v5/forward_ray_shared_fp32_v1'))
    p.add_argument('--warm-start',type=Path,default=Path('out/v5/forward_round3_slowness/best.pt'))
    p.add_argument('--normalization',type=Path,default=Path('out/v5/forward_ray_material_v1/completion.json'))
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--eval-batch',type=int,default=4)
    p.add_argument('--angles-per-record',type=int,default=2)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261105)
    p.add_argument('--eval-every',type=int,default=5)
    p.add_argument('--train-eval-records',type=int,default=64)
    p.add_argument('--min-epochs',type=int,default=40)
    p.add_argument('--plateau-patience',type=int,default=25)
    p.add_argument('--plateau-min-delta',type=float,default=1e-4)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high'); torch.backends.cudnn.benchmark=True
    args.out.mkdir(exist_ok=True,parents=True)
    if (args.out/'last.pt').exists() and not args.resume:
        raise FileExistsError('Use --resume or a new output directory')
    index=read_json(args.cache/'index.json')
    feature_settings=read_json(args.features/'settings.json')
    feature_result=read_json(args.features/'completion.json')
    normalization=read_json(args.normalization)
    old=torch.load(args.warm_start,map_location='cpu',weights_only=True)
    warm_hash=sha256(args.warm_start.read_bytes()).hexdigest()
    index_hash=sha256((args.cache/'index.json').read_bytes()).hexdigest()
    if feature_settings['delay']!=args.delay or feature_settings['warm_start_sha256']!=warm_hash or feature_settings['index_sha256']!=index_hash:
        raise ValueError('Feature cache and training configuration disagree')
    if feature_result['records']!=len(index['records']): raise ValueError('Incomplete feature cache')
    train=[r for r in index['records'] if r['role']=='train']
    val=[r for r in index['records'] if r['role']=='validation']
    if {r['case'] for r in train}&{r['case'] for r in val}: raise ValueError('Case leakage')
    base_train=[r for r in train if r['variant'] in ('oa_breast_original','high_echo_circles_v1')]
    probe=[base_train[i] for i in np.linspace(0,len(base_train)-1,min(len(base_train),args.train_eval_records),dtype=int)]
    spec={**feature_settings['spec'],'architecture':'ray_arrival_v1','base_spec':old['spec'],
          'ray_delay':args.delay,'ray_feature_scales':normalization['feature_scales']}
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('out','resume')}
    settings.update(cache_index_sha256=index_hash,warm_start_sha256=warm_hash,
                    feature_settings_sha256=sha256((args.features/'settings.json').read_bytes()).hexdigest(),
                    normalization_sha256=sha256(args.normalization.read_bytes()).hexdigest(),
                    selection='mean_relative_rf_rms_18_us_to_end',loss_weights=LOSS_WEIGHTS)
    model=make_forward_model(spec); model.base.load_state_dict(old['model']); model.cuda()
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=2e-5)
    if args.resume:
        last=torch.load(args.out/'last.pt',map_location='cpu',weights_only=True)
        # JSON tuples are only used in saved run.json; checkpoint values retain tuples.
        if last['spec']!=spec or last['settings']!=settings: raise ValueError('Resume settings differ')
        model.load_state_dict(last['model']); optimizer.load_state_dict(last['optimizer']); scheduler.load_state_dict(last['scheduler'])
        history,best,stop=last['history'],last['best'],last['stop']; start=last['epoch']+1
    else:
        initial_val=evaluate(model,val,args,VAL_ANGLES); initial_train=evaluate(model,probe,args,VAL_ANGLES)
        best=late_score(initial_val); stop={'best_training_late_rms':late_score(initial_train),'last_improvement_epoch':0}
        history=[{'epoch':0,'validation':initial_val,'training_monitor':initial_train,'seconds':0.}]; start=1
        save_checkpoint(args.out/'best.pt',{'spec':spec,'model':model.state_dict(),'epoch':0,'validation':initial_val})
        write_json(args.out/'baseline_monitor.json',history[0])
        source=args.out/'source'; source.mkdir(exist_ok=True)
        for name in ('forward_model.py','scatter_model.py','ray_propagation.py','ray_model.py','train_ray.py','rf_representation.py'):
            shutil.copy2(Path(__file__).with_name(name),source/name)
    write_json(args.out/'run.json',{**settings,'spec':spec,'train_records':len(train),'validation_records':len(val),
               'trainable_parameters':sum(p.numel() for p in parameters),'frozen_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad),
               'training_monitor_ids':[r['id'] for r in probe]})
    reason='epoch_budget'
    for epoch in range(start,args.epochs+1):
        timer=perf_counter(); total=0.; count=0; model.train()
        for batch in loader(train,args,epoch-1):
            background,indices,features,baseline,target=(v.cuda(non_blocking=True) for v in batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                predicted=model.predict_cached(background,indices,features,baseline)
            loss=loss_fn(predicted,target,spec['output_window_scales'])
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,2.); optimizer.step()
            total+=float(loss.detach())*len(background); count+=len(background)
        row={'epoch':epoch,'train_loss':total/count,'lr':optimizer.param_groups[0]['lr']}
        if epoch==1 or epoch%args.eval_every==0 or epoch==args.epochs:
            monitored=evaluate(model,probe,args,VAL_ANGLES); metrics=evaluate(model,val,args,VAL_ANGLES)
            row.update(training_monitor=monitored,validation=metrics,
                       training_late_rms=late_score(monitored),validation_late_rms=late_score(metrics))
            if not all(math.isfinite(row[k]) for k in ('training_late_rms','validation_late_rms')):
                raise RuntimeError('Non-finite metric')
            if row['validation_late_rms']<best:
                best=row['validation_late_rms']
                save_checkpoint(args.out/'best.pt',{'spec':spec,'model':model.state_dict(),'epoch':epoch,'validation':metrics})
            if row['training_late_rms']<stop['best_training_late_rms']-args.plateau_min_delta:
                stop={'best_training_late_rms':row['training_late_rms'],'last_improvement_epoch':epoch}
        scheduler.step(); row['seconds']=perf_counter()-timer; history.append(row)
        save_checkpoint(args.out/'last.pt',{'spec':spec,'settings':settings,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),'epoch':epoch,'history':history,'best':best,'stop':stop})
        write_json(args.out/'history.json',history); print(json.dumps(row),flush=True)
        if ('training_monitor' in row and epoch>=args.min_epochs and epoch-stop['last_improvement_epoch']>=args.plateau_patience):
            reason='training_late_window_plateau'; break
    angles=tuple(range(11))
    last_train=evaluate(model,base_train,args,angles); last_train['checkpoint_epoch']=history[-1]['epoch']
    write_json(args.out/'training_last_all_angles.json',last_train)
    del model
    model=load_forward_model(args.out/'best.pt','cuda')
    selected=torch.load(args.out/'best.pt',map_location='cpu',weights_only=True)['epoch']
    for role,rows in (('validation',val),('training_best',base_train)):
        metrics=evaluate(model,rows,args,angles,grouped=True); metrics['best_epoch']=selected
        write_json(args.out/(role+'_all_angles.json'),metrics)
    write_json(args.out/'completion.json',{'completed_epochs':history[-1]['epoch'],'best_epoch':selected,
               'stopping_reason':reason,'best_validation_late_rms':best,'stop_state':stop})
    print(json.dumps(read_json(args.out/'completion.json')),flush=True)


if __name__=='__main__': main()
