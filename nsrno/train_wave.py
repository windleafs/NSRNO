"""First bounded training of a native-grid wave-equation residual network."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json, os, time, shutil
import numpy as np
import torch
from scipy.fft import fft,fftshift
from .wave_operator import WaveResidualOperator
from .dataset import read_json,write_json

def save(path,value):
    temporary=path.with_suffix('.partial.pt')
    with temporary.open('wb') as f:
        torch.save(value,f); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)

def sample(row,cache,angle,bin_index):
    medium=torch.from_numpy(np.array(np.load(cache/row['medium'],mmap_mode='r')))[None].cuda()
    iq=np.load(cache/row['iq'],mmap_mode='r')[angle]
    spectrum=fftshift(fft(iq[0]+1j*iq[1],axis=-1,norm='forward'),axes=-1)/np.sqrt(2)
    return medium,torch.from_numpy(np.array(spectrum[:,bin_index-40])).cuda()

@torch.no_grad()
def evaluate(model,rows,cache,learned=True):
    model.eval(); error=energy=0.; residual=[]; traces=[]
    bins=[270,300,330,360,390,420]
    for i,row in enumerate(rows):
        angle=(0,5,10)[i%3]; bin_index=bins[i%len(bins)]
        medium,target=sample(row,cache,angle,bin_index)
        predicted,loss,trace=model(medium,bin_index*40e6/2401,angle,learned,True)
        error+=float((predicted-target).abs().square().sum()); energy+=float(target.abs().square().sum())
        residual.append(float(loss.sqrt())); traces.append(trace)
    return dict(records=len(rows),relative_channel_spectrum_rms=(error/energy)**.5,
                pde_relative_residual_mean=float(np.mean(residual)),pde_residual_traces=traces,
                note='One specified frequency/angle per record; not a full-band or late-time RF score.')

def main():
    p=ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=240); p.add_argument('--iterations',type=int,default=8)
    p.add_argument('--preconditioner',choices=('shifted','shifted_tapered','free_space','free_space_tapered'),default='shifted')
    p.add_argument('--attenuation',choices=('imaginary','relaxation'),default='imaginary')
    p.add_argument('--cache',type=Path,default=Path('out/v5/forward_cache_v1'))
    args=p.parse_args(); torch.set_num_threads(2); torch.manual_seed(20261105)
    rng=np.random.default_rng(20261105); args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'run.json').exists(): raise FileExistsError('Use a fresh experiment directory')
    index=read_json(args.cache/'index.json'); rows=index['records']
    if index['acquisition']['source_f0_hz']!=7.5e6:
        raise ValueError('This first-round trainer samples the original V5 high-frequency band. Adapt frequency sampling and relaxation coefficients before training the 3 MHz cache.')
    train=[r for r in rows if r['role']=='train']; val=[r for r in rows if r['role']=='validation']
    if {r['case'] for r in train}&{r['case'] for r in val}: raise ValueError('Case leakage')
    train_probe=[train[i] for i in np.linspace(0,len(train)-1,6,dtype=int)]
    val_probe=[val[i] for i in np.linspace(0,len(val)-1,12,dtype=int)]
    spec={**index['grid'],'angles_deg':index['acquisition']['angles_deg'],'source_tref_s':rows[0]['source_tref_s']}
    if args.attenuation=='relaxation':
        spec.update(attenuation='relaxation',taus_s=[2e-8,1e-7],eta_per_unit_alpha=[.011490947273079713,.005150122286585719])
    settings=dict(spec=spec,steps=args.steps,iterations=args.iterations,preconditioner=args.preconditioner,train_records=len(train),
        validation_probe=[r['id'] for r in val_probe],training_probe=[r['id'] for r in train_probe],
        cache_index_sha256=sha256((args.cache/'index.json').read_bytes()).hexdigest(),
        selection='channel_spectrum_relative_rms_on_fixed_validation_probe',
        physics_loss_weight=.1,frequency_bins=[252,432],seed=20261105)
    write_json(args.out/'run.json',settings)
    for filename in ('wave_operator.py','train_wave.py'):
        shutil.copy2(Path(__file__).with_name(filename),args.out/filename)
    model=WaveResidualOperator(spec,args.iterations,args.preconditioner).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    initial=evaluate(model,val_probe,args.cache); initial_train=evaluate(model,train_probe,args.cache)
    history=[dict(step=0,validation=initial,training=initial_train)]
    print(json.dumps(history[-1]),flush=True); write_json(args.out/'history.json',history)
    best=initial['relative_channel_spectrum_rms']
    save(args.out/'best.pt',dict(spec=spec,iterations=args.iterations,preconditioner=args.preconditioner,model=model.state_dict(),step=0,validation=initial))
    timer=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train(); row=train[int(rng.integers(len(train)))]; angle=int(rng.integers(11)); bin_index=int(rng.integers(252,433))
        medium,target=sample(row,args.cache,angle,bin_index)
        optimizer.zero_grad(set_to_none=True)
        predicted,physics,_=model(medium,bin_index*40e6/2401,angle)
        data_loss=(predicted-target).abs().square().mean()/target.abs().square().mean().clamp_min(1e-8)
        loss=data_loss+.1*physics
        if not torch.isfinite(loss): raise RuntimeError('Non-finite wave loss')
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm): raise RuntimeError('Non-finite gradient')
        optimizer.step()
        record=dict(step=step,id=row['id'],angle=angle,bin=bin_index,
                    data_loss=float(data_loss.detach()),physics_loss=float(physics.detach()),
                    seconds=time.perf_counter()-timer)
        if step==1 or step%20==0:
            record.update(validation=evaluate(model,val_probe,args.cache),training=evaluate(model,train_probe,args.cache))
            if record['validation']['relative_channel_spectrum_rms']<best:
                best=record['validation']['relative_channel_spectrum_rms']
                save(args.out/'best.pt',dict(spec=spec,iterations=args.iterations,preconditioner=args.preconditioner,model=model.state_dict(),step=step,validation=record['validation']))
            save(args.out/'last.pt',dict(spec=spec,iterations=args.iterations,preconditioner=args.preconditioner,model=model.state_dict(),step=step,optimizer=optimizer.state_dict()))
        history.append(record); write_json(args.out/'history.json',history); print(json.dumps(record),flush=True)
    selected=torch.load(args.out/'best.pt',map_location='cpu',weights_only=True); model.load_state_dict(selected['model'])
    result=dict(selected_step=selected['step'],initial=initial,selected=evaluate(model,val_probe,args.cache),
                fixed_preconditioner_same_response=evaluate(model,val_probe,args.cache,False),
                elapsed_seconds=time.perf_counter()-timer,parameters=sum(p.numel() for p in model.parameters()))
    write_json(args.out/'completion.json',result); print(json.dumps(result),flush=True)

if __name__=='__main__': main()
