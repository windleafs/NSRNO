"""All-angle dense-band preconditioner training, with explicit coverage receipts."""
from pathlib import Path
from argparse import ArgumentParser
import json,os,random,time,math
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nsrno.dense_wave import DenseData,channels,ROOT
from nsrno.conditioned_wave import LearnedPreconditioner,differentiable_gmres
from nsrno.conditioned_data import receiver_response

class DenseTrainingModel(nn.Module):
    def __init__(self,initial):
        super().__init__(); self.preconditioner=LearnedPreconditioner()
        self.preconditioner.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True)['model'])
    def forward(self,system,rhs,iterations):
        field=differentiable_gmres(system,self.preconditioner.setup(system),rhs,iterations)
        relative_residual=(system(field)-rhs).abs().square().sum()/rhs.abs().square().sum().clamp_min(1e-20)
        return channels(system.g,field)*receiver_response(system.frequency),relative_residual

def main():
    p=ArgumentParser(); p.add_argument('--cycles',type=int,default=3); p.add_argument('--iterations',type=int,default=32)
    p.add_argument('--out',type=Path,default=Path('out/v5/forward_round8_dense'))
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--initial',type=Path,default=Path('out/v5/forward_round7_conditioned/last.pt'))
    p.add_argument('--lr',type=float,default=.001); p.add_argument('--energy-weighted',action='store_true'); args=p.parse_args()
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); dist.init_process_group('nccl'); torch.set_num_threads(2)
    torch.manual_seed(20261105); random.seed(20261105+rank)
    args.out.mkdir(parents=True,exist_ok=True); data=DenseData(device=f'cuda:{local}',root=args.root)
    training=[r for r in data.rows if r['role']=='train']; pairs=[(b,a) for b in data.spec['bins'] for a in range(11)]
    if len(pairs)%world: raise ValueError('Pair count must divide GPU count for exact coverage')
    steps_per_cycle=len(pairs)//world; steps=args.cycles*steps_per_cycle
    base=DenseTrainingModel(args.initial).to(local)
    model=DDP(base,device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.Adam(base.parameters(),lr=args.lr)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=args.lr/10)
    settings={'cycles':args.cycles,'steps_per_cycle':steps_per_cycle,'steps':steps,'world_size':world,
        'observations':steps*world,'iterations':args.iterations,'loss':'relative complex channel error + relative PDE residual squared',
        'initial_lr':args.lr,'final_lr':args.lr/10,'seed':20261105,'known_true_alpha':True,'training_media':len(training),
        'energy_weighted':args.energy_weighted,'root':str(args.root),'initial_checkpoint':str(args.initial),
        'material_acquisition_cartesian_product':False,'material_sampling':'deterministic rotating training media',
        'frequency_angle_coverage':'every configured frequency and all11 angles once per cycle','spec':data.spec}
    if rank==0: (args.out/'settings.json').write_text(json.dumps(settings,indent=2))
    counts=np.zeros((data.count,11),np.int32); history=[]; start=time.perf_counter()
    for cycle in range(args.cycles):
        order=list(pairs); random.Random(20261105+cycle).shuffle(order)
        for step_in_cycle in range(steps_per_cycle):
            step=cycle*steps_per_cycle+step_in_cycle
            index=step_in_cycle*world+rank; b,a=order[index]
            row=training[(index*17+cycle*13)%len(training)]
            medium,alpha=data.material(row); system,rhs,_,_=data.assemble(medium,alpha,b,[a])
            optimizer.zero_grad(set_to_none=True)
            predicted,residual=model(system,rhs[0],args.iterations)
            target=data.target(row,b,[a])[0]
            channel_loss=(predicted-target).abs().square().sum()/target.abs().square().sum().clamp_min(1e-20)
            weight=1.
            if args.energy_weighted:
                spectrum=data.spectra[row['id']]
                energy=np.sum(abs(spectrum)**2,axis=(0,1),dtype=np.float64)
                weight=float(data.count*energy[b-data.first]/max(energy.sum(),1e-30))
            loss=weight*channel_loss+residual
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite dense training loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(base.parameters(),2.)
            if not bool(torch.isfinite(norm)): raise RuntimeError('Nonfinite dense training gradient')
            optimizer.step(); schedule.step(); counts[b-data.first,a]+=1
            if (step+1)%20==0 or step==0:
                stats=torch.stack((loss.detach(),channel_loss.detach(),residual.detach()))
                dist.all_reduce(stats); stats/=world
                if rank==0:
                    record={'step':step+1,'cycle':cycle+1,'loss':float(stats[0]),'rf_relative_rms':float(stats[1].sqrt()),
                        'pde_residual':float(stats[2].sqrt()),'grad_norm':float(norm),
                        'elapsed_s':time.perf_counter()-start,'lr':optimizer.param_groups[0]['lr']}
                    history.append(record); print(json.dumps(record),flush=True)
            if (step+1)%200==0 or step_in_cycle==steps_per_cycle-1:
                if rank==0:
                    saved={'model':base.preconditioner.state_dict(),'spec':data.spec,'iterations':args.iterations,
                        'step':step+1,'settings':settings,'optimizer':optimizer.state_dict(),'scheduler':schedule.state_dict()}
                    torch.save(saved,args.out/'last.pt'); (args.out/'history.json').write_text(json.dumps(history,indent=2))
        coverage=torch.from_numpy(counts).to(local); dist.all_reduce(coverage)
        if rank==0:
            receipt={'completed_cycles':cycle+1,'frequency_angle_pairs':len(pairs),'min_exposure':int(coverage.min()),
                'max_exposure':int(coverage.max()),'total_observations':int(coverage.sum()),
                'all_11_angles_and_all_frequencies_seen':bool((coverage>0).all()),'counts':coverage.cpu().tolist()}
            (args.out/'coverage.json').write_text(json.dumps(receipt,indent=2))
            torch.save(saved,args.out/f'cycle_{cycle+1}.pt')
            print(json.dumps({k:v for k,v in receipt.items() if k!='counts'}),flush=True)
    if rank==0:
        (args.out/'training_complete.json').write_text(json.dumps({'completed':True,'steps':steps,'seconds':time.perf_counter()-start},indent=2))
    dist.destroy_process_group()

if __name__=='__main__': main()
