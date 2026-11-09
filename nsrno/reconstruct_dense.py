"""Second joint reconstruction with every dense frequency and all11 angles.

The neural preconditioner is frozen. Forward/adjoint fields meet actual PDE
residual tolerances; material truth is never loaded by this program.
"""
from pathlib import Path
from argparse import ArgumentParser
from datetime import timedelta
import os,json,math,time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nsrno.dense_wave import DenseData,implicit_channels,solve_system,channels
from nsrno.conditioned_wave import LearnedPreconditioner
from nsrno.reconstruct_material import build_medium,regularization,write_json

class MaterialInverse(nn.Module):
    def __init__(self,data,alpha,network,initial=None):
        super().__init__(); g=data.geometry
        self.q=nn.Parameter(torch.zeros(2,g.h,g.w,device=data.device) if initial is None else initial)
        self.data=data; self.network=network; self.register_buffer('alpha',alpha)
        z=torch.arange(g.h,device=data.device)*g.dx+g.spec['z0_m']
        self.register_buffer('support',(z>=.001-1e-7)[:,None].expand(g.h,g.w).float())
    def material(self): return build_medium(self.q,self.support)
    def forward(self,bin_index):
        return implicit_channels(self.material(),self.alpha,self.data,bin_index,list(range(11)),self.network,1e-5)

def save_state(out,step,base,history,coverage,optimizer,scheduler):
    np.save(out/f'medium_step_{step:03d}.npy',base.material().detach().cpu().numpy())
    np.save(out/'reconstructed_medium.npy',base.material().detach().cpu().numpy())
    write_json(out/'history.json',history); write_json(out/'coverage.json',coverage)
    torch.save({'q':base.q.detach(),'step':step,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
        'history':history,'coverage':coverage},out/'resume.pt')

def main():
    p=ArgumentParser(); p.add_argument('--cycles',type=int,default=2)
    p.add_argument('--out',type=Path,default=Path('out/v5/reconstruction_round2_dense_train_200'))
    p.add_argument('--root',type=Path,default=Path('out/v5/conditioned_dense_full_v1'))
    p.add_argument('--checkpoint',type=Path,default=Path('out/v5/forward_round8_dense_full/last.pt'))
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); dist.init_process_group('nccl',timeout=timedelta(minutes=60)); torch.set_num_threads(2)
    torch.manual_seed(20261105); data=DenseData(device=f'cuda:{local}',reference_limit=12,root=args.root,fast_fft=True)
    forward=json.load(open('out/v5/forward_round8_dense_full/forward_summary.json'))
    if not forward['passed_for_reconstruction']: raise RuntimeError('Forward accuracy gate not passed')
    row=next(r for r in data.rows if r['id']=='train_200')
    # Read only RF and known alpha; material truth remains inaccessible here.
    alpha=torch.from_numpy(np.load(row['alpha'])).to(local)
    network=LearnedPreconditioner().to(local); saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    network.load_state_dict(saved['model']); network.eval(); network.requires_grad_(False)
    base=MaterialInverse(data,alpha,network).to(local); model=DDP(base,device_ids=[local],broadcast_buffers=False)
    frequencies=data.spec['bins']; per_cycle=len(frequencies)//world
    if len(frequencies)%world: raise ValueError('Dense frequencies must divide worker count')
    steps=args.cycles*per_cycle; args.out.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.Adam([base.q],lr=.01); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=.002)
    lower=torch.tensor([math.log(1400/1540)/.06,math.log(850/1000)/.08],device=local)[:,None,None]
    upper=torch.tensor([math.log(1700/1540)/.06,math.log(1200/1000)/.08],device=local)[:,None,None]
    settings={'id':row['id'],'case':row['case'],'cycles':args.cycles,'steps':steps,'world_size':world,
        'frequencies':frequencies,'angle_indices':list(range(11)),'receivers':192,'known_true_alpha':True,'truth_used_in_optimization':False,
        'initial_c':1540.,'initial_rho':1000.,'fixed_gel_mm':1.,'native_grid':data.spec,'bounds_c':[1400,1700],'bounds_rho':[850,1200],
        'frequency_order':'ascending complete band each cycle;3 frequencies per synchronous material update',
        'loss':'unbiased full-band complex waveform energy error + weak edge-preserving regularization',
        'forward_tolerance':1e-5,'adjoint_tolerance':1e-5,'material_gradient':'implicit adjoint of converged physical equation',
        'initial_lr':.01,'final_lr':.002,'selection':'complete-pass candidate with smallest independent time-domain full RF misfit, no material truth','root':str(args.root),'checkpoint':str(args.checkpoint),
        'tv_weight':.002,'l2_weight':.0001,'fft_shape':[data.geometry.H,data.geometry.W]}
    start_step=0; previous_elapsed=0.; history=[]; counts=np.zeros(len(frequencies),np.int32)
    if args.resume:
        state=torch.load(args.out/'resume.pt',map_location=f'cuda:{local}',weights_only=True)
        with torch.no_grad(): base.q.copy_(state['q'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start_step=state['step']; history=state.get('history',json.load(open(args.out/'history.json')))
        history=[r for r in history if r['step']<=start_step]
        previous_elapsed=history[-1]['elapsed_s'] if history else 0.
        global_counts=state.get('coverage',json.load(open(args.out/'coverage.json')))['counts']
        counts=np.array([value if i%world==rank else 0 for i,value in enumerate(global_counts)],np.int32)
        if rank==0:
            old_settings=json.load(open(args.out/'settings.json'))
            old_settings['fft_shape_after_resume']=[data.geometry.H,data.geometry.W]; old_settings['fft_implementation_resume_step']=start_step
            write_json(args.out/'settings.json',old_settings)
    elif (args.out/'history.json').exists(): raise FileExistsError('Use --resume for an existing inverse run')
    if rank==0 and not args.resume:
        write_json(args.out/'settings.json',settings)
        np.save(args.out/'initial_medium.npy',base.material().detach().cpu().numpy()); np.save(args.out/'known_alpha.npy',alpha.cpu().numpy())
    started=time.perf_counter()
    for cycle in range(args.cycles):
        for j in range(per_cycle):
            step=cycle*per_cycle+j
            if step<start_step: continue
            b=frequencies[j*world+rank]
            target=data.target(row,b,list(range(11)))
            # Frequency sampling is uniform; scaling makes the expected loss
            # equal the complete-band waveform energy error.
            total_energy=float(data.full_energies[row['id']].sum())
            optimizer.zero_grad(set_to_none=True); begin=time.perf_counter()
            prediction=model(b)
            residual=prediction-target
            data_loss=data.count*residual.abs().square().sum()/total_energy
            penalty,tv,l2=regularization(base.q,base.support); loss=data_loss+penalty
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite dense inverse loss')
            loss.backward(); norm=base.q.grad.norm()
            if not bool(torch.isfinite(norm)): raise RuntimeError('Nonfinite implicit material gradient')
            # DDP has already averaged the material gradient over the three
            # distinct frequencies, including all11 angles at each frequency.
            optimizer.step(); scheduler.step()
            with torch.no_grad(): base.q.copy_(torch.maximum(torch.minimum(base.q,upper),lower)*base.support)
            counts[b-data.first]+=1
            stats=torch.tensor([float(data_loss.detach()),float(penalty.detach()),float(norm),time.perf_counter()-begin],device=local,dtype=torch.float64)
            dist.all_reduce(stats); stats/=world
            if rank==0:
                record={'step':step+1,'cycle':cycle+1,'frequency_bins':[frequencies[j*world+r] for r in range(world)],
                    'sampled_full_band_loss':float(stats[0]),'sampled_relative_rms':math.sqrt(float(stats[0])),
                    'regularization':float(stats[1]),'gradient_norm':float(stats[2]),'mean_worker_seconds':float(stats[3]),
                    'elapsed_s':previous_elapsed+time.perf_counter()-started,'lr':optimizer.param_groups[0]['lr']}
                history.append(record); print(json.dumps(record),flush=True)
            if (step+1)%14==0 or j==per_cycle-1:
                covered=torch.from_numpy(counts).to(local); dist.all_reduce(covered)
                if rank==0:
                    receipt={'completed_updates':step+1,'fully_completed_frequency_passes':cycle+(j==per_cycle-1),
                        'min_frequency_exposure':int(covered.min()),'max_frequency_exposure':int(covered.max()),
                        'total_frequency_batches':int(covered.sum()),'angles_per_frequency':11,
                        'all_frequencies_seen':bool((covered>0).all()),'counts':covered.cpu().tolist()}
                    save_state(args.out,step+1,base,history,receipt,optimizer,scheduler)
    if rank==0:
        write_json(args.out/'completion.json',{'completed':True,'updates':steps,'full_frequency_passes':args.cycles,
            'all_294_frequencies_and_all_11_angles_used':True,'seconds':previous_elapsed+time.perf_counter()-started,'truth_used_in_optimization':False})
    dist.destroy_process_group()

if __name__=='__main__': main()
