"""Gated pure-neural c/rho/tissue-alpha inversion with full-data step acceptance.

True material files are never read. This module refuses an inaccurate surrogate.
"""
from pathlib import Path
from argparse import ArgumentParser
import json,math,time
import numpy as np
import torch
from nsrno.uno_surrogate import UNOSurrogate
from nsrno.uno_data import ROOT,atomic_json,atomic_checkpoint

def main():
    p=ArgumentParser(); p.add_argument('--steps',type=int,default=80); args=p.parse_args()
    evaluation=json.load(open('out/v5/uno_round1_triparam/forward_evaluation.json'))
    if not evaluation['passed_for_inverse']: raise RuntimeError('Pure UNO forward/attenuation-response criterion not passed; inversion is intentionally not run')
    out=Path('out/v5/reconstruction_uno_triparam_train_200'); out.mkdir(parents=True,exist_ok=True)
    if (out/'history.json').exists(): raise FileExistsError('Use a new inverse output; completed records are preserved')
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(14*2**30/torch.cuda.get_device_properties(0).total_memory)
    surrogate=UNOSurrogate(); rows=json.load(open(ROOT/'index.json'))['records']
    observations=[]
    for source in [7.5,3.]:
        selected=[r for r in rows if r['parent_id']=='train_200' and r['source_mhz']==source]
        if selected:
            row=selected[0]; observations.append({'source_mhz':source,'id':row['id'],'iq':torch.from_numpy(np.load(row['iq'])).cuda()})
    if not observations: raise RuntimeError('Independent original RF missing')
    h,w=961,1121; z=torch.arange(h,device='cuda')*5e-5-.003
    support=(z>=.001-1e-7)[:,None].expand(h,w).clone().float()
    support[:24]=0; support[-24:]=0; support[:,:24]=0; support[:,-24:]=0
    edge=torch.from_numpy(np.load(ROOT/'known_edge_alpha.npy')).cuda()[None]
    low=torch.tensor([1400.,850.,.002],device='cuda')[:,None,None]
    high=torch.tensor([1700.,1200.,2.],device='cuda')[:,None,None]
    initial=torch.tensor([1540.,1000.,.5],device='cuda')[:,None,None]
    fraction=(initial-low)/(high-low); q=torch.nn.Parameter(torch.logit(fraction).expand(3,h,w).clone())
    fixed=initial.expand(3,h,w).clone(); fixed[2,z<.001-1e-7]=.002
    def material():
        values=(low+(high-low)*q.sigmoid())*support+fixed*(1-support)
        return torch.cat((values[:2],(values[2]+edge)[None]),0)
    def penalty(m):
        tissue=(m[2]-edge).clamp_min(.002)
        values=torch.stack((torch.log(m[0]/1540)/.06,torch.log(m[1]/1000)/.08,torch.log(tissue/.5)/.5))*support
        dx=values[:,:-1,1:]-values[:,:-1,:-1]; dz=values[:,1:,:-1]-values[:,:-1,:-1]
        return .002*(torch.sqrt(dx.square()+dz.square()+1e-6)-.001).mean()+.0001*values.square().mean()
    settings={'unknowns':['c','rho','tissue_alpha0'],'alpha_power':1.1,'known_numerical_edge_alpha':True,
        'known_true_tissue_alpha':False,'true_material_used_in_optimization':False,'initial_c':1540,'initial_rho':1000,'initial_tissue_alpha0':.5,
        'sources_mhz':[r['source_mhz'] for r in observations],'angles_per_source':11,'rf_samples':2401,
        'update':'full-data neural LBFGS with strong Wolfe; reject steps increasing full RF or regularized objective',
        'parameter_bounds':[[1400,1700],[850,1200],[.002,2]],'steps_budget':args.steps,'selection_includes_initial':True,
        'checkpoint_step':surrogate.checkpoint_step,'physical_solver_in_optimizer':False}
    atomic_json(out/'settings.json',settings); np.save(out/'initial_medium.npy',material().detach().cpu().numpy())
    history=[]; calls=0; start=time.perf_counter(); lr=1.; rejects=0
    def optimizer(): return torch.optim.LBFGS([q],lr=lr,max_iter=1,max_eval=12,history_size=8,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)
    opt=optimizer()
    def evaluate():
        m=material(); result=surrogate.objective(m,observations); result['regularization']=float(penalty(m).detach())
        result['objective']=result['data_loss']+result['regularization']; return result
    current=evaluate(); best=current.copy(); best_q=q.detach().clone(); best_step=0
    atomic_json(out/'initial_objective.json',current)
    for step in range(1,args.steps+1):
        previous=q.detach().clone(); before=current.copy()
        def closure():
            nonlocal calls
            opt.zero_grad(set_to_none=True); m=material(); data_loss,gradient=surrogate.objective_and_gradient(m,observations)
            reg=penalty(m); reg.backward(retain_graph=True); m.backward(gradient); calls+=1
            if not bool(torch.isfinite(q.grad).all()): raise RuntimeError('Nonfinite3-parameter neural gradient')
            return torch.tensor(data_loss+float(reg.detach()),device='cuda')
        opt.step(closure); proposed=evaluate()
        accepted=proposed['objective']<=before['objective']+1e-8 and proposed['relative_rms'][0]<=before['relative_rms'][0]+1e-7
        if accepted: current=proposed; rejects=0
        else:
            with torch.no_grad(): q.copy_(previous)
            current=before; rejects+=1; lr=max(.001,lr*.5); opt=optimizer()
        if current['objective']<best['objective']:
            best=current.copy(); best_q=q.detach().clone(); best_step=step
        record={'step':step,'accepted':accepted,'proposal':proposed,'current':current,'closure_calls':calls,'lr':lr,
            'elapsed_s':time.perf_counter()-start,'allocated_peak_gib':torch.cuda.max_memory_allocated()/2**30}
        history.append(record); print(json.dumps(record),flush=True)
        atomic_json(out/'history.json',history)
        if step%10==0 or rejects>=3:
            np.save(out/f'medium_step_{step:03d}.npy',material().detach().cpu().numpy())
            atomic_checkpoint(out/'last.pt',{'q':q.detach(),'best_q':best_q,'step':step,'best_step':best_step,'history':history,'settings':settings})
        if rejects>=3: break
    with torch.no_grad(): q.copy_(best_q)
    np.save(out/'reconstructed_medium.npy',material().detach().cpu().numpy())
    atomic_json(out/'selected.json',{'step':best_step,'objective':best,'selection':'minimum accepted full-data neural objective including initial; material truth never used'})
    atomic_json(out/'completion.json',{'completed':True,'updates':len(history),'closure_calls':calls,'seconds':time.perf_counter()-start,
        'stop_reason':'three rejected global-data proposals' if rejects>=3 else 'scheduled_budget','best_step':best_step,'truth_used':False,
        'independent_physical_replay_still_required':True})

if __name__=='__main__': main()
