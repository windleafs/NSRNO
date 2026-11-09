"""One necessary memory/gradient implementation check, without wave simulations."""
from pathlib import Path
import json,time,os,threading,subprocess
import numpy as np
import torch
from nsrno.uno_rf import ChannelUNO
from nsrno.uno_data import atomic_json,rf_objective

def main():
    torch.set_num_threads(2); torch.manual_seed(20261105)
    torch.cuda.set_per_process_memory_fraction(14*2**30/torch.cuda.get_device_properties(0).total_memory)
    out=Path('out/v5/uno_triparam_v1'); out.mkdir(parents=True,exist_ok=True)
    alive=[True]; measurements=[]
    def sample():
        while alive[0]:
            try:
                s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
                for line in s.splitlines():
                    pid,mem=line.split(',')
                    if int(pid)==os.getpid(): measurements.append(float(mem))
            except Exception: pass
            time.sleep(.2)
    thread=threading.Thread(target=sample,daemon=True); thread.start()
    net=ChannelUNO().cuda(); medium=torch.empty(1,3,961,1121,device='cuda'); medium[:,0]=1540; medium[:,1]=1000; medium[:,2]=.5
    target=torch.from_numpy(np.load('out/v5/forward_cache_v1/train_200.iq.npy')[5:6]).cuda()/150
    angle=torch.tensor([0.],device='cuda'); source=torch.tensor([7.5],device='cuda')
    result={'device':torch.cuda.get_device_name(0),'parameters':sum(p.numel()*(2 if p.is_complex() else 1) for p in net.parameters()),
        'config':net.config,'phases':{},'no_physical_solver':True,'no_target_alpha_during_input_gradient':True}
    probe=torch.randn(2,2,192,640,device='cuda')
    angles=torch.tensor([-8.,8.],device='cuda'); sources=torch.tensor([3.,7.5],device='cuda')
    transformed=net.restore_source(probe,angles,sources)
    restored=net.restore_source(transformed,angles,sources,inverse=True)
    result['source_clock_roundtrip_relative_rms']=float((restored-probe).norm()/probe.norm())
    if result['source_clock_roundtrip_relative_rms']>1e-6: raise RuntimeError('Source clock transform is not invertible')
    del probe,transformed,restored
    def begin():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); measurements.clear(); torch.cuda.synchronize(); return time.perf_counter()
    def finish(label,t0):
        torch.cuda.synchronize(); result['phases'][label]={'seconds':time.perf_counter()-t0,
            'allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30,
            'sampled_process_gib':max(measurements,default=0)/1024}
    t0=begin()
    with torch.no_grad(): predicted=net(medium,angle,source)
    finish('one_angle_forward_fp32',t0)
    if predicted.shape!=(1,2,192,640): raise RuntimeError(str(predicted.shape))
    del predicted
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4)
    t0=begin()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        predicted=net(medium,angle,source); loss,_=rf_objective(predicted,target)
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    finish('one_angle_training_bf16',t0); del predicted,loss,opt
    net.requires_grad_(False); net.eval(); medium.requires_grad_(True)
    # Full11-angle VJP with no accumulation of11 decoder graphs.
    t0=begin(); feature=net.encode(medium); leaf=feature.detach().requires_grad_(True)
    for a in np.linspace(-8,8,11):
        prediction=net.observe(leaf,torch.tensor([a],device='cuda',dtype=torch.float32),source)
        loss,_=rf_objective(prediction,target); (loss/11).backward()
    feature.backward(leaf.grad); finish('full11_angle_three_parameter_gradient_fp32',t0)
    result['input_gradient_norms']=[float(medium.grad[:,k].norm()) for k in range(3)]
    if not all(np.isfinite(result['input_gradient_norms'])) or min(result['input_gradient_norms'])<=0:
        raise RuntimeError('Missing/invalid material gradients')
    del feature,leaf,prediction,loss
    # Compare the shared-encoder VJP with ordinary end-to-end autograd on2 views.
    medium.grad=None; feature=net.encode(medium); leaf=feature.detach().requires_grad_(True)
    for a in [-8.,8.]:
        p=net.observe(leaf,torch.tensor([a],device='cuda'),source); p.square().mean().backward()
    feature.backward(leaf.grad); shared=medium.grad.clone(); medium.grad=None
    del feature,leaf,p
    for a in [-8.,8.]:
        p=net(medium,torch.tensor([a],device='cuda'),source); p.square().mean().backward()
    result['shared_encoder_gradient_relative_difference']=float((medium.grad-shared).norm()/shared.norm())
    result['passed']=result['shared_encoder_gradient_relative_difference']<2e-4 and max(v['reserved_gib'] for v in result['phases'].values())<16
    alive[0]=False; thread.join(timeout=2)
    atomic_json(out/'memory_profile.json',result); print(json.dumps(result,indent=2),flush=True)
    if not result['passed']: raise RuntimeError('Memory/shared-gradient implementation check failed')

if __name__=='__main__': main()
