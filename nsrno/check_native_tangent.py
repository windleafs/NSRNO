"""One small synthetic correctness check for the newly added tangent kernels.

This is an implementation test, not a patient identifiability experiment.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from .native_time import Config
from .native_tangent import tangent,pullback
from .check_native_time import reference

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(703)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    device='cuda:0';h,w,nt,pml=37,49,350,4
    c=Config(h,w,nt,27,9,pml,31,50e-6,2.5e-9,20e-9,100e-9,.01,.007)
    material=torch.stack([1540+10*torch.randn(h,w),1000+20*torch.randn(h,w),.5+.1*torch.rand(h,w)]).to(device)
    direction=torch.stack([15*torch.randn(h,w),20*torch.randn(h,w),.08*torch.randn(h,w)]).to(device)
    scols=torch.arange(11,38,device=device);rcols=torch.arange(8,41,4,device=device)
    si=(scols+8*w)[:,None].expand(-1,4).contiguous().int()
    ri=(rcols+8*w)[:,None].expand(-1,4).contiguous().int()
    sw=torch.zeros(27,4,device=device);sw[:,0]=1
    rw=torch.zeros(9,4,device=device);rw[:,0]=1
    t=torch.arange(nt,device=device)*c.dt;tau=t-4/7.5e6
    source=(1e5*torch.exp(-.5*(tau*7.5e6)**2)*torch.sin(2*torch.pi*7.5e6*tau))[:,None]*torch.hann_window(27,periodic=False,device=device)[None]
    source=source.contiguous();started=time.time()
    raw,jvp,cp=tangent(material,direction,source,si,sw,ri,rw,c)
    expected_raw,expected_jvp=torch.autograd.functional.jvp(
        lambda m:reference(m,source,si,sw,ri,rw,c),material,direction)
    weight=torch.randn_like(raw)/raw.norm()
    adjoint=pullback(material,source,si,sw,ri,rw,c,cp,weight)
    lhs=float((jvp.double()*weight.double()).sum())
    rhs=float((direction.double()*adjoint.double()).sum())
    report={'kind':'new_kernel_synthetic_test','patient_wave_solves':0,
        'background_relative_RMS':float((raw-expected_raw).norm()/expected_raw.norm()),
        'JVP_vs_independent_autograd_relative_RMS':float((jvp-expected_jvp).norm()/expected_jvp.norm()),
        'JVP_VJP_dot_relative_error':abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-30),
        'seconds':time.time()-started,'peak_Torch_GiB':torch.cuda.max_memory_allocated()/2**30}
    report['passed']=(report['background_relative_RMS']<2e-5
        and report['JVP_vs_independent_autograd_relative_RMS']<5e-4
        and report['JVP_VJP_dot_relative_error']<5e-4)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['passed']:
        raise RuntimeError('New tangent kernel failed its independent implementation check')

if __name__=='__main__':main()
