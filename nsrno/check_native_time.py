"""Check the newly implemented discrete transpose against independent autograd."""
import json,time
import numpy as np
import torch
from torch.nn import functional as F
from nsrno.native_time import Config,native_wave

def derivative(a,axis,positive,dx):
    h,w=a.shape; q=F.pad(a,(4,4,4,4)); out=torch.zeros_like(a)
    for j,v in enumerate((1225/1024,-245/3072,49/5120,-5/7168)):
        first=j+1 if positive else j; second=-j if positive else -j-1
        if axis==0: out=out+v*(q[4:4+h,4+first:4+first+w]-q[4:4+h,4+second:4+second+w])
        else: out=out+v*(q[4+first:4+first+h,4:4+w]-q[4+second:4+second+h,4:4+w])
    return out/dx

def reference(material,source,si,sw,ri,rw,c):
    p=c.pml; m=F.pad(material[None,:,p:-p,p:-p],(p,p,p,p),mode='replicate')[0]
    speed,rho,alpha=m; e1=alpha*speed/1540*c.e1; e2=alpha*speed/1540*c.e2
    A=(1+.5*e1+.5*e2)**2; b1=c.dt*e1/(c.tau1*A); b2=c.dt*e2/(c.tau2*A)
    bx=2/(rho+F.pad(rho[:,1:][None,None],(0,1,0,0),mode='replicate')[0,0])
    bz=2/(rho+F.pad(rho[1:][None,None],(0,0,0,1),mode='replicate')[0,0])
    z,x=torch.meshgrid(torch.arange(c.nz,device=m.device),torch.arange(c.nx,device=m.device),indexing='ij')
    mx=(x<p)|(x>=c.nx-p); mz=(z<p)|(z>=c.nz-p)
    def damping(q,n,shift):
        v=torch.where(q<p,1-(q+shift)/p,torch.where(q>=n-p,1-(n-1-q-shift)/p,0.))
        return 1-c.dt*(2*1540/c.dx)*v**4
    avx,avz=damping(x,c.nx,.5),damping(z,c.nz,.5)
    arx,arz=damping(x,c.nx,0),damping(z,c.nz,0)
    pressure,vx,vz,qx,qz,s1,s2=[torch.zeros_like(speed) for _ in range(7)]; records=[]
    for t in range(c.nt):
        vpx=vx-c.dt*bx*derivative(pressure,0,True,c.dx)
        vpz=vz-c.dt*bz*derivative(pressure,1,True,c.dx)
        qpx=qx-c.dt*rho*derivative(vpx,0,False,c.dx)
        qpz=qz-c.dt*rho*derivative(vpz,1,False,c.dx)
        pn=speed**2*A*(qpx+qpz)-s1-s2
        sn1=(1-c.dt/c.tau1)*s1+b1*(pressure+s1+s2)
        sn2=(1-c.dt/c.tau2)*s2+b2*(pressure+s1+s2)
        vnx=vpx+(avx-1)*vx; vnz=vpz+(avz-1)*vz
        qnx=torch.where(mx,arx*qx-c.dt*rho*derivative(vnx,0,False,c.dx),qpx)
        qnz=torch.where(mz,arz*qz-c.dt*rho*derivative(vnz,1,False,c.dx),qpz)
        injection=torch.zeros(c.nz*c.nx,device=m.device).scatter_add(0,si.flatten().long(),(source[t,:,None]*sw*c.dt/(1540*c.dx)).flatten()).view(c.nz,c.nx)
        records.append((pn.flatten()[ri.long()]*rw).sum(-1))
        pressure,vx,vz,qx,qz,s1,s2=pn,vnx,vnz,qnx+injection,qnz+injection,sn1,sn2
    return torch.stack(records)

def main():
    torch.set_num_threads(1); torch.manual_seed(702); device='cuda'; h,w,nt,p=37,49,350,4
    c=Config(h,w,nt,27,9,p,31,50e-6,2.5e-9,20e-9,100e-9,.01,.007)
    material=torch.stack([1540+10*torch.randn(h,w),1000+20*torch.randn(h,w),.5+.1*torch.rand(h,w)]).to(device).requires_grad_()
    scols=torch.arange(11,38,device=device); rcols=torch.arange(8,41,4,device=device)
    si=(scols+8*w)[:,None].expand(-1,4).contiguous().int(); ri=(rcols+8*w)[:,None].expand(-1,4).contiguous().int()
    sw=torch.zeros(27,4,device=device);sw[:,0]=1; rw=torch.zeros(9,4,device=device);rw[:,0]=1
    t=torch.arange(nt,device=device)*c.dt; tau=t-4/7.5e6
    source=(1e5*torch.exp(-.5*(tau*7.5e6)**2)*torch.sin(2*torch.pi*7.5e6*tau))[:,None]*torch.hann_window(27,periodic=False,device=device)[None]
    source=source.contiguous(); start=time.time()
    actual=native_wave(material,source,si,sw,ri,rw,c); other=reference(material,source,si,sw,ri,rw,c)
    weight=torch.randn_like(actual)/actual.detach().square().mean().sqrt()/actual.numel()
    ga=torch.autograd.grad((actual*weight).sum(),material)[0]
    gr=torch.autograd.grad((other*weight).sum(),material)[0]
    report={'forward_relative_rms':float((actual-other).norm()/other.norm()),'gradient_relative_rms':[float((ga[j]-gr[j]).norm()/gr[j].norm()) for j in range(3)],'seconds':time.time()-start,'memory_GiB':torch.cuda.max_memory_allocated()/2**30}
    print(json.dumps(report),flush=True)
    assert report['forward_relative_rms']<2e-5,report
    assert max(report['gradient_relative_rms'])<5e-4,report

if __name__=='__main__': main()
