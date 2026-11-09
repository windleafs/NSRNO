"""Exact discrete JVP and existing adjoint with one-shot CPU checkpoints."""
from pathlib import Path
import ctypes
import hashlib
import subprocess
import torch
from torch.nn import functional as F
from .native_time import Config,ptr,library as adjoint_library,recompute_tape

_LIB=None

def library():
    global _LIB
    if _LIB is None:
        source=Path(__file__).with_suffix('.cu')
        original=source.with_name('native_time.cu')
        fingerprint=hashlib.sha256(source.read_bytes()+original.read_bytes()).hexdigest()[:16]
        folder=Path('out/v5/time_triparam_v1/build')
        folder.mkdir(parents=True,exist_ok=True)
        target=folder/f'native_tangent_{fingerprint}.so'
        if not target.exists():
            subprocess.run(['/usr/local/cuda-12.2/bin/nvcc','-O3','-std=c++17','-shared',
                '-Xcompiler=-fPIC','-gencode=arch=compute_86,code=sm_86',str(source),'-o',str(target)],check=True)
        _LIB=ctypes.CDLL(str(target.resolve()))
        _LIB.native_tangent.restype=ctypes.c_int
        _LIB.native_tangent.argtypes=[ctypes.POINTER(Config)]+[ctypes.c_void_p]*19
        _LIB.native_error.restype=ctypes.c_char_p
        _LIB.native_error.argtypes=[ctypes.c_int]
    return _LIB

def padded(material,p):
    return F.pad(material[None,:,p:-p,p:-p],(p,p,p,p),mode='replicate')[0].contiguous()

def checkpoints(config):
    shape=((config.nt+config.chunk-1)//config.chunk,7,config.nz,config.nx)
    try:
        return torch.empty(shape,dtype=torch.float32,pin_memory=True)
    except RuntimeError:
        return torch.empty(shape,dtype=torch.float32)

@torch.no_grad()
def tangent(material,direction,source,si,sw,ri,rw,config,save_checkpoints=True):
    """Return background RF, J*direction, and background checkpoints."""
    c=config;device=material.device;shape=(c.nz,c.nx)
    if material.dtype!=torch.float32 or direction.dtype!=torch.float32:
        raise ValueError('Native material and direction must be float32')
    m=padded(material,c.pml);dm=padded(direction,c.pml)
    state=torch.empty((7,*shape),device=device);other=torch.empty_like(state)
    dstate=torch.empty_like(state);dother=torch.empty_like(state)
    vp=torch.empty((2,*shape),device=device);dvp=torch.empty_like(vp)
    k=torch.empty((11,*shape),device=device);dk=torch.empty((7,*shape),device=device)
    raw=torch.empty((c.nt,c.nr),device=device);draw=torch.empty_like(raw)
    cp=checkpoints(c) if save_checkpoints else None
    lib=library()
    status=lib.native_tangent(ctypes.byref(c),*[ptr(t) for t in
        [m,dm,source,si,sw,ri,rw,raw,draw,cp,state,other,vp,dstate,dother,dvp,k,dk]],
        ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
    if status:
        raise RuntimeError(lib.native_error(status).decode())
    return raw,draw,cp

@torch.no_grad()
def pullback(material,source,si,sw,ri,rw,config,cp,residual):
    """Return the exact adjoint in the unpadded material coordinates."""
    c=config;device=material.device;shape=(c.nz,c.nx)
    m=padded(material,c.pml)
    state=torch.empty((7,*shape),device=device);other=torch.empty_like(state)
    vp=torch.empty((2,*shape),device=device);coeff=torch.empty((11,*shape),device=device)
    tape=recompute_tape(c,device)
    adj=torch.empty_like(state);prev=torch.empty_like(state);work=torch.empty((10,*shape),device=device)
    grad=torch.empty_like(material,dtype=torch.float64)
    lib=adjoint_library();residual=residual.contiguous()
    status=lib.native_backward(ctypes.byref(c),*[ptr(t) for t in
        [m,source,si,sw,ri,rw,residual,cp,state,other,vp,coeff,tape,adj,prev,work,grad]],
        ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
    if status:
        raise RuntimeError(lib.native_error(status).decode())
    # Differentiate only replicate padding; never retain a propagation graph.
    with torch.enable_grad():
        original=material.detach().requires_grad_(True)
        result=torch.autograd.grad(padded(original,c.pml),original,grad_outputs=grad.float())[0]
    return result
