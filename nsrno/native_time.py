"""Checkpointed discrete physical time propagator with a plain CUDA ABI."""
from pathlib import Path
import ctypes,hashlib,os,subprocess
# Large pinned checkpoints fragment host memory. Prevent direct THP compaction
# in this process; no machine-wide VM settings are changed.
if os.name=='posix':
    if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise OSError('Cannot disable process transparent huge pages')
import torch
from torch.nn import functional as F

class Config(ctypes.Structure):
    _fields_=[(n,ctypes.c_int) for n in ['nz','nx','nt','ns','nr','pml','chunk']]+[(n,ctypes.c_float) for n in ['dx','dt','tau1','tau2','e1','e2']]

_LIB=None
_TAPES={}

def recompute_tape(config,device):
    # Shots execute sequentially on one CUDA stream. Keep the large allocation
    # intact so later response matrices cannot split its cached allocator block.
    stream=torch.cuda.current_stream(device).cuda_stream
    key=(device.index,stream,config.chunk,config.nz,config.nx)
    if key not in _TAPES:
        _TAPES[key]=torch.empty((config.chunk,7,config.nz,config.nx),device=device)
    return _TAPES[key]
def library():
    global _LIB
    if _LIB is None:
        source=Path(__file__).with_suffix('.cu'); fingerprint=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        folder=Path('out/v5/time_triparam_v1/build'); folder.mkdir(parents=True,exist_ok=True)
        target=folder/f'native_time_{fingerprint}.so'
        if not target.exists():
            subprocess.run(['/usr/local/cuda-12.2/bin/nvcc','-O3','-std=c++17','-shared','-Xcompiler=-fPIC',
                '-gencode=arch=compute_86,code=sm_86',str(source),'-o',str(target)],check=True)
        lib=ctypes.CDLL(str(target.resolve()))
        for name,count in [('native_forward',14),('native_backward',19)]:
            fn=getattr(lib,name); fn.restype=ctypes.c_int
            fn.argtypes=[ctypes.POINTER(Config)]+[ctypes.c_void_p]*(count-1)
        lib.native_error.restype=ctypes.c_char_p; lib.native_error.argtypes=[ctypes.c_int]; _LIB=lib
    return _LIB

def ptr(t): return ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p(0)

class NativeWave(torch.autograd.Function):
    @staticmethod
    def forward(ctx,material,source,si,sw,ri,rw,config):
        if material.dtype!=torch.float32 or not material.is_cuda: raise ValueError('CUDA float32 material required')
        c=config; device=material.device; shape=(c.nz,c.nx)
        state=torch.empty((7,*shape),device=device); other=torch.empty_like(state)
        vp=torch.empty((2,*shape),device=device); coeff=torch.empty((11,*shape),device=device)
        raw=torch.empty((c.nt,c.nr),device=device)
        need_gradient=ctx.needs_input_grad[0]
        cp=None
        if need_gradient:
            cp_shape=((c.nt+c.chunk-1)//c.chunk,7,*shape)
            try: cp=torch.empty(cp_shape,dtype=torch.float32,pin_memory=True)
            except RuntimeError: cp=torch.empty(cp_shape,dtype=torch.float32)
        lib=library(); status=lib.native_forward(ctypes.byref(c),*[ptr(t) for t in [material,source,si,sw,ri,rw,raw,cp,state,other,vp,coeff]],
            ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
        if status: raise RuntimeError(lib.native_error(status).decode())
        if need_gradient: ctx.save_for_backward(material,source,si,sw,ri,rw,cp); ctx.config=c
        return raw

    @staticmethod
    def backward(ctx,gradient):
        material,source,si,sw,ri,rw,cp=ctx.saved_tensors; c=ctx.config; device=material.device; shape=(c.nz,c.nx)
        state=torch.empty((7,*shape),device=device); other=torch.empty_like(state)
        vp=torch.empty((2,*shape),device=device); coeff=torch.empty((11,*shape),device=device)
        tape=recompute_tape(c,device)
        adj=torch.empty_like(state); prev=torch.empty_like(state); work=torch.empty((10,*shape),device=device)
        grad=torch.empty_like(material,dtype=torch.float64); gradient=gradient.contiguous()
        lib=library(); status=lib.native_backward(ctypes.byref(c),*[ptr(t) for t in [material,source,si,sw,ri,rw,gradient,cp,state,other,vp,coeff,tape,adj,prev,work,grad]],
            ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
        if status: raise RuntimeError(lib.native_error(status).decode())
        return grad.float(),None,None,None,None,None,None

def native_wave(material,source,source_index,source_weights,receiver_index,receiver_weights,config):
    p=config.pml
    coefficients=F.pad(material[None,:,p:-p,p:-p],(p,p,p,p),mode='replicate')[0].contiguous()
    return NativeWave.apply(coefficients,source,source_index,source_weights,receiver_index,receiver_weights,config)
