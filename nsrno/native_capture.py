"""Small-memory frequency observations of the already used native propagator."""
from pathlib import Path
import ctypes
import hashlib
import subprocess
from .native_time import Config, ptr
import numpy as np
import torch

_LIB=None
_BUFFERS={}


def library():
    global _LIB
    if _LIB is None:
        source=Path(__file__).with_suffix('.cu')
        digest=hashlib.sha256(source.read_bytes()+source.with_name('native_time.cu').read_bytes()).hexdigest()[:16]
        folder=Path('out/v5/time_triparam_v1/build');folder.mkdir(parents=True,exist_ok=True)
        target=folder/f'native_capture_{digest}.so'
        if not target.exists():
            subprocess.run(['/usr/local/cuda-12.2/bin/nvcc','-O3','-std=c++17','-shared','-Xcompiler=-fPIC',
                            '-gencode=arch=compute_86,code=sm_86',str(source),'-o',str(target)],check=True)
        lib=ctypes.CDLL(str(target.resolve()))
        lib.native_capture.restype=ctypes.c_int
        lib.native_capture.argtypes=[ctypes.POINTER(Config)]+[ctypes.c_void_p]*9+[ctypes.c_int]*2+[ctypes.c_void_p]*5
        lib.native_error.restype=ctypes.c_char_p;lib.native_error.argtypes=[ctypes.c_int]
        _LIB=lib
    return _LIB


@torch.no_grad()
def capture(material,source,si,sw,ri,rw,config,frequencies,stride=10):
    c=config;device=material.device;shape=(c.nz,c.nx)
    if material.dtype!=torch.float32 or source.dtype!=torch.float32:
        raise ValueError('The native update uses CUDA float32 coefficients and source')
    if stride!=10:
        raise ValueError('Use the existing 40-MHz output clock')
    times=np.arange(0,c.nt,stride,dtype=np.float64)*2.5e-9
    duration=2401/40e6
    phase=np.exp(-2j*np.pi*np.asarray(frequencies)[:,None]*times[None,:])/2401
    phase=torch.from_numpy(phase).to(device)
    # Reuse the same block for short tails (e.g. 270 = 8*32 + 14).
    # A second smaller pool would coexist with the first and breach the cap.
    capacity=max(32,1 << (len(frequencies)-1).bit_length())
    key=(device.index,c.nz,c.nx,c.nt,capacity)
    if key not in _BUFFERS:
        state=torch.empty((7,*shape),device=device);other=torch.empty_like(state)
        _BUFFERS[key]=(torch.empty((capacity,*shape),dtype=torch.complex128,device=device),
                       state,other,torch.empty((2,*shape),device=device),
                       torch.empty((11,*shape),device=device),torch.empty((c.nt,c.nr),device=device))
    fields,state,other,vp,coefficients,raw=_BUFFERS[key]
    lib=library()
    status=lib.native_capture(ctypes.byref(c),*[ptr(t) for t in [material,source,si,sw,ri,rw,raw,phase,fields]],
                              len(frequencies),stride,*[ptr(t) for t in [state,other,vp,coefficients]],
                              ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
    if status:raise RuntimeError(lib.native_error(status).decode())
    # Transfer the observed arrays before freeing the large Fourier block.
    return fields[:len(frequencies)].cpu().numpy(),raw.cpu().numpy()
