"""Checkpointed physical scalar-source extension of the frozen native scheme."""
from pathlib import Path
import ctypes
import hashlib
import subprocess
import torch
from .native_time import Config, ptr, library as original_library, recompute_tape
from .native_tangent import padded, checkpoints

_LIB = None


def library():
    global _LIB
    if _LIB is None:
        source = Path(__file__).with_suffix('.cu')
        original = source.with_name('native_time.cu')
        fingerprint = hashlib.sha256(source.read_bytes()+original.read_bytes()).hexdigest()[:20]
        folder = Path.home()/'nsrno_artifacts/native_source_extension_build'
        folder.mkdir(parents=True, exist_ok=True)
        target = folder/f'native_source_extension_{fingerprint}.so'
        if not target.exists():
            subprocess.run(['/usr/local/cuda-12.2/bin/nvcc', '-O3', '-std=c++17', '-shared',
                '-Xcompiler=-fPIC', '-gencode=arch=compute_86,code=sm_86', str(source),
                '-o', str(target)], check=True)
        lib = ctypes.CDLL(str(target.resolve()))
        c, p, f = ctypes.POINTER(Config), ctypes.c_void_p, ctypes.c_float
        signatures = {
            'extension_adjoint': [c]+[p]*12,
            'extension_forward': [c]+[p]*9+[f]+[p]*11,
            'extension_backward': [c]+[p]*10+[f]+[p]*11,
        }
        for name, signature in signatures.items():
            getattr(lib, name).argtypes = signature
            getattr(lib, name).restype = ctypes.c_int
        lib.native_error.restype = ctypes.c_char_p
        lib.native_error.argtypes = [ctypes.c_int]
        _LIB = lib
    return _LIB


class SourceExtensionWorkspace:
    """No full space-time bank. Checkpoints are CPU, scalar forcing is one block."""
    def __init__(self, material, config):
        if material.dtype != torch.float32 or not material.is_cuda:
            raise ValueError('The unchanged native float32 material is required')
        self.material, self.c = material, config
        self.device, self.shape = material.device, (config.nz, config.nx)
        self.m = padded(material, config.pml)
        self.state = torch.empty((7, *self.shape), device=self.device)
        self.other = torch.empty_like(self.state)
        self.adj = torch.empty_like(self.state)
        self.prev = torch.empty_like(self.state)
        self.vp = torch.empty((2, *self.shape), device=self.device)
        self.k = torch.empty((11, *self.shape), device=self.device)
        self.work = torch.empty((10, *self.shape), device=self.device)
        self.force = None
        self.norm = torch.empty((), device=self.device, dtype=torch.float64)
        self.stream = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)

    def check(self, lib, status):
        if status:
            raise RuntimeError(lib.native_error(status).decode())

    @torch.no_grad()
    def physical_forward(self, source, si, sw, ri, rw, need_checkpoints):
        raw = torch.empty((self.c.nt, self.c.nr), device=self.device)
        cp = checkpoints(self.c) if need_checkpoints else None
        lib = original_library()
        values = [self.m, source, si, sw, ri, rw, raw, cp,
                  self.state, self.other, self.vp, self.k]
        self.check(lib, lib.native_forward(ctypes.byref(self.c), *[ptr(t) for t in values], self.stream))
        return raw, cp

    @torch.no_grad()
    def adjoint(self, raw_dual, ri, rw, weight, need_checkpoints):
        cp = checkpoints(self.c) if need_checkpoints else None
        lib = library()
        values = [self.m, raw_dual.contiguous(), ri, rw, weight, cp,
                  self.adj, self.prev, self.k, self.work, self.norm]
        self.check(lib, lib.extension_adjoint(ctypes.byref(self.c), *[ptr(t) for t in values], self.stream))
        return float(self.norm), cp

    def forcing_tape(self):
        if self.force is None:
            self.force = torch.empty((self.c.chunk, *self.shape), device=self.device)
        return self.force

    @torch.no_grad()
    def augmented_forward(self, source, si, sw, ri, rw, raw_dual, weight, cp_dual, scale):
        raw = torch.empty((self.c.nt, self.c.nr), device=self.device)
        cp = checkpoints(self.c)
        lib = library()
        before = [self.m, source, si, sw, ri, rw, raw_dual, weight, cp_dual]
        after = [raw, cp, self.state, self.other, self.vp, self.k,
                 self.adj, self.prev, self.work, self.forcing_tape()]
        self.check(lib, lib.extension_forward(ctypes.byref(self.c), *[ptr(t) for t in before],
            ctypes.c_float(scale), *[ptr(t) for t in after], self.stream))
        return raw, cp

    @torch.no_grad()
    def augmented_pullback(self, source, si, sw, ri, rw, raw_dual, weight,
                           cp_dual, cp_forward, scale):
        grad = torch.empty_like(self.material, dtype=torch.float64)
        tape = recompute_tape(self.c, self.device)
        lib = library()
        before = [self.m, source, si, sw, ri, rw, raw_dual, weight, cp_dual, cp_forward]
        after = [self.state, self.other, self.vp, self.k, self.adj, self.prev,
                 self.work, self.forcing_tape(), tape, grad]
        self.check(lib, lib.extension_backward(ctypes.byref(self.c), *[ptr(t) for t in before],
            ctypes.c_float(scale), *[ptr(t) for t in after], self.stream))
        with torch.enable_grad():
            original = self.material.detach().requires_grad_(True)
            result, = torch.autograd.grad(padded(original, self.c.pml), original,
                                          grad_outputs=grad.float())
        return result
