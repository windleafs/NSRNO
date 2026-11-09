"""Four-channel residual operators, using the exact native state propagator.

Actual solves allocate only a four-channel time block and one CPU adjoint bank.
"""
from pathlib import Path
import ctypes
import hashlib
import subprocess
import numpy as np
import torch
from .native_time import Config, ptr
from .native_tangent import checkpoints
from .native_source_extension import SourceExtensionWorkspace

_LIB = None


def library():
    global _LIB
    if _LIB is None:
        source = Path(__file__).with_suffix('.cu')
        inputs = [source, source.with_name('native_source_extension.cu'), source.with_name('native_time.cu')]
        fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in inputs)).hexdigest()[:20]
        folder = Path.home()/'nsrno_artifacts/native_multi_equation_build'
        folder.mkdir(parents=True, exist_ok=True)
        target = folder/f'native_multi_equation_{fingerprint}.so'
        if not target.exists():
            subprocess.run(['/usr/local/cuda-12.2/bin/nvcc', '-O3', '-std=c++17', '-shared',
                '-Xcompiler=-fPIC', '-gencode=arch=compute_86,code=sm_86', str(source), '-o', str(target)], check=True)
        lib = ctypes.CDLL(str(target.resolve()))
        c, p = ctypes.POINTER(Config), ctypes.c_void_p
        signatures = {'multi_adjoint': [c]+[p]*15,
            'multi_forward': [c]+[p]*11+[ctypes.c_float]+[p]*11,
            'multi_explicit': [c]+[p]*17}
        for name, signature in signatures.items():
            getattr(lib, name).argtypes = signature
            getattr(lib, name).restype = ctypes.c_int
        lib.native_error.argtypes = [ctypes.c_int]
        lib.native_error.restype = ctypes.c_char_p
        _LIB = lib
    return _LIB


def residual_masks(node, config, device):
    node = np.asarray(node, bool)
    if node.shape != (config.nz, config.nx):
        raise ValueError('Explicit node support required')
    fx, fz = np.zeros_like(node), np.zeros_like(node)
    fx[:, :-1] = node[:, :-1]|node[:, 1:]
    fz[:-1] = node[:-1]|node[1:]
    z, x = np.ogrid[:config.nz, :config.nx]
    pml = (z < config.pml)|(z >= config.nz-config.pml)|(x < config.pml)|(x >= config.nx-config.pml)
    if np.any((node|fx|fz)&pml):
        raise ValueError('The fixed material coat and all residual channels must stay outside PML')
    return [torch.from_numpy(v.astype(np.float32)).to(device) for v in [node, fx, fz]]


class MultiEquationWorkspace(SourceExtensionWorkspace):
    def __init__(self, material, config, node_mask):
        super().__init__(material, config)
        self.node, self.fx, self.fz = residual_masks(node_mask, config, material.device)
        self.statistics = None

    def forcing_tape(self):
        if self.force is None:
            self.force = torch.empty((self.c.chunk, 4, *self.shape), device=self.device)
        return self.force

    @torch.no_grad()
    def adjoint(self, raw_dual, ri, rw, need_checkpoints=True, save_full_for_qualification=False):
        cp = checkpoints(self.c) if need_checkpoints else None
        full = torch.empty((self.c.nt, 4, *self.shape), device=self.device) if save_full_for_qualification else None
        lib = library()
        values = [self.m, raw_dual.contiguous(), ri, rw, self.node, self.fx, self.fz,
            cp, self.adj, self.prev, self.k, self.work, self.norm, full]
        self.check(lib, lib.multi_adjoint(ctypes.byref(self.c), *[ptr(v) for v in values], self.stream))
        return float(self.norm), cp, full

    @torch.no_grad()
    def from_dual(self, source, si, sw, ri, rw, raw_dual, cp_dual, with_statistics=False):
        raw = torch.empty((self.c.nt, self.c.nr), device=self.device)
        if with_statistics and self.statistics is None:
            self.statistics = torch.empty((8, *self.shape), device=self.device, dtype=torch.float64)
        lib = library()
        before = [self.m, source, si, sw, ri, rw, raw_dual, self.node, self.fx, self.fz, cp_dual]
        after = [raw, self.state, self.other, self.vp, self.k, self.adj, self.prev, self.work,
            self.forcing_tape(), self.statistics if with_statistics else None]
        self.check(lib, lib.multi_forward(ctypes.byref(self.c), *[ptr(v) for v in before],
            ctypes.c_float(1.), *[ptr(v) for v in after], self.stream))
        return raw, self.statistics if with_statistics else None

    @torch.no_grad()
    def explicit(self, source, si, sw, ri, rw, force=None, with_statistics=False):
        if force is not None and force.shape != (self.c.nt, 4, *self.shape):
            raise ValueError('Only qualification accepts a full explicit forcing tensor')
        raw = torch.empty((self.c.nt, self.c.nr), device=self.device)
        if with_statistics and self.statistics is None:
            self.statistics = torch.empty((8, *self.shape), device=self.device, dtype=torch.float64)
        lib = library()
        values = [self.m, source, si, sw, ri, rw, self.node, self.fx, self.fz, force,
            raw, self.state, self.other, self.vp, self.k, self.statistics if with_statistics else None]
        self.check(lib, lib.multi_explicit(ctypes.byref(self.c), *[ptr(v) for v in values], self.stream))
        return raw, self.statistics if with_statistics else None
