"""Same weighted native pressure inverse, with cuSPARSE triangular solves.

CPU assembly/LU, physical coefficients, compact source weight, and analytic
adjoint are identical to the qualified CPU version. Only the factor solves
and source-weight multiplication stay on GPU. One nrhs-family cache avoids
retaining both the8-source receiver setup and11-transmission workspaces.
"""
import time
import numpy as np
import torch
from .native_compact_weighted_pressure import NativeCompactPressureCorrection as CPUCorrection
from .gpu_sparse_lu import GpuSuperLU
from .dense_wave import adjoint_operator


def gpu_csr(matrix,device):
    matrix=matrix.tocsr();matrix.sort_indices()
    return torch.sparse_csr_tensor(
        torch.as_tensor(np.asarray(matrix.indptr,np.int32),device=device),
        torch.as_tensor(np.asarray(matrix.indices,np.int32),device=device),
        torch.as_tensor(np.asarray(matrix.data,np.complex64),device=device),
        size=matrix.shape,device=device)


class GPUCorrection(CPUCorrection):
    def __init__(self,system):
        super().__init__(system);start=time.time();self.gpu=None
        self.q_gpu=gpu_csr(self.Q,self.device);self.qh_gpu=gpu_csr(self.QH,self.device)
        self.gpu=GpuSuperLU(self.lu,self.device)
        self.metadata.update(backend='GPU cuSPARSE, identical CPU factors and compact Q',
            GPU_LU=self.gpu.metadata,GPU_setup_seconds=time.time()-start,
            GPU_RHS_weight_storage_GiB=sum(t.numel()*t.element_size() for q in [self.q_gpu,self.qh_gpu]
                for t in [q.crow_indices(),q.col_indices(),q.values()])/2**30,
            maximum_cached_workspace_GiB=0.,cache_policy='one RHS count at a time; all normal/adjoint plans for that count')
    @torch.no_grad()
    def __call__(self,rhs,adjoint=False):
        native=self.g.native(rhs).contiguous();leading=native.shape[:-2]
        # The padded native crop normally copies. Retain the nonmutating
        # pressure-operator contract even if a future geometry is unpadded.
        if native.untyped_storage().data_ptr()==rhs.untyped_storage().data_ptr():native=native.clone()
        value=native.reshape(-1,int(np.prod(self.shape)))
        if adjoint:
            self.gpu.solve(value,adjoint=True,out=value)
            value=torch.sparse.mm(self.qh_gpu,value.T).T.contiguous()
        else:
            value=torch.sparse.mm(self.q_gpu,value.T).T.contiguous()
            self.gpu.solve(value,adjoint=False,out=value)
        workspace=self.gpu.workspace_metadata();self.metadata['last_workspace']=workspace
        self.metadata['maximum_cached_workspace_GiB']=max(self.metadata['maximum_cached_workspace_GiB'],
            sum(x['workspaces_GiB'] for x in workspace.values()))
        return self.g.pad_field(value.reshape(*leading,*self.shape))
    def __del__(self):
        try:
            if self.gpu is not None:self.gpu.close()
        except Exception:pass


def two_level_pressure(system,fine_forward,fine_adjoint):
    correction=GPUCorrection(system)
    def forward(rhs):
        value=fine_forward(rhs)
        return value+correction(rhs-system(value))
    def adjoint(rhs):
        value=correction(rhs,adjoint=True)
        return fine_adjoint(rhs)+value-fine_adjoint(adjoint_operator(system,value))
    return forward,adjoint,correction.metadata
