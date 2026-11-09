"""cuSPARSE triangular solves for the EXISTING SciPy pressure LU factors.

Only the linear-algebra backend changes. No physical equation or factor is
approximated here. Use the cuSPARSE library AND enum definitions shipped in
the active PyTorch environment, without compiling against system CUDA11.

Pr A Pc=L U, with SciPy's documented permutation convention:
  A^-1 b = Pc U^-1 L^-1 Pr b;
  A^-H b = Pr.H L^-H U^-H Pc.H b.
All workspaces and dense scratch arrays use PyTorch's capped allocator.
The descriptors keep fixed dense pointers between analysis and solve.
"""
from pathlib import Path
import ctypes as C
import re,threading,time
import numpy as np
import torch


def _enums(text):
    text=re.sub(r'/\*.*?\*/','',text,flags=re.S)
    text=re.sub(r'//[^\n]*','',text)
    result={}
    for match in re.finditer(r'\benum(?:\s+\w+)?\s*\{(.*?)\}',text,re.S):
        value=-1
        for item in match.group(1).split(','):
            token=re.fullmatch(r'\s*(\w+)\s*(?:=\s*(\d+)\s*)?',item)
            if token:
                value=int(token[2]) if token[2] is not None else value+1
                result[token[1]]=value
    return result


class _API:
    def __init__(self):
        packages=Path(torch.__file__).parent.parent/'nvidia'
        header=packages/'cusparse/include/cusparse.h'
        types=packages/'cuda_runtime/include/library_types.h'
        source=header.read_text();self.constants=_enums(source+'\n'+types.read_text())
        self.path=packages/'cusparse/lib/libcusparse.so.12'
        self.lib=C.CDLL(str(self.path))
        self.version=sum(int(re.search(r'#define CUSPARSE_VER_'+name+r'\s+(\d+)',source)[1])*scale
            for name,scale in [('MAJOR',1000),('MINOR',100),('PATCH',1)])
        P=C.c_void_p;I=C.c_int;L=C.c_int64;S=C.c_size_t
        definitions={
            'cusparseCreate':[C.POINTER(P)],'cusparseDestroy':[P],
            'cusparseSetStream':[P,P],'cusparseGetVersion':[P,C.POINTER(I)],
            'cusparseCreateCsr':[C.POINTER(P),L,L,L,P,P,P,I,I,I,I],
            'cusparseSpMatSetAttribute':[P,I,P,S],'cusparseDestroySpMat':[P],
            'cusparseCreateDnMat':[C.POINTER(P),L,L,L,P,I,I],'cusparseDestroyDnMat':[P],
            'cusparseSpSM_createDescr':[C.POINTER(P)],'cusparseSpSM_destroyDescr':[P],
            'cusparseSpSM_bufferSize':[P,I,I,P,P,P,P,I,I,P,C.POINTER(S)],
            'cusparseSpSM_analysis':[P,I,I,P,P,P,P,I,I,P,P],
            'cusparseSpSM_solve':[P,I,I,P,P,P,P,I,I,P],
        }
        for name,args in definitions.items():
            function=getattr(self.lib,name);function.argtypes=args;function.restype=I
        self.lib.cusparseGetErrorString.argtypes=[I];self.lib.cusparseGetErrorString.restype=C.c_char_p
    def k(self,name):
        if name not in self.constants:raise RuntimeError('Matching CUDA header lacks enum '+name)
        return self.constants[name]
    def call(self,name,*args):
        status=getattr(self.lib,name)(*args)
        if status:
            message=self.lib.cusparseGetErrorString(status)
            raise RuntimeError(f'{name} returned {status}: {message.decode() if message else "unknown"}')


_api_instance=None
def _api():
    global _api_instance
    if _api_instance is None:_api_instance=_API()
    return _api_instance


class _Triangular:
    def __init__(self,matrix,device,lower,unit,stream):
        self.api=_api();self.handle=C.c_void_p();self.matrix=C.c_void_p();self.storage=[]
        matrix=matrix.tocsr().astype(np.complex64);matrix.sort_indices();self.n=matrix.shape[0]
        for value,dtype in [(matrix.indptr,torch.int32),(matrix.indices,torch.int32),(matrix.data,torch.complex64)]:
            self.storage.append(torch.from_numpy(np.ascontiguousarray(value)).to(device=device,dtype=dtype))
        self.api.call('cusparseCreate',C.byref(self.handle))
        version=C.c_int();self.api.call('cusparseGetVersion',self.handle,C.byref(version))
        if version.value!=self.api.version:raise RuntimeError('cuSPARSE header/library versions differ')
        self.api.call('cusparseSetStream',self.handle,C.c_void_p(stream))
        self.api.call('cusparseCreateCsr',C.byref(self.matrix),self.n,self.n,matrix.nnz,
            *(C.c_void_p(t.data_ptr()) for t in self.storage),self.api.k('CUSPARSE_INDEX_32I'),
            self.api.k('CUSPARSE_INDEX_32I'),self.api.k('CUSPARSE_INDEX_BASE_ZERO'),self.api.k('CUDA_C_32F'))
        fill=C.c_int(self.api.k('CUSPARSE_FILL_MODE_LOWER' if lower else 'CUSPARSE_FILL_MODE_UPPER'))
        diagonal=C.c_int(self.api.k('CUSPARSE_DIAG_TYPE_UNIT' if unit else 'CUSPARSE_DIAG_TYPE_NON_UNIT'))
        self.api.call('cusparseSpMatSetAttribute',self.matrix,self.api.k('CUSPARSE_SPMAT_FILL_MODE'),C.byref(fill),C.sizeof(fill))
        self.api.call('cusparseSpMatSetAttribute',self.matrix,self.api.k('CUSPARSE_SPMAT_DIAG_TYPE'),C.byref(diagonal),C.sizeof(diagonal))
    def close(self):
        if self.matrix.value:self.api.call('cusparseDestroySpMat',self.matrix);self.matrix=C.c_void_p()
        if self.handle.value:self.api.call('cusparseDestroy',self.handle);self.handle=C.c_void_p()
        self.storage=[]
    def __del__(self):
        try:self.close()
        except Exception:pass


class _Plan:
    def __init__(self,tri,scratch,adjoint):
        self.tri=tri;self.api=tri.api;self.B,self.X=scratch
        self.bdesc=C.c_void_p();self.xdesc=C.c_void_p();self.description=C.c_void_p();self.workspace=None
        self.alpha=(C.c_float*2)(1.,0.)
        self.operation=self.api.k('CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE' if adjoint else 'CUSPARSE_OPERATION_NON_TRANSPOSE')
        nrhs,n=self.B.shape
        for descriptor,tensor in [(self.bdesc,self.B),(self.xdesc,self.X)]:
            self.api.call('cusparseCreateDnMat',C.byref(descriptor),n,nrhs,n,C.c_void_p(tensor.data_ptr()),
                self.api.k('CUDA_C_32F'),self.api.k('CUSPARSE_ORDER_COL'))
        self.api.call('cusparseSpSM_createDescr',C.byref(self.description));size=C.c_size_t()
        arguments=self.arguments()
        self.api.call('cusparseSpSM_bufferSize',*arguments,C.byref(size))
        self.bytes=size.value;self.workspace=torch.empty(max(size.value,1),dtype=torch.uint8,device=self.B.device)
        self.api.call('cusparseSpSM_analysis',*arguments,C.c_void_p(self.workspace.data_ptr()))
    def arguments(self):
        return (self.tri.handle,self.operation,self.api.k('CUSPARSE_OPERATION_NON_TRANSPOSE'),C.byref(self.alpha),
            self.tri.matrix,self.bdesc,self.xdesc,self.api.k('CUDA_C_32F'),self.api.k('CUSPARSE_SPSM_ALG_DEFAULT'),self.description)
    def solve(self):self.api.call('cusparseSpSM_solve',*self.arguments())
    def close(self):
        if self.description.value:self.api.call('cusparseSpSM_destroyDescr',self.description);self.description=C.c_void_p()
        for name in ['bdesc','xdesc']:
            descriptor=getattr(self,name)
            if descriptor.value:self.api.call('cusparseDestroyDnMat',descriptor);setattr(self,name,C.c_void_p())
        self.workspace=None


class GpuSuperLU:
    def __init__(self,lu,device):
        start=time.time();self.device=torch.device(device);self.lock=threading.RLock();self.families={}
        self.stream=torch.cuda.current_stream(self.device).cuda_stream;self.closed=False;self.triangles=[]
        self.n=lu.shape[0];self.permutations={}
        with torch.cuda.device(self.device):
            self.triangles=[_Triangular(lu.L,self.device,True,True,self.stream),
                _Triangular(lu.U,self.device,False,False,self.stream)]
            for key,array in [('r',lu.perm_r),('c',lu.perm_c),('ir',np.argsort(lu.perm_r)),('ic',np.argsort(lu.perm_c))]:
                self.permutations[key]=torch.as_tensor(np.asarray(array,np.int64),device=self.device)
        self.metadata={'N':self.n,'L_nnz':lu.L.nnz,'U_nnz':lu.U.nnz,'cuSPARSE_version':_api().version,
            'library':str(_api().path),'transfer_seconds':time.time()-start,
            'matrix_storage_GiB':sum(t.numel()*t.element_size() for tri in self.triangles for t in tri.storage)/2**30,
            'physical_operator_changed':False,'factorization_changed':False}
    def _family(self,nrhs):
        # Receiver-basis8 and transmission11 batches occur in separate phases.
        # Retaining both families would duplicate four large analysis buffers.
        if self.families and nrhs not in self.families:self.clear_plans()
        if nrhs not in self.families:
            self.families[nrhs]={'scratch':(torch.empty((nrhs,self.n),device=self.device,dtype=torch.complex64),
                torch.empty((nrhs,self.n),device=self.device,dtype=torch.complex64)),'plans':{}}
        return self.families[nrhs]
    def _plan(self,family,which,adjoint):
        key=(which,adjoint)
        if key not in family['plans']:
            plan=_Plan.__new__(_Plan)
            try:plan.__init__(self.triangles[which],family['scratch'],adjoint)
            except BaseException:
                try:plan.close()
                except Exception:pass
                raise
            family['plans'][key]=plan
        return family['plans'][key]
    @torch.no_grad()
    def solve(self,rhs,adjoint=False,out=None):
        if self.closed:raise RuntimeError('GPU LU is closed')
        if rhs.ndim!=2 or rhs.shape[1]!=self.n or rhs.dtype!=torch.complex64 or rhs.device!=self.device:
            raise ValueError('Expected GPU complex64 [nrhs,N]')
        if torch.cuda.current_stream(self.device).cuda_stream!=self.stream:raise RuntimeError('One fixed CUDA stream per GPU LU instance')
        if out is not None and (out.shape!=rhs.shape or out.dtype!=rhs.dtype or out.device!=rhs.device or not out.is_contiguous()):
            raise ValueError('out must be a contiguous caller-owned tensor matching rhs; it may alias rhs')
        with self.lock,torch.cuda.device(self.device):
            family=self._family(len(rhs));B,X=family['scratch']
            if rhs.data_ptr() in [B.data_ptr(),X.data_ptr()]:raise ValueError('rhs must not alias internal solve scratch')
            torch.index_select(rhs,1,self.permutations['ic' if adjoint else 'ir'],out=B)
            order=[1,0] if adjoint else [0,1]
            self._plan(family,order[0],adjoint).solve();B.copy_(X)
            self._plan(family,order[1],adjoint).solve()
            permutation=self.permutations['r' if adjoint else 'c']
            if out is not None:
                if out.data_ptr()==X.data_ptr():raise ValueError('out must not alias internal solve output')
                torch.index_select(X,1,permutation,out=out);return out
            return X[:,permutation].contiguous()
    def workspace_metadata(self):
        return {str(k):{'workspaces_GiB':sum(p.bytes for p in f['plans'].values())/2**30,
            'plans':len(f['plans']),'scratch_GiB':sum(t.numel()*t.element_size() for t in f['scratch'])/2**30}
            for k,f in self.families.items()}
    def clear_plans(self):
        with self.lock:
            torch.cuda.current_stream(self.device).synchronize()
            for family in self.families.values():
                for plan in family['plans'].values():plan.close()
            self.families={}
    def close(self):
        if self.closed:return
        self.clear_plans()
        for tri in self.triangles:tri.close()
        self.triangles=[];self.permutations={};self.closed=True
    def __del__(self):
        try:self.close()
        except Exception:pass
