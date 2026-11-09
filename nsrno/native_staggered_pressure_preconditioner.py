"""Two-level linear pressure preconditioner for the unchanged fine equation.

The native field/material grid remains50um. A100um pressure correction is
factorized on CPU, while the existing learned Fourier inverse handles fine
components. This is a solver preconditioner, not a coarser forward model.
"""
import time
import numpy as np
import torch
from torch.nn import functional as F
from scipy import sparse
from scipy.sparse.linalg import splu
from .dense_wave import adjoint_operator

_DERIVATIVES={}


def derivative_1d(n,dx):
    diagonals=[];offsets=[]
    for j,a in enumerate((9/8,-1/24)):
        for offset,value in [(j+1,a),(-j,-a)]:
            offsets.append(offset);diagonals.append(np.full(n-abs(offset),value/dx,np.float32))
    return sparse.diags(diagonals,offsets,shape=(n,n),format='csr',dtype=np.float32)


def derivatives(h,w,dx):
    key=(h,w,dx)
    if key not in _DERIVATIVES:
        _DERIVATIVES[key]=(sparse.kron(sparse.eye(h,format='csr',dtype=np.float32),derivative_1d(w,dx),format='csr'),
                           sparse.kron(derivative_1d(h,dx),sparse.eye(w,format='csr',dtype=np.float32),format='csr'))
    return _DERIVATIVES[key]


class CoarsePressureCorrection:
    def __init__(self,system):
        start=time.time();g=system.g;self.g=g
        if (g.h-1)%2 or (g.w-1)%2:raise ValueError('Require odd native dimensions for an exact bilinear adjoint')
        self.shape=(g.h,g.w)
        self.device=g.mask.device;h,w=self.shape
        Dx,Dz=derivatives(h,w,g.dx)
        rho=g.native(system.rho).cpu().numpy();a=1000/rho
        mass=g.native(system.mass).cpu().numpy().astype(np.complex64)
        ax=2*a*np.concatenate([a[:,1:],a[:,-1:]],axis=1)/(a+np.concatenate([a[:,1:],a[:,-1:]],axis=1))
        az=2*a*np.concatenate([a[1:],a[-1:]],axis=0)/(a+np.concatenate([a[1:],a[-1:]],axis=0))
        arx=g.arx[0,g.px:g.px+g.w].cpu().numpy()
        arz=g.arz[g.pz:g.pz+g.h,0].cpu().numpy()
        avx=g.avx[0,g.px:g.px+g.w].cpu().numpy()
        avz=g.avz[g.pz:g.pz+g.h,0].cpu().numpy()
        D=system.D
        def axis(plus,face,ar,av,which):
            sv=(D-1)/(D-av);gain=1+(1-av)/D
            m=(ar!=1).astype(np.float32)
            left=m/(D-ar);right=1+(1-m)/(D-ar)
            def expand(v):return np.broadcast_to(v[None,:] if which==0 else v[:,None],(h,w)).ravel()
            flux=(face*(sv[None,:] if which==0 else sv[:,None])).ravel().astype(np.complex64)
            minus=-plus.T.tocsr()
            regular=minus@plus.multiply(flux[:,None])
            provisional=minus@plus.multiply((flux*expand(gain))[:,None])
            return regular.multiply(expand(left)[:,None])+provisional.multiply(expand(right)[:,None])
        spatial=axis(Dx,ax,arx,avx,0)+axis(Dz,az,arz,avz,1)
        scalar=np.complex64((D-1)/D/system.k0**2)
        matrix=(sparse.diags(mass.ravel(),format='csr')+scalar*spatial).astype(np.complex64).tocsc()
        matrix.eliminate_zeros();assembled=time.time()
        self.lu=splu(matrix,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
        factored=time.time()
        kernel=torch.tensor([[.25,.5,.25],[.5,1.,.5],[.25,.5,.25]],device=self.device)
        self.kernel=kernel[None,None].repeat(2,1,1,1)
        self.metadata={'native_shape':[g.h,g.w],'coarse_shape':list(self.shape),
            'native_dx_m':g.dx,'correction_dx_m':g.dx,'matrix_nnz':matrix.nnz,'preconditioner_staggered_order':4,
            'physical_staggered_order_unchanged':8,'PML_update_algebra_unchanged':True,
            'LU_L_nnz':self.lu.L.nnz,'LU_U_nnz':self.lu.U.nnz,
            'assembly_seconds':assembled-start,'factor_seconds':factored-assembled,
            'factor_on':'CPU','forward_operator_changed':False,'material_grid_changed':False}
        del matrix,spatial

    def __call__(self,rhs,adjoint=False):
        native=self.g.native(rhs).contiguous().cpu().numpy();leading=native.shape[:-2]
        value=native.reshape(-1,int(np.prod(self.shape))).T.copy(order='F')
        solution=self.lu.solve(value,trans='H' if adjoint else 'N').T.reshape(*leading,*self.shape)
        return self.g.pad_field(torch.from_numpy(np.ascontiguousarray(solution)).to(self.device))


def two_level_pressure(system,fine_forward,fine_adjoint):
    correction=CoarsePressureCorrection(system)
    def forward(rhs):
        value=fine_forward(rhs)
        return value+correction(rhs-system(value))
    def adjoint(rhs):
        value=correction(rhs,adjoint=True)
        return fine_adjoint(rhs)+value-fine_adjoint(adjoint_operator(system,value))
    return forward,adjoint,correction.metadata
