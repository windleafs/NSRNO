"""Nine-point compact coarse pressure inverse, used only as a preconditioner.

For uniform media the coarse equation is the fourth-order compact Helmholtz
formula L9/k0² + mass*(I+h²*L5/12). The fine physical operator is unchanged.
Variable coefficients and PML are kept in the two directional coarse fluxes.
"""
import time
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu
from .coarse_pressure_preconditioner import CoarsePressureCorrection
from .dense_wave import adjoint_operator


class NativeCompactPressureCorrection(CoarsePressureCorrection):
    def __init__(self,system):
        start=time.time();g=system.g;self.g=g;self.device=g.mask.device
        if (g.h-1)%2 or (g.w-1)%2:raise ValueError('Require odd native dimensions')
        self.shape=h,w=g.h,g.w;dx=g.dx
        def plus(n):return sparse.diags([-np.ones(n),np.ones(n-1)],[0,1],shape=(n,n),format='csr',dtype=np.float32)/dx
        Dx=sparse.kron(sparse.eye(h,format='csr'),plus(w),format='csr')
        Dz=sparse.kron(plus(h),sparse.eye(w,format='csr'),format='csr')
        rho=g.native(system.rho).cpu().numpy();a=1000/rho
        mass=g.native(system.mass).cpu().numpy().astype(np.complex64)
        ax=2*a*np.concatenate([a[:,1:],a[:,-1:]],axis=1)/(a+np.concatenate([a[:,1:],a[:,-1:]],axis=1))
        az=2*a*np.concatenate([a[1:],a[-1:]],axis=0)/(a+np.concatenate([a[1:],a[-1:]],axis=0))
        D=system.D
        def axis(derivative,face,ar,av,which):
            def expand(v):return np.broadcast_to(v[None,:] if which==0 else v[:,None],(h,w)).ravel()
            m=(ar!=1).astype(np.float32);left=m/(D-ar);right=1+(1-m)/(D-ar)
            flux=(face.ravel()*expand((D-1)/(D-av))).astype(np.complex64)
            minus=-derivative.T.tocsr()
            first=minus@derivative.multiply(flux[:,None])
            second=minus@derivative.multiply((flux*expand(1+(1-av)/D))[:,None])
            return ((D-1)/D*(first.multiply(expand(left)[:,None])+second.multiply(expand(right)[:,None]))).astype(np.complex64)
        Sx=axis(Dx,ax,g.arx[0,g.px:g.px+g.w].cpu().numpy(),g.avx[0,g.px:g.px+g.w].cpu().numpy(),0)
        Sz=axis(Dz,az,g.arz[g.pz:g.pz+g.h,0].cpu().numpy(),g.avz[g.pz:g.pz+g.h,0].cpu().numpy(),1)
        inverse_a=sparse.diags(1/a.ravel(),format='csr')
        five=Sx+Sz
        nine=five+(dx*dx/12)*(Sx@inverse_a@Sz+Sz@inverse_a@Sx)
        matrix=(sparse.diags(mass.ravel(),format='csr')+nine/system.k0**2+
            five.multiply((mass/a).ravel()[:,None])*(dx*dx/12)).astype(np.complex64).tocsc()
        matrix.eliminate_zeros();assembled=time.time()
        self.lu=splu(matrix,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True});factored=time.time()
        self.metadata={'native_shape':[g.h,g.w],'coarse_shape':list(self.shape),'native_dx_m':g.dx,
            'correction_dx_m':dx,'format':'nine-point compact, fourth-order constant-medium interior',
            'matrix_nnz':matrix.nnz,'LU_L_nnz':self.lu.L.nnz,'LU_U_nnz':self.lu.U.nnz,
            'assembly_seconds':assembled-start,'factor_seconds':factored-assembled,'factor_on':'CPU',
            'forward_operator_changed':False,'material_grid_changed':False,
            'diagonal_pivot_threshold':0.,'symmetric_ordering_mode':True,'restriction':'identity: native50um'}

    def __call__(self,rhs,adjoint=False):
        value=self.g.native(rhs).contiguous().cpu().numpy().ravel()
        solution=self.lu.solve(value,trans='H' if adjoint else 'N').reshape(self.shape)
        return self.g.pad_field(torch.from_numpy(np.ascontiguousarray(solution)).to(self.device))


def two_level_pressure(system,fine_forward,fine_adjoint):
    correction=NativeCompactPressureCorrection(system)
    def forward(rhs):
        value=fine_forward(rhs)
        return value+correction(rhs-system(value))
    def adjoint(rhs):
        value=correction(rhs,adjoint=True)
        return fine_adjoint(rhs)+value-fine_adjoint(adjoint_operator(system,value))
    return forward,adjoint,correction.metadata
