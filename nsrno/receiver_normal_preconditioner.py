"""Exact receiver-block Woodbury preconditioner in field-solver coordinates.

If P is the existing block field inverse and K=C P, use
    M = (lambda I + K* K)^-1.
All192 receivers are included. K* is stored using pressure alone, exploiting
the analytic block-adjoint relation, so three vector fields per receiver are
unnecessary. This changes neither observations nor the field objective.
"""
import time
import torch


class ReceiverNormalPreconditioner:
    @torch.no_grad()
    def __init__(self,operator,pressure_adjoint,response,penalty):
        start=time.time();self.op=operator;g=operator.g;self.g=g;self.penalty=penalty
        n=g.H*g.W;k=len(g.rx_cols);device=g.mask.device
        self.real=torch.empty((k,n),device=device,dtype=torch.float32)
        self.imag=torch.empty_like(self.real)
        for j in range(k):
            rhs=torch.zeros((g.H,g.W),device=device,dtype=torch.complex64)
            rhs[g.face,g.rx_cols[j]]=complex(response).conjugate()
            w=pressure_adjoint(rhs)*g.mask
            self.real[j].copy_(w.real.flatten());self.imag[j].copy_(w.imag.flatten())
        columns_done=time.time();gram=torch.empty((k,k),device=device,dtype=torch.complex64)
        for j in range(k):
            w=torch.complex(self.real[j],self.imag[j]).reshape(g.H,g.W)
            gram[:,j]=self.project(self.combine_adjoint(self.combine(w)))
        gram=(gram+gram.mH)*.5
        spectrum=torch.linalg.eigvalsh(gram.to(torch.complex128))
        # Conservative roundoff margin in the preconditioner only. Lambda in
        # the field objective and scaled multipliers is never changed.
        shift=max(0.,-float(spectrum[0])*1.01)+max(float(spectrum[-1]),1e-30)*5e-6
        shifted=gram.to(torch.complex128)+torch.eye(k,device=device,dtype=torch.complex128)*(penalty+shift)
        self.cholesky=torch.linalg.cholesky(shifted)
        self.metadata={'receivers':k,'stored_fields':'pressure only; velocities reconstructed by exact block-adjoint relation',
            'storage_GiB':(self.real.numel()+self.imag.numel())*4/2**30,
            'columns_seconds':columns_done-start,'gram_seconds':time.time()-columns_done,
            'seconds':time.time()-start,'Gram_min_eigenvalue':float(spectrum[0]),
            'Gram_max_eigenvalue':float(spectrum[-1]),'preconditioner_roundoff_shift':shift,
            'field_penalty_unchanged':penalty,'all192_receivers':True,'material_derivative_probe':False}

    def combine(self,w):
        op=self.op
        return torch.stack([w,-op.Ix.conj()*op.divergence(w,0,True),
            -op.Iz.conj()*op.divergence(w,1,True)])*self.g.mask

    def combine_adjoint(self,z):
        op=self.op;z=z*self.g.mask
        return (z[0]-op.divergence(op.Ix*z[1],0)-op.divergence(op.Iz*z[2],1))*self.g.mask

    def project(self,q):
        qr=q.real.flatten().contiguous();qi=q.imag.flatten().contiguous()
        return torch.complex(torch.mv(self.real,qr)+torch.mv(self.imag,qi),
            torch.mv(self.real,qi)-torch.mv(self.imag,qr))

    def pressure_combination(self,coeff):
        return torch.complex(torch.mv(self.real.T,coeff.real)-torch.mv(self.imag.T,coeff.imag),
            torch.mv(self.real.T,coeff.imag)+torch.mv(self.imag.T,coeff.real)).reshape(self.g.H,self.g.W)

    @torch.no_grad()
    def __call__(self,z):
        projections=self.project(self.combine_adjoint(z)).to(torch.complex128)
        coeff=torch.cholesky_solve(projections[:,None],self.cholesky)[:,0].to(torch.complex64)
        return (z-self.combine(self.pressure_combination(coeff)))/self.penalty
