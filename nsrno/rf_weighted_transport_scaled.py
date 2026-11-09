"""Exact variable scaling of weighted RF vector transport; CPU only.

phi=sqrt(b)*psi, B=diag(1/a) D diag(sqrt(b)). The dual quadratic
becomes uniform, while the weighted objective and every original RF sample
remain identical to rf_weighted_vector_transport.py.
"""
import time
import numpy as np
from scipy.linalg.lapack import dpttrf,dpttrs
if __package__:
    from .rf_vector_transport import lift,hilbert
else:
    from rf_vector_transport import lift,hilbert


def transport(residual,h,lam,node_weight=None,edge_cost=None,tolerance=1e-6,
              maximum=8000,rho=None,warm=None,check_every=25):
    started=time.perf_counter();g=np.asarray(residual,np.float64)
    if g.ndim!=2 or len(g)<2 or not np.isfinite(g).all() or min(h,lam)<=0:
        raise ValueError('Finite time-by-component residual and positive units required')
    n=len(g);b=np.ones(n) if node_weight is None else np.broadcast_to(np.asarray(node_weight,np.float64),(n,)).copy()
    a=np.sqrt((b[:-1]+b[1:])/2) if edge_cost is None else np.broadcast_to(np.asarray(edge_cost,np.float64),(n-1,)).copy()
    if not all(np.isfinite(x).all() for x in [b,a]) or min(b.min(),a.min())<=0:
        raise ValueError('Weights and transport edge costs must be strictly positive')
    if not 0<tolerance<1 or maximum<1 or check_every<1:raise ValueError('Invalid solver budget')
    sqrtb=np.sqrt(b);left=sqrtb[:-1]/(h*a);right=sqrtb[1:]/(h*a)
    def B(value):return right[:,None]*value[1:]-left[:,None]*value[:-1]
    def BT(value):
        out=np.zeros((n,value.shape[1]));out[:-1]-=left[:,None]*value;out[1:]+=right[:,None]*value
        return out
    gram_diagonal=np.zeros(n);gram_diagonal[:-1]+=left*left;gram_diagonal[1:]+=right*right
    gram_off=-left*right
    bound=gram_diagonal.copy();bound[:-1]+=abs(gram_off);bound[1:]+=abs(gram_off)
    operator_bound=float(bound.max())
    y=sqrtb[:,None]*g
    rho=float(h*min((n-1)*h,10)/(2*np.pi*lam)) if rho is None else float(rho)
    if not np.isfinite(rho) or rho<=0:raise ValueError('Invalid ADMM penalty')
    if not np.any(g):
        return 0.,np.zeros_like(g),None,{'iterations':0,'relative_gap':0.,'converged':True,
            'primal':0.,'dual':0.,'seconds':time.perf_counter()-started}
    z=np.zeros((n-1,g.shape[1]));u=np.zeros_like(z)
    if warm is not None:
        z=np.array(warm['z'],copy=True);u=np.array(warm['scaled_multiplier'],copy=True)/rho
        if z.shape!=(n-1,g.shape[1]) or u.shape!=z.shape:raise ValueError('Wrong warm-start shape')
    def factor(penalty):
        diagonal=1/lam+penalty*gram_diagonal;off=penalty*gram_off
        diagonal,off,info=dpttrf(diagonal,off,overwrite_d=1,overwrite_e=1)
        if info:raise RuntimeError('Scaled dual system is not positive definite')
        return diagonal,off
    diagonal,off=factor(rho);report=None;feasible=None
    for iteration in range(1,maximum+1):
        rhs=np.asfortranarray(y+rho*BT(z-u))
        psi,info=dpttrs(diagonal,off,rhs,overwrite_b=1)
        if info:raise RuntimeError('Scaled tridiagonal dual solve failed')
        bpsi=B(psi);previous=z
        value=bpsi+u
        z=value/np.maximum(1.,np.linalg.norm(value,axis=1))[:,None]
        u=u+bpsi-z
        if iteration%check_every==0 or iteration==maximum:
            multiplier=rho*u;remainder=y-BT(multiplier)
            primal=h*(float(np.linalg.norm(multiplier,axis=1).sum())+.5*lam*float(np.sum(remainder*remainder)))
            violation=float(np.linalg.norm(bpsi,axis=1).max())
            feasible=psi/max(violation,1.)
            dual=h*(float(np.sum(feasible*y))-.5/lam*float(np.sum(feasible*feasible)))
            if dual<0:feasible=np.zeros_like(psi);dual=0.
            gap=primal-dual
            if gap<-1e-9*max(abs(primal),1.):raise RuntimeError('Invalid scaled primal/dual bounds')
            relative=max(gap,0.)/max(primal,1e-300)
            primal_residual=float(np.linalg.norm(bpsi-z))
            dual_residual=float(rho*np.linalg.norm(BT(z-previous)))
            report={'iterations':iteration,'relative_gap':relative,'converged':bool(relative<=tolerance),
                'primal':primal,'dual':dual,'primal_constraint_residual':primal_residual,
                'ADMM_dual_residual':dual_residual,'feasible_dual_gradient_ratio':min(violation,1.),
                'dual_L2_error_bound':float(np.sqrt(max(2*lam*b.max()*gap/h,0.))),
                'scaled_dual_L2_error_bound':float(np.sqrt(max(2*lam*gap/h,0.))),
                'rho':rho,'node_weight_min':float(b.min()),'node_weight_max':float(b.max()),
                'scaled_difference_spectral_squared_upper_bound':operator_bound,
                'same_weighted_target_as_unscaled_solver':True,'seconds':time.perf_counter()-started}
            if report['converged']:break
            new_rho=rho*2 if primal_residual>10*max(dual_residual,1e-30) else (
                rho/2 if dual_residual>10*max(primal_residual,1e-30) else rho)
            if new_rho!=rho:
                u*=rho/new_rho;rho=new_rho;diagonal,off=factor(rho)
    state={'z':z,'scaled_multiplier':rho*u}
    return .5*(report['primal']+report['dual']),h*sqrtb[:,None]*feasible,state,report


def lifted(predicted,observed,h,lam,smoothing=1e-3,**kwargs):
    p=np.asarray(predicted,np.float64);d=np.asarray(observed,np.float64)
    value,v,state,report=transport(lift(p,True,smoothing)-lift(d,True,smoothing),h,lam,**kwargs)
    hp=hilbert(p);envelope=np.sqrt(p*p+hp*hp+smoothing*smoothing)
    gradient=v[:,0]+v[:,2]*p/envelope-hilbert(v[:,1]+v[:,2]*hp/envelope)
    return value,gradient,state,report
