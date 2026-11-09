"""Weighted vector transport on the complete RF time axis, solved on CPU.

Weights act in the convex inner problem. They do not multiply the waveform
before the nonlinear lift, and no trace is normalized by its predicted mass.
"""
import time
import numpy as np
from scipy.linalg.lapack import dpttrf,dpttrs
if __package__:
    from .rf_vector_transport import difference,transpose,lift,hilbert
else:
    from rf_vector_transport import difference,transpose,lift,hilbert


def transport(residual,h,lam,node_weight=None,edge_cost=None,tolerance=1e-6,
              maximum=8000,rho=None,warm=None,check_every=25):
    """T=h min_s sum a|s|+lam/2 sum b|g-D^T s|², b,a positive.

    The dual is h max_{|D phi|<=a} phi.g-|phi|²/(2 lam b).
    ADMM uses a tridiagonal solve and pointwise ball projection; bounds are
    evaluated from feasible primal/dual variables, independently of ADMM's
    residual stopping tests. Gradient h*phi is approximate unless the gap=0.
    """
    started=time.perf_counter();g=np.asarray(residual,np.float64)
    if g.ndim!=2 or len(g)<2 or not np.isfinite(g).all() or min(h,lam)<=0:
        raise ValueError('Finite time-by-component residual and positive units required')
    n=len(g);b=np.ones(n) if node_weight is None else np.broadcast_to(np.asarray(node_weight,np.float64),(n,)).copy()
    a=np.sqrt((b[:-1]+b[1:])/2) if edge_cost is None else np.broadcast_to(np.asarray(edge_cost,np.float64),(n-1,)).copy()
    if not all(np.isfinite(x).all() for x in [b,a]) or min(b.min(),a.min())<=0:
        raise ValueError('Weights and transport edge costs must be finite and strictly positive')
    if not 0<tolerance<1 or maximum<1 or check_every<1:raise ValueError('Invalid solver budget')
    rho=float(h*min((n-1)*h,10)/(2*np.pi*lam*np.median(b))) if rho is None else float(rho)
    if not np.isfinite(rho) or rho<=0:raise ValueError('Invalid ADMM penalty')
    if not np.any(g):
        return 0.,np.zeros_like(g),None,{'iterations':0,'relative_gap':0.,'converged':True,
            'primal':0.,'dual':0.,'seconds':time.perf_counter()-started}
    phi=np.zeros_like(g);z=np.zeros((n-1,g.shape[1]));u=np.zeros_like(z)
    if warm is not None:
        z=np.array(warm['z'],copy=True);u=np.array(warm['multiplier'],copy=True)/rho
        if z.shape!=(n-1,g.shape[1]) or u.shape!=z.shape:raise ValueError('Wrong warm-start shape')
    def factor(penalty):
        diagonal=1/(lam*b)+2*penalty/h**2
        diagonal[[0,-1]]-=penalty/h**2
        off=np.full(n-1,-penalty/h**2)
        diag,off,info=dpttrf(diagonal,off,overwrite_d=1,overwrite_e=1)
        if info:raise RuntimeError('Weighted dual system is not positive definite')
        return diag,off
    diagonal,off=factor(rho);report=None;feasible=None
    for iteration in range(1,maximum+1):
        rhs=np.asfortranarray(g+rho*transpose(z-u,h))
        phi,info=dpttrs(diagonal,off,rhs,overwrite_b=1)
        if info:raise RuntimeError('Tridiagonal dual solve failed')
        dphi=difference(phi,h);previous=z
        value=dphi+u
        z=value/np.maximum(1.,np.linalg.norm(value,axis=1)/a)[:,None]
        u=u+dphi-z
        if iteration%check_every==0 or iteration==maximum:
            multiplier=rho*u
            remainder=g-transpose(multiplier,h)
            primal=h*(float(np.sum(a*np.linalg.norm(multiplier,axis=1)))+
                      .5*lam*float(np.sum(b[:,None]*remainder*remainder)))
            violation=float(np.max(np.linalg.norm(dphi,axis=1)/a))
            feasible=phi/max(violation,1.)
            dual=h*(float(np.sum(feasible*g))-.5/lam*float(np.sum(feasible*feasible/b[:,None])))
            if dual<0:feasible=np.zeros_like(phi);dual=0.
            gap=primal-dual
            if gap<-1e-9*max(abs(primal),1.):raise RuntimeError('Invalid weighted primal/dual bounds')
            relative=max(gap,0.)/max(primal,1e-300)
            primal_residual=float(np.linalg.norm(dphi-z))
            dual_residual=float(rho*np.linalg.norm(transpose(z-previous,h)))
            report={'iterations':iteration,'relative_gap':relative,'converged':bool(relative<=tolerance),
                'primal':primal,'dual':dual,'primal_constraint_residual':primal_residual,
                'ADMM_dual_residual':dual_residual,'feasible_dual_gradient_ratio':min(violation,1.),
                'dual_L2_error_bound':float(np.sqrt(max(2*lam*b.max()*gap/h,0.))),
                'rho':rho,'node_weight_min':float(b.min()),'node_weight_max':float(b.max()),
                'seconds':time.perf_counter()-started}
            if report['converged']:break
            # Standard residual balancing changes only the numerical penalty,
            # never the transport target or the fixed observation weights.
            new_rho=rho*2 if primal_residual>10*max(dual_residual,1e-30) else (
                rho/2 if dual_residual>10*max(primal_residual,1e-30) else rho)
            if new_rho!=rho:
                u*=rho/new_rho;rho=new_rho;diagonal,off=factor(rho)
    state={'z':z,'multiplier':rho*u}
    return .5*(report['primal']+report['dual']),h*feasible,state,report


def lifted(predicted,observed,h,lam,smoothing=1e-3,**kwargs):
    p=np.asarray(predicted,np.float64);d=np.asarray(observed,np.float64)
    value,v,state,report=transport(lift(p,True,smoothing)-lift(d,True,smoothing),h,lam,**kwargs)
    hp=hilbert(p);envelope=np.sqrt(p*p+hp*hp+smoothing*smoothing)
    gradient=v[:,0]+v[:,2]*p/envelope-hilbert(v[:,1]+v[:,2]*hp/envelope)
    return value,gradient,state,report
