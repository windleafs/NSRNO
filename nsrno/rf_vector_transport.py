"""CPU vector L1 transport with quadratic imbalance; no wave solver or GPU."""
import numpy as np


def difference(value,h):
    return np.diff(value,axis=0)/h


def transpose(value,h):
    out=np.zeros((value.shape[0]+1,value.shape[1]),np.float64)
    out[:-1]-=value/h;out[1:]+=value/h
    return out


def transport(residual,h,lam,tolerance=1e-7,maximum=8000,warm=None):
    """min h*sum|sigma| + lam*h/2*sum|g-D^T sigma|².

    Return certified primal/dual values and a feasible approximate dual gradient.
    No claim is made about global convexity after a nonlinear waveform lift.
    """
    g=np.asarray(residual,np.float64)
    if g.ndim!=2 or len(g)<2 or min(h,lam)<=0 or not np.isfinite(g).all():
        raise ValueError('Finite [time,components] residual and positive scales required')
    y=lam*g
    q=np.zeros((len(g)-1,g.shape[1])) if warm is None else np.array(warm,copy=True)
    if q.shape!=(len(g)-1,g.shape[1]):raise ValueError('Wrong warm-start shape')
    if not np.any(g):
        return 0.,np.zeros_like(g),np.zeros_like(q),{'iterations':0,'relative_gap':0.,'converged':True,'primal':0.,'dual':0.}
    step=h*h/4;v=q.copy();momentum=1.
    def objective(q):
        phi=y-transpose(q,h)
        return .5*float(np.sum(phi*phi))+float(np.linalg.norm(q,axis=1).sum())
    previous=objective(q)
    def update(v):
        trial=v+step*difference(y-transpose(v,h),h)
        norm=np.linalg.norm(trial,axis=1,keepdims=True)
        return trial*np.maximum(1-step/np.maximum(norm,1e-300),0)
    report=None;feasible=None
    for iteration in range(1,maximum+1):
        proposed=update(v);value=objective(proposed)
        if value>previous+1e-12*max(abs(previous),1.):
            momentum=1.;v=q;proposed=update(v);value=objective(proposed)
        next_momentum=(1+np.sqrt(1+4*momentum*momentum))/2
        v=proposed+(momentum-1)/next_momentum*(proposed-q)
        q=proposed;momentum=next_momentum;previous=value
        if iteration%25==0 or iteration==maximum:
            phi=y-transpose(q,h)
            violation=float(np.linalg.norm(difference(phi,h),axis=1).max())
            feasible=phi/max(1.,violation)
            primal=h/lam*(.5*float(np.sum(phi*phi))+float(np.linalg.norm(q,axis=1).sum()))
            dual=h*(float(np.sum(feasible*g))-.5/lam*float(np.sum(feasible*feasible)))
            if dual<0:feasible=np.zeros_like(phi);dual=0.
            gap=primal-dual
            if gap<-1e-10*max(abs(primal),1.):raise RuntimeError('Primal/dual sign or scaling error')
            relative=max(gap,0)/max(abs(primal),1e-300)
            report={'iterations':iteration,'relative_gap':relative,'converged':bool(relative<=tolerance),
                'primal':primal,'dual':dual,'dual_gradient_bound':float(np.linalg.norm(difference(feasible,h),axis=1).max()),
                'dual_L2_error_bound':float(np.sqrt(max(2*lam*gap/h,0)))}
            if report['converged']:break
    # d T / d residual = h*phi at the exact dual optimum.
    return .5*(report['primal']+report['dual']),h*feasible,q,report


def hilbert(value):
    value=np.asarray(value,np.float64)
    return np.fft.ifft(-1j*np.sign(np.fft.fftfreq(len(value)))*np.fft.fft(value)).real


def lift(value,analytic=False,smoothing=1e-3):
    p=np.asarray(value,np.float64)
    if analytic:
        q=hilbert(p);return np.stack((p,q,np.sqrt(p*p+q*q+smoothing*smoothing)),axis=1)
    return np.stack((p,np.sqrt(p*p+smoothing*smoothing)),axis=1)


def lifted(predicted,observed,h,lam,analytic=False,smoothing=1e-3,**kwargs):
    p=np.asarray(predicted,np.float64)
    residual=lift(p,analytic,smoothing)-lift(observed,analytic,smoothing)
    value,grad,q,report=transport(residual,h,lam,**kwargs)
    if analytic:
        quadrature=hilbert(p);envelope=np.sqrt(p*p+quadrature*quadrature+smoothing*smoothing)
        derivative=grad[:,0]+grad[:,2]*p/envelope-hilbert(grad[:,1]+grad[:,2]*quadrature/envelope)
    else:
        derivative=grad[:,0]+grad[:,1]*p/np.sqrt(p*p+smoothing*smoothing)
    return value,derivative,q,report


def exact_two_site_scalar(a,b,h=1.,lam=2.):
    """Pred=(a+b,-a+b), observed=(-1,1), signs fixed near (a,b)=(.75,0)."""
    if not (a+b>0 and -a+b<0):raise ValueError('Formula is restricted to the stated sign sector')
    radius=np.hypot(a+1,b)
    if radius<=h/(2*lam):raise ValueError('Formula assumes active transport')
    value=lam*h*(b*b+(a-1)**2)+h*h*radius-h**3/(4*lam)
    grad=np.array([2*lam*h*(a-1)+h*h*(a+1)/radius,2*lam*h*b+h*h*b/radius])
    return value,grad
