"""Exact known-image Galerkin updates with physical boxes and H deflation."""
import numpy as np
from .native_span_qp import solve


class TrueImageSpan:
    def __init__(self,gradient,diagonal,low,high):
        self.shape=np.asarray(gradient).shape
        self.g=np.asarray(gradient,np.float64).reshape(-1)
        self.diagonal=np.asarray(diagonal,np.float64).reshape(-1)
        self.low=np.asarray(low,np.float64).reshape(-1);self.high=np.asarray(high,np.float64).reshape(-1)
        if np.any(self.diagonal<=0) or np.any(self.low>0) or np.any(self.high<0):raise ValueError('Invalid metric or bounds')
        self.basis=np.empty((0,self.g.size));self.images=np.empty_like(self.basis)
        self.gram=np.empty((0,0));self.u=np.zeros_like(self.g);self.Hu=np.zeros_like(self.g)
        self.coefficients=np.empty(0)

    def append(self,direction,image):
        d=np.asarray(direction,np.float64).reshape(-1);h=np.asarray(image,np.float64).reshape(-1)
        norm=np.sqrt(np.dot(d,self.diagonal*d))
        if not np.isfinite(norm) or norm<=1e-20 or np.dot(d,h)<=0:raise ValueError('Degenerate or nonpositive true direction')
        d=d/norm;h=h/norm
        self.basis=np.concatenate([self.basis,d[None]],axis=0)
        self.images=np.concatenate([self.images,h[None]],axis=0)
        raw=self.basis@self.images.T
        asymmetry=np.linalg.norm(raw-raw.T)/max(np.linalg.norm(raw),1e-30)
        if asymmetry>1e-4:raise RuntimeError('Known true images are not symmetric at the declared reference')
        self.gram=.5*(raw+raw.T)
        return {'directions':len(self.basis),'normalization':float(norm),'relative_asymmetry':float(asymmetry)}

    def solve(self):
        if not len(self.basis):return {'directions':0,'model_change':0.,'restricted_stationarity_relative':None}
        values,vectors=np.linalg.eigh(self.gram)
        if values[0]<=max(float(values[-1])*1e-11,1e-30):raise RuntimeError('Known basis lost positive rank; do not drop a physical direction silently')
        rotation=vectors/np.sqrt(values)[None,:]
        white=rotation.T@self.basis;linear=rotation.T@(self.basis@self.g)
        y,report=solve(white,linear,self.low,self.high)
        self.coefficients=rotation@y
        self.u=self.basis.T@self.coefficients;self.Hu=self.images.T@self.coefficients
        value=float(self.g@self.u+.5*self.u@self.Hu)
        if abs(value-report['objective'])>1e-8*max(abs(value),1.):raise RuntimeError('Restricted value did not reconstruct')
        return {**report,'directions':len(self.basis),'model_change':value,
            'minimum_restricted_eigenvalue':float(values[0]),'maximum_restricted_eigenvalue':float(values[-1]),
            'full_projected_norm':self.projected_norm(),'full_space_optimality_claimed':False}

    def projected_norm(self):
        gradient=self.g+self.Hu
        change=self.u-np.clip(self.u-gradient/self.diagonal,self.low,self.high)
        return float(np.sqrt(np.dot(change,self.diagonal*change)))

    def next_direction(self):
        residual=self.g+self.Hu;v=-residual/self.diagonal
        at_low=self.u<=self.low+1e-10;at_high=self.u>=self.high-1e-10
        v[(at_low&(v<0))|(at_high&(v>0))]=0
        feasible_slope=float(residual@v)
        if feasible_slope>=0:raise RuntimeError('No feasible preconditioned residual descent')
        if len(self.basis):
            projection=np.linalg.solve(self.gram,self.images@v)
            direction=v-self.basis.T@projection
        else:direction=v.copy()
        norm=np.sqrt(np.dot(direction,self.diagonal*direction))
        if not np.isfinite(norm) or norm<=1e-20:raise RuntimeError('No independent preconditioned residual direction')
        direction/=norm;slope=float(residual@direction)
        # At active bounds r is not orthogonal to the old span. The deflated
        # p alone need not descend: span(B,p) also contains the feasible v.
        # Joint box minimization over all coefficients must supply the step.
        conjugacy=self.images@direction
        return direction.reshape(self.shape),{'slope_at_current_subspace_solution':slope,
            'normalization':float(norm),'known_H_conjugacy_absolute_max':float(np.max(abs(conjugacy))) if len(conjugacy) else 0.,
            'feasible_preconditioned_residual_slope':feasible_slope,
            'deflated_direction_alone_may_not_descend_at_active_bounds':True,
            'pixelwise_magnitude_clipping':False,'old_exact_images_used_only_at_same_reference':True}
