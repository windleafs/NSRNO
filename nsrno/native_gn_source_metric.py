"""SPD BFGS updates that retain the already computed per-shot curvature.

For H=sum H_s+R and y_s=H_s d, replace the aggregate positive BFGS term
by sum y_s y_s^T/(d^T y_s), including the regularizer R as one component.
This matches H d, is SPD for an SPD starting metric, and adds a PSD
matrix relative to the aggregate update with the same previous metric.
"""
import numpy as np
from .native_gn_coordinates import dot


class SourceResolvedBFGS:
    def __init__(self,base,base_diagonal,mask):
        self.base=base
        self.base_diagonal=base_diagonal.copy()
        self.diagonal=base_diagonal.copy()
        self.mask=mask
        self.pairs=[]
        self.pending_components=None

    def apply(self,x):
        out=self.base(x)
        for negative,denominator,positive in self.pairs:
            out-=np.multiply(negative,dot(negative,x)/denominator,dtype=np.float64)
            for vector,energy in positive:
                out+=np.multiply(vector,dot(vector,x)/energy,dtype=np.float64)
        return out*self.mask

    def update(self,d,y,components=None):
        if components is None and self.pending_components is not None:
            components=self.pending_components
            self.pending_components=None
        v=self.apply(d)
        vs=dot(d,v)
        ys=dot(d,y)
        if not np.isfinite(vs+ys) or vs<=0 or ys<=0:
            raise RuntimeError('Nonpositive source-resolved BFGS curvature')
        if components is None:
            positive=[(y.copy(),ys)]
        else:
            positive=[]
            total=np.zeros_like(y,dtype=np.float64)
            for component in components:
                vector=component['vector']
                energy=dot(d,vector)
                if not np.isfinite(energy) or energy<=0:
                    raise RuntimeError(f'Nonpositive component curvature: {component["name"]}')
                if vector.shape!=y.shape or not np.all(np.isfinite(vector)):
                    raise RuntimeError('Invalid per-shot image')
                total+=vector
                positive.append((vector,energy))
            difference=total-y
            relative=np.sqrt(max(dot(difference,difference),0)/max(dot(y,y),1e-300))
            if relative>1e-10:
                raise RuntimeError(f'Per-shot images changed the native total HVP: {relative}')
        self.pairs.append((v.copy(),vs,positive))
        self.diagonal-=np.multiply(v,v,dtype=np.float64)/vs
        for vector,energy in positive:
            self.diagonal+=np.multiply(vector,vector,dtype=np.float64)/energy
        minimum=float(np.min(self.diagonal[:,self.mask]))
        tolerance=1e-10*max(float(np.max(self.base_diagonal[:,self.mask])),1e-30)
        if minimum < -tolerance:
            raise RuntimeError('Source-resolved BFGS diagonal lost positivity')
        self.diagonal=np.maximum(self.diagonal,self.base_diagonal*1e-12)
        self.diagonal[:,~self.mask]=1.
        residual=self.apply(d)-y
        defect=np.sqrt(max(dot(residual,residual),0)/max(dot(y,y),1e-300))
        if defect>1e-8:
            raise RuntimeError(f'Source-resolved secant equation failed: {defect}')
        return {'pairs':len(self.pairs),'sBs':vs,'sHy':ys,
            'positive_components':len(positive),'source_resolved':components is not None,
            'last_secant_relative_error':defect,
            'true_to_previous_metric_curvature':ys/vs}
