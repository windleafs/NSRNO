"""Independent global CDF-envelope energy; no material or RF data readers."""
from pathlib import Path
import numpy as np

LOWER=np.array([(1400-1540)/80,(1540e3*np.exp(-.25)/1680-1000)/100,-8.])
UPPER=np.array([(1680-1540)/80,(1540e3*np.exp(.25)/1400-1000)/100,2.])


def kernel():
    x=np.arange(-8,9,dtype=np.float64)
    k=np.exp(-.5*(x/2.)**2)
    return k/k.sum()


def make_grid(field):
    low=np.floor((LOWER[field]-.15)*100)/100
    high=np.ceil((UPPER[field]+.15)*100)/100
    return low+np.arange(round((high-low)/.01)+1)*.01


def histogram_cdf(values,grid):
    values=np.asarray(values,np.float64).ravel()
    dx=grid[1]-grid[0]
    position=(values-grid[0])/dx
    left=np.floor(position).astype(np.int64)
    fraction=position-left
    if (not len(values) or not np.isfinite(values).all() or
        left.min()<0 or left.max()+1>=len(grid)):
        raise ValueError('Finite feature samples strictly inside the padded grid required')
    weight=1/len(values)
    counts=(np.bincount(left,weights=(1-fraction)*weight,minlength=len(grid))+
            np.bincount(left+1,weights=fraction*weight,minlength=len(grid)))
    return counts.cumsum(),left


def smooth_cdf(cdf,k):
    r=len(k)//2
    return np.convolve(np.pad(cdf,(r,r),constant_values=(0.,1.)),k,mode='valid')


def smooth_transpose(dual,k):
    r=len(k)//2
    return np.convolve(np.pad(dual,(r,r),constant_values=0.),k,mode='valid')


class GlobalCDFPrior:
    def __init__(self,path=None,state=None):
        if (path is None)==(state is None):
            raise ValueError('One frozen CDF template source required')
        if path is not None:
            with np.load(Path(path),allow_pickle=False) as archive:
                state={key:archive[key].copy() for key in archive.files}
        self.k=np.asarray(state['kernel'],np.float64)
        self.grids=[np.asarray(state[f'grid_{f}'],np.float64) for f in range(3)]
        self.lower=[np.asarray(state[f'lower_{f}'],np.float64) for f in range(3)]
        self.upper=[np.asarray(state[f'upper_{f}'],np.float64) for f in range(3)]
        if not np.allclose(self.k,self.k[::-1],atol=1e-15,rtol=0) or abs(self.k.sum()-1)>1e-12:
            raise ValueError('Fixed normalized symmetric feature-axis kernel required')
        for grid,lo,hi in zip(self.grids,self.lower,self.upper):
            if (grid.ndim!=1 or len(grid)<20 or lo.shape!=grid.shape or hi.shape!=grid.shape or
                not all(np.isfinite(v).all() for v in [grid,lo,hi]) or
                np.any(lo>hi) or np.any(lo<0) or np.any(hi>1) or
                np.any(np.diff(lo)<-1e-12) or np.any(np.diff(hi)<-1e-12) or
                not np.allclose(np.diff(grid),grid[1]-grid[0],atol=1e-12,rtol=0)):
                raise ValueError('Ordered finite CDF envelope and uniform feature grid required')

    def feature_value_gradient(self,values,field,gradient=True):
        values=np.asarray(values,np.float64)
        grid=self.grids[field]
        cdf,left=histogram_cdf(values,grid)
        estimate=smooth_cdf(cdf,self.k)
        residual=estimate-np.clip(estimate,self.lower[field],self.upper[field])
        dx=grid[1]-grid[0]
        value=.5*dx*float(residual@residual)
        derivative=(-smooth_transpose(residual,self.k)[left]/values.size).reshape(values.shape) if gradient else None
        return value,derivative,{'energy':value,'CDF_L2_outside_envelope':float(np.sqrt(2*value)),
            'maximum_CDF_violation':float(np.max(abs(residual))),
            'fraction_grid_with_violation':float(np.mean(abs(residual)>1e-12))}

    def evaluate(self,coordinates,control,gradient=True,region=(slice(80,921),slice(178,943))):
        absolute=coordinates.absolute+np.asarray(control,np.float64)
        zs,xs=region
        values=absolute[:,zs,xs]
        c=1540.*np.exp(.05*values[0])
        rho=1000.*np.exp(.1*values[1])
        features=[(c-1540.)/80.,(rho-1000.)/100.,values[2]]
        chains=[.05*c/80.,.1*rho/100.,np.ones_like(values[2])]
        derivative=np.zeros_like(absolute) if gradient else None
        total,parts=0.,{}
        for field in range(3):
            value,g,details=self.feature_value_gradient(features[field],field,gradient)
            total+=value
            parts[f'f{field}']=details
            if gradient:
                derivative[field,zs,xs]=g*chains[field]
        return total,derivative,{'global_CDF':total,'per_field':parts,
            'three_independent_CDFs':True,'material_spatial_filter':False}
