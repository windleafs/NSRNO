"""SPD block BFGS with a nonlocal base and all saved data secants.

B = A - AD(D'AD)^-1(AD)' + Y(D'Y)^-1Y' + R.
Here Y=H_data D and R is the unchanged exact TV+damping metric.
The base complement annihilates D, so every independent saved direction
keeps its measured total curvature, to the precision of the native HVPs.
"""
import numpy as np
from .native_gn_coordinates import dot


def inverse_sqrt_factor(matrix,name):
    matrix=.5*(matrix+matrix.T)
    diagonal=np.sqrt(np.diag(matrix))
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal<=0):
        raise RuntimeError(f'Nonpositive {name} Gram diagonal')
    correlation=matrix/diagonal[:,None]/diagonal[None,:]
    eigenvalues,eigenvectors=np.linalg.eigh(correlation)
    if eigenvalues.min()<=1e-9*eigenvalues.max():
        raise RuntimeError(f'{name} saved directions need rank-revealing treatment: {eigenvalues.min()}')
    factor=(eigenvectors/np.sqrt(eigenvalues)[None,:])/diagonal[:,None]
    return factor,{'minimum_correlation_eigenvalue':float(eigenvalues.min()),
        'maximum_correlation_eigenvalue':float(eigenvalues.max()),
        'correlation_condition_number':float(eigenvalues.max()/eigenvalues.min())}


def combine(fields,factor):
    result=[]
    for column in range(factor.shape[1]):
        value=np.zeros_like(fields[0],dtype=np.float64)
        for row,field in enumerate(fields):value+=factor[row,column]*field
        result.append(value)
    return result


class MultisecantMetric:
    def __init__(self,base,mask,regularizer,regularizer_diagonal,pairs):
        self.base=base
        self.mask=mask
        self.regularizer=regularizer
        self.regularizer_diagonal=regularizer_diagonal
        self.pairs=list(pairs)
        self.pending_data=None
        self.rebuild()

    def rebuild(self):
        directions=[p[0] for p in self.pairs]
        images=[p[1] for p in self.pairs]
        base_images=[self.base.apply(d) for d in directions]
        count=len(directions)
        G=np.array([[dot(d,y) for y in images] for d in directions])
        GA=np.array([[dot(d,y) for y in base_images] for d in directions])
        symmetry=np.linalg.norm(G-G.T)/max(np.linalg.norm(G),1e-300)
        if symmetry>1e-5:raise RuntimeError(f'Saved native data Gram is asymmetric: {symmetry}')
        positive_factor,positive_report=inverse_sqrt_factor(G,'native_data')
        negative_factor,negative_report=inverse_sqrt_factor(GA,'spectral_base')
        self.positive=combine(images,positive_factor)
        self.negative=combine(base_images,negative_factor)
        nz,nx=directions[0].shape[-2:]
        weights=np.full(self.base.symbol.shape[1],2.)
        weights[0]=1
        if nx%2==0:weights[-1]=1
        values=np.array([np.sum(self.base.symbol[...,j,j].real*weights[None,:])/(nz*nx)
            for j in range(3)])
        self.base_diagonal=np.broadcast_to(values[:,None,None],directions[0].shape).copy()
        self.base_diagonal+=self.regularizer_diagonal
        self.diagonal=self.base_diagonal.copy()
        for field in self.positive:self.diagonal+=field*field
        for field in self.negative:self.diagonal-=field*field
        minimum=float(np.min(self.diagonal[:,self.mask]))
        if minimum < -1e-10*max(float(np.max(self.base_diagonal[:,self.mask])),1e-30):
            raise RuntimeError('Multisecant diagonal lost positivity')
        self.diagonal=np.maximum(self.diagonal,self.base_diagonal*1e-12)
        self.diagonal[:,~self.mask]=1.
        self.report={'data_secants_retained':count,'native_Gram_relative_asymmetry':symmetry,
            'native_data_Gram':positive_report,'spectral_base_Gram':negative_report,
            'prior_metric_changed':False,'data_metric':'PSD spectral base complement plus native-data Nystrom'}

    def apply(self,x):
        result=self.base.apply(x)+self.regularizer(x)
        for field in self.positive:result+=dot(field,x)*field
        for field in self.negative:result-=dot(field,x)*field
        return result*self.mask

    def update(self,d,total_image):
        previous=self.apply(d)
        if self.pending_data is not None:
            data=self.pending_data
            self.pending_data=None
        else:data=total_image-self.regularizer(d)
        self.pairs.append((d.copy(),data.copy()))
        self.rebuild()
        error=self.apply(d)-total_image
        relative=np.sqrt(max(dot(error,error),0)/max(dot(total_image,total_image),1e-300))
        if relative>2e-5:raise RuntimeError(f'New measured secant not retained: {relative}')
        return dict(self.report,sBs=dot(d,previous),sHy=dot(d,total_image),
            last_secant_relative_error=relative,
            true_to_previous_metric_curvature=dot(d,total_image)/dot(d,previous))
