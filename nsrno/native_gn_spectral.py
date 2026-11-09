"""Real self-adjoint PSD convolution fitted from already computed HVPs.

This is a data-normal preconditioner, not a new RF forward solver. The
translation-invariance assumption must be assessed on held-out native HVPs.
"""
import numpy as np
from scipy import fft


class SpectralDataMetric:
    def __init__(self,symbol,mask):
        self.symbol=symbol
        self.mask=mask

    def apply(self,x):
        transformed=fft.rfft2(np.asarray(x,np.float64)*self.mask,
            axes=(-2,-1),norm='ortho',workers=1)
        result=np.einsum('zxab,bzx->azx',self.symbol,transformed,optimize=True)
        return fft.irfft2(result,s=x.shape[-2:],axes=(-2,-1),
            norm='ortho',workers=1)*self.mask


def fit_symbol(paths,ridge=.01):
    """Fit one 3x3 complex symbol per spatial frequency, then impose PSD.

    Per-pair input-norm weighting prevents one large trial direction from
    dominating. No held-out direction or image enters this calculation.
    """
    C=M=None
    for folder in paths:
        d=np.load(folder/'direction.npy')
        y=np.load(folder/'data_image.npy')
        weight=1/max(float(np.einsum('ijk,ijk->',d,d)),1e-300)
        X=fft.rfft2(d,axes=(-2,-1),norm='ortho',workers=1)
        Y=fft.rfft2(y,axes=(-2,-1),norm='ortho',workers=1)
        if C is None:
            C=np.zeros((*X.shape[-2:],3,3),np.complex128)
            M=np.zeros_like(C)
        C+=weight*np.einsum('azx,bzx->zxab',X,X.conj(),optimize=True)
        M+=weight*np.einsum('azx,bzx->zxab',Y,X.conj(),optimize=True)
    trace=np.trace(C,axis1=-2,axis2=-1).real/3
    positive=trace[trace>0]
    if not positive.size:raise ValueError('No excited spatial frequencies')
    regularizer=ridge*trace+float(np.median(positive))*1e-6
    for j in range(3):C[...,j,j]+=regularizer
    # Right solve M C^-1, using the Hermitian input covariance.
    fit=np.linalg.solve(C,np.swapaxes(M.conj(),-1,-2))
    fit=np.swapaxes(fit.conj(),-1,-2)
    hermitian=.5*(fit+np.swapaxes(fit.conj(),-1,-2))
    eigenvectors=None
    eigenvalues,eigenvectors=np.linalg.eigh(hermitian)
    negative_fraction=float(np.mean(eigenvalues<0))
    negative_magnitude=float(np.sum(np.maximum(-eigenvalues,0))/max(np.sum(abs(eigenvalues)),1e-300))
    positive_eigenvalues=eigenvalues[eigenvalues>0]
    if not positive_eigenvalues.size:raise RuntimeError('The fitted symbol has no positive modes')
    floor=float(np.median(positive_eigenvalues))*1e-6
    clipped=np.maximum(eigenvalues,floor)
    symbol=np.einsum('...ik,...k,...jk->...ij',eigenvectors,clipped,
        eigenvectors.conj(),optimize=True)
    # rFFT boundary columns must obey the conjugate pairing in the full
    # first Fourier axis to define a real self-adjoint spatial operator.
    partner=(-np.arange(symbol.shape[0]))%symbol.shape[0]
    for column in [0]+([-1] if d.shape[-1]%2==0 else []):
        boundary=symbol[:,column].copy()
        symbol[:,column]=.5*(boundary+boundary[partner].conj())
    report={'training_pairs':len(paths),'ridge_relative':ridge,
        'input_pair_weighting':'inverse squared input norm',
        'symbol_shape':list(symbol.shape),'negative_unconstrained_eigenvalue_fraction':negative_fraction,
        'negative_unconstrained_eigenvalue_magnitude_fraction':negative_magnitude,
        'PSD_floor':floor,'translation_invariance_is_an_assumption':True,
        'held_out_pair_used_in_fit':False}
    return symbol,report
