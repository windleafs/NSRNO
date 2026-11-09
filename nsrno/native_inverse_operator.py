"""CPU positive spatial/Fourier inverse trained only on saved native images."""
import numpy as np
from scipy import fft
import torch
from torch import nn
from torch.nn import functional as F


LOWER=[(1,0),(2,0),(2,1)]


class SpatialSpectralInverse(nn.Module):
    def __init__(self,mask,gain,symbol_grid=(64,64),gate_grid=(16,16)):
        super().__init__()
        mask=np.asarray(mask,bool)
        self.shape=mask.shape;self.symbol_grid=tuple(symbol_grid)
        self.register_buffer('mask',torch.from_numpy(mask.astype(np.float32))[None,None])
        self.register_buffer('gain',torch.tensor(float(gain),dtype=torch.float32))
        self.log_diagonal_raw=nn.Parameter(torch.zeros(3,*symbol_grid))
        self.off_real=nn.Parameter(torch.zeros(3,*symbol_grid))
        self.off_imag=nn.Parameter(torch.zeros(3,*symbol_grid))
        self.gate_raw=nn.Parameter(torch.zeros(1,3,*gate_grid))

    def initialize_cholesky(self,cholesky):
        # Cholesky is for P/gain, with positive real diagonal.
        with torch.no_grad():
            diagonal=np.stack([cholesky[...,j,j].real for j in range(3)])
            logs=np.clip(np.log(diagonal),-3.99,3.99)
            self.log_diagonal_raw.copy_(torch.from_numpy((4*np.arctanh(logs/4)).astype(np.float32)))
            self.off_real.copy_(torch.from_numpy(np.stack([cholesky[...,i,j].real for i,j in LOWER]).astype(np.float32)))
            self.off_imag.copy_(torch.from_numpy(np.stack([cholesky[...,i,j].imag for i,j in LOWER]).astype(np.float32)))

    def forward(self,value):
        if value.device.type!='cpu':
            raise ValueError('This candidate operates only on CPU')
        single=value.ndim==3
        if single:value=value[None]
        if tuple(value.shape[1:])!=(3,*self.shape):
            raise ValueError('Three full native fields required')
        nz,nx=self.shape;size=(nz,nx//2+1)
        gate=torch.exp(2*torch.tanh(F.interpolate(self.gate_raw,size=self.shape,mode='bilinear',align_corners=True)))
        diagonal=torch.exp(F.interpolate((4*torch.tanh(self.log_diagonal_raw/4))[None],
            size=size,mode='bilinear',align_corners=True)[0])
        real=F.interpolate(self.off_real[None],size=size,mode='bilinear',align_corners=True)[0]
        imag=F.interpolate(self.off_imag[None],size=size,mode='bilinear',align_corners=True)[0]
        lower=torch.zeros((3,3,*size),dtype=torch.complex64,device=value.device)
        for j in range(3):lower[j,j]=diagonal[j]
        for n,(i,j) in enumerate(LOWER):lower[i,j]=torch.complex(real[n],imag[n])
        transformed=torch.fft.fftshift(torch.fft.rfft2(value*self.mask*gate,norm='ortho'),dim=-2)
        temp=torch.einsum('ijzx,bizx->bjzx',lower.conj(),transformed)
        result=torch.einsum('ijzx,bjzx->bizx',lower,temp)+1e-6*transformed
        result=torch.fft.irfft2(torch.fft.ifftshift(result,dim=-2),s=self.shape,norm='ortho')
        result=self.gain*gate*self.mask*result
        return result[0] if single else result


def binned_inverse_initialization(X,Y,grid=(64,64),callback=None):
    # X = sqrt(D) p / ||sqrt(D)p||; Y = H p / sqrt(D) / same norm.
    nz,nx=X.shape[-2:];kx=nx//2+1
    iz=np.rint(np.linspace(0,grid[0]-1,nz)).astype(np.int32)
    ix=np.rint(np.linspace(0,grid[1]-1,kx)).astype(np.int32)
    bins=(iz[:,None]*grid[1]+ix[None,:]).reshape(-1)
    frequency_weights=np.full(kx,2.);frequency_weights[0]=1.
    if nx%2==0:frequency_weights[-1]=1.
    frequency_weights=np.broadcast_to(frequency_weights,(nz,kx)).reshape(-1)
    count=grid[0]*grid[1]
    G=np.zeros((count,3,3),np.complex128);C=np.zeros_like(G)
    rayleigh=np.einsum('ncij,ncij->n',X,Y,dtype=np.float64)
    if np.any(rayleigh<=0):
        raise ValueError('Training curvature is not positive')
    gain=float(np.exp(-np.median(np.log(rayleigh))))
    diagonal_gain=float(rayleigh.sum()/np.einsum('ncij,ncij->',Y,Y,dtype=np.float64))
    def accumulate(value):
        return np.bincount(bins,weights=value.real,minlength=count)+1j*np.bincount(bins,weights=value.imag,minlength=count)
    for n in range(len(X)):
        xf=fft.fftshift(fft.rfft2(X[n],norm='ortho',workers=2),axes=-2).reshape(3,-1)
        yf=fft.fftshift(fft.rfft2(Y[n],norm='ortho',workers=2),axes=-2).reshape(3,-1)
        for i in range(3):
            for j in range(3):
                G[:,i,j]+=accumulate(frequency_weights*yf[i]*yf[j].conj())
                C[:,i,j]+=accumulate(frequency_weights*xf[i]*yf[j].conj())
        if callback:callback(n+1,len(X))
    G=.5*(G+G.conj().swapaxes(-1,-2))
    energy=G.trace(axis1=-2,axis2=-1).real/3
    reference=max(float(np.median(energy[energy>0])),1e-30)
    ridge=1e-4*energy+1e-8*reference
    G+=ridge[:,None,None]*np.eye(3)
    raw=C@np.linalg.inv(G);raw=.5*(raw+raw.conj().swapaxes(-1,-2))
    values,vectors=np.linalg.eigh(raw)
    clipped=np.clip(values,gain*np.exp(-7),gain*np.exp(7))
    positive=(vectors*clipped[:,None,:])@vectors.conj().swapaxes(-1,-2)
    chol=np.linalg.cholesky(positive/gain).reshape(*grid,3,3)
    return chol,gain,diagonal_gain,{'model_gain_from_training_Rayleigh_geometric_median':gain,
        'best_scalar_inverse_gain_from_training':diagonal_gain,'symbol_grid':list(grid),
        'ridge_relative':1e-4,'eigenvalue_range_relative_to_gain':[float(np.exp(-7)),float(np.exp(7))],
        'initialization_is_PSD_projection_of_unconstrained_binned_inverse_fit':True,
        'not_an_exact_constrained_PSD_least_squares_solution':True}


def quality(value,target):
    vv=float(np.sum(value*value));tt=float(np.sum(target*target));cross=float(np.sum(value*target))
    return {'cosine':cross/max(np.sqrt(vv*tt),1e-300),
        'relative_error':float(np.sqrt(np.sum((value-target)**2)/max(tt,1e-300))),
        'norm_gain':float(np.sqrt(vv/max(tt,1e-300)))}


def coarse_balance(rhs,apply,X,Y,gram):
    coefficient=np.linalg.solve(gram,X@rhs.reshape(-1))
    right=rhs.reshape(-1)-Y.T@coefficient
    result=np.asarray(apply(right.reshape(rhs.shape)),np.float64).reshape(-1)
    result-=X.T@np.linalg.solve(gram,Y@result)
    return (X.T@coefficient+result).reshape(rhs.shape)


def structural_check():
    # Real-space symmetry must also hold for complex symbols and odd FFT sizes.
    torch.manual_seed(115)
    model=SpatialSpectralInverse(np.ones((9,11),bool),.1,symbol_grid=(4,5),gate_grid=(3,3))
    with torch.no_grad():
        for p in model.parameters():p.normal_(0,.2)
        a=torch.randn(3,9,11);b=torch.randn_like(a);pa=model(a);pb=model(b)
        left=float((a*pb).sum());right=float((pa*b).sum())
        relative=abs(left-right)/max(abs(left),abs(right),1e-12)
        positive=float((a*pa).sum())
    if relative>2e-5 or positive<=0:
        raise RuntimeError('CPU inverse is not numerically self-adjoint positive')
    return {'small_random_complex_symbol_real_space_symmetry_error':relative,
        'small_random_positive_quadratic':positive,'no_patient_data_or_wave_solves':True}
