"""Source-block data sketches for a fresh nonlinear linearization point.

Each shot contributes its original residual and all newly measured tangent
RF directions. Their block Nyström terms are <= H_s, match H_s D, and
retain g_s g_s'/E_s. Old-reference secants are never imported here.
"""
from pathlib import Path
import numpy as np
from .native_gn_coordinates import dot
from .native_gn_multisecant import inverse_sqrt_factor,combine


class SourceBlockMetric:
    def __init__(self,base,mask,regularizer,regularizer_diagonal,residual_features):
        self.base=base;self.mask=mask
        self.regularizer=regularizer;self.regularizer_diagonal=regularizer_diagonal
        self.residuals={r['name']:dict(r,vector=np.load(r['image'],mmap_mode='r'))
            for r in residual_features}
        if len(self.residuals)!=22:raise ValueError('Need all22 fresh shot residual gradients')
        self.directions=[];self.partial_images=[];self.pending_partials=None
        self.rebuild()

    def rebuild(self):
        self.positive=[];self.negative=[]
        n=len(self.directions)
        smallest=0.
        for name,residual in self.residuals.items():
            g=residual['vector'];energy=float(residual['weighted_residual_energy'])
            if energy<=0:raise RuntimeError('Fresh shot has no residual energy; handle a converged acquisition explicitly')
            self.positive.append(np.asarray(g,np.float64)/np.sqrt(energy))
            if not n:continue
            V=[images[name] for images in self.partial_images]
            G=np.array([[dot(d,v) for v in V] for d in self.directions])
            G=.5*(G+G.T)
            b=np.array([dot(d,g) for d in self.directions])
            K=G-np.outer(b,b)/energy
            K=.5*(K+K.T)
            scales=np.sqrt(np.maximum(np.diag(G),1e-300))
            eigenvalues,eigenvectors=np.linalg.eigh(K/scales[:,None]/scales[None,:])
            smallest=min(smallest,float(eigenvalues.min()))
            maximum=max(float(eigenvalues.max()),1.)
            if eigenvalues.min() < -1e-4*maximum:raise RuntimeError('Fresh source residual/HVP Gram is not PSD')
            keep=eigenvalues>1e-7*maximum
            factor=(eigenvectors[:,keep]/np.sqrt(eigenvalues[keep])[None,:])/scales[:,None]
            for column in range(factor.shape[1]):
                image=np.zeros_like(g,dtype=np.float64)
                for j,v in enumerate(V):
                    image+=np.multiply(v,factor[j,column],dtype=np.float64)
                    image-=np.multiply(g,factor[j,column]*b[j]/energy,dtype=np.float64)
                self.positive.append(image)
        if n:
            images=[self.base.apply(d) for d in self.directions]
            G=np.array([[dot(d,y) for y in images] for d in self.directions])
            factor,report=inverse_sqrt_factor(G,'fresh_spectral_base')
            self.negative=combine(images,factor)
        shape=next(iter(self.residuals.values()))['vector'].shape
        nz,nx=shape[-2:]
        weights=np.full(self.base.symbol.shape[1],2.)
        weights[0]=1
        if nx%2==0:weights[-1]=1
        values=np.array([np.sum(self.base.symbol[...,j,j].real*weights[None,:])/(nz*nx)
            for j in range(3)])
        self.base_diagonal=np.broadcast_to(values[:,None,None],shape).copy()+self.regularizer_diagonal
        self.diagonal=self.base_diagonal.copy()
        for image in self.positive:self.diagonal+=image*image
        for image in self.negative:self.diagonal-=image*image
        if np.min(self.diagonal[:,self.mask]) < -1e-10*max(np.max(self.base_diagonal),1e-30):
            raise RuntimeError('Fresh source-block diagonal lost positivity')
        self.diagonal=np.maximum(self.diagonal,self.base_diagonal*1e-12)
        self.diagonal[:,~self.mask]=1.
        self.report={'fresh_source_residuals':len(self.residuals),'fresh_HVP_directions':n,
            'positive_data_modes':len(self.positive),'old_exact_secants_reused':False,
            'minimum_source_conditional_eigenvalue':smallest,
            'data_sketch_bound':'sum source blocks <= true data normal operator',
            'data_loss_nonnegative_by_source_residual_sketch':True}

    def apply(self,x):
        result=self.base.apply(x)+self.regularizer(x)
        for image in self.positive:result+=dot(image,x)*image
        for image in self.negative:result-=dot(image,x)*image
        return result*self.mask

    def update(self,d,total_image):
        if self.pending_partials is None:raise RuntimeError('Need the new per-shot HVPs')
        rows=self.pending_partials;self.pending_partials=None
        images={row['name']:np.load(row['image'],mmap_mode='r') for row in rows}
        if set(images)!=set(self.residuals):raise RuntimeError('Gradient/HVP shot sets differ')
        previous=dot(d,self.apply(d))
        self.directions.append(d.copy());self.partial_images.append(images)
        self.rebuild()
        difference=self.apply(d)-total_image
        error=np.sqrt(max(dot(difference,difference),0)/max(dot(total_image,total_image),1e-300))
        if error>2e-4:raise RuntimeError(f'Fresh source-block secant mismatch: {error}')
        return dict(self.report,sBs=previous,sHy=dot(d,total_image),
            true_to_previous_metric_curvature=dot(d,total_image)/previous,
            last_secant_relative_error=error)
