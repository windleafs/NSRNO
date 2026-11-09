"""Native source-feature Schur complement added to the residual-aware metric.

N_D+N_source <= H_data and N_D+N_residual <= H_data. Their additive
combination is PSD, preserves measured model directions, and keeps the
gradient Cauchy bound. The sum of its feature terms is at most 2 H_data;
the spectral base complement remains an approximation outside that sketch.
"""
from pathlib import Path
import numpy as np
from .native_gn_residual_metric import ResidualAwareMetric
from .native_gn_coordinates import dot


class SourceEnrichedMetric(ResidualAwareMetric):
    def __init__(self,base,mask,regularizer,regularizer_diagonal,pairs,
                 data_gradient,residual_energy,features):
        self.source_features=features
        self.source_vectors=[f['vector'] if 'vector' in f else
            np.load(Path(f['image']),mmap_mode='r') for f in features]
        self.source_energy_gram=None
        super().__init__(base,mask,regularizer,regularizer_diagonal,pairs,
            data_gradient,residual_energy)

    def rebuild(self):
        super().rebuild()
        D=[p[0] for p in self.pairs]
        Y=[p[1] for p in self.pairs]
        V=self.source_vectors
        self.source_only_vectors=[]
        if not V:
            self.report.update(source_feature_count=0,source_conditional_modes=0)
            return
        Gactual=np.array([[dot(d,y) for y in Y] for d in D])
        G=.5*(Gactual+Gactual.T)
        C=np.array([[dot(d,v) for v in V] for d in D])
        if self.source_energy_gram is None:
            E=np.zeros((len(V),len(V)))
            for i,a in enumerate(self.source_features):
                for j,b in enumerate(self.source_features):
                    if a['name']==b['name']:
                        E[i,j]=dot(D[a['direction_index']],V[j])
            self.source_energy_gram=.5*(E+E.T)
        E=self.source_energy_gram
        K=E-C.T@np.linalg.solve(G,C)
        K=.5*(K+K.T)
        scales=np.sqrt(np.diag(E))
        if np.any(scales<=0):raise RuntimeError('Nonpositive source feature energy')
        normalized=K/scales[:,None]/scales[None,:]
        eigenvalues,eigenvectors=np.linalg.eigh(normalized)
        maximum=max(float(eigenvalues.max()),1.)
        if eigenvalues.min() < -1e-4*maximum:
            raise RuntimeError('Native conditional source feature Gram lost PSD')
        threshold=1e-6*maximum
        keep=eigenvalues>threshold
        factor=(eigenvectors[:,keep]/np.sqrt(eigenvalues[keep])[None,:])/scales[:,None]
        # The actual stored bilinear matrix removes measured directions to
        # native precision, without amplifying its tiny asymmetry in K modes.
        image_coefficients=np.linalg.solve(Gactual,C)@factor
        for column in range(factor.shape[1]):
            value=np.zeros_like(Y[0],dtype=np.float64)
            for coefficient,field in zip(factor[:,column],V):
                value+=np.multiply(field,coefficient,dtype=np.float64)
            for coefficient,field in zip(image_coefficients[:,column],Y):
                value-=np.multiply(field,coefficient,dtype=np.float64)
            self.source_only_vectors.append(value)
            self.positive.append(value)
            self.diagonal+=value*value
        self.diagonal[:,~self.mask]=1.
        self.report.update(source_feature_count=len(V),
            source_conditional_modes=int(keep.sum()),
            source_conditional_eigenvalue_threshold=threshold,
            source_minimum_conditional_eigenvalue=float(eigenvalues.min()),
            source_maximum_conditional_eigenvalue=float(eigenvalues.max()),
            extra_waves_for_source_information=0,
            source_information_is_true_native_data_not_a_stationarity_assumption=True)
