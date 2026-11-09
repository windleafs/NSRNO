"""Augment native secants with the already available residual feature.

With g=J'Wr and E=r'Wr=2F0, H_data >= gg'/E. Adding r to the data-space
sketch preserves this inequality and the nonnegative linearized data loss.
No new wave or adjoint is needed: g and E were computed at the reference.
"""
import numpy as np
from .native_gn_multisecant import MultisecantMetric
from .native_gn_coordinates import dot


class ResidualAwareMetric(MultisecantMetric):
    def __init__(self,base,mask,regularizer,regularizer_diagonal,pairs,
                 data_gradient,residual_energy):
        self.data_gradient=np.asarray(data_gradient,np.float64)
        self.residual_energy=float(residual_energy)
        if self.residual_energy<=0:raise ValueError('Positive reference residual energy required')
        super().__init__(base,mask,regularizer,regularizer_diagonal,pairs)

    def rebuild(self):
        super().rebuild()
        directions=[p[0] for p in self.pairs]
        images=[p[1] for p in self.pairs]
        G=np.array([[dot(d,y) for y in images] for d in directions])
        G=.5*(G+G.T)
        b=np.array([dot(d,self.data_gradient) for d in directions])
        coefficients=np.linalg.solve(G,b)
        energy=self.residual_energy-float(b@coefficients)
        if energy<=self.residual_energy*1e-10:
            raise RuntimeError('Residual sketch needs dependent-feature treatment')
        remaining=self.data_gradient.copy()
        for coefficient,image in zip(coefficients,images):remaining-=coefficient*image
        feature=remaining/np.sqrt(energy)
        self.positive.append(feature)
        self.diagonal+=feature*feature
        self.diagonal[:,~self.mask]=1.
        self.report.update(residual_feature_included=True,
            reference_residual_energy=self.residual_energy,
            residual_energy_after_saved_span=energy,
            data_loss_nonnegative_by_augmented_sketch=True,
            additional_waves_for_residual_feature=0)
