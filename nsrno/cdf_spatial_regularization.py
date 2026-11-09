"""Independent global CDF term added to the frozen neural spatial prior."""
import numpy as np
from .global_cdf_prior import GlobalCDFPrior
from .learned_spatial_energy import LearnedSpatialEnergy


class CDFSpatialRegularization:
    def __init__(self,neural_path,cdf_path):
        self.neural=LearnedSpatialEnergy(neural_path)
        self.cdf=GlobalCDFPrior(cdf_path)

    def evaluate(self,coordinates,control,gradient=True):
        neural,gn,pn=self.neural.evaluate(coordinates,control,gradient=gradient)
        cdf,gc,pc=self.cdf.evaluate(coordinates,control,gradient=gradient)
        derivative=gn+gc if gradient else None
        if gradient and not np.isfinite(derivative).all():
            raise RuntimeError('Nonfinite actual composite prior gradient')
        return neural+cdf,derivative,{'neural_energy':pn['neural_energy'],
            'global_CDF':cdf,'TV':pn['TV'],'global_CDF_details':pc,
            'neural_details':pn,'all_fixed_coefficients_one':True}


def objectives(plan):
    full=plan['families']
    low={name:{**spec,'bands_hz':[[.20e6,.325e6]],'coefficient':.25}
         for name,spec in full.items() if spec['centre']=='low'}
    if len(low)!=2 or len(full)!=4:
        raise RuntimeError('Fixed two low and four full acquisition families required')
    return {key:{'families':families,'sources':[[name,shot] for name in families for shot in range(11)]}
            for key,families in [('low_background',low),('full_joint',full)]}
