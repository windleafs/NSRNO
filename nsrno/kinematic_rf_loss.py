"""CPU adapter around unchanged, existing native-RF kinematic objectives.

This comparison changes physical c/rho paths, not objective hyperparameters.
"""
from types import SimpleNamespace
import math
import numpy as np
import torch
from .adaptive_rf_loss import AdaptiveMatchingLoss
from .differential_phase_loss import DifferentialPhaseLoss


class KinematicRFLoss:
    def __init__(self,observed,depth_bins,depth_weights,source):
        self.observed=torch.as_tensor(observed,dtype=torch.float32,device='cpu')
        if tuple(self.observed.shape[:2])!=(11,192):
            raise ValueError('All eleven transmissions and192 receivers required')
        bins=torch.as_tensor(depth_bins,dtype=torch.int8,device='cpu')
        acquisition=SimpleNamespace(observed=self.observed,source=source)
        self.adaptive=AdaptiveMatchingLoss(acquisition,bins=bins,sigma_us=1.5,
            stabilization=.1,taper_samples=17,filter_energy_floor=1e-3)
        self.differential=DifferentialPhaseLoss(acquisition,self.adaptive,lag_bins=16,spectral_floor=1e-3)
        self.weights=torch.as_tensor(depth_weights,dtype=torch.float32)
        self.actual_difference_hz=self.differential.frequency_difference_hz

    def shot(self,prediction,shot):
        awi=self.adaptive.terms(prediction,shot)
        differential=self.differential.terms(prediction,shot)
        return {'AWI_geometry':sum(w*v for w,v in zip(self.weights,awi)),
                'frequency_difference_geometry':sum(w*v for w,v in zip(self.weights,differential))}

    def evaluate(self,predictions):
        if isinstance(predictions,np.ndarray) and not predictions.flags.writeable:
            predictions=np.array(predictions)
        predictions=torch.as_tensor(predictions,dtype=torch.float32,device='cpu')
        if predictions.shape!=self.observed.shape:raise ValueError('RF prediction shape mismatch')
        values={}
        with torch.no_grad():
            for shot in range(11):
                for name,value in self.shot(predictions[shot],shot).items():
                    values[name]=values.get(name,0.)+float(value)
            error=float((predictions-self.observed).double().square().sum())
            energy=float(self.observed.double().square().sum())
        values['RF_relative_RMS']=math.sqrt(error/energy)
        return values
