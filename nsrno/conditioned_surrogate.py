"""Differentiable inference API for the round-7 frequency surrogate.

The current acquisition cache covers three frequencies and three angles only.
This API returns a complex channel spectrum, not a complete RF trace.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from nsrno.conditioned_wave import FrequencyGeometry,ConditionedFrequencySystem,LearnedPreconditioner,differentiable_gmres
from nsrno.conditioned_data import receiver_response,key

class ConditionedSurrogate(nn.Module):
    def __init__(self,checkpoint,reference_root,device='cpu'):
        super().__init__()
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.spec=saved['spec']; self.iterations=saved['iterations']
        self.geometry=FrequencyGeometry(self.spec)
        self.preconditioner=LearnedPreconditioner(); self.preconditioner.load_state_dict(saved['model'])
        root=Path(reference_root); source_spec=json.load(open(root/'spec.json'))
        for field in ('medium_shape','dx_m','native_dt_s','native_nt','source_f0_hz','taus_s','eta_per_unit_alpha','bins','angle_indices'):
            if self.spec[field]!=source_spec[field]: raise ValueError('Acquisition reference mismatch: '+field)
        with np.load(root/'sources.npz') as sources:
            for b in self.spec['bins']:
                for a in self.spec['angle_indices']:
                    name=key(b,a)
                    self.register_buffer(name+'_line',torch.from_numpy(sources[name].copy()),persistent=False)
                    with np.load(root/'reference'/(name+'.npz')) as ref:
                        for field in ('pressure','operator_pressure','source'):
                            self.register_buffer(name+'_'+field,self.geometry.pad_field(torch.from_numpy(ref[field].copy())),persistent=False)
        self.to(device).eval()

    def forward(self,medium,alpha,frequency_bin,angle_index,iterations=None,return_field=False):
        """Native [2,z,x] c/rho, known [z,x] alpha -> [192] complex channels.

        Input gradients flow through material-dependent propagation and all
        unrolled steps. They are gradients of the finite-iteration surrogate,
        not an implicit gradient of the converged physical solution.
        """
        if frequency_bin not in self.spec['bins'] or angle_index not in self.spec['angle_indices']:
            raise ValueError('Generate the matching homogeneous reference before requesting a new frequency/angle')
        n=self.iterations if iterations is None else int(iterations)
        if n<=0: raise ValueError('iterations must be positive')
        frequency=frequency_bin*40e6/2401; name=key(frequency_bin,angle_index)
        system=ConditionedFrequencySystem(self.geometry,medium,alpha,frequency)
        pressure=getattr(self,name+'_pressure')
        rhs=system.source(getattr(self,name+'_line'))-getattr(self,name+'_source')-system(pressure)+getattr(self,name+'_operator_pressure')
        field=differentiable_gmres(system,self.preconditioner.setup(system),rhs,n)
        result={'channels':self.geometry.channels(field)*receiver_response(frequency),
                'relative_residual':(system(field)-rhs).norm()/rhs.norm().clamp_min(1e-30)}
        if return_field: result['native_scattered_field']=self.geometry.native(field)
        return result
