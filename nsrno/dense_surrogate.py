"""Public dense-band conditional forward interface, independent of Geo-Flow."""
from pathlib import Path
import torch
from torch import nn
from nsrno.dense_wave import DenseData,implicit_channels,solve_system,channels,batched_gmres_cycle
from nsrno.conditioned_wave import LearnedPreconditioner
from nsrno.conditioned_data import receiver_response

class DenseSurrogate(nn.Module):
    def __init__(self,checkpoint='out/v5/forward_round8_dense_full/last.pt',
                 reference_root='out/v5/conditioned_dense_full_v1',device='cuda'):
        super().__init__(); saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.spec=saved['spec']; self.data=DenseData(device=device,reference_limit=12,root=reference_root)
        if self.spec['bins']!=self.data.spec['bins'] or self.spec['angles_deg']!=self.data.spec['angles_deg']:
            raise ValueError('Checkpoint and acquisition references disagree')
        self.preconditioner=LearnedPreconditioner().to(device); self.preconditioner.load_state_dict(saved['model'])
        self.preconditioner.requires_grad_(False); self.eval()

    def forward(self,medium,alpha,frequency_bin,angles=None,tolerance=1e-5):
        """Native c/rho + known alpha -> complex channel spectrum.

        Derivatives are implicit gradients of the converged physical equation.
        Acquisition-only reference fields are fixed; material targets are never
        read. One or more of the11 available angles may be requested.
        """
        if frequency_bin not in self.spec['bins']: raise ValueError('Frequency outside cached dense band')
        angles=list(range(11)) if angles is None else list(angles)
        return implicit_channels(medium,alpha,self.data,frequency_bin,angles,self.preconditioner,tolerance)

    @torch.no_grad()
    def truncated(self,medium,alpha,frequency_bin,angles=None,iterations=32):
        """Return the finite-depth neural result and its actual PDE residual."""
        angles=list(range(11)) if angles is None else list(angles)
        system,rhs,_,_=self.data.assemble(medium,alpha,frequency_bin,angles)
        value=batched_gmres_cycle(system,self.preconditioner.setup(system),rhs,iterations)
        return channels(self.data.geometry,value)*receiver_response(system.frequency),float((system(value)-rhs).norm()/rhs.norm())
