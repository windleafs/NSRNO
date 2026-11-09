"""Exact differentiable coordinates used by Hesse et al. (2013)."""
import torch

C0=1540.
RHO0=1000.
KAPPA0=1/(RHO0*C0**2)

def relative_parameters(medium):
    """[B,2,z,x] c,rho in SI -> gamma_kappa,gamma_rho (dimensionless)."""
    c,rho=medium.unbind(1)
    a=RHO0/rho
    return torch.stack((a*(C0/c).square()-1,1-a),1)

def medium_from_relative(gamma):
    """Inverse map; require gamma_kappa>-1 and gamma_rho<1."""
    gamma_kappa,gamma_rho=gamma.unbind(1)
    a=1-gamma_rho; b=1+gamma_kappa
    return torch.stack((C0*torch.sqrt(a/b),RHO0/a),1)
