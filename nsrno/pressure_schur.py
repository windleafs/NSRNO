"""Exact scalar Schur equation of the native first-order frequency system.

Only the lifted residual metric changes. Physical coefficients, discrete
time relaxation, PML update order, receiver channels and source are retained.
No pressure or material truth is accepted by this module.
"""
import torch
from .first_order_wavefield import FirstOrderWavefield


class PressureSchur(FirstOrderWavefield):
    def __call__(self, pressure):
        mask = self.g.mask
        p = pressure * mask
        value = (self.system.mass * p
                 - self.divergence(self.Ix * self.gradient(p, 0), 0)
                 - self.divergence(self.Iz * self.gradient(p, 1), 1))
        return mask * value + (1 - mask) * pressure

    def adjoint(self, pressure):
        mask = self.g.mask
        z = pressure * mask
        value = (self.system.mass.conj() * z
                 - self.gradient(self.Ix.conj() * self.divergence(z, 0, True), 0, True)
                 - self.gradient(self.Iz.conj() * self.divergence(z, 1, True), 1, True))
        return mask * value + (1 - mask) * pressure

    def scattering_rhs(self, reference_operator, reference_pressure):
        p = reference_pressure * self.g.mask
        return self.g.mask * (
            -(self.system.mass - reference_operator.system.mass) * p
            + self.divergence((self.Ix - reference_operator.Ix) * self.gradient(p, 0), 0)
            + self.divergence((self.Iz - reference_operator.Iz) * self.gradient(p, 1), 1))

    def material_change(self, old_operator, total_pressure):
        """(S_new-S_old)P at fixed pressure, without cancellation of S P."""
        p = total_pressure * self.g.mask
        return self.g.mask * (
            (self.system.mass - old_operator.system.mass) * p
            - self.divergence((self.Ix - old_operator.Ix) * self.gradient(p, 0), 0)
            - self.divergence((self.Iz - old_operator.Iz) * self.gradient(p, 1), 1))

    def readout(self, pressure, response):
        return pressure[..., self.g.face, self.g.rx_cols] * response

    def readout_adjoint(self, receivers, response):
        value = torch.zeros((*receivers.shape[:-1], self.g.H, self.g.W),
                            dtype=receivers.dtype, device=receivers.device)
        value[..., self.g.face, self.g.rx_cols] = receivers * complex(response).conjugate()
        return value
