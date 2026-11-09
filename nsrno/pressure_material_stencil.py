"""Nineteen local material columns for the eighth-order scalar pressure row.

The increments are real q, with (delta r, delta beta, delta t) = .1*q.
Pressure is fixed. No inverse, learned approximation or propagation is used.
"""
import numpy as np
import torch
from .first_order_wavefield import relaxation_inverse


FD_COEFFICIENTS = (1225 / 1024, -245 / 3072, 49 / 5120, -5 / 7168)
DENSITY_OFFSETS = (tuple((0, d) for d in range(-4, 5))
                   + tuple((d, 0) for d in range(-4, 5) if d))
OFFSETS = DENSITY_OFFSETS + ((0, 0), (0, 0))
FIELDS = (0,) * 17 + (1, 2)
PAIRS = tuple((i, j) for i in range(19) for j in range(i, 19))


def sample_offset(value, dz, dx):
    """At row i return value[i+(dz,dx)], with periodic padded FFT indexing."""
    return torch.roll(value, shifts=(-dz, -dx), dims=(-2, -1))


def material_indices(geometry):
    """Fold crop-and-replicate coefficient ghosts to original native nodes."""
    g = geometry
    z = np.clip(np.arange(g.H) - g.pz, g.pml, g.h - g.pml - 1)
    x = np.clip(np.arange(g.W) - g.px, g.pml, g.w - g.pml - 1)
    padded = z[:, None] * g.w + x[None, :]
    return np.stack([
        np.roll(padded, (-dz, -dx), (-2, -1))[g.pz:g.pz + g.h, g.px:g.px + g.w]
        for dz, dx in OFFSETS]).astype(np.int64)


class PressureMaterialStencil:
    @torch.no_grad()
    def __init__(self, operator, total_pressure, unknown=None):
        self.op = operator
        g = operator.g
        if total_pressure.ndim != 3:
            raise ValueError('Expected a batch of fixed total pressures [TX,H,W]')
        device = total_pressure.device
        self.index = torch.from_numpy(material_indices(g)).to(device)
        self.fields = torch.tensor(FIELDS, dtype=torch.long, device=device)
        if unknown is None:
            unknown = np.broadcast_to(np.arange(g.h)[:, None] >= 80, (g.h, g.w)).copy()
        self.unknown = torch.as_tensor(unknown, dtype=torch.bool, device=device)
        if self.unknown.shape != (g.h, g.w):
            raise ValueError('Wrong native unknown mask')
        self.coefficients = torch.zeros((19, len(total_pressure), g.h, g.w),
                                         dtype=total_pressure.dtype, device=device)
        p = total_pressure * g.mask
        for axis in range(2):
            face = operator.Ix if axis == 0 else operator.Iz
            a = operator.system.ax if axis == 0 else operator.system.az
            velocity = -face * operator.gradient(p, axis)
            coefficient = -.05 * a * velocity
            left = operator.left_x if axis == 0 else operator.left_z
            right = operator.right_x if axis == 0 else operator.right_z
            gain = operator.system.gvx if axis == 0 else operator.system.gvz
            for j, weight in enumerate(FD_COEFFICIENTS):
                # Gradient symbol: +weight at +(j+1), -weight at -j.
                # Divergence is its negative adjoint, hence offsets -d.
                for d, grad_weight in ((j + 1, weight / g.dx), (-j, -weight / g.dx)):
                    o = -d
                    dz, dx = (0, o) if axis == 0 else (o, 0)
                    term = (g.dt * 1540 * (-grad_weight)
                            * (left + right * sample_offset(gain, dz, dx))
                            * sample_offset(coefficient, dz, dx) * g.mask)
                    term = g.native(term)
                    second = (dz, dx + 1) if axis == 0 else (dz + 1, dx)
                    self.coefficients[DENSITY_OFFSETS.index((dz, dx))].add_(term)
                    self.coefficients[DENSITY_OFFSETS.index(second)].add_(term)
        s = operator.system
        t = s.alpha * s.c / 1540
        beta = s.a * (1540 / s.c).square()
        M, Mp = relaxation_inverse(s.frequency, g.dt, g.spec['taus_s'],
                                   g.spec['eta_per_unit_alpha'], t)
        self.coefficients[17].copy_(g.native(.1 * M * p * g.mask))
        self.coefficients[18].copy_(g.native(.1 * beta * Mp * p * g.mask))

    def gather(self, q):
        q = q * self.unknown
        return q.flatten(1)[self.fields[:, None, None], self.index]

    def apply(self, q):
        return torch.einsum('sthw,shw->thw', self.coefficients, self.gather(q).to(self.coefficients.dtype))

    def fold(self, local):
        g = self.op.g
        out = torch.zeros((3, g.h * g.w), dtype=local.dtype, device=local.device)
        for s, field in enumerate(FIELDS):
            out[field].scatter_add_(0, self.index[s].flatten(), local[s].flatten())
        return out.reshape(3, g.h, g.w) * self.unknown

    def adjoint(self, residual):
        local = (self.coefficients.conj() * residual[None]).real.sum(1, dtype=torch.float64)
        return self.fold(local)

    @torch.no_grad()
    def accumulate(self, residual, scale, packed_gram, local_right):
        """Accumulate J*J and -J*e, including cross-parameter terms.

        Output is row-local; the CPU normal folds coherent duplicate ghosts.
        Scale is lambda/(group energy * number of acquisition groups).
        """
        K = self.coefficients
        if residual.shape != K.shape[1:]:
            raise ValueError('Expected native residual [TX,h,w]')
        for i in range(19):
            local_right[i].add_(-(K[i].conj() * residual).real.sum(0, dtype=torch.float64), alpha=scale)
        for slot, (i, j) in enumerate(PAIRS):
            packed_gram[slot].add_((K[i].conj() * K[j]).real.sum(0, dtype=torch.float64), alpha=scale)
