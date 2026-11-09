"""Exact anchored integration of a two-dimensional discrete gradient.

Unknown rows start immediately below a known zero-correction coupling row.
No lateral/bottom truth or boundary values are supplied. The adjoint includes
the predicted edge flux; the normal equation does not set true side flux to zero.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class AnchoredGradientIntegration(nn.Module):
    def __init__(self, height=211, width=192):
        super().__init__()
        if height < 1 or width < 1:
            raise ValueError('A nonempty unknown material rectangle is required')
        self.height, self.width = height, width
        dtype = torch.float64
        z = torch.arange(1, height + 1, dtype=dtype)[:, None]
        kz = torch.arange(height, dtype=dtype)[None]
        theta_z = math.pi * (2 * kz + 1) / (2 * height + 1)
        uz = (4. / (2 * height + 1))**.5 * (z * theta_z).sin()
        x = (torch.arange(width, dtype=dtype) + .5)[:, None]
        kx = torch.arange(width, dtype=dtype)[None]
        theta_x = math.pi * kx / width
        ux = (2. / width)**.5 * (x * theta_x).cos()
        ux[:, 0] /= math.sqrt(2.)
        eigenvalues = (4 * (theta_z[0] / 2).sin().square()[:, None] +
                       4 * (theta_x[0] / 2).sin().square()[None])
        self.register_buffer('vertical_basis', uz)
        self.register_buffer('horizontal_basis', ux)
        self.register_buffer('eigenvalues', eigenvalues)
        self.config = {
            'operator': '(G_adjoint G)^-1 G_adjoint',
            'known_boundary': 'zero update at the adjacent coupling row only',
            'unknown_side_and_bottom_truth_supplied': False,
            'gradient_units': 'scaled material change per grid pixel',
            'separable_mixed_boundary_graph_eigensolve': True,
            'internal_precision': 'float64',
            'field_representation_is_not_restricted_to_smooth_functions': True,
            'height': height, 'width': width,
        }

    def gradient(self, field):
        if field.shape[-2:] != (self.height, self.width):
            raise ValueError('Unknown-field shape does not match the integration grid')
        lateral = field[..., 1:] - field[..., :-1]
        vertical = torch.cat((field[..., :1, :],
                              field[..., 1:, :] - field[..., :-1, :]), -2)
        return lateral, vertical

    def adjoint(self, lateral, vertical):
        if lateral.shape[-2:] != (self.height, self.width - 1):
            raise ValueError('Lateral edge shape is invalid')
        if vertical.shape[-2:] != (self.height, self.width):
            raise ValueError('Vertical edges must include the upper coupling anchor')
        horizontal = F.pad(lateral, (1, 0)) - F.pad(lateral, (0, 1))
        axial = vertical - F.pad(vertical[..., 1:, :], (0, 0, 0, 1))
        return horizontal + axial

    def solve(self, right_hand_side):
        value = right_hand_side.to(torch.float64)
        modes = self.vertical_basis.T @ value @ self.horizontal_basis
        return self.vertical_basis @ (modes / self.eigenvalues) @ self.horizontal_basis.T

    def forward(self, lateral, vertical):
        dtype = vertical.dtype
        # Form G* in double precision too; the lowest eigenvalue is small.
        rhs = self.adjoint(lateral.to(torch.float64), vertical.to(torch.float64))
        return self.solve(rhs).to(dtype)
