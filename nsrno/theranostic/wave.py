"""Small 2-D variable-density Helmholtz reference and its Born derivative.

This is a frequency-domain *linear-acoustic* solver.  Solving a heterogeneous
medium retains repeated scattering; it does not model harmonic generation,
elastic shear waves, heating, or the mechanical stack of a therapy probe.

With time dependence exp(-i omega t), the normalized equation is

    div((1 + m_eta) grad(p)) + k0**2 (1 + m_kappa) p = -source,

where m_eta = eta/eta0 - 1 and m_kappa = kappa/kappa0 - 1.  A complex coordinate
stretch on the outer grid implements a quadratic PML.  The stretched equation
is multiplied by sx*sz, giving a complex-symmetric flux discretization.  The
outer ghost values are zero.  The PML is a numerical boundary treatment, not
an estimate of tissue attenuation.

Coordinates exposed by this module use mm, frequency MHz, and sound speed m/s.
All differential operators and quadrature factors use SI units internally.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class Grid:
    """Uniform square grid including its PML; arrays use [z, x] order."""

    x_mm: np.ndarray
    z_mm: np.ndarray
    pml_mm: float

    def __post_init__(self):
        for name in ("x_mm", "z_mm"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
                raise ValueError(f"{name} must be a finite one-dimensional axis with >=3 nodes")
            if np.any(np.diff(values) <= 0):
                raise ValueError(f"{name} must be strictly increasing")
            if not np.allclose(np.diff(values), np.diff(values)[0], rtol=1e-10, atol=1e-10):
                raise ValueError(f"{name} must be uniformly sampled")
            values = values.copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if not np.isclose(np.diff(self.x_mm)[0], np.diff(self.z_mm)[0], rtol=1e-10, atol=1e-10):
            raise ValueError("The Helmholtz grid must have equal x and z spacing")
        if not np.isfinite(self.pml_mm) or self.pml_mm < 0:
            raise ValueError("pml_mm must be finite and nonnegative")
        if 2 * self.pml_mm >= min(np.ptp(self.x_mm), np.ptp(self.z_mm)):
            raise ValueError("PML leaves no physical domain")

    @property
    def shape(self):
        return (self.z_mm.size, self.x_mm.size)

    @property
    def size(self):
        return self.x_mm.size * self.z_mm.size

    @property
    def h_mm(self):
        return float(self.x_mm[1] - self.x_mm[0])

    @property
    def h_m(self):
        return self.h_mm * 1e-3

    @property
    def physical_bounds_mm(self):
        return (
            self.x_mm[0] + self.pml_mm,
            self.x_mm[-1] - self.pml_mm,
            self.z_mm[0] + self.pml_mm,
            self.z_mm[-1] - self.pml_mm,
        )

    @property
    def pml_mask(self):
        x0, x1, z0, z1 = self.physical_bounds_mm
        return ((self.x_mm[None, :] < x0 - 1e-9)
                | (self.x_mm[None, :] > x1 + 1e-9)
                | (self.z_mm[:, None] < z0 - 1e-9)
                | (self.z_mm[:, None] > z1 + 1e-9))


class Solver:
    """Sparse flux-form Helmholtz solver on a known homogeneous exterior.

    ``pml_strength`` defines sigma_max = pml_strength*c0/PML_thickness.
    The stretch is s = 1 + i*sigma/omega with a quadratic spatial profile.
    Matrix factorization is shared by all source right-hand sides in a call.
    No dense inverse of the wave operator is formed.
    """

    def __init__(self, grid, c0_m_s=1540.0, pml_strength=12.0):
        self.grid = grid
        self.c0_m_s = float(c0_m_s)
        self.pml_strength = float(pml_strength)
        if not np.isfinite(self.c0_m_s) or self.c0_m_s <= 0:
            raise ValueError("c0_m_s must be finite and positive")
        if not np.isfinite(self.pml_strength) or self.pml_strength <= 0:
            raise ValueError("pml_strength must be finite and positive")
        nz, nx = grid.shape
        nodes = np.arange(grid.size).reshape(grid.shape)
        # The first two edge groups join grid nodes; the last four terminate
        # at a zero-valued exterior ghost.  Ghost material contrast is zero.
        horizontal_i, horizontal_j = nodes[:, :-1].ravel(), nodes[:, 1:].ravel()
        vertical_i, vertical_j = nodes[:-1, :].ravel(), nodes[1:, :].ravel()
        boundaries = np.concatenate((nodes[:, 0], nodes[:, -1], nodes[0], nodes[-1]))
        self.edge_i = np.concatenate((horizontal_i, vertical_i, boundaries))
        self.edge_j = np.concatenate((horizontal_j, vertical_j, -np.ones(boundaries.size, dtype=int)))
        self._n_horizontal = horizontal_i.size
        self._n_vertical = vertical_i.size
        self._interior_edges = self.edge_j >= 0
        valid = self._interior_edges
        rows = np.concatenate((np.arange(self.edge_i.size), np.flatnonzero(valid)))
        columns = np.concatenate((self.edge_i, self.edge_j[valid]))
        self.edge_average = sparse.csr_matrix(
            (np.full(rows.size, 0.5), (rows, columns)),
            shape=(self.edge_i.size, grid.size),
        )

    def _stretch(self, coordinates, lower, upper, omega):
        if self.grid.pml_mm == 0:
            return np.ones_like(coordinates, dtype=np.complex128)
        width = self.grid.pml_mm
        fraction = np.maximum((lower + width - coordinates) / width,
                              (coordinates - upper + width) / width)
        profile = np.clip(fraction, 0.0, 1.0)**2
        sigma_max = self.pml_strength * self.c0_m_s / (width * 1e-3)
        return 1.0 + 1j * (sigma_max / omega) * profile

    def coefficients(self, frequency_mhz):
        """Return (k0, stretched cell mass, stretched edge flux weights)."""
        if not np.isfinite(frequency_mhz) or frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be finite and positive")
        omega = 2 * np.pi * frequency_mhz * 1e6
        k0 = omega / self.c0_m_s
        x, z, h = self.grid.x_mm, self.grid.z_mm, self.grid.h_mm
        sx = self._stretch(x, x[0], x[-1], omega)
        sz = self._stretch(z, z[0], z[-1], omega)
        sxh = self._stretch((x[:-1] + x[1:]) / 2, x[0], x[-1], omega)
        szh = self._stretch((z[:-1] + z[1:]) / 2, z[0], z[-1], omega)
        sx_outer = self._stretch(np.array([x[0] - h/2, x[-1] + h/2]), x[0], x[-1], omega)
        sz_outer = self._stretch(np.array([z[0] - h/2, z[-1] + h/2]), z[0], z[-1], omega)
        edge_stretch = np.concatenate((
            (sz[:, None] / sxh[None, :]).ravel(),
            (sx[None, :] / szh[:, None]).ravel(),
            sz / sx_outer[0], sz / sx_outer[1],
            sx / sz_outer[0], sx / sz_outer[1],
        ))
        return k0, (sz[:, None] * sx[None, :]).ravel(), edge_stretch

    def _contrast(self, value, name):
        if value is None:
            return np.zeros(self.grid.size)
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real")
        values = np.asarray(raw, dtype=np.float64)
        if values.shape not in (self.grid.shape, (self.grid.size,)):
            raise ValueError(f"{name} shape must be {self.grid.shape} or ({self.grid.size},)")
        values = values.ravel()
        if not np.isfinite(values).all() or np.any(values <= -1):
            raise ValueError(f"{name} must be finite and > -1")
        if np.any(np.abs(values[self.grid.pml_mask.ravel()]) > 1e-12):
            raise ValueError(f"{name} must vanish in the PML")
        return values

    def matrix(self, frequency_mhz, m_kappa=None, m_eta=None):
        """Assemble the complex-symmetric, second-order flux matrix in SI."""
        mk = self._contrast(m_kappa, "m_kappa")
        me = self._contrast(m_eta, "m_eta")
        k0, cell_mass, stretch = self.coefficients(frequency_mhz)
        flux = stretch * (1.0 + self.edge_average @ me) / self.grid.h_m**2
        diagonal = k0**2 * cell_mass * (1.0 + mk)
        np.add.at(diagonal, self.edge_i, -flux)
        valid = self._interior_edges
        np.add.at(diagonal, self.edge_j[valid], -flux[valid])
        nodes = np.arange(self.grid.size)
        row = np.concatenate((nodes, self.edge_i[valid], self.edge_j[valid]))
        column = np.concatenate((nodes, self.edge_j[valid], self.edge_i[valid]))
        values = np.concatenate((diagonal, flux[valid], flux[valid]))
        return sparse.csc_matrix((values, (row, column)), shape=(self.grid.size, self.grid.size))

    def source_weights(self, positions_mm, normals, widths_mm, quadrature=5):
        """Reciprocal normalized finite-line transducer sampling weights.

        Every source has unit integrated source strength.  This does not mean
        equal radiated energy.  Width is measured along the tangent to the
        supplied normal; Gauss-Legendre quadrature and bilinear interpolation
        resolve that line.  The elements radiate to both sides: no rigid baffle
        or metal housing is modeled.  A width of zero is an ideal point element.
        """
        positions = np.asarray(positions_mm, dtype=np.float64)
        normal = np.asarray(normals, dtype=np.float64)
        widths = np.asarray(widths_mm, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2 or not positions.shape[0]:
            raise ValueError("positions_mm must have shape [n_elements, 2] in [x,z] order")
        if normal.shape != positions.shape or widths.shape != (positions.shape[0],):
            raise ValueError("normals or widths_mm have inconsistent element dimensions")
        if not (np.isfinite(positions).all() and np.isfinite(normal).all() and np.isfinite(widths).all()):
            raise ValueError("Element geometry must be finite")
        normal_length = np.linalg.norm(normal, axis=1)
        if np.any(normal_length <= 0) or np.any(widths < 0):
            raise ValueError("Normals must be nonzero and element widths nonnegative")
        if not isinstance(quadrature, (int, np.integer)) or quadrature < 1:
            raise ValueError("quadrature must be a positive integer")
        normal = normal / normal_length[:, None]
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
        nodes, weights = np.polynomial.legendre.leggauss(quadrature)
        points = positions[:, None, :] + tangent[:, None, :] * widths[:, None, None] * nodes[None, :, None] / 2
        x, z = points[..., 0], points[..., 1]
        x0, x1, z0, z1 = self.grid.physical_bounds_mm
        if np.any((x < x0 - 1e-9) | (x > x1 + 1e-9) | (z < z0 - 1e-9) | (z > z1 + 1e-9)):
            raise ValueError("All transducer quadrature points must be outside the PML")
        if np.any((x <= self.grid.x_mm[0]) | (x >= self.grid.x_mm[-1])
                  | (z <= self.grid.z_mm[0]) | (z >= self.grid.z_mm[-1])):
            raise ValueError("Transducers must be strictly inside the computational grid")
        fx = (x - self.grid.x_mm[0]) / self.grid.h_mm
        fz = (z - self.grid.z_mm[0]) / self.grid.h_mm
        ix, iz = np.floor(fx).astype(int), np.floor(fz).astype(int)
        wx, wz = fx - ix, fz - iz
        elements = np.broadcast_to(np.arange(positions.shape[0])[:, None], x.shape).ravel()
        rows, columns, entries = [], [], []
        for dx, dz, coefficient in ((0, 0, (1-wx)*(1-wz)), (1, 0, wx*(1-wz)),
                                    (0, 1, (1-wx)*wz), (1, 1, wx*wz)):
            rows.append(((iz + dz)*self.grid.shape[1] + ix + dx).ravel())
            columns.append(elements)
            entries.append((coefficient * weights[None, :] / 2).ravel())
        result = sparse.csc_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))),
                                   shape=(self.grid.size, positions.shape[0]))
        result.eliminate_zeros()
        return result

    def solve(self, frequency_mhz, source_weights, m_kappa=None, m_eta=None,
              *, return_diagnostics=False, source_chunk=8):
        """Solve all source fields, reusing one sparse LU for the frequency."""
        if np.iscomplexobj(source_weights):
            raise ValueError("Reciprocal aperture weights must be real; apply transmit encoding to solved port fields")
        weights = sparse.csc_matrix(source_weights, dtype=np.float64)
        if weights.shape[0] != self.grid.size or weights.shape[1] == 0:
            raise ValueError("source_weights has incompatible shape")
        if not isinstance(source_chunk, (int, np.integer)) or source_chunk < 1:
            raise ValueError("source_chunk must be a positive integer")
        if not np.isfinite(weights.data).all():
            raise ValueError("source_weights must be finite")
        started = perf_counter()
        matrix = self.matrix(frequency_mhz, m_kappa, m_eta)
        lu = splu(matrix)
        factor_seconds = perf_counter() - started
        fields = np.empty(weights.shape, dtype=np.complex128)
        error_squared, rhs_squared = 0.0, 0.0
        for first in range(0, weights.shape[1], source_chunk):
            end = min(first + source_chunk, weights.shape[1])
            rhs = -weights[:, first:end].toarray().astype(np.complex128) / self.grid.h_m**2
            values = lu.solve(rhs)
            fields[:, first:end] = values
            if return_diagnostics:
                error_squared += float(np.linalg.norm(matrix @ values - rhs)**2)
                rhs_squared += float(np.linalg.norm(rhs)**2)
        if not return_diagnostics:
            return fields
        diagnostics = dict(grid_shape=list(self.grid.shape), nodes=self.grid.size,
                           frequency_mhz=float(frequency_mhz), sources=weights.shape[1],
                           relative_linear_residual=float(np.sqrt(error_squared / max(rhs_squared, np.finfo(float).tiny))),
                           factor_seconds=float(factor_seconds), elapsed_seconds=float(perf_counter() - started),
                           lu_nonzeros=int(lu.L.nnz + lu.U.nnz))
        return fields, diagnostics

    @staticmethod
    def sample(fields, receiver_weights):
        """Apply the reciprocal aperture average; return [transmitter, receiver]."""
        return np.asarray(receiver_weights.T @ fields).T

    def born_data(self, frequency_mhz, source_weights, m_kappa, m_eta,
                  background_fields=None, *, return_diagnostics=False, source_chunk=8):
        """Predict Born data for arbitrary nodal contrasts without an inverse basis.

        This evaluates ``delta_u = -A0**(-1) delta_A u0``, including the
        inverse-density face flux.  ``delta_A`` is exactly linear in the two
        normalized nodal contrasts.  A supplied ``background_fields`` must
        use this grid, frequency, and the same homogeneous source weights.
        The output is reference-subtracted data in [tx,rx] order.
        """
        if np.iscomplexobj(source_weights):
            raise ValueError("Reciprocal aperture weights must be real; apply transmit encoding to solved port fields")
        weights = sparse.csc_matrix(source_weights, dtype=np.float64)
        if weights.shape[0] != self.grid.size or weights.shape[1] == 0:
            raise ValueError("source_weights has incompatible shape")
        if not isinstance(source_chunk, (int, np.integer)) or source_chunk < 1:
            raise ValueError("source_chunk must be a positive integer")
        if not np.isfinite(weights.data).all():
            raise ValueError("source_weights must be finite")
        started = perf_counter()
        background = self.matrix(frequency_mhz)
        perturbation = self.matrix(frequency_mhz, m_kappa, m_eta) - background
        lu = splu(background)
        factor_seconds = perf_counter() - started
        if background_fields is not None:
            background_fields = np.asarray(background_fields, dtype=np.complex128)
            if background_fields.shape != weights.shape or not np.isfinite(background_fields).all():
                raise ValueError("background_fields has incompatible shape or nonfinite values")
        result = np.empty((weights.shape[1], weights.shape[1]), dtype=np.complex128)
        error_squared, rhs_squared = 0.0, 0.0
        for first in range(0, weights.shape[1], source_chunk):
            end = min(first + source_chunk, weights.shape[1])
            if background_fields is None:
                source = -weights[:, first:end].toarray().astype(np.complex128) / self.grid.h_m**2
                incident = lu.solve(source)
            else:
                incident = background_fields[:, first:end]
            rhs = -(perturbation @ incident)
            scattered = lu.solve(rhs)
            result[first:end] = self.sample(scattered, weights)
            if return_diagnostics:
                error_squared += float(np.linalg.norm(background @ scattered - rhs)**2)
                rhs_squared += float(np.linalg.norm(rhs)**2)
        if not return_diagnostics:
            return result
        diagnostics = dict(grid_shape=list(self.grid.shape), frequency_mhz=float(frequency_mhz),
                           sources=weights.shape[1],
                           relative_linear_residual=float(np.sqrt(error_squared / max(rhs_squared, np.finfo(float).tiny))),
                           factor_seconds=float(factor_seconds), elapsed_seconds=float(perf_counter() - started))
        return result, diagnostics

    def born_jacobian(self, frequency_mhz, background_fields, source_indices,
                      receiver_indices, basis, *, pair_chunk=16):
        """Return the exact derivative of this discretization at zero contrast.

        ``basis`` maps each real material coefficient field to grid-node values.
        Output ordering is [tx,rx] flattened, with columns [m_kappa, m_eta].
        ``background_fields`` must be homogeneous fields excited by the same
        unit-strength sampling weights used for source and reciprocal receiver
        elements.  No complex conjugate enters the reciprocal field product;
        inversion of real parameters must nevertheless use Re(J.conj().T r).

        Only nodes/edges touched by the basis participate in the products.
        Temporary arrays are bounded by ``pair_chunk`` times that support.
        """
        fields = np.asarray(background_fields, dtype=np.complex128)
        if fields.ndim != 2 or fields.shape[0] != self.grid.size:
            raise ValueError("background_fields has incompatible shape")
        indices = []
        for value in (source_indices, receiver_indices):
            index = np.asarray(value)
            if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer) or index.size == 0:
                raise ValueError("Source/receiver indices must be nonempty integer vectors")
            if np.any(index < 0) or np.any(index >= fields.shape[1]):
                raise ValueError("Source/receiver index out of bounds")
            indices.append(index.astype(int))
        transmitters, receivers = indices
        if not isinstance(pair_chunk, (int, np.integer)) or pair_chunk < 1:
            raise ValueError("pair_chunk must be a positive integer")
        if np.iscomplexobj(basis):
            raise ValueError("Material basis must be real")
        basis = sparse.csr_matrix(basis, dtype=np.float64)
        basis.eliminate_zeros()
        if basis.shape[0] != self.grid.size or basis.shape[1] == 0 or not np.isfinite(basis.data).all():
            raise ValueError("Material basis has incompatible shape or nonfinite entries")
        active_nodes = np.flatnonzero(np.diff(basis.indptr))
        if np.any(self.grid.pml_mask.ravel()[active_nodes]):
            raise ValueError("Material basis must vanish in the PML")
        edge_basis = (self.edge_average @ basis).tocsr()
        edge_basis.eliminate_zeros()
        active_edges = np.flatnonzero(np.diff(edge_basis.indptr))
        k0, cell_mass, stretch = self.coefficients(frequency_mhz)
        mass_basis = (sparse.diags(k0**2 * cell_mass[active_nodes]) @ basis[active_nodes]).tocsr()
        flux_basis = (sparse.diags(-stretch[active_edges]) @ edge_basis[active_edges]).tocsr()
        gradients = -fields[self.edge_i[active_edges]].copy()
        neighbours = self.edge_j[active_edges]
        valid = neighbours >= 0
        gradients[valid] += fields[neighbours[valid]]
        gradients /= self.grid.h_m
        active_fields = fields[active_nodes]
        tx = np.repeat(transmitters, receivers.size)
        rx = np.tile(receivers, transmitters.size)
        result = np.empty((tx.size, 2 * basis.shape[1]), dtype=np.complex128)
        for first in range(0, tx.size, pair_chunk):
            end = min(first + pair_chunk, tx.size)
            product = active_fields[:, rx[first:end]] * active_fields[:, tx[first:end]]
            result[first:end, :basis.shape[1]] = (mass_basis.T @ product).T * self.grid.h_m**2
            product = gradients[:, rx[first:end]] * gradients[:, tx[first:end]]
            result[first:end, basis.shape[1]:] = (flux_basis.T @ product).T * self.grid.h_m**2
        return result
