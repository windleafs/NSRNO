"""CPU checks for the acoustic model and the controlled aperture experiment.

Run from the repository root with::

    python -m unittest discover -s tests -p 'test_theranostic.py'

These tests check physical conventions and derivatives.  They deliberately do
not assert that an arc must improve a reconstruction: that is an experimental
outcome, not a property to bake into the implementation.
"""

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.special import hankel1

from nsrno.theranostic.inverse import gradient_matrix, real_operator, solve_inverse
from nsrno.theranostic.setup import load_config, physical_maps, validate_config
from nsrno.theranostic.wave import Grid, Solver


CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiments/theranostic_arc/quick.json"


def relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)


class AcousticPhysicsTests(unittest.TestCase):
    """Small fixtures keep the independent numerical checks inexpensive."""

    @classmethod
    def setUpClass(cls):
        axis = np.linspace(-16.0, 16.0, 81)
        cls.solver = Solver(Grid(axis, axis, pml_mm=6.0))
        cls.frequency = 0.15
        positions = np.array([[-6.4, -3.2], [-1.7, -3.1], [3.2, -3.2], [6.3, 1.7]])
        normals = np.array([[0.0, 1.0], [0.2, 1.0], [-0.15, 1.0], [-0.7, 0.7]])
        cls.weights = cls.solver.source_weights(positions, normals, np.array([0.8, 1.2, 0.6, 1.0]))
        cls.fields = cls.solver.solve(cls.frequency, cls.weights)
        cls.reference = cls.solver.sample(cls.fields, cls.weights)
        xx, zz = np.meshgrid(axis, axis)

        def bump(x0, z0, a, b):
            return np.maximum(1 - ((xx - x0) / a)**2 - ((zz - z0) / b)**2, 0)**2

        cls.basis = sparse.csr_matrix(np.column_stack([
            bump(-2.0, 1.0, 2.2, 1.7).ravel(),
            bump(2.0, 4.0, 1.6, 2.1).ravel(),
            bump(-1.5, 6.5, 2.4, 1.3).ravel(),
        ]))
        cls.ncoeff = cls.basis.shape[1]
        cls.direction = np.array([0.7, -0.2, 0.4, -0.25, 0.55, 0.1])
        ports = np.arange(cls.weights.shape[1])
        cls.jacobian = cls.solver.born_jacobian(
            cls.frequency, cls.fields, ports, ports, cls.basis, pair_chunk=3,
        )

    def model_fields(self, coefficients):
        return (self.basis @ coefficients[:self.ncoeff],
                self.basis @ coefficients[self.ncoeff:])

    def data(self, coefficients):
        mk, me = self.model_fields(coefficients)
        fields = self.solver.solve(self.frequency, self.weights, mk, me)
        return self.solver.sample(fields, self.weights)

    def test_zero_contrast_has_zero_scattered_data(self):
        zeros = np.zeros(self.solver.grid.shape)
        fields, diagnostics = self.solver.solve(
            self.frequency, self.weights, zeros, zeros, return_diagnostics=True,
        )
        assert_allclose(self.solver.sample(fields, self.weights) - self.reference, 0, atol=1e-13)
        scattered = self.solver.born_data(self.frequency, self.weights, zeros, zeros)
        assert_allclose(scattered, 0, atol=1e-13)
        self.assertLess(diagnostics["relative_linear_residual"], 1e-12)

    def test_finite_apertures_are_normalized_and_reciprocal(self):
        assert_allclose(np.asarray(self.weights.sum(axis=0)).ravel(), 1, atol=1e-14)
        self.assertGreaterEqual(self.weights.data.min(), 0)
        heterogeneous = self.data(0.15 * self.direction)
        self.assertLess(relative_error(heterogeneous, heterogeneous.T), 1e-12)
        # A complex-symmetric reciprocal wave system is generally not Hermitian.
        self.assertGreater(np.linalg.norm(heterogeneous - heterogeneous.conj().T), 1e-3)

    def test_each_material_derivative_matches_fullwave_difference(self):
        epsilon = 1e-4
        # Reordered non-square subsets catch tx/rx flattening mistakes as well
        # as signs, cell-area factors, and inverse-density face averaging.
        tx, rx = np.array([3, 0]), np.array([2, 1, 0])
        jacobian = self.solver.born_jacobian(
            self.frequency, self.fields, tx, rx, self.basis, pair_chunk=2,
        )
        for parameter, columns in (("kappa", slice(0, self.ncoeff)),
                                   ("eta", slice(self.ncoeff, 2 * self.ncoeff))):
            with self.subTest(parameter=parameter):
                direction = np.zeros(2 * self.ncoeff)
                direction[columns] = [0.7, -0.4, 0.55]
                finite_difference = ((self.data(epsilon * direction)
                                      - self.data(-epsilon * direction)) / (2 * epsilon))
                expected = finite_difference[np.ix_(tx, rx)].ravel()
                self.assertLess(relative_error(jacobian @ direction, expected), 1e-6)

    def test_real_material_gradient_matches_fullwave_loss(self):
        rng = np.random.default_rng(42)
        target = self.reference.ravel() + 0.02 * (
            rng.normal(size=self.reference.size) + 1j * rng.normal(size=self.reference.size)
        )
        residual = self.reference.ravel() - target
        gradient = np.real(self.jacobian.conj().T @ residual)

        def loss(coefficients):
            difference = self.data(coefficients).ravel() - target
            return 0.5 * np.vdot(difference, difference).real

        epsilon = 1e-5
        finite_difference = (loss(epsilon * self.direction) - loss(-epsilon * self.direction)) / (2 * epsilon)
        assert_allclose(gradient @ self.direction, finite_difference, rtol=1e-5, atol=1e-10)

    def test_born_solve_matches_reciprocal_jacobian(self):
        mk, me = self.model_fields(0.08 * self.direction)
        scattered = self.solver.born_data(
            self.frequency, self.weights, mk, me, background_fields=self.fields,
        )
        self.assertLess(relative_error(scattered.ravel(), self.jacobian @ (0.08 * self.direction)), 1e-9)

    def test_born_remainder_decreases_quadratically(self):
        tangent = (self.jacobian @ self.direction).reshape(self.reference.shape)
        errors = []
        for amplitude in (0.08, 0.04, 0.02):
            scattered = self.data(amplitude * self.direction) - self.reference
            errors.append(np.linalg.norm(scattered - amplitude * tangent))
        for coarse, fine in zip(errors[:-1], errors[1:]):
            self.assertGreater(fine / coarse, 0.15)
            self.assertLess(fine / coarse, 0.35)

    def test_outgoing_green_function_and_grid_refinement(self):
        # Independent continuum reference for div(grad p)+k^2 p=-delta and
        # exp(-i omega t).  Grid-aligned point elements avoid interpolation
        # changes being mistaken for finite-difference convergence.
        positions = np.array([[-3.2, 0.0], [3.2, 0.0], [0.0, 4.8]])
        distances_m = np.linalg.norm(positions[:, None] - positions[None, :], axis=2) * 1e-3
        off_diagonal = ~np.eye(len(positions), dtype=bool)
        k0 = 2 * np.pi * self.frequency * 1e6 / 1540.0
        exact = 0.25j * hankel1(0, k0 * distances_m[off_diagonal])
        errors = []
        for intervals in (40, 80):
            axis = np.linspace(-16.0, 16.0, intervals + 1)
            solver = Solver(Grid(axis, axis, pml_mm=6.0))
            weights = solver.source_weights(positions, np.tile([0.0, 1.0], (3, 1)), np.zeros(3))
            measured = solver.sample(solver.solve(self.frequency, weights), weights)[off_diagonal]
            errors.append(relative_error(measured, exact))
        self.assertLess(errors[1], 0.02)
        self.assertLess(errors[1], 0.4 * errors[0])

    def test_material_must_be_positive_and_outside_pml(self):
        invalid = np.zeros(self.solver.grid.shape)
        invalid[40, 40] = -1.0
        with self.assertRaises(ValueError):
            self.solver.matrix(self.frequency, m_kappa=invalid)
        with self.assertRaises(ValueError):
            self.solver.matrix(self.frequency, m_eta=invalid)
        invalid.fill(0)
        invalid[0, 0] = 0.01
        with self.assertRaises(ValueError):
            self.solver.matrix(self.frequency, m_eta=invalid)


class InverseTests(unittest.TestCase):
    def test_augmented_real_operator_has_correct_adjoint_and_objective(self):
        rng = np.random.default_rng(13)
        x, z = np.array([-1.0, 0.5, 2.0]), np.array([0.0, 2.0])
        columns = 2 * len(x) * len(z)
        jacobian = rng.normal(size=(17, columns)) + 1j * rng.normal(size=(17, columns))
        matrix = np.vstack((jacobian.real, jacobian.imag))
        gradient = gradient_matrix(x, z)
        l0, lg = 0.07, 0.31
        operator = real_operator(matrix, gradient, l0, lg)
        model, dual = rng.normal(size=columns), rng.normal(size=operator.shape[0])
        assert_allclose(np.dot(operator @ model, dual), np.dot(model, operator.rmatvec(dual)),
                        rtol=1e-13, atol=1e-12)
        expected = (np.linalg.norm(jacobian @ model)**2 + l0**2 * np.linalg.norm(model)**2
                    + lg**2 * np.linalg.norm(gradient @ model)**2)
        assert_allclose(np.linalg.norm(operator @ model)**2, expected, rtol=1e-13)
        # A constant map has no artificial perimeter or wraparound penalty.
        assert_allclose(gradient @ np.ones(columns), 0, atol=1e-14)

    def test_column_conditioning_preserves_physical_bounds_and_ridge_solution(self):
        # Diagonal complex observations admit an independent analytic bounded
        # ridge solution, including weak and strong columns and active bounds.
        scales = np.array([0.02, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 40.0])
        phases = np.linspace(-0.7, 0.8, len(scales))
        jacobian = np.diag(scales * np.exp(1j * phases))
        truth = np.array([0.8, -0.8, 0.11, -0.13, 0.9, -0.9, 0.08, -0.07])
        config = {"lambda0": 0.07, "lambda_gradient": 0.0, "bound": 0.25, "max_iterations": 100}
        model, diagnostics = solve_inverse(jacobian, jacobian @ truth,
                                           np.array([0.0, 1.0]), np.array([0.0, 1.0]), config)
        expected = np.clip(scales**2 * truth / (scales**2 + config["lambda0"]**2),
                           -config["bound"], config["bound"])
        self.assertFalse(np.iscomplexobj(model))
        assert_allclose(model, expected, rtol=2e-5, atol=2e-6)
        self.assertLessEqual(np.max(np.abs(model)), config["bound"] + 1e-12)
        self.assertTrue(diagnostics["converged"])
        self.assertGreater(diagnostics["bound_fraction"], 0)
        residual = jacobian @ model - jacobian @ truth
        expected_objective = 0.5 * (np.linalg.norm(residual)**2
                                    + config["lambda0"]**2 * np.linalg.norm(model)**2)
        assert_allclose(diagnostics["objective"], expected_objective, rtol=1e-12)


class ExperimentControlTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG_PATH)

    def test_transmit_only_sides_provide_no_same_band_receive_data(self):
        from nsrno.theranostic.experiment import scenario_apertures

        self.config["geometry"]["side_receive"] = True
        enabled = scenario_apertures(self.config)
        self.assertEqual(enabled["flat_dual"], ["center_hf", "flat_lf"])
        self.assertEqual(enabled["arc_dual"], ["center_hf", "arc_lf"])
        self.config["geometry"]["side_receive"] = False
        disabled = scenario_apertures(self.config)
        self.assertEqual(disabled["flat_dual"], disabled["central_hf"])
        self.assertEqual(disabled["arc_dual"], disabled["central_hf"])
        self.assertEqual(disabled["central_hf"], ["center_hf"])
        # This separate virtual control explicitly has its own LF receivers.
        self.assertEqual(disabled["central_dual"], ["center_hf", "center_lf"])

    def test_heldout_ports_do_not_leak_through_reciprocity(self):
        from nsrno.theranostic.experiment import holdout_mask

        n, stride = 9, 4
        heldout = np.asarray(holdout_mask(n, stride), dtype=bool).reshape(n, n)
        assert_allclose(heldout, heldout.T)
        self.assertTrue(heldout.any())
        self.assertTrue((~heldout).any())
        heldout_ports = np.arange(n) % stride == stride - 1
        self.assertTrue(heldout[heldout_ports, :].all())
        self.assertTrue(heldout[:, heldout_ports].all())
        self.assertFalse(heldout[np.ix_(~heldout_ports, ~heldout_ports)].any())

    def test_overlapping_bands_are_rejected(self):
        config = deepcopy(self.config)
        config["bands"]["high_mhz"][0] = config["bands"]["low_mhz"][-1]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_bounds_keep_both_materials_and_speed_positive(self):
        bound = self.config["inversion"]["bound"]
        mk, me = np.array([-bound, bound, 0]), np.array([bound, -bound, 0])
        c0, rho0 = 1540.0, 1000.0
        maps = physical_maps(mk, me, c0, rho0)
        self.assertTrue(np.all(maps["speed_m_s"] > 0))
        self.assertTrue(np.all(maps["density_kg_m3"] > 0))
        assert_allclose(maps["speed_m_s"]**2,
                        maps["inverse_density_m3_kg"] / maps["compressibility_pa_inv"])
        assert_allclose(maps["density_kg_m3"] * maps["inverse_density_m3_kg"], 1)
        for parameter in ("kappa", "eta"):
            with self.subTest(parameter=parameter), self.assertRaises(ValueError):
                physical_maps([-1.0] if parameter == "kappa" else [0.0],
                              [-1.0] if parameter == "eta" else [0.0], c0, rho0)
        for bound in (0, 1, 1.2):
            with self.subTest(bound=bound):
                config = deepcopy(self.config)
                config["inversion"]["bound"] = bound
                with self.assertRaises(ValueError):
                    validate_config(config)


if __name__ == "__main__":
    unittest.main()
