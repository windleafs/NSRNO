"""Bounded real-valued inversion with explicit, shared regularization.

The two coefficient maps are independent. No tissue-endpoint relation,
segmentation prior, fitted channel gain, or truth-selected hyperparameter is
used. Complex observations are represented by stacked real and imaginary parts.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import LinearOperator


def gradient_matrix(x_mm, z_mm):
    """First differences per mm, with no periodic or zero-boundary penalty."""
    nx, nz = len(x_mm), len(z_mm)
    nodes = np.arange(nx * nz).reshape(nz, nx)
    left = np.r_[nodes[:, :-1].ravel(), nodes[:-1, :].ravel()]
    right = np.r_[nodes[:, 1:].ravel(), nodes[1:, :].ravel()]
    spacing = np.r_[np.tile(np.diff(x_mm), nz), np.repeat(np.diff(z_mm), nx)]
    row = np.arange(left.size)
    one = sparse.coo_matrix((np.r_[-1 / spacing, 1 / spacing],
                            (np.r_[row, row], np.r_[left, right])),
                           shape=(left.size, nx * nz)).tocsr()
    return sparse.block_diag((one, one), format="csr")


def real_operator(matrix, gradient, lambda0, lambda_gradient):
    """Augmented least-squares operator for real material coefficients."""
    rows, columns = matrix.shape

    def matvec(value):
        value = np.asarray(value).ravel()
        return np.r_[matrix @ value, lambda0 * value,
                     lambda_gradient * (gradient @ value)]

    def rmatvec(value):
        value = np.asarray(value).ravel()
        return (matrix.T @ value[:rows] + lambda0 * value[rows:rows + columns]
                + lambda_gradient * (gradient.T @ value[rows + columns:]))

    return LinearOperator((rows + columns + gradient.shape[0], columns),
                          matvec=matvec, rmatvec=rmatvec, dtype=np.float64)


def solve_inverse(jacobian, data, model_x, model_z, config):
    """Solve a pre-whitened/scaled system; report optimizer and KKT diagnostics.

    The diagonal change of variables only conditions the numerical solve.
    Both the physical box bounds and regularization are transformed with it.
    """
    matrix = np.ascontiguousarray(np.vstack((jacobian.real, jacobian.imag)))
    observations = np.r_[data.real, data.imag]
    gradient = gradient_matrix(model_x, model_z)
    l0, lg = float(config["lambda0"]), float(config["lambda_gradient"])
    operator = real_operator(matrix, gradient, l0, lg)
    rhs = np.r_[observations, np.zeros(matrix.shape[1] + gradient.shape[0])]
    diagonal = (np.sum(matrix**2, axis=0) + l0**2
                + lg**2 * np.asarray(gradient.power(2).sum(axis=0)).ravel())
    preconditioner = 1 / np.sqrt(np.maximum(diagonal, 1e-16))
    conditioned = LinearOperator(operator.shape,
                                 matvec=lambda v: operator @ (preconditioner * np.asarray(v).ravel()),
                                 rmatvec=lambda v: preconditioner * operator.rmatvec(v),
                                 dtype=np.float64)
    limit = float(config["bound"])
    iterations = int(config["max_iterations"])
    result = lsq_linear(conditioned, rhs, bounds=(-limit / preconditioner, limit / preconditioner),
                        tol=1e-7, lsmr_tol=1e-9, max_iter=iterations,
                        lsmr_maxiter=max(500, 10 * iterations))
    model = preconditioner * result.x
    residual = operator @ model - rhs
    grad = operator.rmatvec(residual)
    projected = grad.copy()
    projected[(model <= -limit + 1e-7) & (grad > 0)] = 0
    projected[(model >= limit - 1e-7) & (grad < 0)] = 0
    initial_gradient = operator.rmatvec(-rhs)
    relative_kkt = float(np.linalg.norm(projected, ord=np.inf) /
                         max(np.linalg.norm(initial_gradient, ord=np.inf), 1e-15))
    unbounded = getattr(result, "unbounded_sol", None)
    # scipy may report status=3 when an unconstrained LSMR iterate is inside
    # bounds even if LSMR stopped at its iteration limit. Keep both diagnostics.
    diagnostics = {
        "scipy_success": bool(result.success), "scipy_status": int(result.status),
        "message": str(result.message), "outer_iterations": int(result.nit),
        "unbounded_lsmr_stop_code": int(unbounded[1]) if unbounded is not None else None,
        "unbounded_lsmr_iterations": int(unbounded[2]) if unbounded is not None else None,
        "relative_projected_gradient_inf": relative_kkt,
        "converged": bool(result.success and relative_kkt < 1e-5),
        "objective": float(.5 * residual @ residual),
        "data_objective": float(.5 * np.linalg.norm(jacobian @ model - data)**2),
        "bound_fraction": float(np.mean(np.abs(model) >= limit - 1e-7)),
        "lambda0": l0, "lambda_gradient": lg,
    }
    return model, diagnostics


def conditional_sensitivity(jacobian):
    """Per-node two-parameter information, conditional on all other nodes.

    These 2x2 submatrices do not account for ambiguity with adjacent pixels and
    must not be interpreted as global identifiability or posterior uncertainty.
    """
    n = jacobian.shape[1] // 2
    a, b = jacobian[:, :n], jacobian[:, n:]
    aa, bb = np.sum(np.abs(a)**2, axis=0), np.sum(np.abs(b)**2, axis=0)
    ab = np.real(np.sum(a.conj() * b, axis=0))
    total = aa + bb
    delta = np.sqrt(np.maximum((aa - bb)**2 + 4 * ab**2, 0))
    smallest, largest = np.maximum((total - delta) / 2, 0), (total + delta) / 2
    return {
        "conditional_eigenvalue_ratio": np.divide(smallest, largest, out=np.zeros_like(total), where=largest > 0),
        "conditional_column_cosine": np.divide(ab, np.sqrt(aa * bb), out=np.zeros_like(ab), where=aa * bb > 0),
        "conditional_information_trace": total,
    }
