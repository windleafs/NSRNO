"""Small bounded joint step from exact all-source native tangent responses."""
import itertools
import numpy as np


def box_quadratic(H, linear):
    """Enumerate the faces of a four-dimensional strictly convex box QP."""
    H, linear = np.asarray(H, np.float64), np.asarray(linear, np.float64)
    n = len(linear)
    if H.shape != (n, n) or not 1 <= n <= 4:
        raise ValueError('At most four actual material directions')
    scale = np.sqrt(np.maximum(np.diag(H), 1e-30))
    normalized = H/np.outer(scale, scale)
    smallest = float(np.linalg.eigvalsh((normalized+normalized.T)*.5)[0])
    damping = max(1e-4, 1e-4-smallest)
    damped = (H+H.T)*.5+damping*np.diag(np.maximum(np.diag(H), 1e-30))
    best = (0., np.zeros(n))
    for face in itertools.product([-1, 0, 1], repeat=n):
        face = np.asarray(face)
        free, fixed = face == 0, face != 0
        x = face.astype(np.float64)
        if free.any():
            x[free] = np.linalg.solve(damped[np.ix_(free, free)],
                -linear[free]-damped[np.ix_(free, fixed)]@x[fixed])
        if np.max(np.abs(x)) > 1+1e-10:
            continue
        x = np.clip(x, -1, 1)
        value = float(linear@x+.5*x@damped@x)
        if value < best[0]:
            best = value, x.copy()
    value, x = best
    grad = linear+damped@x
    residual = x-np.clip(x-grad/np.maximum(np.diag(damped), 1e-30), -1, 1)
    if not np.isfinite(x).all() or np.linalg.norm(residual, np.inf) > 1e-6:
        raise RuntimeError('Bounded joint coefficient problem did not satisfy its KKT condition')
    return x, {'damping_relative_to_directional_diagonal': damping,
               'normalized_undamped_smallest_eigenvalue': smallest,
               'damped_model_change': value, 'projected_KKT_infinity': float(np.max(abs(residual))),
               'undamped_model_change': float(linear@x+.5*x@H@x)}
