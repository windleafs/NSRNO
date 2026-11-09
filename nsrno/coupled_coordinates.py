"""Native c/Z/alpha coordinates, weak-TV curvature and CPU PCG."""
import numpy as np
import torch
from scipy.fft import dctn, idctn
from .reconstruct_time import Parameters


class Coordinates:
    def __init__(self, initial, radii=(.25, .5, .3), tv_weight=1e-3, smoothing_mm=.5):
        if initial.shape != (3, 961, 1121):
            raise ValueError('A native physical-coordinate checkpoint is required')
        self.prefix = np.asarray(initial[:, :80], np.float32).copy()
        self.initial = np.asarray(initial[:, 80:], np.float64).copy()
        self.model = Parameters((961, 1121), 1)
        self.radii = np.asarray(radii, np.float64)[:, None, None]
        self.tv_weight = tv_weight
        h, w = self.initial.shape[-2:]
        kz = np.pi*np.arange(h)/(h*.05)
        kx = np.pi*np.arange(w)/(w*.05)
        self.sobolev = 1/(1+smoothing_mm**2*(kz[:, None]**2+kx[None]**2))
        self.lower = np.asarray([-1.75, -2.5, -8.])[:, None, None]
        self.upper = np.asarray([1.75, 2.5, 2.])[:, None, None]

    def fields(self, u):
        return np.concatenate([self.prefix.astype(u.dtype, copy=False), u], axis=1)

    def material(self, u):
        with torch.no_grad():
            q = torch.from_numpy(self.fields(np.asarray(u, np.float32)))
            return self.model.material(q).numpy()

    @staticmethod
    def tissue_alpha(u):
        with torch.no_grad():
            return (.5*torch.exp(.7*torch.from_numpy(np.asarray(u[2], np.float32)))).numpy()

    def tangent(self, u, material, v):
        c, rho = material[:2, 80:].astype(np.float64)
        alpha = self.tissue_alpha(u).astype(np.float64)
        result = np.zeros_like(material, dtype=np.float32)
        dc = 80*v[0]
        result[:, 80:] = np.stack([dc, rho*(.1*v[1]-dc/c), .7*alpha*v[2]])
        return result

    def transpose(self, u, material, gradient):
        c, rho = material[:2, 80:].astype(np.float64)
        alpha = self.tissue_alpha(u).astype(np.float64)
        gc, gr, ga = np.asarray(gradient[:, 80:], np.float64)
        return np.stack([80*gc-80*rho/c*gr, .1*rho*gr, .7*alpha*ga])

    @staticmethod
    def divergence(dx, dz, shape):
        out = np.zeros(shape, dtype=np.float64)
        out[:, :-1, :-1] -= dx+dz
        out[:, :-1, 1:] += dx
        out[:, 1:, :-1] += dz
        return out

    def regularizer(self, u):
        q = self.fields(u)
        dx = q[:, :-1, 1:]-q[:, :-1, :-1]
        dz = q[:, 1:, :-1]-q[:, :-1, :-1]
        root = np.sqrt(dx*dx+dz*dz+1e-6)
        value = self.tv_weight*float(np.mean(root-.001))
        weight = self.tv_weight/root/root.size
        gradient = self.divergence(weight*dx, weight*dz, q.shape)[:, 80:]
        return value, gradient, weight

    def tv_metric(self, v, weight):
        # Positive lagged-diffusivity majorizer, not the exact TV Hessian.
        full = np.zeros((3, 961, 1121), dtype=np.float64)
        full[:, 80:] = v
        dx = full[:, :-1, 1:]-full[:, :-1, :-1]
        dz = full[:, 1:, :-1]-full[:, :-1, :-1]
        return self.divergence(weight*dx, weight*dz, full.shape)[:, 80:]

    def precondition(self, r):
        return idctn(dctn(r, axes=(-2, -1), type=2, norm='ortho', workers=1)*self.sobolev,
                     axes=(-2, -1), type=2, norm='ortho', workers=1)


def dot(a, b):
    return float(np.vdot(a.reshape(-1), b.reshape(-1)).real)


def pcg(operator, right, precondition, maximum=8, tolerance=.15, callback=None):
    x = np.zeros_like(right)
    r = right.copy()
    z = precondition(r)
    p = z.copy()
    rz = dot(r, z)
    initial = rz
    history = []
    if rz == 0:
        return x, history
    if not rz > 0 or not np.isfinite(rz):
        raise RuntimeError('The initial preconditioned residual is not positive')
    for iteration in range(1, maximum+1):
        hp = operator(p)
        curvature = dot(p, hp)
        if not curvature > 0 or not np.isfinite(curvature):
            raise RuntimeError(f'Nonpositive coupled metric curvature: {curvature}')
        step = rz/curvature
        x += step*p
        r -= step*hp
        z = precondition(r)
        next_rz = dot(r, z)
        if not next_rz >= 0 or not np.isfinite(next_rz):
            raise RuntimeError('Nonfinite preconditioned CG residual')
        relative = float(np.sqrt(next_rz/initial))
        item = {'iteration': iteration, 'preconditioned_relative_residual': relative,
                'positive_curvature': curvature, 'cg_step': step}
        history.append(item)
        if callback:
            callback(item)
        if relative <= tolerance:
            break
        p = z+(next_rz/rz)*p
        rz = next_rz
    return x, history
