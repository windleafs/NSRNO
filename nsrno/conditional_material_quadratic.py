"""Centered conditional material objective in independent c/impedance/alpha boxes.

The change from c/rho to c/(c*rho) is invertible; no shared tissue labels or
spatial material filtering is imposed. Statistics come from all source fields.
"""
import time
import numpy as np
from scipy.optimize import minimize, Bounds


def native_body(shape=(961, 1121), pml=24):
    z, x = np.ogrid[:shape[0], :shape[1]]
    # Hold the one replicated boundary node fixed; the full requested FOV is inside.
    return (z >= 80)&(z < shape[0]-pml-1)&(x >= pml+1)&(x < shape[1]-pml-1)


def right(a):
    return np.concatenate((a[:, 1:], a[:, -1:]), axis=1)


def down(a):
    return np.concatenate((a[1:], a[-1:]), axis=0)


def face_adjoint(x, z):
    out = x+z
    out[:, 1:] += x[:, :-1]
    out[1:] += z[:-1]
    out[:, -1] += x[:, -1]
    out[-1] += z[-1]
    return out


class ConditionalMaterialQuadratic:
    def __init__(self, material, edge, statistics, active, eta, normalization=1.):
        self.base = np.asarray(material, np.float64)
        self.active = np.asarray(active, bool)
        self.edge = np.broadcast_to(edge, self.base.shape[1:]).astype(np.float64)
        self.stats = np.asarray(statistics, np.float64)
        self.b = float(np.sum(eta)/2)
        self.normalization = max(float(normalization), 1e-30)
        if (self.base.shape[0] != 3 or self.active.shape != self.base.shape[1:] or
                self.stats.shape != (8, *self.active.shape) or not np.isfinite(self.stats).all() or
                np.any(self.stats[[0, 2, 4, 6]] < 0) or not self.active.any() or self.b < 0):
            raise ValueError('Finite centered PSD conditional statistics and explicit active nodes required')
        c, rho = self.base[:2, self.active]
        tissue = self.base[2, self.active]-self.edge[self.active]
        if np.any(tissue <= 0):
            raise ValueError('Positive tissue attenuation in the unknown body required')
        self.phi0 = np.stack([np.log(c/1540.)/.05, np.log(c*rho/(1540.*1000.))/.10,
            np.log(tissue/.5)/.70])
        self.lower = np.array([np.log(1400./1540.)/.05, -2.5, -8.])[:, None]
        self.upper = np.array([np.log(1680./1540.)/.05, 2.5, 2.])[:, None]
        if np.any(self.phi0 < self.lower-1e-6) or np.any(self.phi0 > self.upper+1e-6):
            raise ValueError('The actual starting material must respect the original physical domain')
        c, rho, tissue, h, K, R = self.fields(self.phi0)
        self.rho0, self.K0, self.R0 = rho.copy(), K.copy(), R.copy()
        self.shape = self.phi0.shape
        root = 1+self.b*h
        u = tissue*c/1540.
        jacK = np.stack([.05*K*(1+2*self.b*h/root), .10*K, .70*K*2*self.b*u/root])
        Rh = (1-self.b*h)/root**3
        jacR = np.stack([.05*h*Rh, np.zeros_like(h), .70*u*Rh])
        face_diagonal = face_adjoint(self.stats[4].copy(), self.stats[6].copy())[self.active]
        diagonal = self.stats[0, self.active]*jacK**2+self.stats[2, self.active]*jacR**2
        diagonal[0] += (.05*rho)**2*face_diagonal
        diagonal[1] += (.10*rho)**2*face_diagonal
        diagonal /= self.normalization
        floors = np.array([max(float(np.median(d[d > 0]))*1e-6, 1e-30)
            if np.any(d > 0) else 1. for d in diagonal])[:, None]
        self.scale = 1/np.sqrt(np.maximum(diagonal, floors))
        self.bound_lower = ((self.lower-self.phi0)/self.scale).ravel()
        self.bound_upper = ((self.upper-self.phi0)/self.scale).ravel()

    def fields(self, phi):
        c = 1540.*np.exp(.05*phi[0])
        rho = 1000.*np.exp(.10*phi[1]-.05*phi[0])
        tissue = .5*np.exp(.70*phi[2])
        h = (tissue+self.edge[self.active])*c/1540.
        root = 1+self.b*h
        if np.max(self.b*h) >= 1:
            raise ValueError('The proved small-h physical branch must be retained')
        return c, rho, tissue, h, rho*c*c*root*root, h/(root*root)

    def value_gradient(self, vector):
        phi = self.phi0+self.scale*np.asarray(vector, np.float64).reshape(self.shape)
        c, rho, tissue, h, K, R = self.fields(phi)
        dK, dR = K-self.K0, R-self.R0
        drho = np.zeros(self.active.shape, np.float64)
        drho[self.active] = rho-self.rho0
        sx, sz = drho+right(drho), drho+down(drho)
        aK, gK, aR, gR = [v[self.active] for v in self.stats[:4]]
        aX, gX, aZ, gZ = self.stats[4:]
        value = (np.sum(gK*dK+.5*aK*dK*dK)+np.sum(gR*dR+.5*aR*dR*dR)
            +np.sum(gX*sx+.5*aX*sx*sx)+np.sum(gZ*sz+.5*aZ*sz*sz))
        qK, qR = gK+aK*dK, gR+aR*dR
        grho = face_adjoint(gX+aX*sx, gZ+aZ*sz)[self.active]
        root = 1+self.b*h
        Rh = (1-self.b*h)/root**3
        u = tissue*c/1540.
        gc = .05*(K*(1+2*self.b*h/root)*qK+h*Rh*qR-rho*grho)
        gz = .10*(K*qK+rho*grho)
        ga = .70*(K*2*self.b*u/root*qK+u*Rh*qR)
        gradient = np.stack([gc, gz, ga])*self.scale/self.normalization
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError('Nonfinite actual conditional objective')
        return float(value/self.normalization), gradient.ravel()

    def material(self, vector):
        phi = self.phi0+self.scale*np.asarray(vector).reshape(self.shape)
        c, rho, tissue, *_ = self.fields(phi)
        result = self.base.copy()
        result[0, self.active] = c
        result[1, self.active] = rho
        result[2, self.active] = tissue+self.edge[self.active]
        return result.astype(np.float32)

    def optimize(self, max_iterations=250, max_seconds=600, gtol=1e-8):
        started = time.time()
        best = {'x': np.zeros(self.phi0.size), 'value': 0., 'iterations': 0}
        history = []
        class Limit(Exception):
            pass
        def objective(x):
            if time.time()-started >= max_seconds:
                raise Limit()
            value, grad = self.value_gradient(x)
            if value < best['value']:
                best.update(x=x.copy(), value=value)
            return value, grad
        def callback(x):
            best['iterations'] += 1
            if best['iterations'] == 1 or best['iterations']%10 == 0:
                history.append({'iteration': best['iterations'], 'best_normalized_change': best['value'],
                    'seconds': time.time()-started})
        try:
            result = minimize(objective, best['x'], method='L-BFGS-B', jac=True,
                bounds=Bounds(self.bound_lower, self.bound_upper), callback=callback,
                options={'maxiter': max_iterations, 'maxcor': 6, 'ftol': 1e-12, 'gtol': gtol, 'maxls': 30})
            status, message = int(result.status), str(result.message)
        except Limit:
            status, message = -1, 'fixed conditional CPU time reached;best already evaluated point retained'
        value, gradient = self.value_gradient(best['x'])
        projected = best['x']-np.clip(best['x']-gradient, self.bound_lower, self.bound_upper)
        return self.material(best['x']), {'status': status, 'message': message,
            'iterations': best['iterations'], 'seconds': time.time()-started,
            'normalized_conditional_change': value, 'conditional_change': value*self.normalization,
            'projected_gradient_infinity_norm': float(np.max(abs(projected))),
            'independent_full_native_fields': True, 'material_lowpass': False,
            'physical_domain_retained': True, 'not_claimed_global_conditional_optimum': True, 'history': history}
