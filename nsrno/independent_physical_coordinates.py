"""Independent log c/rho/tissue-alpha controls with the original physical domain.

Prepared coordinate/prior algebra, not yet used in a patient inverse run.
Projection is Euclidean in the scaled controls; the original c and c*rho
constraints are retained, not replaced by truth-derived density bounds.
"""
import numpy as np


SCALES = np.array([.05, .10, .70], np.float64)[:, None, None]


class IndependentPhysicalCoordinates:
    def __init__(self, material, edge):
        self.m = np.asarray(material, np.float64)
        if self.m.ndim!=3 or self.m.shape[0]!=3 or not np.isfinite(self.m).all():
            raise ValueError('Finite independent three-field native material required')
        self.edge = np.broadcast_to(edge, self.m.shape[1:]).astype(np.float64)
        self.mask = np.broadcast_to(np.arange(self.m.shape[1])[:, None]>=80, self.m.shape[1:])
        self.tissue = self.m[2]-self.edge
        if (np.any(self.m[:2]<=0) or np.any(self.tissue[self.mask]<=0)):
            raise ValueError('Positive c, rho, and unknown tissue attenuation required')
        self.absolute = np.stack([np.log(self.m[0]/1540.)/.05,
            np.log(self.m[1]/1000.)/.10,
            np.log(np.maximum(self.tissue, 1e-30)/.5)/.70])
        self.c_lower, self.c_upper = np.log(np.array([1400., 1680.])/1540.)/.05
        self.logZ_lower, self.logZ_upper = -.25, .25
        self.alpha_lower, self.alpha_upper = -5.6/.7, 1.4/.7

    def material(self, control):
        control = np.asarray(control, np.float64)
        if control.shape!=self.m.shape or not np.isfinite(control).all():
            raise ValueError('Finite native independent physical controls required')
        increment = SCALES*control
        result = np.stack([self.m[0]*np.exp(increment[0]),
            self.m[1]*np.exp(increment[1]),
            self.edge+self.tissue*np.exp(increment[2])])
        result[:, ~self.mask] = self.m[:, ~self.mask]
        return result.astype(np.float32)

    def pullback(self, material_gradient, control):
        material = self.material(control).astype(np.float64)
        return np.stack([.05*material[0]*material_gradient[0],
            .10*material[1]*material_gradient[1],
            .70*(material[2]-self.edge)*material_gradient[2]])*self.mask

    def project(self, control, active_mask):
        """Exact scaled-Euclidean projection onto original c/Z/alpha constraints.

        In the (log c/.05, log rho/.10) plane, feasible points lie inside two
        c bounds and two parallel logZ=.05*phi_c+.10*phi_rho bounds. The nearest
        point is the input if feasible, or the closest point on one of four
        boundary segments. No sound-speed or density image smoothing occurs.
        """
        control = np.asarray(control, np.float64)
        active_mask = np.asarray(active_mask, bool)&self.mask
        if control.shape!=self.m.shape or active_mask.shape!=self.m.shape[1:]:
            raise ValueError('Full independent controls and explicit original active mask required')
        if not np.isfinite(control).all():
            raise ValueError('Finite proposed controls required')
        absolute = self.absolute+control
        c, rho = absolute[:2]
        logZ = .05*c+.10*rho
        valid = ((c>=self.c_lower)&(c<=self.c_upper)&
                 (logZ>=self.logZ_lower)&(logZ<=self.logZ_upper))
        best_c, best_rho = c.copy(), rho.copy()
        distance = np.where(valid, 0., np.inf)

        def consider(new_c, new_rho):
            nonlocal best_c, best_rho, distance
            candidate_distance = (new_c-c)**2+(new_rho-rho)**2
            choose = candidate_distance<distance
            best_c = np.where(choose, new_c, best_c)
            best_rho = np.where(choose, new_rho, best_rho)
            distance = np.minimum(distance, candidate_distance)

        for boundary in [self.c_lower, self.c_upper]:
            projected_rho = np.clip(rho, (self.logZ_lower-.05*boundary)/.10,
                                   (self.logZ_upper-.05*boundary)/.10)
            consider(boundary, projected_rho)
        normal_squared = .05**2+.10**2
        for boundary in [self.logZ_lower, self.logZ_upper]:
            projected_c = np.clip(c-.05*(logZ-boundary)/normal_squared,
                                  self.c_lower, self.c_upper)
            projected_rho = (boundary-.05*projected_c)/.10
            consider(projected_c, projected_rho)
        projected = np.stack([best_c, best_rho,
            np.clip(absolute[2], self.alpha_lower, self.alpha_upper)])
        return (projected-self.absolute)*active_mask[None]

    def regularization(self, control, weight=.001):
        """Independent physical-field TV; ordinary value/gradient, no data HVP."""
        if control.shape!=self.m.shape or not weight>=0:
            raise ValueError('Native controls and nonnegative TV weight required')
        physical = (self.absolute+control)[:, 80:]
        dx = physical[:, :-1, 1:]-physical[:, :-1, :-1]
        dz = physical[:, 1:, :-1]-physical[:, :-1, :-1]
        radius = np.sqrt(dx*dx+dz*dz+1e-6)
        value = weight*float(np.mean(radius-.001))
        scale = weight/radius/radius.size
        gradient = np.zeros_like(control)
        body = gradient[:, 80:]
        gx, gz = scale*dx, scale*dz
        body[:, :-1, :-1] -= gx+gz
        body[:, :-1, 1:] += gx
        body[:, 1:, :-1] += gz
        return value, gradient
