"""Recovery-only accounting with cached-field diagnostics; no pressure solve.

Receiver-space primitives for a simultaneous pressure/material update.

These are NOT the currently running conditional-pressure reconstruction.
P is the existing pressure inverse approximation. Identities involving S^-1
are exact only for an exact P; actual joint candidates must be checked by S.
All material increments q are real, with delta(r,beta,t)=.1*q.
"""
import numpy as np
import torch
from .first_order_wavefield import relaxation_inverse


LOCAL_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def inner_real(a, b):
    """Double-precision real complex inner product for objective accounting."""
    result = torch.zeros((), dtype=torch.float64, device=a.device)
    x, y = a.flatten(), b.flatten()
    for start in range(0, len(x), 131072):
        result += (x[start:start+131072].to(torch.complex128).conj()
                   * y[start:start+131072].to(torch.complex128)).real.sum()
    return result


class ReceiverMetric:
    """lambda*(lambda I+Gamma)^-1 in all-receiver data coordinates."""
    @torch.no_grad()
    def __init__(self, gram, penalty):
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or penalty <= 0:
            raise ValueError('Expected a square receiver Gram and positive penalty')
        self.gram = (gram.to(torch.complex128)+gram.to(torch.complex128).mH)*.5
        self.penalty = float(penalty)
        values, vectors = torch.linalg.eigh(self.gram)
        shifted = values+self.penalty
        if not torch.isfinite(shifted).all() or float(shifted[0]) <= 0:
            raise RuntimeError('Nonpositive receiver metric; objective penalty is not changed')
        self.Q = (vectors*torch.sqrt(self.penalty/shifted)[None])@vectors.mH
        self.cholesky = torch.linalg.cholesky(self.gram+torch.eye(len(gram),
            device=gram.device, dtype=torch.complex128)*self.penalty)

    @classmethod
    def from_field_preconditioner(cls, preconditioner):
        """Recover its actual Gram, removing the preconditioner-only shift."""
        chol = preconditioner.cholesky
        penalty = preconditioner.penalty
        shift = preconditioner.metadata['preconditioner_roundoff_shift']
        gram = chol@chol.mH-torch.eye(len(chol), device=chol.device,
            dtype=chol.dtype)*(penalty+shift)
        return cls(gram, penalty)

    def solve(self, receivers):
        result = torch.cholesky_solve(receivers.T.to(torch.complex128), self.cholesky).T
        return result.to(receivers.dtype)

    def whiten_data(self, receivers):
        # Each batch row is a receiver vector; transpose has no conjugation.
        return (receivers.to(torch.complex128)@self.Q.T).to(receivers.dtype)

    @torch.no_grad()
    def whiten_basis_in_place(self, real, imag, chunk=65536):
        """Consume raw rows w_j=P* C* e_j, returning whitened receiver rows.

        Rows transform as conj(Q)W so <w_tilde_i,Jq>=(Q KJq)_i. This
        destroys the raw field-preconditioner basis; no further normal CG
        may use that preconditioner. The caller must release its wrapper.
        """
        if real.shape != imag.shape or len(real) != len(self.Q):
            raise ValueError('Receiver basis shape does not match the metric')
        dtype = torch.complex64 if real.dtype == torch.float32 else torch.complex128
        transform = self.Q.conj().to(dtype)
        for start in range(0, real.shape[1], chunk):
            end = min(start+chunk, real.shape[1])
            block = transform@torch.complex(real[:, start:end], imag[:, start:end])
            real[:, start:end].copy_(block.real)
            imag[:, start:end].copy_(block.imag)
        return real, imag


class PressureBornColumns:
    """Folded material Born columns for one receiver and every transmission.

    The density column includes the four neighbouring face contributions.
    Crop/replicate ghost columns are coherently folded before squaring. No
    full [receiver,transmission,material-pixel] tensor is retained.
    """
    @torch.no_grad()
    def __init__(self, operator, total_pressure, unknown=None):
        self.op = operator
        self.g = g = operator.g
        if total_pressure.ndim != 3:
            raise ValueError('Expected all-transmission total pressure [TX,H,W]')
        self.p = total_pressure*g.mask
        z = np.clip(np.arange(g.H)-g.pz, g.pml, g.h-g.pml-1)
        x = np.clip(np.arange(g.W)-g.px, g.pml, g.w-g.pml-1)
        self.index = torch.as_tensor(z[:, None]*g.w+x[None],
                                     dtype=torch.long, device=total_pressure.device)
        self.ix = self.index.roll(-1, -1)
        self.iz = self.index.roll(-1, -2)
        if unknown is None:
            unknown = np.broadcast_to(np.arange(g.h)[:, None] >= 80, (g.h, g.w)).copy()
        self.unknown = torch.as_tensor(unknown, dtype=torch.bool, device=total_pressure.device)
        if self.unknown.shape != (g.h, g.w):
            raise ValueError('Wrong native material unknown mask')
        s = operator.system
        t = s.alpha*s.c/1540
        beta = s.a*(1540/s.c).square()
        M, Mp = relaxation_inverse(s.frequency, g.dt, g.spec['taus_s'],
                                   g.spec['eta_per_unit_alpha'], t)
        self.beta_column = .1*M*self.p
        self.t_column = .1*beta*Mp*self.p
        vx = -operator.Ix*operator.gradient(self.p, 0)
        vz = -operator.Iz*operator.gradient(self.p, 1)
        self.x_face_column = -.05*s.ax*vx
        self.z_face_column = -.05*s.az*vz

    def fold(self, values, index=None):
        index = self.index if index is None else index
        leading = values.shape[:-2]
        flat = values.reshape(-1, self.g.H*self.g.W)
        indices = index.flatten()[None].expand(len(flat), -1)
        shape = (len(flat), self.g.h*self.g.w)
        real = torch.zeros(shape, dtype=values.real.dtype, device=values.device)
        imag = torch.zeros_like(real)
        real.scatter_add_(1, indices, flat.real)
        imag.scatter_add_(1, indices, flat.imag)
        return torch.complex(real, imag).reshape(*leading, self.g.h, self.g.w)*self.unknown

    @torch.no_grad()
    def receiver_columns(self, adjoint_pressure):
        if adjoint_pressure.shape != (self.g.H, self.g.W):
            raise ValueError('Use one receiver basis at a time')
        w = adjoint_pressure*self.g.mask
        fx = self.x_face_column*self.op.divergence(w, 0, True).conj()
        fz = self.z_face_column*self.op.divergence(w, 1, True).conj()
        result = torch.empty((3, len(self.p), self.g.h, self.g.w),
                             dtype=self.p.dtype, device=self.p.device)
        result[0].copy_(self.fold(fx))
        result[0].add_(self.fold(fx, self.ix))
        result[0].add_(self.fold(fz))
        result[0].add_(self.fold(fz, self.iz))
        del fx, fz
        result[1].copy_(self.fold(self.beta_column*w.conj()))
        result[2].copy_(self.fold(self.t_column*w.conj()))
        return result

    def jvp(self, real_increment):
        if real_increment.shape != (3, self.g.h, self.g.w) or real_increment.is_complex():
            raise ValueError('Expected a real normalized native material increment')
        q = (real_increment*self.unknown).flatten(1)[:, self.index]
        density_x = self.x_face_column*(q[0]+q[0].roll(-1, -1))
        density_z = self.z_face_column*(q[0]+q[0].roll(-1, -2))
        return self.g.mask*(self.beta_column*q[1]+self.t_column*q[2]
            +self.op.divergence(density_x, 0)+self.op.divergence(density_z, 1))

    @torch.no_grad()
    def receiver_statistics(self, basis_real, basis_imag, whitened_residual, scale):
        """Return six same-node data curvature blocks and projected gradient.

        Off-node curvature is omitted. Scale is 1/(group energy * number
        of groups), because lambda is already included by data whitening.
        """
        if whitened_residual.shape != (len(self.p), len(basis_real)):
            raise ValueError('Need every transmission and receiver residual')
        blocks = torch.zeros((6, self.g.h, self.g.w), dtype=torch.float64, device=self.p.device)
        gradient = torch.zeros((3, self.g.h, self.g.w), dtype=torch.float64, device=self.p.device)
        for receiver in range(len(basis_real)):
            w = torch.complex(basis_real[receiver], basis_imag[receiver]).reshape(self.g.H, self.g.W)
            columns = self.receiver_columns(w)
            r = whitened_residual[:, receiver, None, None]
            gradient.add_((columns.conj()*r[None]).real.sum(1, dtype=torch.float64), alpha=scale)
            for k, (i, j) in enumerate(LOCAL_PAIRS):
                blocks[k].add_((columns[i].conj()*columns[j]).real.sum(0, dtype=torch.float64), alpha=scale)
            del w, columns, r
        return blocks, gradient


@torch.no_grad()
def projected_receiver_residual(operator, current, rhs, target, response, forward, metric,
                                return_raw=False):
    equation = operator(current)-rhs
    data = operator.readout(current, response)-target
    # Exact elimination includes the old field gradient, even when a
    # relative field-normal tolerance leaves nonzero g_x.
    residual = operator.readout(forward(equation), response)-data
    whitened = metric.whiten_data(residual)
    return (whitened, residual) if return_raw else whitened


@torch.no_grad()
def _joint_accounting(old_operator, new_operator, reference, current, rhs, target,
                      response, delta, penalty, real_increment, line_scale,
                      linear_source=None, unknown=None, old_residuals=None):
    e, rd = ((old_operator(current)-rhs, old_operator.readout(current, response)-target)
             if old_residuals is None else old_residuals)
    trial = current+delta
    # New augmented source includes its exact coefficient change.
    new_rhs = rhs-new_operator.material_change(old_operator, reference)
    new_e = new_operator(trial)-new_rhs
    new_rd = new_operator.readout(trial, response)-target
    if linear_source is None:
        jvp = PressureBornColumns(old_operator, reference+current, unknown=unknown).jvp(real_increment)
    else:
        jvp = linear_source
    delta_data = old_operator.readout(delta, response)
    delta_equation = old_operator(delta)+jvp
    slope = inner_real(rd, delta_data)+penalty*inner_real(e, delta_equation)
    data_curvature = inner_real(delta_data, delta_data)
    equation_curvature = penalty*inner_real(delta_equation, delta_equation)
    nonlinear_remainder = new_e-e-delta_equation
    remainder_halfnorm = .5*penalty*inner_real(nonlinear_remainder, nonlinear_remainder)
    conditional_material_curvature = penalty*inner_real(jvp, jvp)
    before_data = .5*inner_real(rd, rd)
    before_equation = .5*penalty*inner_real(e, e)
    after_data = .5*inner_real(new_rd, new_rd)
    after_equation = .5*penalty*inner_real(new_e, new_e)
    return trial, {'data_halfcost_change': float(after_data-before_data),
        'equation_halfcost_change': float(after_equation-before_equation),
        'joint_halfcost_change': float(after_data+after_equation-before_data-before_equation),
        'joint_linear_slope': float(slope), 'line_scale': line_scale,
        'actual_joint_linear_curvature': float(data_curvature+equation_curvature),
        'actual_linear_data_curvature': float(data_curvature),
        'actual_linear_equation_curvature': float(equation_curvature),
        'nonlinear_equation_remainder_halfnorm': float(remainder_halfnorm),
        'linearized_joint_halfcost_change': float(slope+.5*(data_curvature+equation_curvature)),
        'actual_conditional_material_curvature': float(conditional_material_curvature),
        'field_delta_norm': float(delta.norm()), 'truth_read': False,
        'exact_physical_operators_in_acceptance': True,
        'pressure_inverse_is_approximate': True}, -new_e, -new_rd


@torch.no_grad()
def joint_pressure_trial(old_operator, new_operator, reference, current, rhs,
                         target, response, forward, adjoint, metric,
                         real_increment=None, line_scale=1., linear_source=None,
                         unknown=None, projection_raw=None):
    """A simultaneous GN direction with exact NONLINEAR candidate accounting.

    Field elimination uses Jq, not the finite nonlinear coefficient change.
    line_scale also scales the old-field-gradient correction, ensuring the
    joint line returns to its current field and material at zero. No bank is
    mutated; the complete first trial can be cached for cheap backtracking.
    """
    if not 0 <= line_scale <= 1:
        raise ValueError('Line search scale must lie in [0,1]')
    e = old_operator(current)-rhs
    rd = old_operator.readout(current, response)-target
    jvp = (PressureBornColumns(old_operator, reference+current, unknown=unknown).jvp(real_increment)
           if linear_source is None else linear_source)
    pressure = forward(line_scale*e+jvp)
    receiver_rhs = old_operator.readout(pressure, response)-line_scale*rd
    coefficients = metric.solve(receiver_rhs)
    projected = adjoint(old_operator.readout_adjoint(coefficients, response))
    delta = -pressure+forward(projected)
    trial, record, y, yd = _joint_accounting(old_operator, new_operator, reference, current, rhs,
        target, response, delta, metric.penalty, real_increment, line_scale,
        linear_source=jvp, unknown=unknown, old_residuals=(e, rd))
    if projection_raw is not None:
        # receiver_rhs = alpha*(K e-r_d)+K Jq. No extra wave solve is
        # needed to compare conditional and reduced curvature on THIS step.
        image = receiver_rhs-line_scale*projection_raw
        conditional = metric.penalty*inner_real(jvp, jvp)
        reduced = metric.penalty*inner_real(image, metric.solve(image))
        record.update(conditional_curvature_along_material=float(conditional),
            receiver_reduced_curvature_along_material=float(reduced),
            reduced_over_conditional_curvature=float(reduced/max(float(conditional), 1e-300)),
            conditional_material_linear_slope=float(metric.penalty*inner_real(e, jvp)),
            receiver_projected_material_linear_slope=float(metric.penalty*inner_real(image, metric.solve(projection_raw))),
            curvature_diagnostic_extra_wave_solves=0)
    return trial, record, y, yd


@torch.no_grad()
def scaled_joint_pressure_trial(old_operator, new_operator, reference, current,
                               rhs, target, response, first_trial, penalty,
                               real_increment=None, line_scale=1.,
                               linear_source=None, unknown=None):
    """Exact nonlinear backtracking along one cached joint field direction.

    Neither an inverse nor a receiver basis is needed. The material proposal
    and its real_increment must already correspond to this line_scale.
    Keeping first_trial immutable preserves the same direction on all lines.
    """
    if not 0 <= line_scale <= 1:
        raise ValueError('Line search scale must lie in [0,1]')
    delta = line_scale*(first_trial-current)
    return _joint_accounting(old_operator, new_operator, reference, current, rhs,
        target, response, delta, penalty, real_increment, line_scale,
        linear_source=linear_source, unknown=unknown)
