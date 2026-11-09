"""CPU algebra of NEW joint kernels; no patient or material FD experiments."""
from types import SimpleNamespace
import json
import numpy as np
import torch
from .conditioned_wave import FrequencyGeometry, ConditionedFrequencySystem
from .pressure_schur import PressureSchur
from .pressure_joint_kernels import (ReceiverMetric, PressureBornColumns,
                                     joint_pressure_trial, scaled_joint_pressure_trial, LOCAL_PAIRS)


def relative(a, b):
    return float((a-b).norm()/max(float(a.norm()), float(b.norm()), 1e-30))


def born_algebra():
    spec = {'medium_shape': [2, 64, 64], 'dx_m': 50e-6, 'x0_m': -.0016,
        'z0_m': -.0016, 'native_dt_s': 5e-9, 'pml': 24,
        'eta_per_unit_alpha': [.015, .025], 'taus_s': [12e-9, 35e-9]}
    g = FrequencyGeometry(spec)
    for name, value in list(g._buffers.items()):
        if value.is_complex():
            g._buffers[name] = value.to(torch.complex128)
        elif value.is_floating_point():
            g._buffers[name] = value.double()
    # Double precision symbols keep this an algebra check, not a float32
    # Fourier/stencil roundoff comparison.
    def symbol(q):
        coeff = (1225/1024, -245/3072, 49/5120, -5/7168)
        return sum(a*(torch.exp(1j*(j+1)*q)-torch.exp(-1j*j*q))
                   for j, a in enumerate(coeff))/g.dx
    g.Dx = symbol(2*torch.pi*torch.fft.fftfreq(g.W, dtype=torch.float64))[None]
    g.Dz = symbol(2*torch.pi*torch.fft.fftfreq(g.H, dtype=torch.float64))[:, None]
    c = 1540+8*torch.randn(g.h, g.w, dtype=torch.float64)
    rho = 1000+9*torch.randn_like(c)
    alpha = .4+.01*torch.randn_like(c)
    op = PressureSchur(ConditionedFrequencySystem(g, torch.stack([c, rho]), alpha, 2.5e6))
    p = torch.randn(2, g.H, g.W, dtype=torch.complex128)*g.mask
    unknown = np.ones((g.h, g.w), dtype=bool)
    born = PressureBornColumns(op, p, unknown=unknown)
    q = torch.randn(3, g.h, g.w, dtype=torch.float64)
    W = torch.randn(3, g.H, g.W, dtype=torch.complex128)*g.mask
    columns = torch.stack([born.receiver_columns(w) for w in W])
    direct = (W.conj()[None]*born.jvp(q)[:, None]).flatten(2).sum(2)
    from_columns = torch.einsum('rathw,ahw->tr', columns, q.to(torch.complex128))
    transpose_error = relative(direct, from_columns)
    wf = W.flatten(1)
    gamma = wf.conj()@wf.T
    penalty = .01*float(torch.linalg.eigvalsh(gamma)[-1])
    metric = ReceiverMetric(gamma, penalty)
    real, imag = W.real.flatten(1).clone(), W.imag.flatten(1).clone()
    metric.whiten_basis_in_place(real, imag, chunk=731)
    white_columns = torch.stack([born.receiver_columns(torch.complex(a, b).reshape(g.H, g.W))
                                  for a, b in zip(real, imag)])
    white_values = torch.einsum('rathw,ahw->tr', white_columns, q.to(torch.complex128))
    whitening_error = relative(white_values, metric.whiten_data(direct))
    residual = torch.randn(2, 3, dtype=torch.complex128)
    scale = .17
    blocks, gradient = born.receiver_statistics(real, imag, metric.whiten_data(residual), scale)
    weighted_residual = penalty*metric.solve(residual)
    explicit_gradient = torch.einsum('rathw,tr->ahw', columns.conj(), weighted_residual).real*scale
    gradient_error = relative(gradient, explicit_gradient)
    # Three-node directions include coherently replicated PML ghost columns.
    local_errors = []
    for z, x in [(24, 24), (39, 39), (32, 31)]:
        v = torch.tensor([.7, -.4, .2], dtype=torch.float64)
        response = (white_columns[..., z, x]*v[None, :, None]).sum(1)
        expected = float(response.abs().square().sum()*scale)
        actual = 0.
        for k, (i, j) in enumerate(LOCAL_PAIRS):
            actual += float(blocks[k, z, x]*v[i]*v[j])*(1 if i == j else 2)
        local_errors.append(abs(actual-expected)/max(abs(actual), abs(expected), 1e-30))
    return {'Born_transpose_relative': transpose_error,
        'receiver_whitening_relative': whitening_error,
        'projected_material_gradient_relative': gradient_error,
        'same_node_coupled_curvature_relative': local_errors}


class DensePressure:
    def __init__(self, matrix, channels):
        self.matrix, self.channels = matrix, channels

    def __call__(self, values):
        return values@self.matrix.T

    def readout(self, values, response):
        return values@self.channels.T

    def readout_adjoint(self, values, response):
        return values@self.channels.conj()

    def material_change(self, previous, total):
        return total@(self.matrix-previous.matrix).T


def joint_algebra():
    n, k, tx = 7, 3, 2
    matrix = 3*torch.eye(n, dtype=torch.complex128)+torch.randn(n, n, dtype=torch.complex128)/5
    channels = torch.randn(k, n, dtype=torch.complex128)/3
    inverse = torch.linalg.inv(matrix)
    forward = lambda x: x@inverse.T
    adjoint = lambda x: x@inverse.conj()
    gamma = channels@inverse@inverse.mH@channels.mH
    penalty = .13
    metric = ReceiverMetric(gamma, penalty)
    old = DensePressure(matrix, channels)
    reference = torch.randn(tx, n, dtype=torch.complex128)
    current = torch.randn(tx, n, dtype=torch.complex128)
    rhs = torch.randn_like(current)
    target = torch.randn(tx, k, dtype=torch.complex128)
    line = .37
    tangent = .1*torch.randn(n, n, dtype=torch.complex128)
    # A nonlinear coefficient proposal differs from its GN linear source.
    change = line*tangent+line**2*.02*torch.randn(n, n, dtype=torch.complex128)
    new = DensePressure(matrix+change, channels)
    source = (reference+current)@(line*tangent).T
    raw_projection = old.readout(forward(old(current)-rhs), 1.)-(old.readout(current, 1.)-target)
    trial, record, y, yd = joint_pressure_trial(old, new, reference, current,
        rhs, target, 1., forward, adjoint, metric, line_scale=line, linear_source=source,
        projection_raw=raw_projection)
    e, rd = old(current)-rhs, old.readout(current, 1)-target
    hxx = channels.mH@channels+penalty*matrix.mH@matrix
    gx = rd@channels.conj()+penalty*e@matrix.conj()
    right = line*gx+penalty*source@matrix.conj()
    exact_delta = -torch.linalg.solve(hxx, right.T).T
    field_error = relative(trial-current, exact_delta)
    kv = old.readout(forward(source), 1.)
    expected_curvature = float(penalty*(kv.conj()*metric.solve(kv)).real.sum())
    curvature_error = abs(record['receiver_reduced_curvature_along_material']-expected_curvature)/max(abs(expected_curvature), 1e-30)
    before = .5*(rd.abs().square().sum()+penalty*e.abs().square().sum())
    new_rhs = rhs-new.material_change(old, reference)
    enew, dnew = new(trial)-new_rhs, new.readout(trial, 1)-target
    after = .5*(dnew.abs().square().sum()+penalty*enew.abs().square().sum())
    objective_error = abs(record['joint_halfcost_change']-float(after-before)) / max(abs(float(after-before)), 1e-30)
    dual_error = max(relative(y, -enew), relative(yd, -dnew))
    zero, zero_record, _, _ = joint_pressure_trial(old, old, reference, current,
        rhs, target, 1., forward, adjoint, metric, line_scale=0., linear_source=torch.zeros_like(current))
    zero_error = max(relative(zero, current), abs(zero_record['joint_halfcost_change']))
    first_new = DensePressure(matrix+tangent, channels)
    first, _, _, _ = joint_pressure_trial(old, first_new, reference, current, rhs,
        target, 1., forward, adjoint, metric, line_scale=1.,
        linear_source=(reference+current)@tangent.T)
    scaled, scaled_record, _, _ = scaled_joint_pressure_trial(old, new, reference,
        current, rhs, target, 1., first, penalty, line_scale=line, linear_source=source)
    scaled_error = relative(scaled, trial)
    scaled_cost_error = abs(scaled_record['joint_halfcost_change']-record['joint_halfcost_change']) / max(abs(record['joint_halfcost_change']), 1e-30)
    return {'simultaneous_field_elimination_relative': field_error,
        'nonlinear_joint_objective_accounting_relative': objective_error,
        'consistent_joint_dual_relative': dual_error, 'zero_joint_line_error': zero_error,
        'cached_joint_backtracking_relative': scaled_error,
        'cached_joint_objective_accounting_relative': scaled_cost_error,
        'same_step_curvature_diagnostic_relative': curvature_error}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(4903)
    report = {'scope': 'NEW receiver-space joint kernels on synthetic CPU arrays',
        'GPU_used': False, 'patient_propagation': 0, 'material_FD_experiments': 0}
    report.update(born_algebra())
    report.update(joint_algebra())
    errors = [v for k, v in report.items() if k.endswith(('_relative', '_error'))]
    errors = [item for value in errors for item in (value if isinstance(value, list) else [value])]
    report['passed'] = bool(max(errors) < 1e-10)
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise RuntimeError('New joint-kernel CPU algebra failed')


if __name__ == '__main__':
    main()
