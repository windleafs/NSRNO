"""One small CPU code-consistency check, not a patient identifiability probe."""
import json
from types import SimpleNamespace
import numpy as np
import torch
from .conditioned_wave import FrequencyGeometry, ConditionedFrequencySystem
from .pressure_schur import PressureSchur
from .pressure_material_stencil import PressureMaterialStencil, material_indices
from .pressure_material_normal import PressureMaterialNormal


def main():
    torch.set_num_threads(1)
    torch.manual_seed(731)
    np.random.seed(731)
    spec = {'medium_shape': [2, 64, 64], 'dx_m': 50e-6, 'x0_m': -.0016,
        'z0_m': -.0016, 'native_dt_s': 5e-9, 'pml': 24,
        'eta_per_unit_alpha': [.015, .025], 'taus_s': [12e-9, 35e-9]}
    g = FrequencyGeometry(spec)
    for name, value in list(g._buffers.items()):
        if value.is_floating_point():
            g._buffers[name] = value.double()
        elif value.is_complex():
            g._buffers[name] = value.to(torch.complex128)
    # Rebuild the finite difference symbol in double precision for this check.
    def symbol(q):
        co = (1225/1024, -245/3072, 49/5120, -5/7168)
        return sum(a*(torch.exp(1j*(j+1)*q)-torch.exp(-1j*j*q))
                   for j, a in enumerate(co)) / g.dx
    g.Dx = symbol(2*torch.pi*torch.fft.fftfreq(g.W, dtype=torch.float64))[None]
    g.Dz = symbol(2*torch.pi*torch.fft.fftfreq(g.H, dtype=torch.float64))[:, None]
    g.rx_cols = torch.arange(g.px+24, g.px+40)
    g.face = g.pz+26
    r = 1 + .025*torch.randn(g.h, g.w, dtype=torch.float64)
    b = 1 + .035*torch.randn_like(r)
    t = .3 + .015*torch.randn_like(r)
    primitive = torch.stack([r, b, t])
    def operator(v):
        rr, bb, tt = v
        c = 1540/torch.sqrt(rr*bb)
        material = torch.stack([c, 1000*rr])
        return PressureSchur(ConditionedFrequencySystem(g, material, tt*1540/c, 2.3e6))
    op = operator(primitive)
    p = torch.randn(3, g.H, g.W, dtype=torch.complex128) * g.mask
    z = torch.randn_like(p) * g.mask
    inner = lambda a, b: torch.vdot(a.flatten(), b.flatten())
    relative = lambda a, b: float((a-b).norm()/max(float(a.norm()), float(b.norm()), 1e-30))
    schur_relative = relative(op(p), op.system(p))
    lhs, rhs = inner(op(p), z), inner(p, op.adjoint(z))
    pressure_adjoint = float(abs(lhs-rhs)/max(float(abs(lhs)), float(abs(rhs)), 1e-30))
    unknown = np.ones((g.h, g.w), dtype=bool)
    stencil = PressureMaterialStencil(op, p, unknown=unknown)
    q = torch.randn(3, g.h, g.w, dtype=torch.float64)
    h = 1e-4
    fd = g.native((operator(primitive + .1*h*q)(p)-operator(primitive - .1*h*q)(p))/(2*h))
    jq = stencil.apply(q)
    jacobian_relative = relative(fd, jq)
    residual = torch.randn_like(jq)
    jl = float((jq.conj()*residual).real.sum())
    jr = float((q*stencil.adjoint(residual)).sum())
    material_adjoint = abs(jl-jr)/max(abs(jl), abs(jr), 1e-30)
    gram = torch.zeros((190, g.h, g.w), dtype=torch.float64)
    right = torch.zeros((19, g.h, g.w), dtype=torch.float64)
    stencil.accumulate(residual, .7, gram, right)
    variables = SimpleNamespace(shape=(g.h, g.w), size=g.h*g.w, unknown=unknown)
    normal = PressureMaterialNormal(variables, material_indices(g), gram.numpy(), right.numpy())
    expected = .7*stencil.adjoint(jq).numpy()
    packed_normal = float(np.linalg.norm(normal.apply(q.numpy())-expected)/np.linalg.norm(expected))
    packed_right = float(np.linalg.norm(normal.rhs+.7*stencil.adjoint(residual).numpy()) /
                         np.linalg.norm(normal.rhs))
    # Explicit column diagonals also exercise duplicate PML ghost indices.
    chosen = [(0, 24, 24), (0, 39, 39), (1, 24, 25), (2, 39, 38)]
    diagonal_errors = []
    for field, zz, xx in chosen:
        unit = np.zeros((3, g.h, g.w))
        unit[field, zz, xx] = 1
        explicit = normal.apply(unit)[field, zz, xx]
        recorded = normal.diagonal[field, zz, xx]
        diagonal_errors.append(float(abs(explicit-recorded)/max(abs(explicit), abs(recorded), 1e-30)))
    report = {'scope': 'new scalar-pressure code algebra, synthetic CPU arrays only',
        'Schur_vs_existing_pressure_relative': schur_relative,
        'pressure_adjoint_relative': pressure_adjoint,
        'new_material_stencil_relative': jacobian_relative,
        'new_material_adjoint_relative': material_adjoint,
        'packed_normal_relative': packed_normal, 'packed_right_relative': packed_right,
        'duplicate_ghost_diagonal_relative': diagonal_errors,
        'patient_wave_solves': 0, 'patient_truth_read': False, 'GPU_used': False}
    report['passed'] = bool(schur_relative < 1e-11 and pressure_adjoint < 1e-11
        and jacobian_relative < 2e-7 and material_adjoint < 1e-11
        and packed_normal < 1e-11 and packed_right < 1e-11
        and max(diagonal_errors) < 1e-11)
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise RuntimeError('New pressure algebra check failed')


if __name__ == '__main__':
    main()
