"""One independent known-optimum synthetic conditional problem;no acoustic waves."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from .conditional_material_quadratic import ConditionalMaterialQuadratic, right, down
from .low_frequency_0p5 import atomic, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('Do not repeat an already completed new conditional qualification')
    a.out.mkdir(parents=True)
    start = time.time()
    shape = (8, 9)
    active = np.zeros(shape, bool)
    active[2:6, 2:7] = True
    base = np.broadcast_to(np.array([1540., 1000., .5])[:, None, None], (3, *shape)).copy()
    z, x = np.indices(shape)
    target = base.copy()
    target[0, active] += (45*np.sin(.7*z+.4*x))[active]
    target[1, active] += (32*np.cos(.3*z-.8*x))[active]
    target[2, active] *= np.exp((.15*np.sin(.9*z-.2*x))[active])
    eta = np.array([.008899079785215043, .0070595340096893805])
    b = eta.sum()/2
    def coefficients(m):
        h = m[2]*m[0]/1540
        return m[1]*m[0]**2*(1+b*h)**2, h/(1+b*h)**2
    K0, R0 = coefficients(base)
    K1, R1 = coefficients(target)
    drho = target[1]-base[1]
    fx, fz = np.zeros(shape, bool), np.zeros(shape, bool)
    fx[:, :-1] = active[:, :-1]|active[:, 1:]
    fz[:-1] = active[:-1]|active[1:]
    aK, aR = active/K0**2, active/R0**2
    aX, aZ = fx/(2*1000.)**2, fz/(2*1000.)**2
    stats = np.stack([aK, -aK*(K1-K0), aR, -aR*(R1-R0),
        aX, -aX*(drho+right(drho)), aZ, -aZ*(drho+down(drho))])
    solver = ConditionalMaterialQuadratic(base, 0., stats, active, eta)
    recovered, details = solver.optimize(max_iterations=500, max_seconds=30, gtol=1e-10)
    errors = [float(np.sqrt(np.mean((recovered[f, active]-target[f, active])**2))) for f in range(3)]
    thresholds = [2e-3, 2e-3, 1e-5]
    passed = all(e < t for e, t in zip(errors, thresholds)) and details['conditional_change'] < 0
    result = {'status': 'independent_synthetic_conditional_known_optimum_passed' if passed else 'failed_for_inspection',
        'errors_c_rho_alpha': errors, 'thresholds': thresholds, 'optimizer': details,
        'seconds': time.time()-start, 'new_wave_solves': 0, 'patient_arrays_read': False,
        'new_patient_derivative_probes': 0, 'source_SHA256': {name: digest(Path(__file__).with_name(name))
            for name in ['conditional_material_quadratic.py', Path(__file__).name]}}
    atomic(a.out/'result.json', result)
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('New conditional solver did not recover the independent known optimum')


if __name__ == '__main__':
    main()
