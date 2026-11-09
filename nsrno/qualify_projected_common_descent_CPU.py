"""One synthetic check for the new weighted projection and min-max solver."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from .independent_physical_coordinates import IndependentPhysicalCoordinates, SCALES
from .projected_common_descent import weighted_project, common_direction, CAPS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(Path.cwd().resolve()/'out/v5') or out.exists():
        raise ValueError('New same-project synthetic qualification output only')
    started = time.time()
    shape = (88, 9)
    anchor = np.stack([np.full(shape, 1540.), np.full(shape, 1000.), np.full(shape, .53)])
    coordinates = IndependentPhysicalCoordinates(anchor, np.full(shape, .03))
    active = coordinates.mask.copy()
    lateral = np.ones(shape[1], bool)
    rng = np.random.default_rng(20261109)
    proposed = rng.normal(size=anchor.shape)*5*active[None]
    metric = np.exp(rng.normal(size=anchor.shape)*1.3)
    projection = weighted_project(coordinates, proposed, metric, active)
    absolute = coordinates.absolute+projection
    Z = .05*absolute[0]+.1*absolute[1]
    feasible = bool(np.all(absolute[0, active] >= coordinates.c_lower-1e-12) and
        np.all(absolute[0, active] <= coordinates.c_upper+1e-12) and
        np.all(Z[active] >= coordinates.logZ_lower-1e-12) and
        np.all(Z[active] <= coordinates.logZ_upper+1e-12) and
        np.all(absolute[2, active] >= coordinates.alpha_lower-1e-12) and
        np.all(absolute[2, active] <= coordinates.alpha_upper+1e-12))
    # Independent dense boundary sampling, not the analytic projection formula.
    vertices = [np.array([c, (z-.05*c)/.1]) for c, z in
        [(coordinates.c_lower, coordinates.logZ_lower),
         (coordinates.c_upper, coordinates.logZ_lower),
         (coordinates.c_upper, coordinates.logZ_upper),
         (coordinates.c_lower, coordinates.logZ_upper)]]
    fraction = np.linspace(0., 1., 10001)
    boundary = np.concatenate([v[None]+fraction[:, None]*(vertices[(i+1)%4]-v)[None]
                              for i, v in enumerate(vertices)])
    gaps = []
    for x in range(5):
        point = (coordinates.absolute+proposed)[:2, 80, x]
        result = absolute[:2, 80, x]
        B = metric[:2, 80, x]
        distance = float(np.sum((result-point)**2/B))
        if distance <= 1e-20:
            reference = 0.
        else:
            reference = float(np.min(np.sum((boundary-point)**2/B, axis=1)))
        gaps.append((reference-distance)/max(1., reference))
    projection_reference_passed = min(gaps) >= -1e-10 and max(gaps) < 5e-6
    rows = []
    control = np.zeros_like(anchor)
    for name in ['equal', 'mixed', 'opposite', 'boundary']:
        gR = rng.normal(size=anchor.shape)*active[None]
        if name == 'equal':
            gD = gR.copy()
        elif name == 'opposite':
            gD = -gR
        else:
            gD = (.3*gR+rng.normal(size=anchor.shape))*active[None]
        trial_control = control.copy()
        if name == 'boundary':
            trial_control[0, active] = coordinates.c_lower
            trial_control[1, active] = (coordinates.logZ_upper-.05*coordinates.c_lower)/.1
            trial_control[2, active] = coordinates.alpha_lower
        direction, details = common_direction(coordinates, trial_control, gD, gR, active, lateral)
        capped = float(np.max(abs(direction)/CAPS)) <= 1+1e-10
        actual = coordinates.absolute+trial_control+direction
        z_actual = .05*actual[0]+.1*actual[1]
        physical = bool(np.all(actual[0, active] >= coordinates.c_lower-1e-10) and
            np.all(actual[0, active] <= coordinates.c_upper+1e-10) and
            np.all(z_actual[active] >= coordinates.logZ_lower-1e-10) and
            np.all(z_actual[active] <= coordinates.logZ_upper+1e-10) and
            np.all(actual[2, active] >= coordinates.alpha_lower-1e-10) and
            np.all(actual[2, active] <= coordinates.alpha_upper+1e-10))
        descent = details.get('RF_slope', 0.) < 0 and details.get('prior_slope', 0.) < 0
        expected = not details['actual_common_step_available'] if name == 'opposite' else descent
        rows.append({'case': name, 'passed': bool(expected and capped and physical),
            'status': details['status'], 'RF_slope': details.get('RF_slope'),
            'prior_slope': details.get('prior_slope'), 'dual_data_weight': details.get('dual_data_weight'),
            'max_physical_log_increment': np.max(abs(SCALES*direction), axis=(1, 2)).tolist()})
    passed = feasible and projection_reference_passed and all(r['passed'] for r in rows)
    result = {'status': 'weighted_projection_and_common_descent_synthetic_CPU_passed' if passed
              else 'weighted_projection_and_common_descent_synthetic_CPU_failed',
        'seconds': time.time()-started, 'projection_feasible': feasible,
        'independent_dense_boundary_relative_gaps': gaps, 'cases': rows,
        'source_SHA256': digest(Path(__file__).with_name('projected_common_descent.py')),
        'qualification_SHA256': digest(__file__), 'patient_arrays_read': False,
        'new_acoustic_solves': 0, 'new_patient_derivative_probes': 0, 'CUDA_used': False}
    out.mkdir()
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('New optimizer failed its one necessary synthetic qualification')


if __name__ == '__main__':
    main()
