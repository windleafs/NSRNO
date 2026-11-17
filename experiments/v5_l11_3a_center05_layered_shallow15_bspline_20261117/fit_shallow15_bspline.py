"""Fit one warm-start cubic spline over the entire lateral FOV to 15 mm depth."""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch


HERE = Path(__file__).resolve().parent
RUN_ROOT = Path(os.environ.get('NSRNO_V5_RUN_ROOT', HERE.parents[1] / 'out')).resolve()
OUT = RUN_ROOT / HERE.name
PILOT = RUN_ROOT / 'v5_l11_3a_center05_layered_blocks_pilot_20261117'
FULL_SPLINE = RUN_ROOT / 'v5_l11_3a_center05_layered_full_bspline_20261117'
PATCH_SPLINE = RUN_ROOT / 'v5_l11_3a_center05_layered_bspline_20261117'
sys.path[:0] = [str(PILOT), str(FULL_SPLINE), str(PATCH_SPLINE)]
from fit_bspline import LevelSet, STAGES
from fit_full_bspline import X_BOUNDS_MM, normalized_scores, spline_axis
from fit_global_baseline import CachedNearField, FixedCenterFit, TrainingOperator, audit, load_problem, record_maps
from fit_cold_transfer import atomic_npz, coarse_coordinates


DEPTH_MM = (0., 15.)
CASE_NAMES = ('projected_hard', 'fitted_soft', 'fitted_hard')
SPLITS = ('train', 'holdout', 'full')


class ShallowLevelSet(LevelSet):
    """Represent shallow labels with one tensor-product spline across the width."""

    def __init__(self, x, z, labels, pitch, op):
        shallow = z <= DEPTH_MM[1] + 1e-9
        shallow_count = int(shallow.sum())
        np.testing.assert_array_equal(shallow, np.arange(len(z)) < shallow_count)
        self.knots_x, bx, dx = spline_axis(x, X_BOUNDS_MM)
        self.knots_z, bz, dz = spline_axis(z[shallow], DEPTH_MM)
        signed_distance = distance_transform_edt(labels, sampling=pitch) - distance_transform_edt(~labels, sampling=pitch)
        target = signed_distance[shallow]
        # Least squares for B_z C B_x^T using two small one-dimensional basis matrices.
        coefficients = np.linalg.pinv(bz) @ target @ np.linalg.pinv(bx).T
        self.distance_projection_rmse_mm = float(np.sqrt(np.mean((bz @ coefficients @ bx.T - target)**2)))
        self.bx, self.dx, self.bz, self.dz = (op.tensor(v, torch.float64) for v in (bx, dx, bz, dz))
        self.initial = op.tensor(coefficients, torch.float64)
        self.free = torch.ones_like(self.initial, dtype=torch.bool)
        self.deep_phase = op.tensor(labels[~shallow], torch.float64)
        self.op = op

    def hard(self, coefficients):
        return torch.cat((super().hard(coefficients), self.deep_phase))

    def soft(self, coefficients, temperature):
        return torch.cat((super().soft(coefficients, temperature), self.deep_phase))


class ShallowFixedCenterFit(FixedCenterFit):
    def __init__(self, op, observations, metadata, initial_labels):
        super().__init__(op, observations, metadata)
        self.initial = op.tensor(initial_labels, torch.float64)
        self.deep_coverage = super().coverage(self.initial)
        self.shallow = op.tensor((op.z <= DEPTH_MM[1] + 1e-9)[:, None], torch.bool)
        self.endpoints = self.endpoints[self.center_index:self.center_index+1]

    def coverage(self, phase):
        # Freeze the material after interface smoothing, which crosses the depth cut.
        return torch.where(self.shallow, super().coverage(phase), self.deep_coverage)


def run(device):
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    torch.set_num_threads(4)
    acquisition, observed, train, holdout, positive, ref_c, ref_rho, shell, shell_endpoints = load_problem()
    with np.load(PILOT / 'baseline_result.npz', allow_pickle=False) as archive:
        baseline = dict(archive)
    for key in ('x_mm', 'z_mm', 'frequencies_mhz', 'angles_deg'):
        np.testing.assert_array_equal(baseline[key], acquisition[key])
    for key, value in (('observed', observed), ('train_mask', train), ('holdout_mask', holdout),
                       ('depth_speed_m_s', ref_c), ('depth_density_kg_m3', ref_rho), ('fractions_estimated_shell', shell)):
        np.testing.assert_array_equal(baseline[key], value)

    x, z = acquisition['x_mm'], acquisition['z_mm']
    op = CachedNearField(x, z, acquisition['frequencies_mhz'], acquisition['drive'], device=device,
                         depth_speed_m_s=ref_c, depth_density_kg_m3=ref_rho)
    adjusted = op.adjusted_observations(observed)
    original = op.tensor(observed, torch.complex128)
    diagnostics = audit(op, observed, adjusted)
    metadata = dict(fixed_fractions=shell, fixed_endpoints=shell_endpoints)
    training = TrainingOperator(op, train)
    labels = baseline['labels'].astype(bool)
    fit = ShallowFixedCenterFit(training, adjusted.index_select(1, training.indices), metadata, labels)
    fit.data_norm2 = float(np.linalg.norm(observed[:, train])**2)
    np.testing.assert_array_equal(fit.endpoints[0].cpu().numpy(), baseline['endpoints_c_rho'])
    label_x, label_z = coarse_coordinates(x, fit.tile), coarse_coordinates(op.z, fit.tile)
    level = ShallowLevelSet(label_x, label_z, labels, fit.pitch, op)
    projected = level.hard(level.initial)
    change_count = int(torch.count_nonzero(projected != fit.initial))
    setup_seconds = perf_counter() - started
    print(f'SHALLOW SPLINE: setup {setup_seconds:.2f}s, controls {tuple(level.initial.shape)}, '
          f'projection changed {change_count} labels', flush=True)

    (_, projected, soft, hard), fitted, terminal, trace = level.optimize(fit)
    np.testing.assert_array_equal((soft.cpu().numpy() >= .5), hard.cpu().numpy().astype(bool))
    deep_labels = label_z > DEPTH_MM[1]+1e-9
    np.testing.assert_array_equal(hard.cpu().numpy()[deep_labels], labels[deep_labels])
    reader = ShallowFixedCenterFit(op, adjusted, metadata, labels)
    initial = record_maps(reader, acquisition, positive, reader.initial, 0, adjusted, original)
    closure = float(np.linalg.norm(initial['prediction']-baseline['prediction'])/np.linalg.norm(observed))
    if not np.isfinite(closure) or closure >= 2e-5:
        raise ArithmeticError(f'Baseline replay relative error {closure:.3g}')

    rows, scores = [], []
    deep = z > DEPTH_MM[1]+1e-9
    for name, phase in zip(CASE_NAMES, (projected, soft, hard)):
        image = record_maps(reader, acquisition, positive, phase, 0, adjusted, original)
        for key in ('gland_fraction', 'speed_m_s', 'density_kg_m3', 'normalized_contrast'):
            np.testing.assert_array_equal(image[key][..., deep, :], baseline[key][..., deep, :])
        residual, data_cost = normalized_scores(image['prediction'], observed, train, holdout)
        scores.append(dict(case=name, relative_residual=residual, data_cost=data_cost))
        rows.append(image)
        print('SHALLOW CASE', scores[-1], flush=True)
    selected = next(row for row in trace['stages'] if row['stage'] == trace['hard_selected_stage'])
    np.testing.assert_allclose(scores[-1]['data_cost'][0], selected['hard']['train_data_term'], atol=2e-7, rtol=0.)
    atomic_npz(OUT / 'shallow15_bspline_results.npz', case_names=CASE_NAMES, split_names=SPLITS, x_mm=x, z_mm=z,
               x_label_mm=label_x, z_label_mm=label_z, angles_deg=acquisition['angles_deg'],
               frequencies_mhz=acquisition['frequencies_mhz'], observed=observed, train_mask=train, holdout_mask=holdout,
               knots_x_mm=level.knots_x, knots_z_mm=level.knots_z, coefficients_initial=level.initial.cpu().numpy(),
               coefficients_fitted=fitted.cpu().numpy(), coefficients_terminal=terminal.cpu().numpy(),
               label_phase=[phase.cpu().numpy() for phase in (projected, soft, hard)],
               endpoints_c_rho=baseline['endpoints_c_rho'], **{key: [row[key] for row in rows] for key in rows[0]})
    elapsed = perf_counter() - started
    summary = dict(cases=scores, split_names=SPLITS, fit=trace, depths_mm=DEPTH_MM, lateral_bounds_mm=X_BOUNDS_MM,
                   coefficient_grid_shape=list(level.initial.shape), temperatures_and_budgets=STAGES,
                   projection_label_changes=change_count, unmodified_deep_pixels=int(deep.sum()*x.size),
                   baseline_replay_relative_error=closure, cache_checks=diagnostics,
                   setup_seconds=setup_seconds, elapsed_seconds=elapsed)
    target = OUT / 'shallow15_bspline_summary.json'
    pending = target.with_suffix('.tmp')
    pending.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    pending.replace(target)
    op.close()
    print(f'SHALLOW FINISHED {elapsed:.2f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    run(parser.parse_args().device)
