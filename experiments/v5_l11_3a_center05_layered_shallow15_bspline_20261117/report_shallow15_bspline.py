"""Plot the full-width, shallow-only B-spline fit and score it against V5."""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RUN_ROOT = Path(os.environ.get('NSRNO_V5_RUN_ROOT', HERE.parents[1] / 'out')).resolve()
OUT = RUN_ROOT / HERE.name
PILOT = RUN_ROOT / 'v5_l11_3a_center05_layered_blocks_pilot_20261117'
ORACLE = RUN_ROOT / 'v5_l11_3a_center05_full_oracle_20261117'
COLD = RUN_ROOT / 'v5_l11_3a_center05_layered_full_bspline_20261117'


def rms(error):
    return float(np.sqrt(np.mean(np.square(error))))


def dice(left, right):
    return float(2 * np.count_nonzero(left & right) / max(left.sum() + right.sum(), 1))


def kappa(c, rho):
    return 100. * (1000. * 1540.**2 / (rho * c**2) - 1.)


CASES = ('projected_hard', 'fitted_soft', 'fitted_hard')
FIELDS = ('gland_fraction', 'speed_m_s', 'density_kg_m3')
DISPLAY_METHODS = ('global_binary', 'projected_hard', 'fitted_hard')


def score_maps(maps, actual, shell, cell_area):
    gland, speed, density = maps
    labels_true, density_true, kappa_true = actual
    gel, skin = shell
    fat = 1. - gel - skin - gland
    labels = np.argmax(np.stack((gel, skin, fat, gland)), axis=0)
    return dict(fat_dice=dice(labels == 2, labels_true == 2),
                gland_dice=dice(labels == 3, labels_true == 3),
                density_rmse_kg_m3=rms(density-density_true),
                compressibility_rmse_pp=rms(kappa(speed, density)-kappa_true),
                gland_area_mm2=float(gland.sum(dtype=np.float64)*cell_area))


def score(maps, prediction, observed, masks, actual, shell, cell_area):
    metrics = score_maps(maps, actual, shell, cell_area)
    for part, mask in masks.items():
        metrics[part+'_relative_residual'] = float(np.linalg.norm(prediction[:, mask]-observed[:, mask]) /
                                                   np.linalg.norm(observed[:, mask]))
    return metrics


def render(x, z, panels, metrics):
    plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
                         'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.edgecolor': '#24344a', 'text.color': '#152438'})
    fig = plt.figure(figsize=(15.2, 5.6), dpi=185)
    grid = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, .027], hspace=.44, wspace=.12,
                           left=.064, right=.931, bottom=.105, top=.758)
    fig.text(.065, .962, 'V5 浅层 0–15 mm 的全宽 B 样条重建', fontsize=21, fontweight='bold', va='top')
    fig.text(.066, .893, '0.5 MHz 宽带直发 · 分层近场 Weyl–Born · 固定材料端点 · 横向 ±28 mm',
             fontsize=10.5, color='#415369', va='top')
    names = ('V5 真值', '原全局二值', 'B 样条投影初值', 'B 样条拟合结果')
    bars = ('腺体体积分数', '密度（kg/m³）', '压缩率对比度（%）')
    keys = ('gland_dice', 'density_rmse_kg_m3', 'compressibility_rmse_pp')
    formats = ('腺体 Dice {:.3f}', 'RMSE {:.2f} kg/m³', 'RMSE {:.2f} 个百分点')
    cmap_specs = (('viridis', 0., 1.), ('viridis', 940., 1100.), ('RdBu_r', -20., 20.))
    dx, dz = np.diff(x).mean(), np.diff(z).mean()
    extent = (x[0]-dx/2, x[-1]+dx/2, z[-1]+dz/2, z[0]-dz/2)
    for row, (images, (cmap, low, high)) in enumerate(zip(panels, cmap_specs)):
        vmin = min(low, *(im.min() for im in images))
        vmax = max(high, *(im.max() for im in images))
        if row == 2:
            vmin, vmax = -max(-vmin, vmax), max(-vmin, vmax)
        for col, matrix in enumerate(images):
            ax = fig.add_subplot(grid[row, col])
            image = ax.imshow(matrix, cmap=cmap, extent=extent, origin='upper', interpolation='nearest',
                              vmin=vmin, vmax=vmax, aspect='auto')
            ax.set(xlim=(-28, 28), ylim=(15, 0), xticks=[-20, -10, 0, 10, 20], yticks=[0, 5, 10, 15])
            if row == 0:
                ax.set_title(names[col], fontsize=11.4, fontweight='bold', pad=8)
            if col == 0:
                ax.set_ylabel('深度（mm）')
            else:
                ax.tick_params(labelleft=False)
                value = metrics[DISPLAY_METHODS[col-1]][keys[row]]
                ax.text(.981, .06, formats[row].format(value), transform=ax.transAxes,
                        color='#152438', ha='right', va='bottom', fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=.24', alpha=.93))
            if row == 2:
                ax.set_xlabel('横向（mm）')
            else:
                ax.tick_params(labelbottom=False)
        fig.colorbar(image, cax=fig.add_subplot(grid[row, 4]), label=bars[row])
    path = OUT / 'v5_shallow15_bspline_comparison.png'
    fig.savefig(path, dpi=185)
    plt.close(fig)
    return path


def main():
    with (np.load(OUT/'shallow15_bspline_results.npz', allow_pickle=False) as fit,
          np.load(PILOT/'baseline_result.npz', allow_pickle=False) as baseline,
          np.load(ORACLE/'full_material.npz', allow_pickle=False) as truth,
          np.load(COLD/'full_bspline_results.npz', allow_pickle=False) as cold):
        np.testing.assert_array_equal(fit['case_names'], CASES)
        for key in ('x_mm', 'z_mm', 'angles_deg', 'frequencies_mhz', 'train_mask', 'holdout_mask', 'observed'):
            np.testing.assert_array_equal(fit[key], baseline[key])
            np.testing.assert_array_equal(cold[key], baseline[key])
        x, z = fit['x_mm'], fit['z_mm']
        np.testing.assert_allclose(x, truth['x_mm'], atol=1e-10, rtol=0.)
        np.testing.assert_allclose(z, truth['z_mm'], atol=1e-10, rtol=0.)
        shallow = (z >= -1e-8) & (z <= 15.+1e-8)
        positive = z > 1e-8
        np.testing.assert_allclose([x[0], x[-1], z[shallow][0], z[shallow][-1]], [-28., 28., 0., 15.], atol=1e-10)
        cell_area = float(np.diff(x).mean()*np.diff(z).mean())
        observed, train, holdout = (fit[key] for key in ('observed', 'train_mask', 'holdout_mask'))
        np.testing.assert_array_equal(train, ~holdout)
        np.testing.assert_allclose(fit['angles_deg'][holdout], [-24., -8., 8., 24.], rtol=0., atol=1e-10)
        masks = dict(train=train, holdout=holdout, all=train|holdout)

        arrays = {key: fit[key] for key in FIELDS}
        baseline_arrays = {key: baseline[key] for key in FIELDS}
        for key in FIELDS:
            values, reference = arrays[key], baseline_arrays[key]
            np.testing.assert_equal(values.shape, (len(CASES), z.size, x.size))
            assert np.isfinite(values).all()
            np.testing.assert_array_equal(values[:, ~shallow].astype(reference.dtype),
                                          np.broadcast_to(reference[~shallow], values[:, ~shallow].shape),
                                          err_msg=f'{key} changed outside the shallow fitting range')
        np.testing.assert_array_equal(truth['tissue_codes'], [0, 2, 3, 4, 5])
        gel, gland, fat, skin, vessel = truth['fractions'][:, shallow].astype(float)
        gland_true = gland + vessel
        density_true, speed_true = (truth[k][shallow].astype(float) for k in ('truth_density_kg_m3', 'truth_speed_m_s'))
        kappa_true = kappa(speed_true, density_true)
        actual = (np.argmax(np.stack((gel, skin, fat, gland_true)), axis=0), density_true, kappa_true)
        shell = np.zeros((2, z.size, x.size), dtype=float)
        shell[0, ~positive] = 1.
        shell[:, positive] = baseline['fractions_estimated_shell']
        shell = shell[:, shallow]

        model_maps = {'global_binary': [baseline_arrays[k][shallow].astype(float) for k in FIELDS]}
        model_maps.update({name: [arrays[k][i, shallow].astype(float) for k in FIELDS] for i, name in enumerate(CASES)})
        pred = {'global_binary': baseline['prediction']}
        pred.update(zip(CASES, fit['prediction']))
        cold_index = cold['case_names'].tolist().index('fitted_hard')
        pred['full_fov_cold_spline_crop'] = cold['prediction'][cold_index]
        model_maps['full_fov_cold_spline_crop'] = [cold[k][cold_index, shallow].astype(float) for k in FIELDS]
        metrics = {name: score(maps, pred[name], observed, masks, actual, shell, cell_area)
                   for name, maps in model_maps.items()}
        metrics['truth'] = dict(gland_area_mm2=float(gland_true.sum()*cell_area))
        shallow_z = z[shallow]
        depth_edges = np.r_[0, np.searchsorted(shallow_z + 1e-8, [5., 10.]), shallow_z.size]
        bands = []
        for lower, upper, start, stop in zip((0., 5., 10.), (5., 10., 15.), depth_edges[:-1], depth_edges[1:]):
            part = slice(start, stop)
            band = {name: score_maps([m[part] for m in model_maps[name]], [m[part] for m in actual],
                                     shell[:, part], cell_area) for name in DISPLAY_METHODS}
            band['truth'] = dict(gland_area_mm2=float(gland_true[part].sum()*cell_area))
            bands.append(dict(depth_range_mm=[lower, upper], metrics=band))
        panels = [[gland_true], [density_true], [kappa_true]]
        for name in DISPLAY_METHODS:
            gland, speed, density = model_maps[name]
            panels[0].append(gland)
            panels[1].append(density)
            panels[2].append(kappa(speed, density))
        figure = render(x, shallow_z, panels, metrics)

    record = dict(scoring_extent_mm=[[-28., 28.], [0., 15.]], shallow_grid_shape=[int(shallow.sum()), x.size],
                  residual_domain='all recorded frequencies and receive channels for the indicated transmit split',
                  metrics=metrics, depth_bands=bands)
    output = OUT/'shallow15_bspline_metrics.json'
    output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(output)
    print(figure)


if __name__ == '__main__':
    main()
