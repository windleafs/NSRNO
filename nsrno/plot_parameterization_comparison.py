"""Post-hoc comparison of RF-selected material snapshots; never used by inversion."""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument('--physical', type=Path, required=True)
    parser.add_argument('--travel', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--sample', default='train_200')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    truth = np.concatenate([
        np.load(f'out/v5/forward_cache_v1/{args.sample}.medium.npy'),
        np.load(f'out/v5/uno_triparam_v1/alpha/{args.sample}.npy')[None],
    ])
    def snapshot(path):
        if (path/'rf_selection.json').exists():
            return np.load(path/'material.npy'), json.loads((path/'rf_selection.json').read_text())
        return np.load(path/'stage0_best_material.npy'), json.loads((path/'stage0_best.json').read_text())
    physical, physical_selection = snapshot(args.physical)
    travel, travel_selection = snapshot(args.travel)
    maps = [truth, np.load(args.seed), physical, travel]
    selections = [physical_selection, travel_selection]
    if selections[0]['step'] != selections[1]['step'] or selections[0]['stage'] != selections[1]['stage']:
        raise ValueError('Compare the same stage and update count')
    z = (np.arange(truth.shape[1])-60)*.05
    x = (np.arange(truth.shape[2])-560)*.05
    iz = (z >= 0) & (z <= 43)
    ix = abs(x) <= 19.1
    mask = (z[:, None] >= 1) & (z[:, None] <= 43) & (abs(x)[None] <= 19.1)
    names = ['c_m_s', 'density_kg_m3', 'alpha_dB_cm_MHz1p1']
    titles = ['Sound speed (m/s)', 'Density (kg/m³)', 'Attenuation [dB/(cm MHz^1.1)]']
    ranges = [(1450, 1610), (930, 1140), (0, 1.05)]
    step = selections[0]['step']
    stage = selections[0]['stage']
    columns = ['Truth', 'RF-derived input', f'Physical / step {step}', f'Travel / step {step}']
    report = {'sample': args.sample, 'evaluation_only': True,
              'selection': 'Only complete RF objective selected both snapshots',
              'physical_selection': selections[0], 'travel_selection': selections[1],
              'unknown_evaluation_depth_mm': [1, 43], 'metrics': {}}
    fig, axes = plt.subplots(3, 4, figsize=(13, 13), layout='constrained')
    fig.get_layout_engine().set(h_pad=.12, hspace=.06)
    for j, name in enumerate(names):
        report['metrics'][name] = {}
        target = truth[j][mask].astype(np.float64)
        report['metrics'][name]['Truth'] = {'mean': float(target.mean()),
            'spatial_standard_deviation': float(target.std())}
        for col, material in enumerate(maps):
            ax = axes[j, col]
            im = ax.imshow(material[j][iz][:, ix], extent=[x[ix][0], x[ix][-1], z[iz][-1], z[iz][0]],
                           aspect='equal', interpolation='none', cmap='viridis',
                           vmin=ranges[j][0], vmax=ranges[j][1])
            annotation = ''
            if col:
                a = material[j][mask].astype(np.float64)
                corr = float(np.corrcoef(a, target)[0, 1]) if a.std() > 1e-10 else None
                rmse = float(np.sqrt(np.mean((a-target)**2)))
                report['metrics'][name][columns[col]] = {
                    'rmse': rmse, 'correlation': corr,
                    'mean': float(a.mean()), 'spatial_standard_deviation': float(a.std())}
                annotation = f'\nRMSE {rmse:.3g}; r {corr:.3f}' if corr is not None else f'\nRMSE {rmse:.3g}; r undefined'
            ax.set_title(columns[col]+'\n'+titles[j]+annotation, fontsize=9)
            if j == 2:
                ax.set_xlabel('Lateral (mm)')
            ax.set_ylabel('Depth (mm)')
        fig.colorbar(im, ax=axes[j, :], shrink=.72)
    phase = ('3 MHz plane/diverging RF stage' if stage == 0
             else 'Full plane LF/HF and diverging RF stage; each follows its own 64-update LF result')
    fig.suptitle(f'{args.sample}: same original RF-derived input and geometric objective\n'
                 f'{phase}; truth is used only for independent evaluation', fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix('.png'), dpi=160)
    plt.close(fig)
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
