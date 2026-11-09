from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    q = json.loads(a.run.read_text())
    if q['status'] != 'finished':
        raise ValueError('Wait for all predeclared candidate speeds')
    c = np.array([r['c_m_s'] for r in q['curves']])
    score = np.array([r['joint_CF'] for r in q['curves']])
    by_depth = np.array([r['depth_class_CF'] for r in q['curves']])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    axes[0].plot(c, score / score.max(), color='black', lw=2)
    axes[0].axvline(q['selected_c_m_s'], color='black', ls=':', label='Selected 1530 m/s')
    near = c[score >= .95 * score.max()]
    axes[0].axvspan(near.min(), near.max(), color='tab:blue', alpha=.1, label='Score >= 95% of peak')
    axes[0].set_title('Global observed-channel coherence')
    for j, label in enumerate(['5-12 mm', '12-25 mm', '25-39 mm']):
        axes[1].plot(c, by_depth[:, j] / by_depth[:, j].max(), lw=1.7, label=label + ' reference arrival class')
    axes[1].set_title('Separate arrival classes; peaks differ')
    for ax in axes:
        ax.axvline(1540, color='tab:red', ls='--', lw=1, label='Reference 1540 m/s')
        ax.set_xlabel('Trial tissue speed (m/s)')
        ax.set_ylabel('Coherence / curve maximum')
        ax.set_xlim(1400, 1660)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc='lower center')
    fig.suptitle('Propagation initialization only; no local material reconstruction', fontsize=10)
    fig.savefig(a.run.with_name('coherence_curves.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
