"""Decisive CPU data-only check of timing losses; not a material reconstruction."""
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import hashlib, json, time
import numpy as np
import torch
from .adaptive_rf_loss import AdaptiveMatchingLoss
from .differential_phase_loss import DifferentialPhaseLoss
from .rf_depth_weights import arrival_bins, WEIGHTS, NAMES


def main():
    p = ArgumentParser()
    p.add_argument('--sample', default='train_200')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sigma-us', type=float, default=1.5)
    p.add_argument('--step-samples', type=int, default=4)
    p.add_argument('--max-shift-us', type=float, default=4.)
    p.add_argument('--shift-arrival-min', type=int, choices=range(5), default=0)
    p.add_argument('--filter-energy-floor', type=float, default=1e-3)
    p.add_argument('--compare-differential-phase', action='store_true')
    args = p.parse_args()
    import ctypes
    ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)
    torch.set_num_threads(1)
    index = json.loads(Path('out/v5/point_tx_3mhz_v1/index.json').read_text())
    row = next(r for r in index['records'] if r['id'] == args.sample)
    observed = torch.from_numpy(np.load(row['rf']))
    layout = np.load(index['source_layout'])
    geometry = np.load('out/v5/time_triparam_v1/geometry.npz')
    acquisition = SimpleNamespace(source='point3', observed=observed, device='cpu', f0=3e6,
        source_positions=layout['transmit_x_m'], geometry={k: geometry[k] for k in geometry.files})
    bins = torch.from_numpy(arrival_bins(acquisition))
    awi = AdaptiveMatchingLoss(acquisition, bins=bins, sigma_us=args.sigma_us,
                              filter_energy_floor=args.filter_energy_floor, cache_cpu=True)
    phase = DifferentialPhaseLoss(acquisition, awi, cache_cpu=True) if args.compare_differential_phase else None
    # Original geometric residual is compared under exactly its fixed normalization.
    weights = torch.tensor(WEIGHTS, dtype=torch.float64)
    denominator = torch.stack([observed.double().square()[bins == k].sum() for k in range(5)])
    max_shift = round(args.max_shift_us*40)
    shifts = np.arange(-max_shift, max_shift+1, args.step_samples).tolist()
    if 0 not in shifts:
        shifts.append(0)
        shifts.sort()
    scores = []
    start = time.time()
    with torch.no_grad():
        # Retain earlier arrivals; only the specified late component is shifted.
        components = [observed[shot]*awi.windows(shot)[args.shift_arrival_min:].square().sum(0)
                      for shot in range(len(observed))]
        for count, shift in enumerate(shifts):
            awi_terms = torch.zeros(5, dtype=torch.float64)
            rf_terms = torch.zeros(5, dtype=torch.float64)
            phase_terms = torch.zeros(5, dtype=torch.float64)
            for shot in range(len(observed)):
                target = observed[shot]
                component = components[shot]
                prediction = target-component
                if shift > 0:
                    prediction[:, shift:] += component[:, :-shift]
                elif shift < 0:
                    prediction[:, :shift] += component[:, -shift:]
                else:
                    prediction.copy_(target)
                awi_terms += torch.stack(awi.terms(prediction, shot)).double()
                if phase is not None:
                    phase_terms += torch.stack(phase.terms(prediction, shot)).double()
                difference = prediction-target
                rf_terms += torch.stack([difference[bins[shot] == k].double().square().sum()/denominator[k]
                                         for k in range(5)])
            scores.append({'shift_us': shift/40, 'RF': .5*float(weights@rf_terms),
                           'AWI': .5*float(weights@awi_terms),
                           'RF_by_arrival': (.5*rf_terms).tolist(), 'AWI_by_arrival': (.5*awi_terms).tolist()})
            if phase is not None:
                scores[-1].update(DifferentialPhase=.5*float(weights@phase_terms),
                                  DifferentialPhase_by_arrival=(.5*phase_terms).tolist())
            if count % 10 == 0:
                print(json.dumps({'done': count+1, 'total': len(shifts), 'elapsed_s': time.time()-start}), flush=True)
        # Amplitude tests include collapse: AWI should penalize zero, but be blind to nonzero scaling.
        amplitude = []
        for gain in [0., .25, 1., 4.]:
            terms = sum((torch.stack(awi.terms(gain*observed[shot], shot)).double()
                         for shot in range(len(observed))), torch.zeros(5, dtype=torch.float64))
            amplitude.append({'gain': gain, 'AWI': .5*float(weights@terms), 'RF': .5*(gain-1)**2})
            if phase is not None:
                terms = sum((torch.stack(phase.terms(gain*observed[shot], shot)).double()
                             for shot in range(len(observed))), torch.zeros(5, dtype=torch.float64))
                amplitude[-1]['DifferentialPhase'] = .5*float(weights@terms)
    self_score = next(s for s in scores if s['shift_us'] == 0)
    diagnostics = {}
    methods = ['RF', 'AWI']+(['DifferentialPhase'] if phase is not None else [])
    for name in methods:
        t = np.asarray([r['shift_us'] for r in scores])
        y = np.asarray([r[name] for r in scores])
        derivative = np.gradient(y, t)
        region = (abs(t) >= 1) & (abs(t) <= 3)
        diagnostics[name] = {'minimum_shift_us': float(t[y.argmin()]),
            'fraction_restoring_derivative_1_to_3us': float(np.mean(derivative[region]*t[region] > 0)),
            'self_penalty': self_score[name]}
    report = {'evaluation_only': True, 'material_truth_used': False,
        'scope': 'Temporal shifts of a fixed observed component; not propagation or three-map recovery',
        'sample': args.sample, 'source': 'point3', 'shots': len(observed), 'receivers': observed.shape[1],
        'shift_arrival_min': args.shift_arrival_min,
        'sigma_us': args.sigma_us, 'stabilization': .1,
        'filter_energy_floor': args.filter_energy_floor, 'arrival_names': NAMES,
        'differential_phase_hz': phase.frequency_difference_hz if phase is not None else None,
        'arrival_weights': WEIGHTS, 'scores': scores, 'amplitude': amplitude,
        'diagnostics': diagnostics, 'elapsed_s': time.time()-start,
        'sources_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ['adaptive_rf_loss.py', 'differential_phase_loss.py', 'check_awi_shift.py', 'rf_depth_weights.py']}}
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for name in methods:
        axes[0].plot([s['shift_us'] for s in scores], [s[name] for s in scores], label=name)
    axes[0].set(xlabel='Applied RF delay (µs)', ylabel='Geometric weighted objective',
                title=f'All 11 shots and 192 receivers; shifted classes >= {args.shift_arrival_min}')
    axes[0].legend()
    for k, name in enumerate(NAMES):
        axes[1].plot([s['shift_us'] for s in scores], [s['AWI_by_arrival'][k] for s in scores], label=name)
    axes[1].set(xlabel='Applied RF delay (µs)', ylabel='AWI objective', title='Arrival classes; not true depth localization')
    axes[1].legend(fontsize=7)
    fig.savefig(args.out.with_suffix('.png'), dpi=160)
    plt.close(fig)
    print(json.dumps({'diagnostics': diagnostics, 'amplitude': amplitude, 'elapsed_s': report['elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
