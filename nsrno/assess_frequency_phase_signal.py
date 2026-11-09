"""Direct synthetic pulse experiment, not a material-sensitivity/propagation probe."""
from pathlib import Path
import argparse
import json
import os
import time
import numpy as np
import torch
from .frequency_phase_increment import FrequencyPhaseIncrement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Explicit CPU only; no GPU work during this signal experiment')
    args.out = args.out.resolve()
    if not args.out.is_relative_to(Path.cwd().resolve()/'out/v5') or args.out.exists():
        raise RuntimeError('New bounded same-project signal artifact only; do not overwrite')
    args.out.mkdir()
    torch.set_num_threads(1)
    started = time.time()
    n, dt, f0 = 3601, 2.5e-8, 3e6
    frequency = np.fft.rfftfreq(n, dt)
    mask = (frequency>=1.5e6)&(frequency<=4.5e6)
    t = np.arange(n)*dt
    def pulse(at):
        tau = t-at
        return np.exp(-.5*(tau*f0)**2)*np.sin(2*np.pi*f0*tau)
    def coefficients(signal):
        return np.fft.rfft(signal, norm='ortho')[mask]
    early = coefficients(pulse(10e-6))
    late = coefficients(.08*pulse(55e-6))
    single = coefficients(pulse(20e-6))
    shifts = np.linspace(-4e-6, 4e-6, 321)
    rows = {}
    with torch.no_grad():
        for kind, observation in [('single_common_delay', single), ('weak_late_echo_delay', early+late)]:
            data = torch.from_numpy(observation[None, None])
            global_phase = FrequencyPhaseIncrement(data, mask, [1.5e6, 4.5e6], local=False)
            local_phase = FrequencyPhaseIncrement(data, mask, [1.5e6, 4.5e6], local=True)
            curves = {'RF':[], 'global_phase_increment':[], 'local_phase_increment':[]}
            for tau in shifts:
                rotation = np.exp(-2j*np.pi*frequency[mask]*tau)
                prediction = single*rotation if kind=='single_common_delay' else early+late*rotation
                value = torch.from_numpy(prediction[None])
                curves['RF'].append(float(.5*np.sum(abs(prediction-observation)**2)/np.sum(abs(observation)**2)))
                curves['global_phase_increment'].append(float(global_phase.loss(value, 0)))
                curves['local_phase_increment'].append(float(local_phase.loss(value, 0)))
                if time.time()-started>45:
                    raise TimeoutError('Fixed direct-signal budget exceeded; do not start a repeated sweep')
            stats = {}
            for name, values in curves.items():
                values = np.asarray(values)
                tolerance = max(1e-12, float(values.max())*1e-8)
                minima = np.flatnonzero((values[1:-1]<values[:-2]-tolerance)&
                                       (values[1:-1]<values[2:]-tolerance))+1
                stats[name] = {'strict_local_minima_us':(shifts[minima]*1e6).tolist(),
                    'value_at_zero':float(values[len(shifts)//2]),
                    'value_at_plus1us':float(values[np.argmin(abs(shifts-1e-6))]),
                    'value_at_plus4us':float(values[-1])}
            rows[kind] = {'curves':curves, 'statistics':stats,
                'local_configuration':local_phase.config}
        gain_values = {}
        # Use the same single-pulse observation for the amplitude counterexample.
        gain_phase = FrequencyPhaseIncrement(torch.from_numpy(single[None, None]), mask,
                                            [1.5e6, 4.5e6], local=False)
        for gain in [.5, 1., 2.]:
            gain_values[str(gain)] = {'RF':float(.5*(gain-1)**2),
                'global_phase_increment':float(gain_phase.loss(torch.from_numpy((single*gain)[None]), 0))}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for axis, (kind, row) in zip(axes, rows.items()):
        for name, values in row['curves'].items():
            values = np.asarray(values)
            axis.plot(shifts*1e6, values/max(float(values.max()), 1e-30), label=name)
        axis.set_title(kind.replace('_', ' '));axis.set_xlabel('Delay error (us)')
        axis.set_ylabel('Objective / its own maximum');axis.grid(alpha=.25);axis.legend(fontsize=8)
    fig.savefig(args.out/'signal_objectives.png', dpi=170);plt.close(fig)
    result = {'status':'direct_frequency_phase_signal_experiment_finished',
        'seconds':time.time()-started, 'shifts_us':(shifts*1e6).tolist(), 'cases':rows,
        'amplitude_counterexample':gain_values, 'patient_arrays_read':False,
        'new_acoustic_solves':0, 'new_material_derivative_or_identifiability_probes':0,
        'GPU_work':False, 'joint_imaging_success_claimed':False}
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status':result['status'], 'seconds':result['seconds'],
        'statistics':{name:row['statistics'] for name,row in rows.items()},
        'amplitude_counterexample':gain_values, 'new_acoustic_solves':0}), flush=True)


if __name__=='__main__':
    main()
