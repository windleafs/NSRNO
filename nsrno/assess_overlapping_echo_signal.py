"""Adversarial direct-signal example: unresolved echoes in one fixed time window."""
from pathlib import Path
import argparse
import json
import os
import time
import numpy as np
import torch
from .frequency_phase_increment import FrequencyPhaseIncrement
from .local_window_amplitude import LocalWindowAmplitude


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU direct signals only; no patient or acoustic work')
    if args.out.exists() or not args.out.is_relative_to(Path.cwd().resolve()/'out/v5'):
        raise RuntimeError('New same-project scalar/figure artifact; do not repeat or overwrite')
    args.out.mkdir()
    torch.set_num_threads(1)
    started = time.time()
    n, dt, f0 = 3601, 2.5e-8, 3e6
    f = np.fft.rfftfreq(n, dt)
    mask = (f>=1.5e6)&(f<=4.5e6)
    t = np.arange(n)*dt
    def pulse(at, gain):
        tau = t-at
        return np.fft.rfft(gain*np.exp(-.5*(tau*f0)**2)*np.sin(2*np.pi*f0*tau),
                           norm='ortho')[mask]
    fixed, moving = pulse(20e-6, 1.), pulse(20.25e-6, .08)
    data = fixed+moving
    observed = torch.from_numpy(data[None, None])
    phase = FrequencyPhaseIncrement(observed, mask, [1.5e6, 4.5e6])
    amplitude = LocalWindowAmplitude(phase, observed)
    shifts = np.linspace(-4e-6, 4e-6, 321)
    curves = {'RF':[], 'local_phase':[], 'local_magnitude':[]}
    with torch.no_grad():
        for tau in shifts:
            prediction = fixed+moving*np.exp(-2j*np.pi*f[mask]*tau)
            value = torch.from_numpy(prediction[None])
            curves['RF'].append(float(.5*np.sum(abs(prediction-data)**2)/np.sum(abs(data)**2)))
            curves['local_phase'].append(float(phase.loss(value, 0)))
            curves['local_magnitude'].append(float(amplitude.loss(value, 0)))
            if time.time()-started>20:
                raise TimeoutError('Fixed direct-signal budget exceeded')
    i = int(np.argmin(abs(shifts-1e-6)))
    scales = {k:v[i] for k,v in curves.items()}
    if min(scales.values())<=1e-15:
        raise ValueError('Nonzero fixed +1us direct-signal scales required')
    curves = {k:np.array(v) for k,v in curves.items()}
    curves['RF_plus_phase'] = .2*curves['RF']/scales['RF']+.8*curves['local_phase']/scales['local_phase']
    curves['magnitude_plus_phase'] = .2*curves['local_magnitude']/scales['local_magnitude']+.8*curves['local_phase']/scales['local_phase']
    statistics = {}
    for name, values in curves.items():
        tolerance = max(1e-12, float(values.max())*1e-8)
        minima = np.flatnonzero((values[1:-1]<values[:-2]-tolerance)&
                               (values[1:-1]<values[2:]-tolerance))+1
        statistics[name] = {'strict_local_minima_us':(shifts[minima]*1e6).tolist(),
            'value_at_zero':float(values[len(shifts)//2])}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for name in ['RF', 'local_phase', 'local_magnitude']:
        values = curves[name]
        axes[0].plot(shifts*1e6, values/max(float(values.max()), 1e-30), label=name)
    for name in ['RF_plus_phase', 'magnitude_plus_phase']:
        axes[1].plot(shifts*1e6, curves[name], label=name)
    axes[0].set_ylabel('Objective / its own maximum')
    axes[1].set_ylabel('Both terms scaled at +1us')
    for axis in axes:
        axis.set_xlabel('Weak echo delay error (us)');axis.grid(alpha=.25);axis.legend(fontsize=8)
    fig.suptitle('Same-window echoes: 20 and 20.25 us; amplitude ratio 0.08')
    fig.savefig(args.out/'overlap_objectives.png', dpi=170);plt.close(fig)
    result = {'status':'overlapping_echo_direct_signal_counterexample_finished',
        'seconds':time.time()-started, 'echo_times_us':[20., 20.25], 'weak_amplitude_ratio':.08,
        'window_sigma_us':6., 'shifts_us':(shifts*1e6).tolist(), 'statistics':statistics,
        'curves':{k:v.tolist() for k,v in curves.items()}, 'scales_at_plus1us':scales,
        'patient_or_material_arrays_read':False, 'new_acoustic_solves':0,
        'new_material_derivative_probes':0, 'GPU_work':False,
        'patient_scale_result_or_imaging_success_claimed':False}
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','seconds','statistics','echo_times_us',
        'weak_amplitude_ratio','new_acoustic_solves']}), flush=True)


if __name__=='__main__':
    main()
