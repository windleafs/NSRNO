"""Direct signal amplitude counterpart; reuses existing RF and phase curves."""
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
    parser.add_argument('--existing-signal-result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU direct signal study only')
    project = Path.cwd().resolve()/'out/v5'
    args.out, args.existing_signal_result = args.out.resolve(), args.existing_signal_result.resolve()
    if any(not p.is_relative_to(project) for p in [args.out, args.existing_signal_result]):
        raise ValueError('Existing independent nsrno signal results only')
    if args.out.exists():
        raise RuntimeError('No overwrite or repeated sweep')
    old = json.loads(args.existing_signal_result.read_text())
    if old['status']!='direct_frequency_phase_signal_experiment_finished' or old['patient_arrays_read']:
        raise ValueError('Exact existing CPU-only pulse result required')
    args.out.mkdir()
    torch.set_num_threads(1)
    started = time.time()
    n, dt, f0 = 3601, 2.5e-8, 3e6
    frequency = np.fft.rfftfreq(n, dt)
    mask = (frequency>=1.5e6)&(frequency<=4.5e6)
    t, shifts = np.arange(n)*dt, np.asarray(old['shifts_us'])*1e-6
    def coefficients(at, amplitude=1.):
        tau = t-at
        signal = amplitude*np.exp(-.5*(tau*f0)**2)*np.sin(2*np.pi*f0*tau)
        return np.fft.rfft(signal, norm='ortho')[mask]
    single, early, late = coefficients(20e-6), coefficients(10e-6), coefficients(55e-6, .08)
    rows, gain_and_sign = {}, {}
    with torch.no_grad():
        for kind, observation in [('single_common_delay', single), ('weak_late_echo_delay', early+late)]:
            observed = torch.from_numpy(observation[None, None])
            phase = FrequencyPhaseIncrement(observed, mask, [1.5e6, 4.5e6])
            amplitude = LocalWindowAmplitude(phase, observed)
            values = []
            for tau in shifts:
                rotation = np.exp(-2j*np.pi*frequency[mask]*tau)
                prediction = single*rotation if kind=='single_common_delay' else early+late*rotation
                values.append(float(amplitude.loss(torch.from_numpy(prediction[None]), 0)))
                if time.time()-started>20:
                    raise TimeoutError('Fixed direct-signal amplitude budget exceeded')
            RF = np.asarray(old['cases'][kind]['curves']['RF'])
            P = np.asarray(old['cases'][kind]['curves']['local_phase_increment'])
            A = np.asarray(values)
            i = int(np.argmin(abs(shifts-1e-6)))
            if min(A[i], P[i], RF[i])<=1e-15:
                raise ValueError('Nonzero +1us example scales required')
            curves = {'RF_plus_phase':.2*RF/RF[i]+.8*P/P[i],
                      'magnitude_plus_phase':.2*A/A[i]+.8*P/P[i]}
            statistics = {}
            for name, value in curves.items():
                tolerance = max(1e-12, float(value.max())*1e-8)
                ix = np.flatnonzero((value[1:-1]<value[:-2]-tolerance)&
                                   (value[1:-1]<value[2:]-tolerance))+1
                statistics[name] = {'strict_local_minima_us':(shifts[ix]*1e6).tolist(),
                    'value_at_zero':float(value[len(shifts)//2])}
            rows[kind] = {'statistics':statistics, 'curves':{k:v.tolist() for k,v in curves.items()},
                'local_magnitude_curve':values, 'scales_at_plus1us':{'RF':RF[i], 'phase':P[i], 'amplitude':A[i]}}
            gain_and_sign[kind] = {}
            for gain in [-1., .5, 1., 2.]:
                prediction = torch.from_numpy((gain*observation)[None])
                gain_and_sign[kind][str(gain)] = {'local_phase':float(phase.loss(prediction, 0)),
                    'local_magnitude':float(amplitude.loss(prediction, 0)),
                    'RF':float(.5*(gain-1)**2)}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for axis, (kind, row) in zip(axes, rows.items()):
        for name, values in row['curves'].items():
            value = np.array(values)
            axis.plot(shifts*1e6, value, label=name)
        axis.set_title(kind.replace('_', ' '));axis.set_xlabel('Delay error (us)')
        axis.set_ylabel('Objective (both terms scaled at +1us)');axis.set_ylim(0, 2)
        axis.grid(alpha=.25);axis.legend(fontsize=8)
    fig.savefig(args.out/'combined_objectives.png', dpi=170);plt.close(fig)
    result = {'status':'direct_local_magnitude_phase_comparison_finished',
        'seconds':time.time()-started, 'shifts_us':old['shifts_us'], 'cases':rows,
        'gain_and_sign_counterexamples':gain_and_sign, 'scales_are_not_patient_initial_scales':True,
        'original_RF_and_phase_curves_reused':True, 'patient_arrays_read':False,
        'new_acoustic_solves':0, 'new_material_derivative_probes':0, 'GPU_work':False,
        'sign_invariance_means_final_signed_RF_stage_required':True,
        'joint_imaging_success_claimed':False}
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status', 'seconds', 'gain_and_sign_counterexamples']}|
        {'statistics':{k:v['statistics'] for k,v in rows.items()}}), flush=True)


if __name__=='__main__':
    main()
