"""CPU-only time-window statistics from already saved actual native RF.

Only aggregate energies and relative errors are written. RF time is reported
directly; it is not relabelled as depth for sources with different pulse delays.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from .low_frequency_0p5 import atomic,digest,frequency_mask,SAMPLE,CASE

EDGES_US=[0.,18.,30.,42.,54.,66.,90.025]


def decode(coefficients,bins,selected):
    spectrum=np.zeros((192,1801),np.complex128)
    spectrum[:,bins[selected]]=coefficients[:,selected]
    return np.fft.irfft(spectrum,n=3601,axis=-1,norm='ortho')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--balanced-objective',action='store_true')
    p.add_argument('--source-clock',action='store_true')
    a=p.parse_args()
    if not a.root.resolve().is_relative_to(Path.cwd().resolve()) or not a.out.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError('Keep the analysis in this independent nsrno project')
    if a.out.exists():
        raise RuntimeError('Keep the existing analysis; do not overwrite')
    run=json.loads((a.root/'run.json').read_text())
    if run['sample']!=SAMPLE or run['case']!=CASE:
        raise ValueError('Only the authorized development case')
    incremental=(a.root/'native_plan.json').exists()
    plan=json.loads((a.root/('native_plan.json' if incremental else 'anchor_plan.json')).read_text())
    low_path=Path(plan['data_root'])/'index.json'
    low=json.loads(low_path.read_text())
    high=json.loads(Path(plan['high_index']).read_text())
    if (digest(low_path)!=plan['data_index_SHA256']
            or digest(plan['high_index'])!=plan['high_index_SHA256']):
        raise ValueError('Observed RF indices changed')
    low_rows={(r['family'],r['shot']):r for r in low['shots'] if r['kind']==SAMPLE}
    stored={}
    if a.baseline:
        if not incremental and run.get('new_native_gradients',0)<1:
            raise ValueError('The actual native origin gradient has not completed')
        material_hash=run['native_anchor_material_SHA256']
        token=1
        if incremental:
            stored={(r['family'],r['shot']):r for r in plan['reference_RF']}
    else:
        if run['status'] not in ['frequency_multilevel_stage_finished',
                                 'native_frequency_incremental_stage_finished']:
            raise ValueError('A completed actual native result is required')
        material_hash=run['material_SHA256']
        receipt=a.root/'final/actual_native_acceptance.json'
        if receipt.exists():
            final=json.loads(receipt.read_text())
            if final['material_SHA256']!=material_hash:
                raise ValueError('Result material identity changed')
            token=final['actual_native_RF']['token']
            stored={(r['family'],r['shot']):r for r in final.get('RF_source_records',[])}
        else:
            if material_hash!=run['native_anchor_material_SHA256']:
                raise ValueError('No matching physically measured RF')
            token=1
    clock=np.arange(3601)*.025
    edges=[-20.,0.,12.,24.,36.,48.,60.,90.025] if a.source_clock else EDGES_US
    if a.source_clock:
        with np.load('out/v5/time_triparam_v1/geometry.npz') as data:
            source_x=data['x']
        with np.load(Path(plan['data_root'])/'source_layout.npz') as data:
            source_x=source_x[data['source_columns']]
    number_windows=len(edges)-1
    full_error=np.zeros(number_windows)
    full_observation_mass=np.zeros(number_windows)
    source_clock_references={}
    frequency=np.fft.rfftfreq(3601,2.5e-8)
    groups={}
    input_hashes=[]
    provenance_count=0
    started=time.perf_counter()
    for name,spec in plan['families'].items():
        low_centre=spec['centre']=='low'
        bins=np.flatnonzero(frequency_mask() if low_centre else high['DFT_mask'])
        frequencies=frequency[bins]
        bandsets=({f'band{j:02d}':tuple(b) for j,b in enumerate(spec['bands_hz'])}
                  if a.balanced_objective else
                  {'current_union':(min(b[0] for b in spec['bands_hz']),
                                    max(b[1] for b in spec['bands_hz']))})
        if not low_centre:
            if not a.balanced_objective:
                bandsets['full_3MHz_valid_band']=(1.5e6,4.5e6)
            row=high['families'][spec['source']]
            if digest(row['observed_coefficients'])!=row['observed_coefficients_SHA256']:
                raise ValueError('Observed high-frequency coefficients changed')
            high_observed=np.load(row['observed_coefficients'],mmap_mode='r')
        totals={key:{'observed_energy':np.zeros(number_windows),
                     'residual_energy':np.zeros(number_windows),
                     'frequency_observed_energy':0.,'frequency_residual_energy':0.}
                for key in bandsets}
        for shot in range(11):
            reference_us=0.
            if a.source_clock:
                reference_us=1e6*4/(.5e6 if low_centre else 3e6)
                if spec['source']=='lf':
                    reference_us-=1e6*float(np.min(
                        source_x*np.sin(np.deg2rad(low['angles_deg'][shot]))/1540.))
            source_clock_references.setdefault(name,[]).append(reference_us)
            propagation_clock=clock-reference_us
            windows=[(propagation_clock>=lo)&(propagation_clock<hi)
                     for lo,hi in zip(edges[:-1],edges[1:])]
            key=(name,shot)
            if key in stored:
                reference=stored[key]
                path=Path(reference['path'])
                if digest(path)!=reference['SHA256']:
                    raise ValueError('Known saved native RF changed')
            else:
                matches=list((a.root/'workers').glob(
                    f'rank*/fine_RF/token{token:04d}/{name}_{shot:02d}.npy'))
                if len(matches)!=1:
                    raise ValueError('Incomplete or repeated saved native RF')
                path=matches[0]
            prediction=np.load(path)
            input_hashes.append({'family':name,'shot':shot,'SHA256':digest(path)})
            if low_centre:
                row=low_rows[spec['source'],shot]
                if digest(row['path'])!=row['SHA256']:
                    raise ValueError('Observed low-frequency RF changed')
                observed=np.load(row['path'])
            else:
                observed=np.asarray(high_observed[shot])
            if observed.shape!=prediction.shape or observed.shape!=(192,len(bins)):
                raise ValueError('RF coefficient layout mismatch')
            provenance_count+=1
            for label,(lo,hi) in bandsets.items():
                selected=(frequencies>=lo)&(frequencies<=hi)
                target=decode(observed,bins,selected)
                residual=decode(prediction-observed,bins,selected)
                totals[label]['frequency_observed_energy']+=2*float(np.sum(abs(observed[:,selected])**2))
                totals[label]['frequency_residual_energy']+=2*float(np.sum(abs((prediction-observed)[:,selected])**2))
                for j,window in enumerate(windows):
                    totals[label]['observed_energy'][j]+=float(np.sum(target[:,window]**2))
                    totals[label]['residual_energy'][j]+=float(np.sum(residual[:,window]**2))
        groups[name]={}
        for label,values in totals.items():
            energy=values['observed_energy'];error=values['residual_energy']
            checks=[abs(energy.sum()-values['frequency_observed_energy'])/
                    max(values['frequency_observed_energy'],1e-30),
                    abs(error.sum()-values['frequency_residual_energy'])/
                    max(values['frequency_residual_energy'],1e-30)]
            if max(checks)>1e-10:
                raise RuntimeError('Time/frequency energy calculation inconsistent')
            groups[name][label]={'whole_record_relative_RMS':float(np.sqrt(error.sum()/energy.sum())),
                'windows':[{'RF_time_us':[lo,hi],
                    'observed_energy_fraction':float(energy[j]/energy.sum()),
                    'residual_energy_fraction':float(error[j]/max(error.sum(),1e-30)),
                    'relative_RMS':float(np.sqrt(error[j]/max(energy[j],1e-30)))}
                    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:]))],
                'time_frequency_energy_relative_consistency':max(checks)}
        if a.balanced_objective:
            observed_mass=sum((v['observed_energy']/v['observed_energy'].sum()
                              for v in totals.values()),np.zeros(number_windows))/len(totals)
            error_mass=sum((v['residual_energy']/v['observed_energy'].sum()
                           for v in totals.values()),np.zeros(number_windows))/len(totals)
            full_error+=spec['coefficient']*error_mass
            full_observation_mass+=2*spec['coefficient']*observed_mass
            groups[name]={'balanced_current_objective':{
                'whole_record_relative_RMS':float(np.sqrt(error_mass.sum())),
                'family_RF_objective':spec['coefficient']*float(error_mass.sum()),
                'energy_measure':'mean of subband-normalized temporal energy',
                'windows':[{'RF_time_us':[lo,hi],
                    'observed_energy_fraction':float(observed_mass[j]),
                    'residual_energy_fraction':float(error_mass[j]/max(error_mass.sum(),1e-30)),
                    'relative_RMS':float(np.sqrt(error_mass[j]/max(observed_mass[j],1e-30)))}
                    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:]))]}}
    if provenance_count!=44:
        raise RuntimeError('The complete actual native source set is required')
    result={'status':'cached_native_RF_time_window_summary_finished','seconds':time.perf_counter()-started,
        'root':str(a.root),'baseline':a.baseline,'native_RF_token':token,
        'input_saved_native_RF_hashes':input_hashes,
        'material_SHA256':material_hash,'actual_saved_RF_sources':provenance_count,
        'source_families':groups,'RF_dt_s':2.5e-8,'RF_nt':3601,
        'RF_time_not_depth':True,
        'source_clock_aligned':a.source_clock,'known_source_clock_references_us':source_clock_references,
        'source_clock_rule':'4/f0; plane-wave centre x=0 adds -min(x*sin(theta)/1540)',
        'union_band_metric_not_equal_to_balanced_subband_objective':not a.balanced_objective,
        'only_aggregate_RF_statistics':True,'GPU_work':False,'new_wave_propagations':0,
        'new_material_derivatives':0,'patient_material_or_truth_arrays_read':False,
        'reserved_case_unread':'Neg_47_Left'}
    if a.balanced_objective:
        expected=(run.get('native_anchor_RF_loss') if a.baseline else
                  run.get('final_actual_native_RF_loss'))
        consistency=None if expected is None else abs(full_error.sum()-expected)/max(expected,1e-30)
        if consistency is not None and consistency>1e-10:
            raise RuntimeError('Temporal decomposition does not match the actual native objective')
        result['global_balanced_RF_objective']={
            'value':float(full_error.sum()),'expected_saved_native_value':expected,
            'relative_consistency':consistency,
            'windows':[{'RF_time_us':[lo,hi],
                'normalized_observation_weight_fraction':float(full_observation_mass[j]),
                'RF_objective_contribution':float(full_error[j]),
                'RF_objective_fraction':float(full_error[j]/max(full_error.sum(),1e-30))}
                for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:]))]}
    atomic(a.out,result)
    summary={}
    for name,bands in groups.items():
        summary[name]={}
        for label,v in bands.items():
            late=slice(4,6) if a.source_clock else slice(3,5)
            early=slice(0,2) if a.source_clock else slice(0,1)
            late_energy=sum(w['observed_energy_fraction'] for w in v['windows'][late])
            late_error=sum(w['residual_energy_fraction'] for w in v['windows'][late])
            summary[name][label]={
                'whole_record_relative_RMS':v['whole_record_relative_RMS'],
                'early_window_us':[edges[0],12.] if a.source_clock else [0.,18.],
                'early_normalized_observation_fraction':sum(w['observed_energy_fraction'] for w in v['windows'][early]),
                'late_window_us':[36.,60.] if a.source_clock else [42.,66.],
                'late_normalized_observation_fraction':late_energy,
                'late_residual_fraction':late_error,
                'late_relative_RMS':v['whole_record_relative_RMS']*float(np.sqrt(late_error/max(late_energy,1e-30)))}
    print(json.dumps({'status':result['status'],'seconds':result['seconds'],
        'baseline':a.baseline,'source_family_RF_aggregate_statistics':summary,
        'global_balanced_RF_objective':result.get('global_balanced_RF_objective'),
        'source_clock_aligned':a.source_clock,
        'RF_time_not_depth':True,'only_aggregate_statistics_exported':True,
        'new_wave_propagations':0,'new_material_derivatives':0,'GPU_work':False,
        'patient_material_or_truth_arrays_read':False}))


if __name__=='__main__':
    main()

