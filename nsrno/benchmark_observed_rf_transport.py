"""CPU time/units study of two existing observed RF traces; no wave solver call."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import numpy as np
import torch
from .time_acquisition import TimeAcquisition
from .rf_depth_weights import DepthWeights,WEIGHTS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--sample',default='train_200')
    parser.add_argument('--solver',choices=['unscaled','scaled'],default='unscaled')
    parser.add_argument('--compare',type=Path);args=parser.parse_args()
    if args.solver=='scaled':
        from .rf_weighted_transport_scaled import lifted
        solver_source='rf_weighted_transport_scaled.py'
    else:
        from .rf_weighted_vector_transport import lifted
        solver_source='rf_weighted_vector_transport.py'
    if args.sample!='train_200':raise ValueError('Use the existing development material only')
    if args.out.exists():raise RuntimeError('Preserve existing CPU result')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Require CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(1);args.out.mkdir();started=time.time()
    source_hash={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ['rf_vector_transport.py',solver_source,'benchmark_observed_rf_transport.py']}
    run={'status':'observed_RF_CPU_units_and_time','created_unix_s':started,'pid':os.getpid(),
         'sample':args.sample,'GPU_used':False,'wave_solves':0,'material_or_truth_arrays_read':False,
         'reserved_case_unread':'Neg_47_Left','source_SHA256':source_hash,
         'actual_reconstructed_RF_test':False,'counterfactual_waveform_only':True,'solver':args.solver}
    (args.out/'run.json').write_text(json.dumps(run,indent=2));rows=[]
    for source in ['lf','point3']:
        acquisition=TimeAcquisition(source,args.sample,device='cpu');depth=DepthWeights(acquisition)
        bins=depth.bins.numpy();counts=np.bincount(bins.reshape(-1),minlength=5)
        if np.any(counts==0):raise RuntimeError('Every fixed arrival window must be present')
        denominators=np.array(depth.den);coefficients=.5*np.array(WEIGHTS)/denominators
        amplitude=np.sqrt(denominators[0]/counts[0])
        shot=5;receiver=96;observed=acquisition.observed[shot,receiver].numpy().astype(np.float64)/amplitude
        b=coefficients[bins[shot,receiver]]/coefficients[0]
        # Retain the complete original RF time axis. This is a fixed data-space
        # counterfactual, not a prediction from a perturbed acoustic material.
        prediction=np.zeros_like(observed);prediction[20:]=.9*observed[:-20]
        value,gradient,_,report=lifted(prediction,observed,.025,5.,node_weight=b,
            tolerance=1e-6,maximum=12000)
        physical_factor=amplitude**2*coefficients[0]/(5.*.025)
        np.savez_compressed(args.out/f'{source}_trace.npz',observed_RF=observed*amplitude,
            synthetic_prediction_RF=prediction*amplitude,node_weight=b,gradient_normalized_RF=gradient)
        row={'source':source,'shot':shot,'receiver':receiver,'RF_samples':len(observed),
             'RF_dt_us':.025,'complete_original_RF_record':True,'fixed_arrival_window_counts':counts.tolist(),
             'window_RF_energies':denominators.tolist(),'fixed_original_L2_coefficients':coefficients.tolist(),
             'fixed_amplitude_unit_from_early_observed_RMS':amplitude,
             'node_weight_definition':'original fixed coefficient / earliest-window coefficient',
             'edge_cost_definition':'sqrt(mean of two adjacent node weights)',
             'lambda_us_per_early_RMS_amplitude_unit':5.,'smoothing_in_early_RMS_amplitude_unit':.001,
             'objective_before_fixed_external_scale':value,'candidate_external_scale':physical_factor,
             'counterfactual_shift_us':.5,'counterfactual_amplitude_gain':.9,
             'solver':report,'not_a_reconstruction_or_material_derivative_test':True}
        rows.append(row);(args.out/'progress.json').write_text(json.dumps(rows,indent=2))
        del acquisition,depth,bins
    result={'sources':rows,'estimated_serial_channel_CPU_seconds_per_3_worker_material_gradient':
        sum(row['solver']['seconds']*11*192 for row in rows)/3,
        'timing_extrapolation_not_real_gradient_measurement':True,
        'all_selected_inners_converged':all(row['solver']['converged'] for row in rows),
        'no_new_GPU_or_wave_solves':True,'no_patient_material_arrays_read':True,
        'no_three_parameter_reconstruction_claim':True,'not_optimized_lambda':True}
    if args.compare:
        previous=json.loads((args.compare/'result.json').read_text())
        comparison=[]
        for row,before in zip(rows,previous['sources']):
            if any(row[key]!=before[key] for key in ['source','shot','receiver','fixed_amplitude_unit_from_early_observed_RMS',
                'window_RF_energies','fixed_original_L2_coefficients','lambda_us_per_early_RMS_amplitude_unit']):
                raise RuntimeError('Cannot compare different observed-RF objectives')
            lo=max(row['solver']['dual'],before['solver']['dual'])
            hi=min(row['solver']['primal'],before['solver']['primal'])
            if lo>hi+1e-8*max(abs(lo),1.):raise RuntimeError('Same-objective solver bounds do not overlap')
            comparison.append({'source':row['source'],'certified_objective_intervals_overlap':True,
                'unscaled_seconds':before['solver']['seconds'],'scaled_seconds':row['solver']['seconds'],
                'observed_speed_ratio':before['solver']['seconds']/row['solver']['seconds'],
                'unscaled_converged':before['solver']['converged'],'scaled_converged':row['solver']['converged'],
                'unscaled_relative_gap':before['solver']['relative_gap'],'scaled_relative_gap':row['solver']['relative_gap']})
        result['same_target_numerical_comparison']=comparison
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    run.update(status='observed_RF_CPU_units_and_time_finished',finished_unix_s=time.time(),seconds=time.time()-started)
    (args.out/'run.json').write_text(json.dumps(run,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
