"""CPU scalar decomposition of an already-saved original44 RF phase objective."""
from pathlib import Path
import argparse
import json
import os
import time
import numpy as np
import torch
from .frequency_multilevel_acquisition import LowHalfMHzAcquisition
from .full90_3mhz_acquisition import Full90ThreeMHzAcquisition
from .frequency_phase_increment import FrequencyPhaseIncrement
from .low_frequency_0p5 import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU only; no acoustic call or material array is needed')
    args.root, args.out = args.root.resolve(), args.out.resolve()
    project = Path.cwd().resolve()
    if any(not p.is_relative_to(project/'out/v5') for p in [args.root, args.out]):
        raise ValueError('Only existing nsrno results and new scalar records')
    if args.out.exists():
        raise RuntimeError('Do not overwrite or repeat a completed analysis')
    run = json.loads((args.root/'run.json').read_text())
    if run['status'] not in ['encoded_native_depth_stage_finished', 'local_phase_native_stage_finished']:
        raise RuntimeError('Only a completed frozen material and its saved original44 predictions')
    plan = json.loads((args.root/'native_plan.json').read_text())
    record = json.loads((args.root/'final/actual_native_acceptance.json').read_text())
    if record['material_SHA256']!=run['material_SHA256']:
        raise ValueError('Frozen result identity changed')
    references = {(r['family'], int(r['shot'])):r for r in record['RF_source_records']}
    if len(references)!=44:
        raise ValueError('Full original44 source cache required')
    args.out.mkdir()
    torch.set_num_threads(1)
    started, families = time.time(), {}
    weighted_phase = None
    with torch.no_grad():
        for name, spec in plan['families'].items():
            a = (LowHalfMHzAcquisition(plan['data_root'], spec['source'], 'cpu')
                 if spec['centre']=='low' else
                 Full90ThreeMHzAcquisition(plan['high_index'], spec['source'], 'cpu'))
            phase = FrequencyPhaseIncrement(a.observed, a.mask.numpy(),
                [spec['bands_hz'][0][0], spec['bands_hz'][-1][1]])
            values = torch.zeros(len(phase.windows), dtype=torch.float64)
            for shot in range(11):
                row = references[name, shot]
                if digest(row['path'])!=row['SHA256']:
                    raise ValueError('Saved original source prediction changed')
                prediction = torch.from_numpy(np.load(row['path']))
                value = phase._windowed(prediction)
                transfer = (value/value.abs().clamp_min(phase.prediction_floor[shot]))*phase.unit_data[shot].conj()
                discrepancy = (1-(transfer[..., 1:]*transfer[..., :-1].conj()).real).clamp_min(0)/phase.phase_scale
                values += (phase.pair_weight[shot]*discrepancy).sum((1, 2))/phase.window_denominator
            energy = phase.pair_weight.sum((0, 2, 3))
            weighted = values*spec['coefficient']/len(phase.windows)
            weighted_phase = weighted.clone() if weighted_phase is None else weighted_phase+weighted
            families[name] = {'coefficient':spec['coefficient'],
                'window_centres_us':phase.config['window_centres_us'],
                'phase_before_window_average':values.tolist(),
                'weighted_contribution_to_global_phase':weighted.tolist(),
                'data_pair_energy_relative_to_strongest_window':(energy/energy.max()).tolist(),
                'effective_weight_after_weak_window_floor':(energy/phase.window_denominator).tolist()}
            del phase, a
    result = {'status':'cached_RF_phase_windows_CPU_finished', 'seconds':time.time()-started,
        'root':str(args.root), 'material_SHA256':run['material_SHA256'],
        'families':families, 'weighted_global_phase_per_window':weighted_phase.tolist(),
        'weighted_global_phase':float(weighted_phase.sum()),
        'original_source_predictions_read':44, 'new_acoustic_solves':0,
        'new_material_derivative_probes':0, 'patient_truth_or_material_arrays_read':False,
        'GPU_work':False, 'RF_time_is_not_tissue_depth':True}
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__=='__main__':
    main()
