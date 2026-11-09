"""Use the corrected TRAIN prior at the identical saved full44 gradient point."""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable THP')
from pathlib import Path
import argparse
import json
import os
import resource
import time
import numpy as np
from .frozen_score_guidance import FrozenScoreGuidance
from .independent_physical_coordinates import IndependentPhysicalCoordinates
from .boxed_common_descent import common_box_direction
from .low_frequency_0p5 import atomic, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    a = p.parse_args()
    project, source = Path.cwd().resolve(), Path(__file__).parent
    for name in ['model_root', 'out', 'artifact_root']:
        setattr(a, name, getattr(a, name).resolve())
    if (a.model_root != project/'out/v5/anatomical_material_denoisers_v1' or
            not a.out.is_relative_to(project/'out/v5') or a.out.exists() or
            not a.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or a.artifact_root.exists()):
        raise ValueError('Fresh samepoint inputs for the once-trained anatomical population only')
    fit = json.loads((a.model_root/'run.json').read_text())
    if (fit['status'] != 'anatomical_TRAIN_material_denoisers_training_finished' or
            fit['completed_steps'] != 3000 or fit['training_case'] != 'Neg_07_Left' or
            fit['development_arrays_read'] or fit['reserved_arrays_read'] or fit['RF_arrays_read'] or
            fit['population']['retained_materials_per_field'] != [466, 516, 516] or
            Path('/proc', str(fit['pid'])).exists()):
        raise RuntimeError('Fixed final TRAIN-only training must be completed and gone')
    base = json.loads((project/'out/v5/fixed_model_score_inputs_CPU_v1/run.json').read_text())
    comparison = project/'out/v5/reconstruction_fixed_model_score_v1'
    compared = json.loads((comparison/'run.json').read_text())
    workers = json.loads((comparison/'worker_processes.json').read_text())
    if (base['status'] != 'actual_fixed_model_score_CPU_inputs_prepared' or
            compared['status'] != 'matched_score_actual_stage_finished' or
            any(Path('/proc', str(pid)).exists() for pid in [base['pid'], compared['pid'], *workers['worker_pids']])):
        raise RuntimeError('The previous samepoint control must be finished')
    model_path = a.model_root/'products/prior.pt'
    hashes = {**base['source_SHA256'], **fit['source_SHA256'], Path(__file__).name: digest(source/Path(__file__).name)}
    for name, sha in hashes.items():
        if digest(source/name) != sha:
            raise RuntimeError('Frozen source changed: '+name)
    for key in ['material', 'data_gradient', 'cdf']:
        if digest(base[key+'_path']) != base[key+'_SHA256']:
            raise RuntimeError('Same material, full44 gradient and original CDF must be retained')
    if digest(model_path) != fit['prior_SHA256']:
        raise RuntimeError('The new fixed final model changed')
    plan = json.loads((Path(base['parent_root'])/'native_plan.json').read_text())
    a.out.mkdir()
    a.artifact_root.mkdir()
    (a.out/'arrays').symlink_to(a.artifact_root, target_is_directory=True)
    started = time.time()
    keep = ['parent_root', 'material_path', 'control_path', 'material_SHA256', 'data_gradient_path',
        'data_gradient_SHA256', 'initial_RF', 'RF_source_records', 'cdf_path', 'cdf_SHA256']
    run = {k: base[k] for k in keep}
    run.update(status='preparing_anatomical_population_score_CPU_inputs', pid=os.getpid(),
        created_unix_s=started, source_SHA256=hashes, neural_path=str(model_path.resolve()),
        neural_SHA256=fit['prior_SHA256'], population_root=str(a.model_root),
        compared_score_root=str(comparison), compared_score_RF=compared['final_RF'],
        compared_score_material_SHA256=compared['material_SHA256'],
        same_frozen_CDF_and_same_completed_gradient=True, no_new_wave_solves=True,
        new_patient_derivative_probes=0, patient_truth_read=False, reserved_arrays_read=False)
    atomic(a.out/'run.json', run)
    coordinates = IndependentPhysicalCoordinates(np.load(plan['native_anchor_material']),
        np.load(project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'))
    control = np.load(base['control_path']).astype(np.float64)
    if not np.array_equal(coordinates.material(control), np.load(base['material_path'])):
        raise RuntimeError('The identical saved physical material is required')
    prior = FrozenScoreGuidance(model_path, base['cdf_path'])
    guide, parts = prior.guide(coordinates, control)
    z, x = np.ogrid[:961, :1121]
    active = (z >= 80)&(z < 937)&(x >= 24)&(x < 1097)
    lateral = abs((np.arange(1121)-560)*.05) <= 19.1
    direction, details = common_box_direction(coordinates, control,
        np.load(base['data_gradient_path']), guide, active, lateral)
    for name, value in [('score_guide', guide), ('direction', direction)]:
        path = a.out/'arrays'/f'{name}.npy'
        np.save(path, value)
        run[name+'_path'], run[name+'_SHA256'] = str(path.resolve()), digest(path)
    run.update(status='actual_fixed_model_score_CPU_inputs_prepared', initial_guide_parts=parts,
        direction_details=details, seconds=time.time()-started,
        maximum_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    atomic(a.out/'run.json', run)
    print(json.dumps({k: v for k, v in run.items() if k not in ['RF_source_records', 'source_SHA256']}), flush=True)


if __name__ == '__main__':
    main()
