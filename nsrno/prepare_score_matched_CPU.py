"""Prepare one fixed-model actual score direction from the completed point05.

No patient truth arrays, wave propagation, Jacobian probes or GPU use.
"""
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
    p.add_argument('--parent-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    args = p.parse_args()
    project, source = Path.cwd().resolve(), Path(__file__).parent
    for name in ['parent_root', 'out', 'artifact_root']:
        setattr(args, name, getattr(args, name).resolve())
    if (args.parent_root != project/'out/v5/reconstruction_global_CDF_continuation_v1' or
            not args.out.is_relative_to(project/'out/v5') or args.out.exists() or
            not args.artifact_root.is_relative_to(Path.home()/'nsrno_artifacts') or args.artifact_root.exists()):
        raise ValueError('Fresh same-point CPU inputs only')
    old = json.loads((args.parent_root/'run.json').read_text())
    closed = json.loads((args.parent_root/'scientific_boundary_CPU_finalization.json').read_text())
    workers = json.loads((args.parent_root/'worker_processes.json').read_text())
    if (closed['status'] != 'completed_saved_boundary_CPU_finalization' or
            old['pid'] != 3619749 or old['start_ticks'] != 11407646 or
            any(Path('/proc', str(pid)).exists() for pid in [old['pid'], *workers['worker_pids']])):
        raise RuntimeError('Original actual trajectory must be terminal and finalized')
    point = args.parent_root/'steps/point05'
    details = json.loads((point/'direction_details.json').read_text())
    gradient = json.loads((args.parent_root/'jobs/0009_full_joint_gradient.json').read_text())
    material_path, control_path = (point/'start_material.npy').resolve(), (point/'start_control.npy').resolve()
    expected_material = '37020ad42b1738dbed9a13a2a33659b9aa508b3485de06cbf274e8f53e516493'
    expected_gradient = 'f33d0abba34e9c7a55925e1bae6c73c22466dba8cb16e4560d56fbad768a6581'
    if (digest(material_path) != expected_material or details['start_material_SHA256'] != expected_material or
            gradient['token'] != 9 or gradient['kind'] != 'gradient' or gradient['point'] != 5 or
            gradient['objective'] != 'full_joint' or gradient['source_count'] != 44 or
            gradient['gradient_SHA256'] != expected_gradient or digest(gradient['gradient_path']) != expected_gradient or
            details['data_gradient_SHA256'] != expected_gradient or len(gradient['RF_source_records']) != 44):
        raise RuntimeError('Only the exact completed same-material full44 gradient can be reused')
    hashes = dict(old['source_SHA256'])
    for name, sha in hashes.items():
        if digest(source/name) != sha:
            raise RuntimeError('A frozen prior/physical source changed')
    for name in [Path(__file__).name, 'frozen_score_guidance.py']:
        hashes[name] = digest(source/name)
    original = json.loads((Path(old['inputs_root'])/'run.json').read_text())
    for name in ['neural', 'cdf']:
        if digest(original[name+'_path']) != original[name+'_SHA256']:
            raise RuntimeError('Same fixed TRAIN products required')
    plan = json.loads((args.parent_root/'native_plan.json').read_text())
    if (digest(plan['native_anchor_material']) != old['native_anchor_material_SHA256'] or
            digest(project/'out/v5/uno_triparam_v1/known_edge_alpha.npy') != plan['known_edge_SHA256']):
        raise RuntimeError('Original coordinate anchor and known absorption must remain fixed')
    for row in gradient['RF_source_records']:
        if digest(row['path']) != row['SHA256']:
            raise RuntimeError('The required-gradient baseline cache changed')
    args.out.mkdir()
    args.artifact_root.mkdir()
    (args.out/'arrays').symlink_to(args.artifact_root, target_is_directory=True)
    started = time.time()
    run = {'status': 'preparing_fixed_model_score_CPU_inputs', 'pid': os.getpid(),
        'created_unix_s': started, 'parent_root': str(args.parent_root),
        'source_SHA256': hashes, 'material_path': str(material_path), 'control_path': str(control_path),
        'material_SHA256': expected_material, 'data_gradient_path': gradient['gradient_path'],
        'data_gradient_SHA256': expected_gradient, 'initial_RF': gradient['loss'],
        'RF_source_records': gradient['RF_source_records'],
        'neural_path': original['neural_path'], 'neural_SHA256': original['neural_SHA256'],
        'cdf_path': original['cdf_path'], 'cdf_SHA256': original['cdf_SHA256'],
        'no_new_wave_solves': True, 'new_patient_derivative_probes': 0,
        'patient_truth_read': False, 'reserved_arrays_read': False}
    atomic(args.out/'run.json', run)
    coordinates = IndependentPhysicalCoordinates(np.load(plan['native_anchor_material']),
        np.load(project/'out/v5/uno_triparam_v1/known_edge_alpha.npy'))
    control = np.load(control_path).astype(np.float64)
    if not np.array_equal(coordinates.material(control), np.load(material_path)):
        raise RuntimeError('The unchanged physical coordinates must reproduce this exact material')
    prior = FrozenScoreGuidance(original['neural_path'], original['cdf_path'])
    guide, parts = prior.guide(coordinates, control)
    z, x = np.ogrid[:961, :1121]
    active = (z >= 80)&(z < 937)&(x >= 24)&(x < 1097)
    lateral = abs((np.arange(1121)-560)*.05) <= 19.1
    direction, direction_details = common_box_direction(coordinates, control,
        np.load(gradient['gradient_path']), guide, active, lateral)
    for name, value in [('score_guide', guide), ('direction', direction)]:
        path = args.out/'arrays'/f'{name}.npy'
        np.save(path, value)
        run[name+'_path'], run[name+'_SHA256'] = str(path.resolve()), digest(path)
    run.update(status='actual_fixed_model_score_CPU_inputs_prepared', initial_guide_parts=parts,
        direction_details=direction_details, seconds=time.time()-started,
        maximum_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    atomic(args.out/'run.json', run)
    print(json.dumps({k: v for k, v in run.items() if k not in ['RF_source_records', 'source_SHA256']}), flush=True)


if __name__ == '__main__':
    main()
