"""Post-hoc failure diagnosis using the completed, already-computed RF gradient only."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from .native_gn_coordinates import Coordinates
from .native_gn_json import atomic
from .v5_materials import evaluation_material


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.out.exists():
        raise RuntimeError('Preserve existing post-hoc diagnosis')
    run = read(root / 'run.json')
    if run['status'] != 'native_RF_L2_control_finished':
        raise RuntimeError('Wait for completed original RF reconstruction and actual maps')
    workers = read(root / 'worker_processes.json')
    if any(pid and Path('/proc', str(pid)).exists()
           for pid in [run['pid'], run.get('child_pid'), *workers['worker_pids']]):
        raise RuntimeError('Original reconstruction processes have not all exited')
    if not (root / 'step001/maps.png').exists():
        raise RuntimeError('Actual full material maps are missing')
    material = np.load(root / 'accepted_material.npy').astype(np.float64)
    if sha(root / 'accepted_material.npy') != run['material_SHA256']:
        raise RuntimeError('Gradient/accepted material lineage changed')
    if sha(root / 'step001/material.npy') != run['material_SHA256']:
        raise RuntimeError('Final material changed')
    point = read(root / 'accepted_point.json')
    if point['accepted_step'] != run['accepted_steps']:
        raise RuntimeError('Accepted gradient is not at returned material')
    initial = np.load(root / 'initial_material.npy')
    edge = np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    co = Coordinates(initial, edge)
    u = np.load(root / 'accepted_log_coordinates.npy')
    if not np.array_equal(co.material(u), material.astype(np.float32)):
        raise RuntimeError('Accepted coordinates do not reproduce material')
    metric = np.broadcast_to(np.sqrt(np.asarray(run['constant_diagonal_values']))[:, None, None],
                             material.shape).copy()
    metric[:, ~co.mask] = 1.
    saved_scaled = np.load(root / 'accepted_scaled_gradient.npy')
    tv, tv_gradient, _, _ = co.regularization(u, run['TV_weight'])
    if abs(tv - point['TV']) > 1e-12:
        raise RuntimeError('Saved TV and current TV disagree')
    data_u = saved_scaled * metric - tv_gradient
    # u_c=log(c)/.05, u_Z=log(c*rho)/.10, u_a=log(tissue_alpha)/.70.
    # Independent physical log variables give g_vc=g_uc+.5*g_uZ, g_vrho=g_uZ.
    attenuation_gradient = np.divide(data_u[2], .70 * (material[2] - edge),
                                     out=np.zeros_like(data_u[2]), where=co.mask)
    physical = np.stack([(data_u[0] + .5 * data_u[1]) / (.05 * material[0]),
                         data_u[1] / (.10 * material[1]), attenuation_gradient])
    # Truth is read only after optimization, for diagnosis, never for an update or model choice.
    target = evaluation_material(run['sample']).astype(np.float64)
    if target.shape != material.shape:
        raise RuntimeError('Post-hoc truth grid differs')
    z = (np.arange(material.shape[1]) - 60) * .05
    x = (np.arange(material.shape[2]) - 560) * .05
    fov = (abs(x)[None, :] <= 19.1) & (z[:, None] >= 1) & (z[:, None] <= 43)
    masks = {'FOV_1_43mm': fov, 'depth_1_5mm': fov & (z[:, None] < 5),
             'depth_5_12mm': fov & (z[:, None] >= 5) & (z[:, None] < 12),
             'depth_12_25mm': fov & (z[:, None] >= 12) & (z[:, None] < 25),
             'depth_25_43mm': fov & (z[:, None] >= 25)}
    regions = {}
    names = ['c_m_s', 'density_kg_m3', 'tissue_alpha_dB_cm_MHz1p1']
    for name, mask in masks.items():
        fields = {}
        for j, field in enumerate(names):
            g = physical[j][mask]
            error = target[j][mask] - material[j][mask]
            dot = float(np.dot(g, error))
            norm = float(np.linalg.norm(g) * np.linalg.norm(error))
            fields[field] = {
                'RF_directional_derivative_toward_true_material': dot,
                'negative_RF_gradient_cosine_with_true_minus_estimate': -dot / norm if norm > 0 else None,
                'gradient_L2_native_physical_units': float(np.linalg.norm(g)),
                'truth_minus_estimate_RMS': float(np.sqrt(np.mean(error ** 2))),
            }
        regions[name] = {
            'fields': fields,
            'joint_RF_directional_derivative_toward_truth_in_this_region':
                sum(v['RF_directional_derivative_toward_true_material'] for v in fields.values()),
        }
    report = {
        'observed_unix_s': time.time(), 'root': str(root), 'sample': run['sample'],
        'material_SHA256': run['material_SHA256'], 'accepted_step': run['accepted_steps'],
        'RF_loss': point['original_RF_loss'], 'gradient_evaluation': point['evaluation'],
        'saved_gradient_SHA256': sha(root / 'accepted_scaled_gradient.npy'),
        'new_wave_solves': 0, 'new_material_derivatives': 0, 'GPU_work': False,
        'truth_read_only_for_completed_post_hoc_diagnosis': True,
        'truth_used_by_optimizer': False, 'reserved_case_unread': 'Neg_47_Left',
        'TV_gradient_subtracted_exactly_in_original_log_coordinates': True,
        'independent_c_rho_alpha_gradient_not_mislabelled_logZ_as_density': True,
        'path': 'm(s)=m+s*(truth-m), changed only inside each specified region',
        'derivatives_are_local_not_finite_RF_path_values': True,
        'no_global_uniqueness_convergence_or_identifiability_claim': True,
        'no_direction_is_used_for_reconstruction': True, 'regions': regions,
    }
    args.out.mkdir(); atomic(args.out / 'saved_gradient_diagnosis.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
