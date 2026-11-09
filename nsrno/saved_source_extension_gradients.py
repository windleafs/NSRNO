"""Exact reuse of completed true-source gradients at the same unchanged material."""
from pathlib import Path
import json
import numpy as np
from .low_frequency_0p5 import digest


def load_saved_batches(parent, material_sha, family_names, calibration):
    parent = Path(parent)
    run = json.loads((parent/'run.json').read_text())
    if (run['status'] != 'source_extension_native_trajectory_finished' or
        run['accepted_ray_steps'] != 0 or run['initial_material_SHA256'] != material_sha or
        run['material_SHA256'] != material_sha or
        run['source_extension_work_ledger']['completed_gradient_jobs'] != 2):
        raise RuntimeError('Require the two completed gradients of an unchanged accepted material')
    receipts = sorted((parent/'jobs').glob('*_actual_corrected_gradient.json'))
    if len(receipts) != 2:
        raise RuntimeError('Inspect all actual gradient receipts; never silently recompute')
    loss, gradient, batches, provenance = 0., np.zeros((3, 961, 1121), np.float64), [], []
    for receipt in receipts:
        row = json.loads(receipt.read_text())
        if row['kind'] != 'ray_gradient' or set(row['sources']) != set(family_names):
            raise RuntimeError('Incomplete true-source gradient batch')
        total = np.zeros_like(gradient)
        files = []
        for family in family_names:
            shot = row['sources'][family]
            if row['details'][family]['calibration'] != calibration[family]:
                raise RuntimeError('The saved derivative used another objective calibration')
            name = f'token{row["token"]:04d}_{family}_{shot:02d}_control_gradient.npy'
            matches = list((parent/'workers_wri').glob('pool*/rank*/'+name))
            if len(matches) != 1:
                raise RuntimeError('A saved source gradient is missing or ambiguous')
            p = matches[0]
            g = np.load(p)
            if g.shape != gradient.shape or g.dtype != np.float64 or not np.isfinite(g).all():
                raise RuntimeError('Invalid saved native gradient; do not replace with a new probe')
            total += 11*g
            files.append({'family': family, 'shot': shot, 'path': str(p.resolve()), 'SHA256': digest(p)})
        gradient += total/2
        loss += row['loss']/2
        batches.append(row['sources'])
        provenance.append({'receipt': str(receipt.resolve()), 'receipt_SHA256': digest(receipt),
                           'original_gradient_token': row['token'], 'source_gradient_files': files})
    result = {'token': 'reused_two_same_material_actual_gradient_batches', 'loss': loss,
        'source_batches': batches, 'original_native_gradient_jobs_reused': 2,
        'new_native_gradient_jobs': 0, 'new_acoustic_propagations': 0,
        'material_SHA256': material_sha, 'source_provenance': provenance,
        'this_is_not_a_new_patient_material_derivative_probe': True}
    return result, gradient
