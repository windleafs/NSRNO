"""CPU mechanism diagnosis using existing training finite-pair RF only.

No new forward solve, material derivative, development truth or optimizer.
"""
from pathlib import Path
import argparse
import json
import os
import time
import numpy as np
from .central_secant_data import TrainingPairs, array, tissue_label
from .native_gn_json import atomic


def symmetry(plus, minus):
    p = plus.astype(np.float64, copy=False).ravel()
    m = minus.astype(np.float64, copy=False).ravel()
    p2 = float(np.dot(p, p)); m2 = float(np.dot(m, m))
    cross = float(np.dot(p, m))
    odd2 = max(0., .25 * (p2 + m2 - 2 * cross))
    even2 = max(0., .25 * (p2 + m2 + 2 * cross))
    return {
        'plus_minus_cosine': cross / max(np.sqrt(p2 * m2), 1e-300),
        'odd_energy_fraction': odd2 / max(odd2 + even2, 1e-300),
        'even_energy_fraction': even2 / max(odd2 + even2, 1e-300),
        'plus_RMS': float(np.sqrt(p2 / p.size)),
        'minus_RMS': float(np.sqrt(m2 / p.size)),
        'odd_RMS': float(np.sqrt(odd2 / p.size)),
        'even_RMS': float(np.sqrt(even2 / p.size)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('Keep existing computed result')
    started = time.time()
    index = json.loads(a.index.read_text())
    pairs = TrainingPairs(index)
    rows = []
    for kind, groups in pairs.groups.items():
        for members in groups:
            minus, plus = sorted(members, key=lambda r: r['pair_sign'])
            base = pairs.central[(plus['pair_group'], plus['valid_samples'])]
            assert {r['role'] for r in [base, minus, plus]} == {'train'}
            assert {r['case'] for r in [base, minus, plus]} == {'Neg_07_Left'}
            bf = array(base['raw_rf']).astype(np.float64)
            dp = array(plus['raw_rf']).astype(np.float64) - bf
            dm = array(minus['raw_rf']).astype(np.float64) - bf
            parameter = {'pure_c': 0, 'pure_rho': 1, 'alpha': 2}[kind]
            bmaterial = tissue_label(base, pairs.edge)[parameter, 5:].astype(np.float64)
            mp = tissue_label(plus, pairs.edge)[parameter, 5:].astype(np.float64) - bmaterial
            mm = tissue_label(minus, pairs.edge)[parameter, 5:].astype(np.float64) - bmaterial
            rows.append({
                'kind': kind, 'pair_group': plus['pair_group'],
                'baseline_id': base['id'], 'valid_samples': plus['valid_samples'],
                'RF': symmetry(dp, dm), 'material': symmetry(mp, mm),
                'existing_finite_changes_are_not_infinitesimal_derivatives': True,
            })
    groups = {}
    for kind in pairs.groups:
        values = [r for r in rows if r['kind'] == kind]
        groups[kind] = {'count': len(values)}
        for name in ['RF', 'material']:
            groups[kind][name] = {key: {
                'mean': float(np.mean([r[name][key] for r in values])),
                'min': float(np.min([r[name][key] for r in values])),
                'max': float(np.max([r[name][key] for r in values])),
            } for key in ['plus_minus_cosine', 'odd_energy_fraction', 'even_energy_fraction']}
    result = {'finished_unix_s': time.time(), 'seconds': time.time() - started,
              'groups': groups, 'details': rows, 'new_wave_solves': 0,
              'new_material_derivatives': 0, 'GPU_work': False,
              'development_arrays_read': False, 'reserved_case_unread': 'Neg_47_Left',
              'purpose': 'decide whether a signed linear residual update is a justified local approximation',
              'warning': 'finite input RF need not be odd even if target material changes are opposite'}
    a.out.mkdir()
    atomic(a.out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}), flush=True)


if __name__ == '__main__':
    main()
