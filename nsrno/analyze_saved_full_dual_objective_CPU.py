"""Decompose the completed material step using its saved native products only.

No task-result traversal, patient truth, new propagation or derivative probe.
The candidate's first required K action was evaluated at the same old y;
this permits a fixed-direction operator comparison, not a global gain claim.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from .full_source_dual_cg import inner, dual_bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists():
        print(out.read_text(), flush=True)
        return
    torch.set_num_threads(1)
    started = time.time()
    run = json.loads((root/'run.json').read_text())
    if run['status'] != 'full_source_dual_native_stage_finished':
        raise RuntimeError('Analyze a completed fixed experiment only')
    parent = Path(run['parent_root'])
    parameters = json.loads((root/'full_dual_calibration.json').read_text())
    step = json.loads((root/'steps/step01.json').read_text())
    line = step['lines'][0]
    rows = {}
    totals = {k: 0. for k in ['physical_RF_before_same4', 'physical_RF_after_same4',
        'D0_recurrence', 'D1_recurrence', 'data_term', 'operator_term', 'inner_readjustment']}
    for name, shot in step['sources'].items():
        rank = 0 if name in ['lf', 'lf3'] else 1
        suffix = Path(f'pool01/rank{rank:02d}/tasks')
        initial = ((root/'workers_full'/suffix/f'token0002/{name}_{shot:02d}')
            if name in ['lf', 'point3'] else
            (parent/'workers_full'/suffix/f'token0001/{name}_{shot:02d}'))
        candidate = root/'workers_full'/suffix/f'token0003/{name}_{shot:02d}'
        old = torch.load(initial/'dual_state.pt', map_location='cpu', weights_only=False)
        new = torch.load(candidate/'dual_state.pt', map_location='cpu', weights_only=False)
        with np.load(candidate/'operator_actions/action000.npz') as saved:
            p, K1y0 = saved['p'].copy(), torch.from_numpy(saved['Kp'].copy())
        if not np.array_equal(p, old['y'].numpy()) or not torch.equal(old['weights'], new['weights']):
            raise RuntimeError('Warm required action did not use the identical old dual vector')
        w, lam = old['weights'], old['lambda_squared']
        if lam != new['lambda_squared']:
            raise RuntimeError('Regularization changed within this step')
        y0, y1, r0, r1 = old['y'], new['y'], old['rhs'], new['rhs']
        aa0 = inner(y0, old['Ky'], w)
        aa1 = inner(y0, K1y0, w)
        D0 = dual_bounds(r0, y0, old['Ky'], lam, w)['dual']
        D1 = dual_bounds(r1, y1, new['Ky'], lam, w)['dual']
        D1_at_y0 = dual_bounds(r1, y0, K1y0, lam, w)['dual']
        data = inner(y0, r1-r0, w)
        operator = -.5*(aa1-aa0)
        relax = D1-D1_at_y0
        factor = 22*parameters[name]['normalization']
        native_D0 = parameters[name]['anchor_actual_inner_bounds']['dual']
        row = {'physical_RF_before_same4': 11*inner(r0, r0, w),
            'physical_RF_after_same4': 11*inner(r1, r1, w),
            'D0_recurrence': factor*D0, 'D1_recurrence': factor*D1,
            'data_term': factor*data, 'operator_term': factor*operator,
            'inner_readjustment': factor*relax,
            'fixed_old_dual_direction_new_to_old_quadratic_energy_ratio': aa1/aa0,
            'native_start_D_minus_recurrence_D': factor*(native_D0-D0),
            'algebra_identity_residual': factor*(D1-D0-data-operator-relax),
            'old_and_new_y_are_full_receiver_vectors': True,
            'first_required_new_K_action_direction_exactly_matches_old_y': True}
        rows[name] = row
        for k in totals:
            totals[k] += row[k]
    totals['actual_D0'] = line['full_dual_before']
    totals['actual_D1'] = line['full_dual_after']
    totals['actual_D_change'] = totals['actual_D1']-totals['actual_D0']
    totals['recurrence_D_change'] = totals['D1_recurrence']-totals['D0_recurrence']
    totals['native_minus_recurrence_change'] = totals['actual_D_change']-totals['recurrence_D_change']
    result = {'status': 'completed_saved_full_dual_step_objective_decomposition_CPU',
        'seconds': time.time()-started, 'rows': rows, 'totals': totals,
        'new_acoustic_propagations': 0, 'new_material_derivative_probes': 0,
        'patient_truth_read': False, 'complete_job_JSON_exported': False,
        'operator_energy_ratio_is_one_fixed_direction_not_global_gain': True,
        'recurrence_quadratic_decomposition_is_a_numerical_estimate': True}
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
