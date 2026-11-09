"""CPU postprocessing of fixed inverse results; all development alpha pairs."""
from pathlib import Path
import argparse
import json
import os
import time
import numpy as np


def response(prediction, target):
    result = []
    for v, t in zip(prediction[:, 5:], target[:, 5:]):
        v = v.astype(np.float64).ravel()
        t = t.astype(np.float64).ravel()
        vn, tn = np.linalg.norm(v), np.linalg.norm(t)
        result.append({
            'predicted_rms': float(np.sqrt(np.mean(v*v))),
            'target_rms': float(np.sqrt(np.mean(t*t))),
            'relative_error': float(np.linalg.norm(v-t)/tn) if tn > 1e-5 else None,
            'cosine': float(np.dot(v,t)/(vn*tn)) if tn > 1e-5 and vn > 1e-10 else None,
            'RMS_gain': float(vn/tn) if tn > 1e-5 else None,
        })
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, default=Path('out/v5/response_constrained_experiment_v1'))
    p.add_argument('--previous', type=Path, default=Path('out/v5/inverse_angular_spectral_uno_v1'))
    p.add_argument('--index', type=Path, default=Path('out/v5/inverse_angular_spectral_v1/index.json'))
    p.add_argument('--out', type=Path, default=Path('out/v5/response_constrained_comparison_v1'))
    a = p.parse_args()
    parent = json.loads((a.parent/'run.json').read_text())
    if parent['status'] != 'constrained_training_and_evaluations_finished':
        raise RuntimeError('Wait for the existing fixed training/evaluation process')
    for pid in [parent['pid'], parent.get('child_pid')]:
        if pid and Path(f'/proc/{pid}').exists():
            raise RuntimeError('The previous GPU process must actually exit first')
    if a.out.exists():
        raise RuntimeError('Read existing comparison; do not overwrite it')
    current = Path(parent['model'])
    previous = json.loads((a.previous/'evaluation/metrics.json').read_text())
    new = json.loads((current/'evaluation/metrics.json').read_text())
    before = {r['id']:r for r in previous['cases']}
    after = {r['id']:r for r in new['cases']}
    if set(before) != set(after):
        raise ValueError('Different development records')
    rows = json.loads(a.index.read_text())['records']
    groups = {}
    for r in rows:
        if r['role'] != 'train' and r['kind'] == 'alpha':
            groups.setdefault(r['pair_group'], []).append(r)
    details, maps = [], []
    for group in sorted(groups):
        minus, plus = sorted(groups[group], key=lambda r:r['pair_sign'])
        exact = np.load(plus['label'])-np.load(minus['label'])
        values = []
        for run in [a.previous, current]:
            values.append(np.load(run/'evaluation'/f"{plus['id']}.prediction.npy")-
                          np.load(run/'evaluation'/f"{minus['id']}.prediction.npy"))
        detail = {'pair_group':group, 'parent_id':minus['parent_id'],
                  'previous':response(values[0],exact), 'current':response(values[1],exact)}
        details.append(detail)
        maps.append((exact, *values))
    strata = {}
    for name, metric in [('previous',previous),('current',new)]:
        strata[name] = {}
        for has_lf in [False, True]:
            selected = [r for r in metric['cases'] if r['kind'] == 'original'
                        and r['has_low_frequency'] == has_lf]
            strata[name]['HF_and_3MHz' if has_lf else 'HF_only'] = {
                'records':len(selected),
                'mean_RMSE_c_rho_alpha':np.mean([r['model']['rmse'] for r in selected],axis=0).tolist(),
                'training_mean_RMSE_c_rho_alpha':np.mean([r['training_mean']['rmse'] for r in selected],axis=0).tolist(),
            }
    alpha_summary = {}
    for name in ['previous','current']:
        alpha_summary[name] = {
            'pairs':len(details),
            'alpha_mean_relative_error':float(np.mean([r[name][2]['relative_error'] for r in details])),
            'alpha_mean_cosine':float(np.mean([r[name][2]['cosine'] for r in details])),
            'alpha_mean_RMS_gain':float(np.mean([r[name][2]['RMS_gain'] for r in details])),
            'c_mean_cross_talk_RMS_m_s':float(np.mean([r[name][0]['predicted_rms'] for r in details])),
            'rho_mean_cross_talk_RMS_kg_m3':float(np.mean([r[name][1]['predicted_rms'] for r in details])),
        }
    result = {'development_used_for_method_design':True,
              'new_test_case':False, 'training_or_parameter_updates':False,
              'patient_wave_solves':0, 'created_unix_s':time.time(), 'pid':os.getpid(),
              'full_FOV_required_for_joint_imaging':True,
              'development_alpha_response':alpha_summary,'independent_alpha_pairs':details,
              'original_records_by_sources':strata,
              'selected_cases':{k:{'previous':before[k],'current':after[k]}
                                for k in ['train_200','val_059']},
              'training_constraint_certificate':parent['training_response_constraints']}
    a.out.mkdir()
    (a.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(len(details),3,figsize=(11,3.7*len(details)),layout='constrained')
    bound = max(float(np.max(abs(v[0][2,5:]))) for v in maps)
    for row,(detail,triplet) in enumerate(zip(details,maps)):
        for col,image in enumerate(triplet):
            ax = axes[row,col]
            im = ax.imshow(image[2],extent=[-19.1,19.1,43,0],aspect='equal',cmap='RdBu_r',
                           vmin=-bound,vmax=bound,interpolation='none')
            title = ['Target independent alpha change','Previous UNO','Response-constrained UNO'][col]
            if col:
                q = detail[['previous','current'][col-1]]
                title += (f"\nrelative error {q[2]['relative_error']:.3f}, gain {q[2]['RMS_gain']:.3f}"
                          f"\nc/rho cross-talk RMS {q[0]['predicted_rms']:.2f}/{q[1]['predicted_rms']:.2f}")
            ax.set_title(title,fontsize=9)
            ax.set_xlabel('Lateral (mm)')
            ax.set_ylabel(detail['pair_group']+'\nDepth (mm)')
            fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('All development alpha-only pairs; frozen final models\n'
                 'Delta alpha in dB/(cm MHz^1.1); c and rho targets unchanged')
    fig.savefig(a.out/'all_development_alpha_responses.png',dpi=150)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ['independent_alpha_pairs','selected_cases']}),flush=True)


if __name__ == '__main__':
    main()
