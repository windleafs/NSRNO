"""CPU comparison on identical labels, including existing calibration pairs."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from .native_gn_json import atomic


def response(pred_minus,pred_plus,target_minus,target_plus):
    delta=(pred_plus-pred_minus)[:,5:].astype(np.float64).reshape(3,-1)
    exact=(target_plus-target_minus)[:,5:].astype(np.float64).reshape(3,-1)
    result=[]
    for j in range(3):
        v=delta[j];t=exact[j];norm=np.linalg.norm(t);pnorm=np.linalg.norm(v)
        result.append({'target_difference_rms':float(np.sqrt(np.mean(t*t))),
            'predicted_difference_rms':float(np.sqrt(np.mean(v*v))),
            'relative_difference_error':float(np.linalg.norm(v-t)/norm) if norm>1e-5 else None,
            'difference_cosine':float(np.dot(v,t)/(norm*pnorm)) if norm>1e-5 and pnorm>1e-10 else None})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--old-run',type=Path,default=Path('out/v5/inverse_uno_v1'))
    p.add_argument('--new-run',type=Path,default=Path('out/v5/inverse_angular_spectral_uno_v1'))
    p.add_argument('--old-index',type=Path,default=Path('out/v5/inverse_focus_v1/index.json'))
    p.add_argument('--new-index',type=Path,default=Path('out/v5/inverse_angular_spectral_v1/index.json'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    old_path=args.old_run/'evaluation/metrics.json';new_path=args.new_run/'evaluation/metrics.json'
    if not new_path.exists():raise RuntimeError('Fixed-checkpoint evaluation has not finished; keep waiting on the existing pipeline')
    if args.out.exists():raise RuntimeError('Inspect existing comparison; do not overwrite it')
    old=json.loads(old_path.read_text());new=json.loads(new_path.read_text())
    old_index=json.loads(args.old_index.read_text());new_index=json.loads(args.new_index.read_text())
    oi={r['id']:r for r in old_index['records']};ni={r['id']:r for r in new_index['records']}
    if set(oi)!=set(ni):raise RuntimeError('Different observation/target record sets')
    for key in oi:
        if Path(oi[key]['label']).resolve()!=Path(ni[key]['label']).resolve():
            raise RuntimeError(f'Different labels for {key}')
    oc={r['id']:r for r in old['cases']};nc={r['id']:r for r in new['cases']}
    if set(oc)!=set(nc):raise RuntimeError('Different evaluation subsets')
    pairs_old={r['pair_group']:r for r in old['independent_pairs']}
    pairs_new={r['pair_group']:r for r in new['independent_pairs']}
    if set(pairs_old)!=set(pairs_new):raise RuntimeError('Different pre-existing independent pairs')
    pair_summary=[]
    for kind in sorted({r['kind'] for r in pairs_new.values()}):
        names=[name for name,row in pairs_new.items() if row['kind']==kind]
        fields=[]
        for j in range(3):
            a=[pairs_old[k]['maps_c_rho_alpha'][j]['relative_difference_error'] for k in names]
            b=[pairs_new[k]['maps_c_rho_alpha'][j]['relative_difference_error'] for k in names]
            fields.append({'old_mean_relative_error':float(np.mean([v for v in a if v is not None])) if any(v is not None for v in a) else None,
                'new_mean_relative_error':float(np.mean([v for v in b if v is not None])) if any(v is not None for v in b) else None})
        pair_summary.append({'kind':kind,'pairs':len(names),'maps_c_rho_alpha':fields})
    controls=[]
    for key in sorted(nc):
        suffix=next((s for s in ['_cswap','_calibuni','_reseed'] if key.endswith(s)),None)
        if suffix is None:continue
        parent=key[:-len(suffix)]
        if parent not in nc:continue
        t0=np.load(ni[parent]['label']);t1=np.load(ni[key]['label'])
        target_rms=np.sqrt(np.mean((t1[:,5:].astype(np.float64)-t0[:,5:])**2,axis=(1,2)))
        changed=target_rms>np.array([1e-3,1e-3,1e-6])
        row={'base':parent,'variant':key,'target_difference_rms':target_rms.tolist(),
            'fields_that_change':np.flatnonzero(changed).tolist(),
            'single_parameter_control':bool(changed.sum()==1),'predictions':{}}
        for name,run in [('old',args.old_run),('new',args.new_run)]:
            p0=np.load(run/'evaluation'/f'{parent}.prediction.npy')
            p1=np.load(run/'evaluation'/f'{key}.prediction.npy')
            row['predictions'][name]=response(p0,p1,t0,t1)
        controls.append(row)
    selected={key:{'old':oc[key]['model'],'new':nc[key]['model'],
        'new_by_depth_mm':nc[key].get('by_depth_mm')} for key in ['train_200','val_059'] if key in nc}
    report={'CPU_only':True,'patient_wave_solves':0,'evaluation_records':len(nc),
        'identical_label_paths_verified':True,'old_step':old['step'],'new_step':new['step'],
        'old_summary':old['summary'],'new_summary':new['summary'],
        'selected_cases':selected,'independent_pair_summary':pair_summary,
        'existing_calibration_controls':controls,
        'new_weights_SHA256':new.get('weights_sha256'),
        'comparison_code_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'joint_native_reconstruction_success_claimed':False}
    args.out.mkdir(parents=True);atomic(args.out/'comparison.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='existing_calibration_controls'}),flush=True)


if __name__=='__main__':main()
