"""CPU comparison of completed raw RF and focused operators; no new inference."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def mean_or_none(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def response(value,exact):
    result=[]
    for v,t in zip(value[:,5:].astype(np.float64),exact[:,5:].astype(np.float64)):
        vn,tn=np.linalg.norm(v),np.linalg.norm(t)
        result.append({'predicted_RMS':float(np.sqrt(np.mean(v*v))),
            'target_RMS':float(np.sqrt(np.mean(t*t))),
            'relative_error':float(np.linalg.norm(v-t)/tn) if tn>1e-5 else None,
            'cosine':float(np.sum(v*t)/(vn*tn)) if tn>1e-5 and vn>1e-10 else None,
            'RMS_gain':float(vn/tn) if tn>1e-5 else None})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('out/v5/raw_rf_inverse_experiment_v1'))
    p.add_argument('--out',type=Path,default=Path('out/v5/raw_rf_inverse_comparison_v1'))
    a=p.parse_args();run=read(a.root/'run.json');model=Path(run['model']);training=read(model/'run.json')
    if run['status']!='raw_RF_inverse_training_and_evaluations_finished':
        raise RuntimeError('Wait for the existing experiment and all CPU maps')
    for pid in {run['pid'],run.get('child_pid'),training['pid']}:
        if pid and Path(f'/proc/{pid}').exists():raise RuntimeError('Existing experiment still has a live process')
    if a.out.exists():raise RuntimeError('Inspect the already written comparison')
    index_path=Path(training['index'])
    if hashlib.sha256(index_path.read_bytes()).hexdigest()!=training['index_sha256']:
        raise RuntimeError('Fixed input index changed')
    index=read(index_path);rows=index['records'];development=[r for r in rows if r['role']!='train']
    models={'angular_spectral':Path('out/v5/inverse_angular_spectral_uno_v1'),
        'response_constrained':Path('out/v5/inverse_response_constrained_uno_v1'),
        'coverage_60':Path('out/v5/inverse_material_coverage_60_v1'),
        'coverage_90':Path('out/v5/inverse_material_coverage_90_v1'),'raw_RF':model}
    metrics={name:read(path/'evaluation/metrics.json') for name,path in models.items()}
    by_id={name:{r['id']:r for r in metric['cases']} for name,metric in metrics.items()}
    ids={r['id'] for r in development}
    if len(ids)!=32 or any(not ids<=set(v) for v in by_id.values()):raise ValueError('Different development records')
    comparison={}
    for name,values in by_id.items():
        comparison[name]={}
        for label,chosen in [('original_LF_available',[r for r in development if r['kind']=='original']),
            ('original_paired_window',[r for r in development if r['id'] in ['train_200','val_059']]),
            ('original_existing_short_window',[r for r in development if r['kind']=='original' and r['id'] not in ['train_200','val_059']]),
            ('independent_alpha',[r for r in development if r['kind']=='alpha']),
            ('trajectory',[r for r in development if r['kind']=='trajectory'])]:
            selected=[values[r['id']] for r in chosen]
            comparison[name][label]={'records':len(selected),
                'mean_RMSE_c_rho_alpha':np.mean([r['model']['rmse'] for r in selected],axis=0).tolist(),
                'its_training_mean_RMSE':np.mean([r['training_mean']['rmse'] for r in selected],axis=0).tolist()}
    groups={}
    for row in development:
        if row['kind']=='alpha':groups.setdefault(row['pair_group'],[]).append(row)
    alpha_details=[];images=[]
    for group,members in sorted(groups.items()):
        minus,plus=sorted(members,key=lambda r:r['pair_sign'])
        exact=np.load(plus['label'])-np.load(minus['label'])
        predicted={name:np.load(path/'evaluation'/f"{plus['id']}.prediction.npy")-
            np.load(path/'evaluation'/f"{minus['id']}.prediction.npy") for name,path in models.items()}
        alpha_details.append({'pair_group':group,'parent_id':minus['parent_id'],
            'models':{name:response(value,exact) for name,value in predicted.items()}})
        images.append((exact,predicted))
    alpha_summary={}
    for name in models:
        values=[p['models'][name] for p in alpha_details]
        alpha_summary[name]={'pairs':len(values),
            'alpha_mean_relative_error':float(np.mean([v[2]['relative_error'] for v in values])),
            'alpha_mean_cosine':mean_or_none([v[2]['cosine'] for v in values]),
            'alpha_mean_RMS_gain':float(np.mean([v[2]['RMS_gain'] for v in values])),
            'c_mean_cross_talk_RMS_m_s':float(np.mean([v[0]['predicted_RMS'] for v in values])),
            'rho_mean_cross_talk_RMS_kg_m3':float(np.mean([v[1]['predicted_RMS'] for v in values]))}
    previous_pairs=read('out/v5/material_coverage_training_pairs_90_v1/metrics.json')['pairs']
    old={p['pair_group']:p for p in previous_pairs}
    raw={p['pair_group']:p for p in metrics['raw_RF']['independent_pairs'] if p['role']=='train'}
    metadata={r['pair_group']:r for r in rows if r['role']=='train' and 'field_scale_mm' in r and 'pair_group' in r}
    individual=[]
    for group,meta in sorted(metadata.items()):
        changed=meta['changed_parameter'];detail={}
        for name,pair in [('coverage_90',old[group]),('raw_RF',raw[group])]:
            value=pair['maps_c_rho_alpha'][changed]
            detail[name]={'relative_error':value['relative_difference_error'],
                'cosine':value['difference_cosine'],
                'RMS_gain':value['predicted_difference_rms']/value['target_difference_rms'],
                'all_parameters':pair['maps_c_rho_alpha']}
        individual.append({'pair_group':group,'kind':meta['kind'],'parent_id':meta['parent_id'],
            'field_scale_mm':meta['field_scale_mm'],'models':detail})
    if len(individual)!=24:raise ValueError('Missing one or more new independent training responses')
    native={name:{sample:read(path/'evaluation'/f'{sample}_native_full.metrics.json')
        for sample in ['train_200','val_059']} for name,path in models.items() if name in ['coverage_60','coverage_90','raw_RF']}
    report={'created_unix_s':time.time(),'no_new_inference_training_or_wave_solves':True,
        'fixed_final_raw_step':training['completed_steps'],'training_stop_reason':training['stop_reason'],
        'training_seconds':training['seconds'],'raw_model_config':training['model'],
        'matched_development_ids':sorted(ids),'coarse_original_maps':comparison,
        'development_alpha_summary':alpha_summary,'development_alpha_pairs':alpha_details,
        'raw_training_constraints':metrics['raw_RF']['training_constraints'],
        'new_training_pairs_by_geometry_and_scale':individual,'geometry_and_field_scale_are_confounded':True,
        'native_maps':native,'raw_training_resources':read(model/'resource_summary.json'),
        'raw_evaluation_resources':read(model/'evaluation/resource_summary.json'),
        'model_weights_SHA256':{name:hashlib.sha256((path/'weights.pt').read_bytes()).hexdigest() for name,path in models.items()},
        'input_policy_warning':'Raw operator uses LF only; previous models can use HF and LF. '
            'Representation, architecture, initialization and training batches also differ. This is not a single-factor ablation.',
        'development_used_for_method_design':True,'development_independent_c_rho_pairs_present':False,
        'development_simultaneously_independent_three_fields_present':False,'reserved_test_case_unread':'Neg_47_Left',
        'native_RF_score_of_raw_model_not_yet_computed':True,'joint_imaging_success_claimed':False}
    a.out.mkdir()
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    bound=max(float(np.max(abs(exact[2,5:]))) for exact,_ in images)
    fig,axes=plt.subplots(len(images),3,figsize=(11,3.7*len(images)),layout='constrained')
    for row,((exact,predicted),detail) in enumerate(zip(images,alpha_details)):
        for col,(title,value,name) in enumerate([('Target independent alpha change',exact[2],None),
            ('Focused UNO, 90 us',predicted['coverage_90'][2],'coverage_90'),
            ('Raw complex RF inverse UNO',predicted['raw_RF'][2],'raw_RF')]):
            ax=axes[row,col]
            im=ax.imshow(value,extent=[-19.1,19.1,43,0],aspect='equal',cmap='RdBu_r',vmin=-bound,vmax=bound,interpolation='none')
            if name:
                q=detail['models'][name][2];title+=f"\nrelative error {q['relative_error']:.3f}, gain {q['RMS_gain']:.3f}"
            ax.set_title(title,fontsize=9);ax.set_xlabel('Lateral (mm)');ax.set_ylabel(detail['pair_group']+'\nDepth (mm)')
            fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('All development independent attenuation changes; same material labels\nDelta alpha [dB/(cm MHz^1.1)]')
    fig.savefig(a.out/'all_development_alpha_responses.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,kind,title in zip(axes,['pure_c','pure_rho','alpha'],['Sound speed','Density','Attenuation']):
        chosen=sorted([r for r in individual if r['kind']==kind],key=lambda r:r['field_scale_mm'])
        for name,marker in [('coverage_90','o'),('raw_RF','s')]:
            ax.scatter([r['field_scale_mm'] for r in chosen],[r['models'][name]['relative_error'] for r in chosen],marker=marker,label=name)
        ax.axhline(.4,color='gray',linestyle='--',linewidth=1,label='0.4 reference')
        ax.set_xscale('log');ax.set_xlabel('Nominal field scale (mm)');ax.set_ylabel('Relative response error');ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('All new fixed training pairs; geometry and field scale are confounded',fontsize=10)
    fig.savefig(a.out/'training_responses_by_geometry_and_scale.png',dpi=160);plt.close(fig)
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'out':str(a.out),'development_alpha_summary':alpha_summary,
        'coarse_original_maps':comparison,'raw_training_constraints':report['raw_training_constraints'],
        'fixed_final_raw_step':training['completed_steps'],'training_stop_reason':training['stop_reason']}),flush=True)


if __name__=='__main__':main()
