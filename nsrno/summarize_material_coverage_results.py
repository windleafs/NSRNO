"""CPU report of actual matched-window models, maps and parameter responses."""
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


def response(predicted,exact):
    result=[]
    for value,target in zip(predicted[:,5:],exact[:,5:]):
        value=value.astype(np.float64).ravel();target=target.astype(np.float64).ravel()
        vn,tn=np.linalg.norm(value),np.linalg.norm(target)
        result.append({'predicted_RMS':float(np.sqrt(np.mean(value*value))),
                       'target_RMS':float(np.sqrt(np.mean(target*target))),
                       'relative_error':float(np.linalg.norm(value-target)/tn) if tn>1e-5 else None,
                       'cosine':float(np.dot(value,target)/(vn*tn)) if tn>1e-5 and vn>1e-10 else None,
                       'RMS_gain':float(vn/tn) if tn>1e-5 else None})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('out/v5/material_coverage_experiment_v1'))
    p.add_argument('--data',type=Path,default=Path('out/v5/independent_material_3mhz_v1'))
    p.add_argument('--out',type=Path,default=Path('out/v5/material_coverage_comparison_v1'))
    a=p.parse_args()
    run=read(a.root/'run.json')
    if run['status']!='material_coverage_training_and_evaluations_finished':
        raise RuntimeError('Wait for the existing fixed-budget experiment; no new GPU work')
    for pid in [run['pid'],run.get('child_pid')]:
        if pid and Path(f'/proc/{pid}').exists():raise RuntimeError('Current experiment still exists')
    if a.out.exists():raise RuntimeError('Inspect the already written comparison')
    models={'angular_spectral':Path('out/v5/inverse_angular_spectral_uno_v1'),
            'response_constrained':Path('out/v5/inverse_response_constrained_uno_v1'),
            'coverage_60':Path(run['controls']['60']['model']),
            'coverage_90':Path(run['controls']['90']['model'])}
    metrics={name:read(path/'evaluation/metrics.json') for name,path in models.items()}
    by_id={name:{r['id']:r for r in metric['cases']} for name,metric in metrics.items()}
    sets=[set(r) for r in by_id.values()]
    if any(s!=sets[0] for s in sets):raise ValueError('Different evaluation records')
    material_plan=read(a.data/'material_plan.json')
    original_index=read(Path(material_plan['source_index']))
    development=[r for r in original_index['records'] if r['role']!='train']
    groups={}
    for r in development:
        if r['kind']=='alpha':groups.setdefault(r['pair_group'],[]).append(r)
    details=[];images=[]
    for group in sorted(groups):
        minus,plus=sorted(groups[group],key=lambda r:r['pair_sign'])
        exact=np.load(plus['label'])-np.load(minus['label'])
        predicted={name:np.load(path/'evaluation'/f"{plus['id']}.prediction.npy")-
                        np.load(path/'evaluation'/f"{minus['id']}.prediction.npy")
                   for name,path in models.items()}
        detail={'group':group,'parent_id':minus['parent_id'],
                'models':{name:response(value,exact) for name,value in predicted.items()}}
        details.append(detail);images.append((exact,predicted))
    alpha_summary={}
    for name in models:
        q=[d['models'][name] for d in details]
        alpha_summary[name]={'pairs':len(q),
            'alpha_mean_relative_error':float(np.mean([v[2]['relative_error'] for v in q])),
            'alpha_mean_cosine':mean_or_none([v[2]['cosine'] for v in q]),
            'alpha_mean_RMS_gain':float(np.mean([v[2]['RMS_gain'] for v in q])),
            'c_mean_cross_talk_RMS_m_s':float(np.mean([v[0]['predicted_RMS'] for v in q])),
            'rho_mean_cross_talk_RMS_kg_m3':float(np.mean([v[1]['predicted_RMS'] for v in q]))}
    strata={}
    for name,metric in metrics.items():
        strata[name]={}
        original=[r for r in metric['cases'] if r['kind']=='original']
        for group in ['HF_only','HF_LF_existing_60us','HF_LF_paired_window_control']:
            if group=='HF_only':selected=[r for r in original if not r['has_low_frequency']]
            elif group=='HF_LF_existing_60us':
                selected=[r for r in original if r['has_low_frequency'] and r['id'] not in ['train_200','val_059']]
            else:selected=[r for r in original if r['id'] in ['train_200','val_059']]
            strata[name][group]={'records':len(selected),
                'mean_RMSE_c_rho_alpha':np.mean([r['model']['rmse'] for r in selected],axis=0).tolist(),
                'training_mean_RMSE':np.mean([r['training_mean']['rmse'] for r in selected],axis=0).tolist(),
                'record_us_in_this_model':90 if name=='coverage_90' and group=='HF_LF_paired_window_control' else 60}
    training={}
    new_pair_meta={r['pair_group']:r for r in material_plan['records']
                   if r['role']=='train' and 'pair_group' in r}
    for window in ['60','90']:
        training[window]={'constraints':read(a.root/f'final_training_constraints_{window}.json'),
                          'fixed_pairs':read(Path(run['controls'][window]['training_pairs'])/'metrics.json')}
        individual=[]
        for pair in training[window]['fixed_pairs']['pairs']:
            if pair['pair_group'] not in new_pair_meta:continue
            meta=new_pair_meta[pair['pair_group']];changed=meta['changed_parameter']
            value=pair['maps_c_rho_alpha'][changed]
            individual.append({'pair_group':pair['pair_group'],'kind':pair['kind'],
                'context':meta['context'],'parent_id':meta['parent_id'],
                'field_scale_mm':meta['field_scale_mm'],'changed_parameter':changed,
                'changed_relative_error':value['relative_difference_error'],
                'changed_cosine':value['difference_cosine'],
                'changed_RMS_gain':value['predicted_difference_rms']/value['target_difference_rms'],
                'all_parameter_responses':pair['maps_c_rho_alpha']})
        if len(individual)!=24:raise ValueError('Missing independent responses at one or more field scales')
        training[window]['new_pairs_by_geometry_and_scale']=individual
        training[window]['individual_changed_error_summary']={}
        for kind in ['pure_c','pure_rho','alpha']:
            chosen=[r for r in individual if r['kind']==kind]
            errors=[r['changed_relative_error'] for r in chosen]
            training[window]['individual_changed_error_summary'][kind]={
                'pairs':len(chosen),'maximum':float(max(errors)),
                'pairs_above_0p4':sum(v>.4 for v in errors),
                'note':'The training constraint bounds the mean squared response error, not every individual pair.'}
    native={}
    for name in ['coverage_60','coverage_90']:
        native[name]={sample:read(models[name]/'evaluation'/f'{sample}_native_full.metrics.json')
                      for sample in ['train_200','val_059']}
    generated=read(a.data/'generated_index.json');late=[]
    for row in generated['records']:
        wave=np.load(row['rf90'],mmap_mode='r')
        energy=np.sum(np.square(wave,dtype=np.float64),axis=(0,1));den=max(float(energy.sum()),1e-30)
        late.append({'id':row['id'],'role':row['role'],'kind':row['kind'],
                     'energy_fraction_60_75us':float(energy[2400:3000].sum()/den),
                     'energy_fraction_75_90us':float(energy[3000:].sum()/den)})
    report={'created_unix_s':time.time(),'fixed_final_models':True,'no_training_or_GPU_work':True,
            'development_used_for_method_design':True,'reserved_test_case_unread':'Neg_47_Left',
            'no_new_case_generalization_claimed':True,
            'development_independent_c_and_rho_pairs_present':False,
            'development_simultaneous_independent_material_fields_present':False,
            'remaining_evidence_gap':'Development c/rho maps have correlated original tissue fields. '
                'Accurate maps alone do not prove response to independent c/rho changes on an unseen geometry. '
                'This experiment also lacks a development state with all three spatial fields changing independently together.',
            'inputs':'coverage_60 and coverage_90 differ in matched LF record duration; '
                'both use the same new material directions, source-mask policy, warm start and optimization budget.',
            'development_alpha_summary':alpha_summary,'development_alpha_pairs':details,
            'original_records_by_sources_and_window':strata,'training':training,'native_maps':native,
            'geometry_and_field_scale_are_confounded':True,
            'RF_added_time_energy':late,
            'model_weights_SHA256':{name:hashlib.sha256((path/'weights.pt').read_bytes()).hexdigest()
                                    for name,path in models.items()}}
    a.out.mkdir();(a.out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    bound=max(float(np.max(abs(exact[2,5:]))) for exact,_ in images)
    fig,axes=plt.subplots(len(images),3,figsize=(11,3.7*len(images)),layout='constrained')
    for row,((exact,values),detail) in enumerate(zip(images,details)):
        for col,(title,image) in enumerate([('Target independent alpha change',exact[2]),
                                           ('Coverage UNO, 60 us',values['coverage_60'][2]),
                                           ('Coverage UNO, 90 us',values['coverage_90'][2])]):
            ax=axes[row,col]
            im=ax.imshow(image,extent=[-19.1,19.1,43,0],aspect='equal',cmap='RdBu_r',
                         vmin=-bound,vmax=bound,interpolation='none')
            if col:
                r=detail['models'][['coverage_60','coverage_90'][col-1]][2]
                title+=f"\nrelative error {r['relative_error']:.3f}, gain {r['RMS_gain']:.3f}"
            ax.set_title(title,fontsize=9);ax.set_xlabel('Lateral (mm)');ax.set_ylabel(detail['group']+'\nDepth (mm)')
            fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('All development independent attenuation changes; same material labels\n'
                 'Delta alpha in dB/(cm MHz^1.1); fixed final checkpoints')
    fig.savefig(a.out/'all_development_alpha_responses.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,kind,title in zip(axes,['pure_c','pure_rho','alpha'],['Sound speed','Density','Attenuation']):
        for window,marker in [('60','o'),('90','s')]:
            chosen=sorted([r for r in training[window]['new_pairs_by_geometry_and_scale'] if r['kind']==kind],
                          key=lambda r:r['field_scale_mm'])
            ax.scatter([r['field_scale_mm'] for r in chosen],[r['changed_relative_error'] for r in chosen],
                       marker=marker,label=window+' us')
        ax.axhline(.4,color='gray',linestyle='--',linewidth=1,label='0.4 reference')
        ax.set_xscale('log');ax.set_xlabel('Field scale (mm; different geometry at each point)')
        ax.set_ylabel('Changed-parameter relative response error');ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('All new fixed training pairs; a mean constraint can hide individual failures\n'
                 'Geometry and field scale are confounded; this is not an isolated scale experiment',fontsize=10)
    fig.savefig(a.out/'training_responses_by_geometry_and_scale.png',dpi=160);plt.close(fig)
    print(json.dumps({'out':str(a.out),'development_alpha_summary':alpha_summary,
                      'original_records_by_sources_and_window':strata,
                      'new_training_constraints':{k:v['constraints'] for k,v in training.items()},
                      'individual_new_pair_error_summary':{k:v['individual_changed_error_summary'] for k,v in training.items()},
                      'development_independent_c_and_rho_pairs_present':False,
                      'development_simultaneous_independent_material_fields_present':False}),flush=True)


if __name__=='__main__':main()
