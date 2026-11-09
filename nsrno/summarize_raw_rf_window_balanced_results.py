"""CPU before/after comparison of fixed raw and paired-window operators."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np
from .native_gn_json import atomic


def read(path):return json.loads(Path(path).read_text())


def alpha_summary(pairs,window):
    values=[p[window] for p in pairs]
    finite=lambda xs:[v for v in xs if v is not None]
    return {'pairs':len(values),'mean_relative_error':float(np.mean([v[2]['relative_error'] for v in values])),
        'mean_cosine':float(np.mean(finite([v[2]['cosine'] for v in values]))) if finite([v[2]['cosine'] for v in values]) else None,
        'mean_RMS_gain':float(np.mean([v[2]['RMS_gain'] for v in values])),
        'c_mean_cross_talk_RMS':float(np.mean([v[0]['predicted_RMS'] for v in values])),
        'rho_mean_cross_talk_RMS':float(np.mean([v[1]['predicted_RMS'] for v in values]))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('out/v5/raw_rf_window_balanced_experiment_v1'))
    p.add_argument('--out',type=Path,default=Path('out/v5/raw_rf_window_balanced_comparison_v1'))
    a=p.parse_args();run=read(a.root/'run.json');model=Path(run['model']);training=read(model/'run.json')
    evaluation=read(model/'evaluation/run.json');window_run=read(model/'window_evaluation/run.json')
    if [run['status'],training['status'],evaluation['status'],window_run['status']]!=[
        'raw_RF_inverse_training_and_evaluations_finished','training_finished','evaluation_finished','window_evaluation_finished']:
        raise RuntimeError('Complete all current stages first')
    for r in [run,training,evaluation,window_run]:
        for key in ['pid','child_pid']:
            if r.get(key) and Path('/proc',str(r[key])).exists():raise RuntimeError('Existing process has not exited')
    expected=hashlib.sha256((model/'weights.pt').read_bytes()).hexdigest()
    if expected!=evaluation['weights_SHA256'] or expected!=window_run['weights_SHA256']:raise RuntimeError('Mixed checkpoints')
    if a.out.exists():raise RuntimeError('Preserve existing CPU comparison')
    subprocess.run([sys.executable,'-m','nsrno.summarize_raw_rf_inverse_results','--root',str(a.root),'--out',str(a.out)],
        check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=subprocess.DEVNULL)
    old=read('out/v5/raw_rf_inverse_comparison_v1/comparison.json');new=read(a.out/'comparison.json')
    old_window=read('out/v5/raw_rf_record_window_diagnosis_v1/metrics.json')
    new_window=read(model/'window_evaluation/metrics.json')
    if old['matched_development_ids']!=new['matched_development_ids']:raise RuntimeError('Different development states')
    windows={}
    for label,data in [('raw_6000',old_window),('paired_window_final',new_window)]:
        cases=[r for r in data['cases'] if r['kind']=='original']
        windows[label]={'original_cases':cases,
            'all_10_mean_actual_window_prediction_difference_RMS_c_rho_alpha':np.mean([
                r['prediction_difference_RMS_from_RF90_c_rho_alpha']['matched_RF60'] for r in data['cases']],axis=0).tolist(),
            'independent_alpha':{window:alpha_summary(data['independent_alpha_pairs'],window) for window in ['RF60','RF90']}}
    report={'created_unix_s':time.time(),'no_new_inference_training_or_wave_solves':True,
        'raw_6000_weights_SHA256':old['model_weights_SHA256']['raw_RF'],'paired_weights_SHA256':expected,
        'paired_fixed_steps':training['completed_steps'],'paired_stop_reason':training['stop_reason'],
        'paired_training_seconds':training['seconds'],'same_development_ids':new['matched_development_ids'],
        'same_material_labels_and_global_RF_scale':True,'same_network_architecture':True,
        'training_policy_and_added_updates_both_changed':True,'not_an_independent_test_case_comparison':True,
        'native_maps':{'raw_6000':old['native_maps']['raw_RF'],'paired_window_final':new['native_maps']['raw_RF']},
        'coarse_original_maps':{'raw_6000':old['coarse_original_maps']['raw_RF'],'paired_window_final':new['coarse_original_maps']['raw_RF']},
        'training_constraints':{'raw_6000':old['raw_training_constraints'],'paired_window_final':new['raw_training_constraints']},
        'development_alpha_default_input':{'raw_6000':old['development_alpha_summary']['raw_RF'],
            'paired_window_final':new['development_alpha_summary']['raw_RF']},
        'matched_record_windows':windows,
        'all_new_independent_training_pairs':{'raw_6000':old['new_training_pairs_by_geometry_and_scale'],
            'paired_window_final':new['new_training_pairs_by_geometry_and_scale']},
        'paired_resources':new['raw_training_resources'],'development_used_for_method_design':True,
        'development_independent_c_rho_pairs_present':False,'development_simultaneous_independent_three_fields_present':False,
        'reserved_test_case_unread':'Neg_47_Left','native_RF_score_not_computed':True,'joint_imaging_success_claimed':False}
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    index=read(training['index']);groups={}
    for row in index['records']:
        if row['role']!='train' and row['kind']=='alpha':groups.setdefault(row['pair_group'],[]).append(row)
    images=[]
    for group,members in sorted(groups.items()):
        minus,plus=sorted(members,key=lambda r:r['pair_sign'])
        exact=np.load(plus['label'])[2]-np.load(minus['label'])[2]
        predictions=[]
        for source in [Path('out/v5/inverse_raw_rf_uno_v1'),model]:
            predictions.append(np.load(source/'evaluation'/f"{plus['id']}.prediction.npy")[2]-
                np.load(source/'evaluation'/f"{minus['id']}.prediction.npy")[2])
        images.append((group,exact,predictions))
    if len(images)!=4:raise ValueError('Expected all four development alpha pairs')
    bound=max(float(np.max(abs(exact[5:]))) for _,exact,_ in images)
    fig,axes=plt.subplots(4,3,figsize=(11,15),layout='constrained')
    for row,(group,exact,predictions) in enumerate(images):
        for col,(title,value) in enumerate(zip(['Target independent attenuation change','Raw RF, 6000 updates','Paired-window fine tuning'],[exact,*predictions])):
            ax=axes[row,col];im=ax.imshow(value,extent=[-19.1,19.1,43,0],aspect='equal',cmap='RdBu_r',vmin=-bound,vmax=bound,interpolation='none')
            ax.set_title(title,fontsize=9);ax.set_xlabel('Lateral (mm)');ax.set_ylabel(group+'\nDepth (mm)');fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('Same fixed development pairs, default full-record input\nDelta alpha [dB/(cm MHz^1.1)]; no new physical solves',fontsize=11)
    fig.savefig(a.out/'alpha_before_after.png',dpi=150);plt.close(fig)
    atomic(a.out/'before_after.json',report)
    atomic(a.out/'before_after_run.json',{'status':'CPU_comparison_finished','pid':os.getpid(),'finished_unix_s':time.time(),
        'new_wave_solves':0,'new_inferences':0,'joint_imaging_success_claimed':False})
    print(json.dumps({'paired_fixed_steps':training['completed_steps'],'paired_training_seconds':training['seconds'],
        'training_constraints':report['training_constraints'],'development_alpha_default_input':report['development_alpha_default_input'],
        'native_RMSE':{name:{sample:[detail['maps'][key]['reconstruction']['rmse'] for key in
            ['c_m_s','density_kg_m3','alpha_dB_cm_MHz1p1']] for sample,detail in cases.items()}
            for name,cases in report['native_maps'].items()},
        'matched_record_window_comparison':{name:{'all_10_mean_actual_window_prediction_difference_RMS_c_rho_alpha':
            details['all_10_mean_actual_window_prediction_difference_RMS_c_rho_alpha'],
            'independent_alpha':details['independent_alpha'],
            'original_case_prediction_differences':{case['id']:case['prediction_difference_RMS_from_RF90_c_rho_alpha']
                for case in details['original_cases']}} for name,details in windows.items()}}),flush=True)


if __name__=='__main__':main()
