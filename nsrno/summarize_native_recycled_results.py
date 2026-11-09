"""Post-hoc CPU report of completed true-image Krylov reconstruction."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import numpy as np
from .native_gn_json import atomic


def read(path):return json.loads(Path(path).read_text())


def optional(path):return read(path) if Path(path).exists() else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('out/v5/reconstruction_native_recycled_krylov_v1_train_200'))
    p.add_argument('--out',type=Path,default=Path('out/v5/native_recycled_krylov_comparison_v1'))
    a=p.parse_args();run=read(a.root/'run.json');workers=optional(a.root/'worker_processes.json') or {}
    if run['status'] not in ['planned_native_recycled_GN_steps_finished','failed']:
        raise RuntimeError('Wait for the existing native workflow')
    pids={run['pid'],run.get('child_pid'),*workers.get('worker_pids',[])}
    if any(pid and Path('/proc',str(pid)).exists() for pid in pids):raise RuntimeError('Existing process has not exited')
    if a.out.exists():raise RuntimeError('Preserve existing result summary')
    folder=a.root/'step001';restored=optional(folder/'restored.json');history=[]
    for path in sorted(folder.glob('hvp*/progress.json')):
        row=read(path);proposal=optional(path.parent/'proposal.json')
        history.append({'checkpoint':str(path.parent.resolve()),'progress':row,'proposal':proposal})
    jobs=[read(path) for path in sorted((a.root/'jobs').glob('*.json'))]
    hvps=[row for row in jobs if row['kind']=='hvp'];forwards=[row for row in jobs if row['kind']=='forward']
    direction=optional(folder/'direction.json');update=optional(folder/'material_update.json')
    line_search=optional(folder/'line_search.json');maps=optional(folder/'maps.metrics.json')
    resource=optional(a.root/'resource_summary.json')
    if run['status']=='planned_native_recycled_GN_steps_finished':
        if any(value is None for value in [restored,direction,update,line_search,maps,resource]):
            raise RuntimeError('Completed status has missing required evidence')
        if len(hvps)!=direction['new_HVPs'] or len(history)!=len(hvps):raise RuntimeError('Incomplete true-HVP checkpoint inventory')
    uniform=optional('out/v5/inverse_raw_rf_window_balanced_v1/evaluation/train_200_native_full.metrics.json')
    native={}
    if maps:
        for key,value in maps['maps'].items():
            native[key]={'physical_incumbent_before':value['initial'],'final_returned_material':value['reconstruction'],
                'by_depth':value['by_depth'],'uniform_initial':uniform['maps'][key]['initial'] if uniform else None}
    material=folder/'material.npy'
    report={'created_unix_s':time.time(),'source_run':run,'source_root':str(a.root.resolve()),
        'source_summary_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'no_new_training_inference_HVP_or_forward_wave_solves':True,'no_GPU_work':True,
        'all_original_processes_actually_exited':True,'initial_true_subspace_solution':restored,
        'new_direction_history':history,'true_native_HVP_jobs':hvps,'native_forward_jobs':forwards,
        'direction_result':direction,'material_update':update,'physical_line_search':line_search,
        'full_native_maps':native,'resources':resource,
        'returned_material_SHA256':hashlib.sha256(material.read_bytes()).hexdigest() if material.exists() else None,
        'final_map_is_unchanged_incumbent_if_no_candidate_accepted':bool(update and not update['accepted']),
        'HVP_walltime_sum_seconds':sum(row['seconds'] for row in hvps),
        'full_native_RF_forward_walltime_sum_seconds':sum(row['seconds'] for row in forwards),
        'geometry_time_windows_are_arrival_proxies_not_unique_depth_locations':True,
        'local_GN_accuracy_is_not_global_imaging_success':True,'joint_imaging_success_claimed':False,
        'development_case_used_for_method_design':'Neg_35_Left','reserved_test_case_unread':'Neg_47_Left',
        'independent_three_field_and_unseen_case_reconstruction_not_tested':True}
    a.out.mkdir();atomic(a.out/'comparison.json',report)
    if restored and history:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        counts=[restored['directions'],*[row['progress']['directions'] for row in history]]
        relative=[restored['full_projected_KKT_relative'],*[row['progress']['full_projected_KKT_relative'] for row in history]]
        changes=[restored['model_change'],*[row['progress']['model_change'] for row in history]]
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        axes[0].plot(counts,relative,'o-');axes[0].axhline(.001,color='gray',linestyle='--',label='original inner target')
        axes[0].set_xlabel('Total true HVP directions');axes[0].set_ylabel('Full projected residual / initial residual');axes[0].legend()
        axes[1].plot(counts,-np.asarray(changes),'o-');axes[1].set_xlabel('Total true HVP directions');axes[1].set_ylabel('Predicted quadratic decrease')
        fig.suptitle('Fixed native linearization; neither curve measures material recovery',fontsize=11)
        fig.savefig(a.out/'true_quadratic_history.png',dpi=160);plt.close(fig)
    atomic(a.out/'run.json',{'status':'CPU_result_summary_finished','pid':os.getpid(),'finished_unix_s':time.time(),
        'native_source_status':run['status'],'new_wave_solves':0,'joint_imaging_success_claimed':False})
    print(json.dumps({'native_status':run['status'],'new_completed_HVPs':len(hvps),'saved_new_direction_checkpoints':len(history),
        'direction_result':direction,'material_update':update,'full_native_maps':native,
        'sampled_peak_owned_MiB':resource.get('sampled_peak_owned_MiB') if resource else None,
        'native_error':run.get('error')}),flush=True)


if __name__=='__main__':main()
