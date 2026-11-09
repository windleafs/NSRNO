"""CPU-only full-map and search-allocation report after the spatial joint run."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import numpy as np
from .native_gn_json import atomic


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.root;run=read(root/'run.json')
    if run['status']!='native_spatial_Krylov_finished':
        raise RuntimeError('Wait for the actual spatial reconstruction')
    workers=read(root/'worker_processes.json')
    pids={run['pid'],run.get('child_pid'),*workers['worker_pids']}
    if any(pid and Path('/proc',str(pid)).exists() for pid in pids):
        raise RuntimeError('Original parent or child still exists')
    if args.out.exists():
        raise RuntimeError('Preserve existing report')
    folder=root/'step001';update=read(folder/'material_update.json')
    restored=read(folder/'restored.json');direction=read(folder/'direction.json')
    maps=read(folder/'maps.metrics.json');lines=read(folder/'line_search.json')
    resource=read(root/'resource_summary.json')
    if sha(folder/'material.npy')!=run['material_SHA256'] or run['material_SHA256']!=update['material_SHA256']:
        raise RuntimeError('Returned material evidence changed')
    if sha(folder/'stage0_initial_material.npy')!=run['incumbent_material_SHA256']:
        raise RuntimeError('Physical incumbent comparison changed')
    if sha(folder/'linear_reference_material.npy')!=run['fixed_reference_SHA256']:
        raise RuntimeError('Exact linear reference changed')
    paths=[Path(v) for v in read(folder/'fixed_HVP_manifest.json')['paths']]
    if len(paths)!=run['reused_complete_HVPs']+run['new_HVPs']:
        raise RuntimeError('True-image manifest is incomplete')
    history=[{'path':str(path),'progress':read(path/'progress.json'),'proposal':read(path/'proposal.json')}
        for path in paths[run['reused_complete_HVPs']:]]
    jobs=[read(path) for path in sorted((root/'jobs').glob('*.json'))]
    hvps=[job for job in jobs if job['kind']=='hvp']
    if len(hvps)!=run['new_HVPs']:
        raise RuntimeError('New native work inventory is incomplete')
    if any(job['kind']=='gradient' for job in jobs):
        raise RuntimeError('Unexpected duplicate gradient job')
    source=Path(run['source_root'])
    before=read(source/'step001/incumbent_RF.json');after=read(folder/'incumbent_RF.json')
    weights=np.array([.1,.2,.25,.3,.15]);windows={}
    for name in ('lf','point3'):
        first=np.array(before['groups'][name]['geometry_window_relative_RMS'])
        last=np.array(after['groups'][name]['geometry_window_relative_RMS'])
        windows[name]={'before_relative_RMS':first.tolist(),'after_relative_RMS':last.tolist(),
            'native_objective_decrease_contribution':(.25*weights*(first*first-last*last)).tolist()}
    m=np.load(folder/'material.npy');base=np.load(folder/'stage0_initial_material.npy')
    delta=m.astype(np.float64)-base
    z=(np.arange(961)-60)*.05;x=(np.arange(1121)-560)*.05
    unknown=np.broadcast_to(z[:,None]>=1,(961,1121))
    roi=(abs(x)[None,:]<=19.1)&(z[:,None]>=1)&(z[:,None]<=43)
    masks={'full_FOV_1_to_43mm':roi,
        'lateral_outside_FOV':(abs(x)[None,:]>19.1)&(z[:,None]>=1)&(z[:,None]<=43),
        'deeper_than_43mm':(abs(x)[None,:]<=19.1)&(z[:,None]>43),
        'outside_side_and_depth':(abs(x)[None,:]>19.1)&(z[:,None]>43)}
    total=(delta[:,unknown]**2).sum(axis=1);regions={}
    for name,mask in masks.items():
        energy=(delta[:,mask]**2).sum(axis=1)
        regions[name]={'pixels':int(mask.sum()),
            'physical_update_RMS':np.sqrt(np.mean(delta[:,mask]**2,axis=1)).tolist(),
            'fraction_of_actual_squared_update':np.divide(energy,total,out=np.zeros_like(energy),where=total>0).tolist()}
    result={'created_unix_s':time.time(),'source_root':str(root.resolve()),'source_run':run,
        'summary_program_SHA256':sha(__file__),'all_original_processes_actually_exited':True,
        'no_new_wave_solves_training_or_inference':True,'no_GPU_work':True,
        'fixed_reference_RF':read(folder/'baseline_RF.json'),'physical_incumbent_RF':before,'returned_RF':after,
        'restored_exact_state':restored,'new_true_direction_history':history,'direction_result':direction,
        'material_update':update,'physical_line_search':lines,'full_native_maps':maps['maps'],
        'physical_update_regions':regions,'RF_window_comparison':windows,
        'geometry_windows_are_arrival_proxies_not_unique_depth_locations':True,
        'new_native_jobs':jobs,'new_HVP_walltime_sum_seconds':sum(job['seconds'] for job in hvps),
        'resources':resource,'returned_material_SHA256':sha(folder/'material.npy'),
        'unchanged_incumbent_is_returned_if_candidate_not_accepted':not update['accepted'],
        'full_KKT_and_ROI_KKT_are_separate':True,
        'ROI_search_allocation_is_not_established_as_sole_failure_cause':True,
        'reserved_test_case_unread':'Neg_47_Left','joint_imaging_success_claimed':False,
        'independent_field_and_unseen_case_recovery_not_established':True}
    args.out.mkdir();atomic(args.out/'comparison.json',result)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    counts=[0]+list(range(1,len(history)+1))
    full=[restored['full_projected_KKT_relative']]+[row['progress']['full_projected_KKT_relative'] for row in history]
    local=[restored['ROI_projected_KKT_relative']]+[row['progress']['ROI_projected_KKT_relative'] for row in history]
    decrease=[0]+[restored['model_change']-row['progress']['model_change'] for row in history]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].semilogy(counts,full,'o-',label='Full field')
    axes[0].semilogy(counts,local,'s-',label='Full target FOV')
    axes[0].axhline(.001,color='gray',linestyle='--',label='Full-field target')
    axes[0].set_xlabel('New true HVPs; six earlier images reused')
    axes[0].set_ylabel('Projected residual / initial value for that region');axes[0].legend()
    axes[1].plot(counts,decrease,'o-')
    for index,row in enumerate(history,1):
        axes[1].annotate(row['proposal']['proposal_block'],(index,decrease[index]),fontsize=8)
    axes[1].set_xlabel('New true HVPs');axes[1].set_ylabel('Additional quadratic decrease from restored span')
    fig.suptitle('Numerical convergence at one fixed reference; neither curve proves material recovery',fontsize=10)
    fig.savefig(args.out/'spatial_krylov_history.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    titles=['Sound-speed update (m/s)','Density update (kg/m³)','Attenuation update [dB/(cm MHz^1.1)]']
    for j,ax in enumerate(axes):
        scale=max(float(np.percentile(abs(delta[j][unknown]),99)),1e-12)
        im=ax.imshow(delta[j],extent=[x[0],x[-1],z[-1],z[0]],aspect='equal',
            interpolation='none',cmap='RdBu_r',vmin=-scale,vmax=scale)
        ax.add_patch(Rectangle((-19.1,1),38.2,42,fill=False,color='black',linewidth=1.2))
        ax.set_xlabel('Lateral (mm)');ax.set_ylabel('Depth (mm)');ax.set_title(titles[j],fontsize=9)
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle('Actual accepted physical change; outline is the complete target FOV; colors clip at 99th percentile',fontsize=10)
    fig.savefig(args.out/'physical_update_location.png',dpi=160);plt.close(fig)
    atomic(args.out/'run.json',{'status':'CPU_spatial_result_summary_finished','pid':os.getpid(),
        'finished_unix_s':time.time(),'new_wave_solves':0,'joint_imaging_success_claimed':False})
    print(json.dumps({'material_update':update,'full_native_maps':maps['maps'],
        'physical_update_regions':regions,'RF_window_comparison':windows,
        'sampled_peak_owned_MiB':resource['sampled_peak_owned_MiB']}),flush=True)


if __name__=='__main__':
    main()
