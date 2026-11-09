"""CPU-only evidence after a completed conditional or joint native stage."""
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


def stats(value, target, region):
    v=value[region].astype(np.float64);t=target[region].astype(np.float64)
    return {'rmse':float(np.sqrt(np.mean((v-t)**2))),
        'mae':float(np.mean(abs(v-t))),
        'correlation':float(np.corrcoef(v,t)[0,1]) if min(v.std(),t.std())>1e-10 else None}


def spatial_edge_stats(value, target, mask):
    # Image-space finite differences, never RF Jacobians or an identifiability test.
    energy_v=energy_t=cross=0.
    for axis in (0,1):
        v=np.diff(value.astype(np.float64),axis=axis)
        t=np.diff(target.astype(np.float64),axis=axis)
        valid=(mask[1:]&mask[:-1]) if axis==0 else (mask[:,1:]&mask[:,:-1])
        v=v[valid];t=t[valid]
        energy_v+=float(v@v);energy_t+=float(t@t);cross+=float(v@t)
    return {'signed_spatial_edge_cosine':cross/np.sqrt(energy_v*energy_t) if energy_v*energy_t>0 else None,
        'spatial_edge_norm_relative_to_truth':np.sqrt(energy_v/energy_t) if energy_t>0 else None,
        'not_a_material_identifiability_test':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.root;run=read(root/'run.json')
    if run['status']!='native_block_GN_stage_finished':
        raise RuntimeError('Wait for the current native stage to finish')
    workers=read(root/'worker_processes.json')
    pids={run['pid'],run.get('child_pid'),*workers.get('worker_pids',[])}
    if any(pid and Path('/proc',str(pid)).exists() for pid in pids):
        raise RuntimeError('The original parent or child still exists')
    if args.out.exists():
        raise RuntimeError('Preserve existing summaries')
    if run['sample']!='train_200':
        raise RuntimeError('This summary is restricted to the current development experiment')
    folder=root/'step001';update=read(folder/'material_update.json')
    baseline=read(folder/'baseline_RF.json');lines=read(folder/'line_search.json')
    returned_rf=read(folder/'incumbent_RF.json')
    rf_windows={source:{'initial_relative_RMS':baseline['groups'][source]['geometry_window_relative_RMS'],
        'returned_relative_RMS':returned_rf['groups'][source]['geometry_window_relative_RMS']}
        for source in ('lf','point3')}
    direction=read(folder/'direction.json');maps=read(folder/'maps.metrics.json')
    resource=read(root/'resource_summary.json')
    prior_resource=read(root/'prior_resource_summary.json') if (root/'prior_resource_summary.json').exists() else None
    phase_peaks=dict(resource['sampled_peak_owned_MiB'])
    if prior_resource:
        for device,peak in prior_resource['sampled_peak_owned_MiB'].items():
            phase_peaks[device]=max(phase_peaks.get(device,0),peak)
    material=folder/'material.npy';source=Path(run['source_root'])/'step001/material.npy'
    if sha(source)!=run['source_material_SHA256'] or sha(material)!=run['material_SHA256']:
        raise RuntimeError('Physical material evidence changed')
    if sha(material)!=update['material_SHA256']:
        raise RuntimeError('Final material hash disagrees with accepted update')
    base=np.load(folder/'stage0_initial_material.npy');m=np.load(material)
    if base.shape!=(3,961,1121) or m.shape!=base.shape or m.dtype!=np.float32:
        raise RuntimeError('Full native materials required')
    if not np.array_equal(base,np.load(source)):
        raise RuntimeError('Stage initialization differs from its accepted source')
    invariant={'c_bitwise_unchanged':bool(np.array_equal(base[0],m[0])),
        'alpha_bitwise_unchanged':bool(np.array_equal(base[2],m[2]))}
    old_Z=base[0].astype(np.float64)*base[1];new_Z=m[0].astype(np.float64)*m[1]
    invariant['maximum_relative_impedance_change']=float(np.max(abs(new_Z-old_Z)/old_Z))
    if run['block']=='impedance' and not (invariant['c_bitwise_unchanged'] and invariant['alpha_bitwise_unchanged']):
        raise RuntimeError('An inactive material field changed')
    if run['block']=='speed' and (invariant['maximum_relative_impedance_change']>5e-7 or not invariant['alpha_bitwise_unchanged']):
        raise RuntimeError('An inactive impedance or attenuation field changed')
    jobs=[read(path) for path in sorted((root/'jobs').glob('*.json'))]
    history=[{'checkpoint':str(path.parent.resolve()),'progress':read(path)}
        for path in sorted(folder.glob('hvp*/progress.json'))]
    hvps=[job for job in jobs if job['kind']=='hvp']
    if len(history)!=run['new_HVPs'] or len(hvps)!=len(history):
        raise RuntimeError('True-HVP evidence inventory is incomplete')
    # Truth is first read here, after all original processes have exited.
    from .v5_materials import evaluation_material
    target=evaluation_material(run['sample'])
    z=(np.arange(m.shape[1])-60)*.05;x=(np.arange(m.shape[2])-560)*.05
    mask=(abs(x)[None,:]<=19.1)&(z[:,None]>=1)&(z[:,None]<=43)
    truth_Z=target[0].astype(np.float64)*target[1]
    impedance={'units':'MRayl','phase_role':'reflection-state initialization; not independent density recovery',
        'initial':stats(old_Z/1e6,truth_Z/1e6,mask),
        'returned':stats(new_Z/1e6,truth_Z/1e6,mask),
        'initial_spatial_edges':spatial_edge_stats(old_Z,truth_Z,mask),
        'returned_spatial_edges':spatial_edge_stats(new_Z,truth_Z,mask),
        'by_depth':{}}
    for lo,hi in [(1,5),(5,12),(12,25),(25,43)]:
        region=mask&(z[:,None]>=lo)&(z[:,None]<hi)
        impedance['by_depth'][f'{lo}-{hi}']={'initial':stats(old_Z/1e6,truth_Z/1e6,region),
            'returned':stats(new_Z/1e6,truth_Z/1e6,region)}
    report={'created_unix_s':time.time(),'source_root':str(root.resolve()),'source_run':run,
        'source_summary_SHA256':sha(__file__),'all_original_processes_actually_exited':True,
        'no_new_training_inference_HVP_or_forward_wave_solves':True,'no_GPU_work':True,
        'baseline_RF':baseline,'returned_RF':returned_rf,'RF_window_comparison':rf_windows,
        'RF_windows_are_geometry_arrival_proxies_not_unique_depth_locations':True,
        'direction':direction,'material_update':update,'physical_line_search':lines,
        'true_HVP_history':history,'true_native_jobs':jobs,'resources':resource,
        'prior_interrupted_segment_resources':prior_resource,'whole_phase_sampled_peak_owned_MiB':phase_peaks,
        'same_reference_restoration':read(root/'restored.json') if (root/'restored.json').exists() else None,
        'physical_field_invariants':invariant,'full_native_maps':maps['maps'],'impedance':impedance,
        'initialization_only':run['stage_is_initialization_only'],
        'conditional_residual_is_not_full_joint_residual':run['block']!='joint',
        'returned_material_SHA256':sha(material),'joint_imaging_success_claimed':False,
        'reserved_test_case_unread':'Neg_47_Left',
        'independent_three_field_and_unseen_case_reconstruction_not_tested':True}
    args.out.mkdir();atomic(args.out/'comparison.json',report)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    iz=(z>=0)&(z<=43);ix=abs(x)<=19.1
    extent=[x[ix][0],x[ix][-1],z[iz][-1],z[iz][0]]
    fig,axes=plt.subplots(1,4,figsize=(14,5.2),layout='constrained')
    for col,value in enumerate([truth_Z,old_Z,new_Z,new_Z-truth_Z]):
        error=col==3
        image=axes[col].imshow(value[iz][:,ix]/1e6,extent=extent,aspect='equal',
            interpolation='none',cmap='RdBu_r' if error else 'viridis',
            vmin=-.25 if error else 1.25,vmax=.25 if error else 1.9)
        axes[col].set_title(['Truth','Input physical material','Returned physical material','Returned minus truth'][col],fontsize=9)
        axes[col].set_xlabel('Lateral (mm)');axes[col].set_ylabel('Depth (mm)')
        fig.colorbar(image,ax=axes[col],shrink=.75,label='MRayl')
    fig.suptitle('Impedance Z = rho c; a conditional initialization is not three-field recovery',fontsize=11)
    fig.savefig(args.out/'impedance.png',dpi=160);plt.close(fig)
    if history:
        counts=[row['progress']['new_HVPs'] for row in history]
        residual=[row['progress']['conditional_block_relative'] for row in history]
        changes=[row['progress']['model_change'] for row in history]
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        axes[0].semilogy(counts,residual,'o-')
        axes[0].axhline(run['conditional_block_relative_target'],color='gray',linestyle='--')
        axes[0].set_xlabel('New true HVP directions');axes[0].set_ylabel('Active-block residual / initial residual')
        axes[1].plot(counts,-np.asarray(changes),'o-')
        axes[1].set_xlabel('New true HVP directions');axes[1].set_ylabel('Predicted quadratic decrease')
        fig.suptitle('Conditional numerical convergence does not establish joint imaging',fontsize=11)
        fig.savefig(args.out/'conditional_history.png',dpi=160);plt.close(fig)
    atomic(args.out/'run.json',{'status':'CPU_block_result_summary_finished','pid':os.getpid(),
        'finished_unix_s':time.time(),'new_wave_solves':0,'initialization_only':run['stage_is_initialization_only'],
        'joint_imaging_success_claimed':False})
    print(json.dumps({'native_status':run['status'],'block':run['block'],'new_completed_HVPs':len(hvps),
        'direction':direction,'material_update':update,'full_native_maps':maps['maps'],
        'impedance':impedance,'RF_window_comparison':rf_windows,'physical_field_invariants':invariant,
        'sampled_peak_owned_MiB':phase_peaks}),flush=True)


if __name__=='__main__':
    main()
