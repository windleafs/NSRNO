"""CPU-only analysis of an immutable physical checkpoint while GN may continue."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from .native_gn_json import atomic


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();cp=root/'physical_checkpoints'/a.checkpoint
    if cp.resolve().parent!=root/'physical_checkpoints':raise ValueError('Invalid checkpoint name')
    if a.out.exists():raise RuntimeError('Preserve existing analysis')
    run=read(root/'run.json');update=read(cp/'material_update.json');maps=read(cp/'maps.metrics.json')
    if not (cp/'maps.png').exists():raise RuntimeError('Wait for actual checkpoint maps')
    if run.get('child_pid') and Path(run.get('map_folder',''))==cp:
        raise RuntimeError('Wait for this checkpoint map process to finish')
    history=read(root/'step001/line_search.json')['checkpoints']
    index=next(i for i,row in enumerate(history) if Path(row['checkpoint']).resolve()==cp)
    before_folder=Path(history[index-1]['checkpoint']) if index else Path(run['source_root'])/'step001'
    before=read(before_folder/'incumbent_RF.json');after=read(cp/'incumbent_RF.json')
    if before['loss']!=update['native_data_before'] or after['loss']!=update['native_data_after']:
        raise RuntimeError('Checkpoint RF lineage disagrees')
    if sha(cp/'material.npy')!=update['material_SHA256']:
        raise RuntimeError('Returned checkpoint material changed')
    if sha(cp/'stage0_initial_material.npy')!=sha(before_folder/'material.npy'):
        raise RuntimeError('Consecutive physical starting material changed')
    base=np.load(cp/'stage0_initial_material.npy');material=np.load(cp/'material.npy')
    if base.shape!=(3,961,1121) or material.shape!=base.shape:raise ValueError('Full native fields required')
    delta=material.astype(np.float64)-base
    z=(np.arange(961)-60)*.05;x=(np.arange(1121)-560)*.05
    unknown=np.broadcast_to(z[:,None]>=1,(961,1121))
    roi=(abs(x)[None,:]<=19.1)&(z[:,None]>=1)&(z[:,None]<=43)
    masks={'full_FOV_1_to_43mm':roi,
        'lateral_outside_FOV':(abs(x)[None,:]>19.1)&(z[:,None]>=1)&(z[:,None]<=43),
        'deeper_than_43mm':(abs(x)[None,:]<=19.1)&(z[:,None]>43),
        'outside_side_and_depth':(abs(x)[None,:]>19.1)&(z[:,None]>43)}
    total=np.sum(delta[:,unknown]**2,axis=1);regions={}
    for name,mask in masks.items():
        energy=np.sum(delta[:,mask]**2,axis=1)
        regions[name]={'physical_update_RMS':np.sqrt(np.mean(delta[:,mask]**2,axis=1)).tolist(),
            'fraction_of_actual_squared_update':np.divide(energy,total,out=np.zeros_like(energy),where=total>0).tolist()}
    weights=np.array([.1,.2,.25,.3,.15]);windows={}
    for source in ('lf','point3'):
        first=np.array(before['groups'][source]['geometry_window_relative_RMS'])
        last=np.array(after['groups'][source]['geometry_window_relative_RMS'])
        windows[source]={'before_relative_RMS':first.tolist(),'after_relative_RMS':last.tolist(),
            'objective_decrease_contribution':(.25*weights*(first*first-last*last)).tolist()}
    contributions=sum((np.array(v['objective_decrease_contribution']) for v in windows.values()),np.zeros(5))
    decrease=before['loss']-after['loss']
    if abs(float(contributions.sum())-decrease)>1e-8:
        raise RuntimeError('Existing RF window statistics disagree with recorded objective')
    initial_root=Path(run['source_root']);initial_file=initial_root/'step001/maps.metrics.json'
    if not initial_file.exists():
        initial_run=read(initial_root/'run.json')
        physical_folder=Path(initial_run['latest_physical_checkpoint'])
        if sha(physical_folder/'material.npy')!=sha(before_folder/'material.npy') and index==0:
            raise RuntimeError('Interrupted source maps do not match the physical stage seed')
        initial_file=physical_folder/'maps.metrics.json'
    initial_maps=read(initial_file)
    overall_file=Path(run.get('original_source_physical_root',run['source_root']))/'step001/maps.metrics.json'
    overall_maps=read(overall_file) if overall_file.exists() else initial_maps
    result={'observed_unix_s':time.time(),'checkpoint':str(cp),'summary_source_SHA256':sha(__file__),
        'previous_physical_folder':str(before_folder),'previous_material_SHA256':sha(cp/'stage0_initial_material.npy'),
        'returned_material_SHA256':sha(cp/'material.npy'),'fixed_reference_SHA256':run['fixed_reference_SHA256'],
        'material_update':update,'full_checkpoint_maps':maps,'round_initial_material_maps':initial_maps,
        'overall_initial_material_maps':overall_maps,
        'physical_line_search':read(cp/'line_search.json'),'physical_update_regions':regions,'RF_windows':windows,
        'RF_objective_decrease_by_geometry_window':contributions.tolist(),
        'fraction_of_RF_decrease_by_geometry_window':(contributions/decrease).tolist() if decrease else None,
        'no_new_wave_solves_training_or_inference':True,'no_GPU_work':True,'truth_arrays_not_read':True,
        'geometry_windows_are_arrival_proxies_not_unique_scatterer_depths':True,
        'full_native_model_may_still_be_running':True,'joint_imaging_success_claimed':False,
        'reserved_case_unread':'Neg_47_Left'}
    a.out.mkdir();atomic(a.out/'comparison.json',result)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    titles=['Sound-speed update (m/s)','Density update (kg/m³)','Attenuation update [dB/(cm MHz^1.1)]']
    for j,ax in enumerate(axes):
        scale=max(float(np.percentile(abs(delta[j][unknown]),99)),1e-12)
        im=ax.imshow(delta[j],extent=[x[0],x[-1],z[-1],z[0]],aspect='equal',interpolation='none',
            cmap='RdBu_r',vmin=-scale,vmax=scale)
        ax.add_patch(Rectangle((-19.1,1),38.2,42,fill=False,color='black',linewidth=1.2))
        ax.set_xlabel('Lateral (mm)');ax.set_ylabel('Depth (mm)');ax.set_title(titles[j],fontsize=9)
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle(f'{a.checkpoint}: actual change from the preceding accepted material; full FOV outlined',fontsize=10)
    fig.savefig(a.out/'physical_update_location.png',dpi=160);plt.close(fig)
    print(json.dumps({'checkpoint':str(cp),'update':update,'full_maps':maps['maps'],
        'physical_update_regions':regions,'RF_decrease_fractions':result['fraction_of_RF_decrease_by_geometry_window']}),flush=True)


if __name__=='__main__':main()
