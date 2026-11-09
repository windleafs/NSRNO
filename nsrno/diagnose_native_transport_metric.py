"""Diagnose the existing diagonal metric using an immutable saved native gradient."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import numpy as np
from .native_gn_coordinates import Coordinates, physical_blocks
from .native_gn_json import atomic


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gradient_physical(g):
    return np.stack([g[0] + .5*g[1], g[1], g[2]])


def direction_physical(u):
    return np.stack([u[0], u[1] - .5*u[0], u[2]])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.out.exists():raise RuntimeError('Preserve existing metric diagnosis')
    run=read(root/'run.json')
    if run['sample']!='train_200' or run['accepted_steps']!=0:
        raise RuntimeError('Use only the immutable zero-accepted-step original transport run')
    workers=read(root/'worker_processes.json')
    if any(Path('/proc',str(pid)).exists() for pid in [run['pid'],*workers['worker_pids']]):
        raise RuntimeError('The source gradient must be immutable after explicit stop')
    point=read(root/'accepted_point.json')
    if point['accepted_step']!=0 or point['evaluation']!=1:raise RuntimeError('Expected original base gradient')
    seed=np.load(root/'initial_material.npy')
    if sha(root/'accepted_material.npy')!=run['source_material_SHA256']:
        raise RuntimeError('Protected starting material changed')
    edge=np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')
    co=Coordinates(seed,edge);zero=np.zeros_like(seed,dtype=np.float64)
    lo,hi=co.bounds(trust=(1.,1.,10.));_,gtv,_,tvdiag=co.regularization(zero,.001)
    parent=read(Path(run['source_root'])/'run.json')
    reference=read(Path(parent['reference_source_root'])/'run.json')
    statistics=Path(run['variable_scaling_statistics'])
    if sha(statistics)!='acd29954a613f156e95530c9dd67d413e6f1ea9a1b1492ec51eb58a354bfa961':
        raise RuntimeError('Protected variable metric statistics changed')
    with np.load(statistics) as data:blocks=physical_blocks(data['blocks'],seed,edge)
    diagonal=np.maximum(reference['preconditioner_scale']*blocks[:3]+tvdiag,1e-30)
    metric=np.sqrt(diagonal);metric[:,~co.mask]=1.
    gx=np.load(root/'accepted_scaled_gradient.npy')
    total=gx*metric;data_gradient=total-gtv
    projected=np.clip(-gx,lo*metric,hi*metric)/metric
    unit=co.material(projected);stream=io.BytesIO();np.save(stream,unit)
    digest=hashlib.sha256(stream.getvalue()).hexdigest()
    if digest!='34ba3334fa2eb4639b70009395eaa660a545e133bfc5584b262d0719b0bc110d':
        raise RuntimeError('The metric does not reproduce the actually rejected unit trial')
    z=(np.arange(961)-60)*.05;x=(np.arange(1121)-560)*.05
    masks={
        'FOV':co.mask&(abs(x)[None,:]<=19.1)&(z[:,None]<=43),
        'lateral_outside':co.mask&(abs(x)[None,:]>19.1)&(z[:,None]<=43),
        'deeper_inside':co.mask&(abs(x)[None,:]<=19.1)&(z[:,None]>43),
        'deeper_outside':co.mask&(abs(x)[None,:]>19.1)&(z[:,None]>43)}
    fields={'native_data_gradient_physical_coordinates':gradient_physical(data_gradient),
            'total_gradient_physical_coordinates':gradient_physical(total),
            'preconditioned_unprojected_direction':direction_physical(-total/diagonal),
            'actual_bounded_unit_direction':direction_physical(projected)}
    summaries={}
    for name,array in fields.items():
        array[:,~co.mask]=0
        energy=np.sum(array[:,co.mask]**2,axis=1)
        summaries[name]={region:{
            'RMS':np.sqrt(np.mean(array[:,mask]**2,axis=1)).tolist(),
            'squared_norm_fraction':np.divide(np.sum(array[:,mask]**2,axis=1),energy,
                out=np.zeros(3),where=energy>0).tolist()} for region,mask in masks.items()}
    metric_statistics={region:[
        {'quantiles_1_10_50_90_99':np.percentile(diagonal[j][mask],[1,10,50,90,99]).tolist()}
        for j in range(3)] for region,mask in masks.items()}
    report={'source_root':str(root),'source_SHA256':sha(Path(__file__)),
        'physical_source_SHA256':run['source_material_SHA256'],
        'gradient_SHA256':sha(root/'accepted_scaled_gradient.npy'),
        'statistic_SHA256':sha(statistics),'unit_direction_material_SHA256':digest,
        'coordinates':['log(c)/.05','log(rho)/.10','log(tissue alpha)/.70'],
        'original_diagonal_coordinates':['log(c)/.05','log(Z)/.10','log(tissue alpha)/.70'],
        'regional_spatial_norms':summaries,'original_diagonal_statistics':metric_statistics,
        'RF_inner_gradient_error_not_zero':True,
        'different_objects_compared_by_spatial_fractions_not_absolute_gradient_amplitudes':True,
        'CPU_only':True,'new_waves_material_gradients_or_Jacobians':False,
        'truth_arrays_not_read':True,'reserved_case_unread':'Neg_47_Left',
        'does_not_prove_unpreconditioned_reconstruction_would_succeed':True,
        'joint_reconstruction_achieved':False}
    args.out.mkdir();atomic(args.out/'comparison.json',report)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    labels=['data gradient','total gradient','metric direction','bounded direction']
    colors=['#3c8b74','#d28b38','#5173a2','#9666a4']
    for j,ax in enumerate(axes):
        bottom=np.zeros(4)
        for region,color in zip(masks,colors):
            values=np.array([summaries[name][region]['squared_norm_fraction'][j] for name in fields])
            ax.bar(np.arange(4),values,bottom=bottom,label=region,color=color);bottom+=values
        ax.set_xticks(np.arange(4),labels,rotation=20)
        ax.set(ylim=(0,1),ylabel='Fraction of within-field squared norm',
               title=['Sound-speed coordinate','Density coordinate','Tissue-attenuation coordinate'][j])
    axes[-1].legend(fontsize=7)
    fig.suptitle('Saved base gradient: spatial distribution before and after the existing approximate metric',fontsize=10)
    fig.savefig(args.out/'spatial_metric_effect.png',dpi=170);plt.close(fig)
    print(json.dumps({'regional_spatial_norms':summaries,'diagonal_medians':{
        region:[row['quantiles_1_10_50_90_99'][2] for row in values]
        for region,values in metric_statistics.items()},
        'new_waves_or_gradients':False}),flush=True)


if __name__=='__main__':main()
