"""Fit the missing source-resolved normal information using saved vectors.

For U=JD, V_sj=J_s'W_s J_s d_j, build the exact feature-space Schur
complement after projecting out U. No extra native propagation is used.
The held-out direction's image is used only for the final comparison.
"""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from .native_gn_coordinates import dot
from .native_gn_json import atomic


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--train-count',type=int,default=12)
    p.add_argument('--heldout-number',type=int,default=13)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise RuntimeError('Use a new fitting directory')
    started=time.time()
    paths=[Path(p) for p in json.loads((args.source_root/'step001/fixed_HVP_manifest.json').read_text())['paths']]
    if not 0<args.train_count<args.heldout_number<=len(paths):raise ValueError('Require a held-out image')
    training=paths[:args.train_count]
    D=[np.load(p/'direction.npy',mmap_mode='r') for p in training]
    Y=[np.load(p/'data_image.npy',mmap_mode='r') for p in training]
    G=np.array([[dot(d,y) for y in Y] for d in D])
    G=.5*(G+G.T)
    features=[]
    for j,path in enumerate(training):
        metadata=path/'source_curvature.json'
        if not metadata.exists():continue
        for row in json.loads(metadata.read_text())['shots']:
            features.append(dict(row,direction_index=j))
    if not features:raise RuntimeError('No per-shot images in the training prefix')
    V=[np.load(f['image'],mmap_mode='r') for f in features]
    C=np.array([[dot(d,v) for v in V] for d in D])
    E=np.zeros((len(V),len(V)))
    for i,a in enumerate(features):
        for j,b in enumerate(features):
            if a['name']==b['name']:E[i,j]=dot(D[a['direction_index']],V[j])
    E=.5*(E+E.T)
    coefficients=np.linalg.solve(G,C)
    K=E-C.T@coefficients
    K=.5*(K+K.T)
    normalization=np.sqrt(np.diag(E))
    normalized=K/normalization[:,None]/normalization[None,:]
    eigenvalues,eigenvectors=np.linalg.eigh(normalized)
    maximum=max(float(eigenvalues.max()),1.)
    if eigenvalues.min() < -1e-4*maximum:
        raise RuntimeError(f'Source feature Schur complement is not PSD: {eigenvalues.min()}')
    threshold=1e-6*maximum
    keep=eigenvalues>threshold
    factor=(eigenvectors[:,keep]/np.sqrt(eigenvalues[keep])[None,:])/normalization[:,None]
    heldout=paths[args.heldout_number-1]
    d=np.load(heldout/'direction.npy')
    actual=np.load(heldout/'data_image.npy')
    ycross=np.array([dot(y,d) for y in Y])
    vcross=np.array([dot(v,d) for v in V])
    residual_cross=vcross-C.T@np.linalg.solve(G,ycross)
    projection=factor.T@residual_cross
    data_source_curvature=float(projection@projection)
    data_aggregate_curvature=float(ycross@np.linalg.solve(G,ycross))
    reference=json.loads((args.source_root/'fixed_model_reference.json').read_text())
    origin=Path(reference['origin_step_folder'])
    gradient=np.load(origin/'data_gradient.npy')
    energy=2*json.loads((origin/'baseline_RF.json').read_text())['loss']
    b=np.array([dot(q,gradient) for q in D])
    a=np.linalg.solve(G,b)
    remaining_energy=energy-float(b@a)
    residual_gradient_cross=dot(d,gradient)-float(ycross@a)
    gradient_curvature=residual_gradient_cross**2/remaining_energy
    true=dot(d,actual)
    row=json.loads((heldout/'progress.json').read_text())
    args.out.mkdir(parents=True)
    np.savez(args.out/'feature_gram.npz',G=G,C=C,E=E,K=K,
        conditional_inverse_factor=factor,eigenvalues=eigenvalues)
    atomic(args.out/'features.json',{'training_paths':[str(p) for p in training],
        'features':features,'reference':reference})
    report={'kind':'source_residual_Nystrom_information_fit_without_new_waves',
        'training_direction_count':len(D),'source_feature_count':len(V),
        'conditional_modes_kept':int(keep.sum()),'conditional_eigenvalue_threshold':threshold,
        'minimum_conditional_eigenvalue':float(eigenvalues.min()),
        'maximum_conditional_eigenvalue':float(eigenvalues.max()),
        'heldout_HVP':args.heldout_number,'true_data_curvature':true,
        'aggregate_Nystrom_data_curvature':data_aggregate_curvature,
        'extra_source_residual_curvature':data_source_curvature,
        'aggregate_plus_source_to_true_ratio':(data_aggregate_curvature+data_source_curvature)/true,
        'extra_gradient_residual_curvature':gradient_curvature,
        'source_plus_gradient_feature_to_true_ratio':(data_aggregate_curvature+data_source_curvature+gradient_curvature)/true,
        'current_approximate_total_curvature':row['compact_secant']['sBs'],
        'source_features_do_not_use_heldout_image':True,
        'GPU_used':False,'new_wave_solves':0,'material_truth_read':False,
        'used_for_model_selection_not_independent_material_test':True,
        'generator_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seconds':time.time()-started}
    atomic(args.out/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
