"""CPU-only surface initialization from a previously RF-conditioned prediction.

This creates a separate, unaccepted material candidate. It neither runs
propagation nor freezes any material variables in the later inversion.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--incumbent',type=Path,required=True)
    p.add_argument('--prediction',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--full-to-mm',type=float,default=3.)
    p.add_argument('--zero-from-mm',type=float,default=5.)
    p.add_argument('--geometry',type=Path,default=Path('out/v5/time_triparam_v1/geometry.json'))
    p.add_argument('--edge',type=Path,default=Path('out/v5/uno_triparam_v1/known_edge_alpha.npy'))
    args=p.parse_args()
    if args.out.exists():raise RuntimeError('Use a separate new candidate directory')
    if not 1.<args.full_to_mm<args.zero_from_mm<=43.:
        raise ValueError('Require 1 < full depth < zero depth <= 43 mm')
    started=time.time()
    meta=json.loads(args.geometry.read_text())
    incumbent=np.load(args.incumbent)
    prediction=np.load(args.prediction)
    if incumbent.dtype!=np.float32 or prediction.shape!=incumbent.shape:
        raise ValueError('Require native float32 materials on the identical grid')
    if incumbent.shape!=(3,*meta['shape']):raise ValueError('Geometry/material grid mismatch')
    edge=np.broadcast_to(np.load(args.edge),incumbent.shape[1:])
    if not np.all(np.isfinite(incumbent)) or not np.all(np.isfinite(prediction)):
        raise ValueError('Nonfinite material input')
    # Use the same FOV-depth convention as evaluate_time_material and the
    # cached RF-conditioned image. The sparse interpolation origin differs.
    rows=np.arange(incumbent.shape[1])
    depth_mm=(rows-meta['face'])*meta['dx']*1000
    phase=np.clip((depth_mm-args.full_to_mm)/(args.zero_from_mm-args.full_to_mm),0,1)
    weight=.5*(1+np.cos(np.pi*phase))
    weight[(rows<80)|(depth_mm>=args.zero_from_mm)]=0
    weight=np.broadcast_to(weight[:,None],incumbent.shape[1:])
    active=weight>0
    blend=weight[active]
    output=incumbent.copy()
    for channel in [0,1]:
        old=incumbent[channel,active].astype(np.float64)
        learned=prediction[channel,active].astype(np.float64)
        if np.any(old<=0) or np.any(learned<=0):raise ValueError('Nonpositive speed or density')
        output[channel,active]=np.exp((1-blend)*np.log(old)+blend*np.log(learned)).astype(np.float32)
    old=incumbent[2,active].astype(np.float64)-edge[active].astype(np.float64)
    learned=prediction[2,active].astype(np.float64)-edge[active].astype(np.float64)
    if np.any(old<=0) or np.any(learned<=0):raise ValueError('Nonpositive tissue attenuation')
    tissue=np.exp((1-blend)*np.log(old)+blend*np.log(learned))
    output[2,active]=(edge[active].astype(np.float64)+tissue).astype(np.float32)
    if not np.all(np.isfinite(output)):raise RuntimeError('Nonfinite seed output')
    if not np.array_equal(output[:,~active],incumbent[:,~active]):
        raise RuntimeError('Initialization changed pixels outside the surface support')
    if not np.array_equal(output[:,:80],incumbent[:,:80]):
        raise RuntimeError('Initialization changed the known prefix')
    if np.any((output[2]-edge)[active]<=0):raise RuntimeError('Float32 seed lost tissue positivity')
    delta=output.astype(np.float64)-incumbent
    args.out.mkdir(parents=True)
    material=args.out/'candidate_material_unaccepted.npy'
    np.save(material,output)
    report={'kind':'RF_conditioned_superficial_initialization_candidate',
        'candidate_material':str(material.resolve()),'candidate_SHA256':digest(material),
        'incumbent':str(args.incumbent.resolve()),'incumbent_SHA256':digest(args.incumbent),
        'cached_RF_conditioned_prediction':str(args.prediction.resolve()),
        'prediction_SHA256':digest(args.prediction),'geometry_SHA256':digest(args.geometry),
        'generator_SHA256':digest(__file__),'native_shape':list(output.shape),
        'native_dx_m':meta['dx'],'FOV_zero_grid_row':meta['face'],'known_prefix_rows':80,
        'full_prediction_to_mm':args.full_to_mm,'zero_prediction_from_mm':args.zero_from_mm,
        'transition':'cosine in depth, geometric in c/rho/tissue-alpha',
        'depth_selection_basis':'declared development design from previously observed superficial illumination and cached inverse-UNO maps; not an independently validated optimal depth',
        'RMS_change_in_surface_c_rho_total_alpha':np.sqrt(np.mean(delta[:,active]**2,axis=1)).tolist(),
        'RMS_change_over_all_unknown_pixels':np.sqrt(np.mean(delta[:,80:]**2,axis=(1,2))).tolist(),
        'all_other_pixels_bitwise_unchanged':True,'known_prefix_bitwise_unchanged':True,
        'subsequent_unknowns':'all native c/rho/tissue-alpha pixels remain independent; no surface or deep variables frozen',
        'material_truth_read':False,'GPU_used':False,'patient_wave_solves':0,
        'native_RF_evaluation_performed':False,'accepted_physical_reconstruction':False,
        'seconds':time.time()-started}
    (args.out/'initialization.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
