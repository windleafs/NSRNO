"""Same-pair CPU control for global scale versus nonlocal metric fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from .native_gn_coordinates import physical_blocks,dot
from .physical_box_direction import local_apply


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--train-count',type=int,default=10)
    p.add_argument('--heldout-number',type=int,default=11)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise RuntimeError('New output only')
    started=time.time()
    paths=[Path(p) for p in json.loads((args.source_root/'step001/fixed_HVP_manifest.json').read_text())['paths']]
    if not 0<args.train_count<args.heldout_number<=len(paths):raise ValueError('Not a held-out pair')
    origin=Path(json.loads((args.source_root/'fixed_model_reference.json').read_text())['origin_step_folder'])
    run=json.loads((origin.parent/'run.json').read_text())
    material=np.load(origin/'old_material.npy')
    with np.load(run['preconditioner_statistics']) as data:
        blocks=physical_blocks(data['blocks'],material,
            np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'))
    numerator=denominator=0.
    mask=np.broadcast_to(np.arange(material.shape[1])[:,None]>=80,material.shape[1:])
    for path in paths[:args.train_count]:
        d=np.load(path/'direction.npy')
        y=np.load(path/'data_image.npy')
        local=local_apply(blocks,d)*mask
        weight=1/max(dot(d,d),1e-300)
        numerator+=weight*dot(local,y)
        denominator+=weight*dot(local,local)
    scale=max(numerator/max(denominator,1e-300),1e-12)
    heldout=paths[args.heldout_number-1]
    d=np.load(heldout/'direction.npy')
    data=np.load(heldout/'data_image.npy')
    prior=np.load(heldout/'total_image.npy')-data
    prediction=scale*local_apply(blocks,d)*mask
    error=prediction-data
    result={'kind':'weighted_least_squares_scalar_recalibration_of_existing_local_data_metric',
        'training_count':args.train_count,'heldout_number':args.heldout_number,
        'training_input_norm_weights':'same inverse input-norm-squared as spectral fit',
        'fitted_positive_scale':scale,'unconstrained_scale':numerator/max(denominator,1e-300),
        'true_data_curvature':dot(d,data),'predicted_data_curvature':dot(d,prediction),
        'predicted_to_true_data_curvature':dot(d,prediction)/dot(d,data),
        'data_image_relative_error':np.sqrt(dot(error,error)/dot(data,data)),
        'predicted_to_true_total_curvature':dot(d,prediction+prior)/dot(d,data+prior),
        'material_truth_read':False,'GPU_used':False,'new_wave_solves':0,
        'generator_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seconds':time.time()-started}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
