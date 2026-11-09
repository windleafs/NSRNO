"""Evaluate an old frozen spectral metric on two saved unseen directions."""
from pathlib import Path
import argparse
import json
import os
import resource
import time
import traceback
import numpy as np
from .native_gn_recycled_state import RecycledState,digest
from .native_gn_coordinates import dot,physical_blocks
from .native_gn_spectral import SpectralDataMetric
from .physical_box_direction import local_apply
from .native_gn_json import atomic


def quality(direction,prediction,target,diagonal):
    energy=dot(direction,target);difference=prediction-target
    pp=float(np.sum(prediction*prediction/diagonal));yy=float(np.sum(target*target/diagonal))
    cross=float(np.sum(prediction*target/diagonal));oracle=max(0.,cross/max(pp,1e-300))
    return {'true_data_curvature':energy,'predicted_data_curvature':dot(direction,prediction),
        'curvature_prediction_over_true':dot(direction,prediction)/energy,
        'image_relative_error':float(np.linalg.norm(difference)/np.linalg.norm(target)),
        'D_inverse_weighted_image_relative_error':float(np.sqrt(np.sum(difference*difference/diagonal)/
            np.sum(target*target/diagonal))),
        'D_inverse_weighted_image_cosine':cross/max(np.sqrt(pp*yy),1e-300),
        'oracle_positive_scalar_on_this_test_pair':oracle,
        'oracle_rescaled_D_inverse_relative_error':float(np.sqrt(max(0.,yy-2*oracle*cross+oracle*oracle*pp)/yy)),
        'oracle_uses_the_test_target_not_independent_validation':True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,default=Path('out/v5/native_recycled_metric_analysis_v1/HVP_manifest_snapshot.json'))
    p.add_argument('--symbol',type=Path,default=Path('out/v5/native_gn_spectral_v1/symbol.npy'))
    p.add_argument('--out',type=Path,default=Path('out/v5/native_spectral_recycled_holdout_v1'))
    a=p.parse_args()
    if a.out.exists():raise RuntimeError('Preserve saved held-out assessment')
    snapshot=json.loads(a.snapshot.read_text());paths=[Path(v) for v in snapshot['paths'][15:]]
    if len(paths)!=2:raise RuntimeError('Use exactly the earlier fixed seventeen-direction snapshot')
    metadata=json.loads((a.symbol.parent/'report.json').read_text())
    if digest(a.symbol)!=metadata['symbol_SHA256']:raise RuntimeError('Frozen symbol changed')
    training={str(Path(v).resolve()) for v in metadata['training_paths']}
    if any(str(path.resolve()) in training for path in paths):raise RuntimeError('Held-out direction was fitted')
    a.out.mkdir();(a.out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    record={'status':'CPU_frozen_spectral_assessment','pid':os.getpid(),'created_unix_s':time.time(),
        'source_SHA256':digest(__file__),'symbol_SHA256':metadata['symbol_SHA256'],
        'symbol_training_paths':sorted(training),'held_out_paths':[str(p) for p in paths],
        'new_wave_solves':0,'GPU_work':False,'no_refitting':True,'truth_read':False}
    atomic(a.out/'run.json',record);started=time.time()
    try:
        state=RecycledState(Path('out/v5/reconstruction_native_gn_enriched_v1_train_200'))
        if metadata['fixed_material_reference']!=state.reference:raise RuntimeError('Different native reference')
        with np.load(state.preconditioner) as data:
            blocks=state.scale*physical_blocks(data['blocks'],state.material,state.co.edge)
        model=SpectralDataMetric(np.load(a.symbol,mmap_mode='r'),state.co.mask)
        rows=[]
        for path in paths:
            d=np.load(path/'direction.npy');target=np.load(path/'data_image.npy')
            regularizer=state.h_tv(d)+state.damping*d;total=np.load(path/'total_image.npy')
            if np.linalg.norm(total-target-regularizer)>1e-5*np.linalg.norm(total):
                raise RuntimeError('Different fixed native metric in the held-out checkpoint')
            predictions={'diagonal_data_metric':blocks[:3]*d,'local_block_data_metric':local_apply(blocks,d),
                'frozen_nonlocal_spectral_data_metric':model.apply(d)}
            details={}
            for name,value in predictions.items():
                detail=quality(d,value,target,state.diagonal)
                # These directions were generated after removing the first 15
                # true H images. Separate their still-unknown image component.
                error=(value-target).reshape(-1)
                projection=state.span.images.T@np.linalg.solve(state.span.gram,state.span.basis@error)
                complement=(error-projection).reshape(d.shape)
                detail['known_15_image_complement_D_inverse_error']=float(np.sqrt(
                    np.sum(complement*complement/state.diagonal)/np.sum(total*total/state.diagonal)))
                details[name]=detail
            rows.append({'path':str(path),'direction_SHA256':digest(path/'direction.npy'),
                'data_image_SHA256':digest(path/'data_image.npy'),'models':details})
            atomic(a.out/'progress.json',{'completed':len(rows),'total':len(paths)})
        result={'pairs':rows,'frozen_symbol_SHA256':metadata['symbol_SHA256'],'symbol_refitted':False,
            'no_new_wave_solves_or_GPU_work':True,'same_native_reference':True,
            'not_a_test_of_inverse_metric_action_or_future_convergence':True,
            'not_a_new_RF_forward_operator':True,'joint_imaging_success_claimed':False,
            'seconds':time.time()-started,'peak_CPU_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        atomic(a.out/'metrics.json',result)
        record.update(status='CPU_frozen_spectral_assessment_finished',finished_unix_s=time.time(),seconds=time.time()-started)
        atomic(a.out/'run.json',record);print(json.dumps(result),flush=True)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time());atomic(a.out/'run.json',record);raise


if __name__=='__main__':main()
