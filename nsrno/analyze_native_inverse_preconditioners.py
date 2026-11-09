"""CPU inverse-action comparison on two genuinely unseen saved H directions."""
from pathlib import Path
import argparse
import json
import os
import resource
import time
import traceback
import numpy as np
from .native_gn_recycled_state import RecycledState,digest
from .native_gn_coordinates import physical_blocks,inverse_blocks
from .native_gn_spectral import SpectralDataMetric
from .native_gn_json import atomic


def quality(value,target,diagonal):
    vv=float(np.sum(value*value*diagonal));tt=float(np.sum(target*target*diagonal));cross=float(np.sum(value*target*diagonal))
    scalar=max(0.,cross/max(vv,1e-300))
    return {'D_relative_inverse_error':float(np.sqrt(np.sum((value-target)**2*diagonal)/tt)),
        'D_cosine':cross/max(np.sqrt(vv*tt),1e-300),'D_norm_gain':float(np.sqrt(vv/tt)),
        'oracle_positive_rescale':scalar,
        'oracle_rescaled_D_error':float(np.sqrt(max(0.,tt-2*scalar*cross+scalar*scalar*vv)/tt)),
        'oracle_rescale_uses_target_and_is_diagnostic_only':True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('out/v5/native_inverse_preconditioner_assessment_v1'))
    a=p.parse_args()
    if a.out.exists():raise RuntimeError('Preserve existing inverse assessment')
    snapshot=Path('out/v5/native_recycled_metric_analysis_v1/HVP_manifest_snapshot.json')
    paths=[Path(v) for v in json.loads(snapshot.read_text())['paths'][15:]]
    if len(paths)!=2:raise RuntimeError('Use the same fixed pair of held-out directions')
    symbol_path=Path('out/v5/native_gn_spectral_v1/symbol.npy');metadata=json.loads((symbol_path.parent/'report.json').read_text())
    if digest(symbol_path)!=metadata['symbol_SHA256']:raise RuntimeError('Frozen symbol changed')
    if any(str(path) in metadata['training_paths'] for path in paths):raise RuntimeError('Test direction was used in fitting')
    a.out.mkdir();(a.out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    record={'status':'CPU_inverse_action_assessment','pid':os.getpid(),'created_unix_s':time.time(),
        'source_SHA256':digest(__file__),'symbol_SHA256':metadata['symbol_SHA256'],
        'new_wave_solves':0,'GPU_work':False,'truth_read':False,'no_metric_refitting':True}
    atomic(a.out/'run.json',record);started=time.time()
    try:
        state=RecycledState(Path('out/v5/reconstruction_native_gn_enriched_v1_train_200'))
        if state.reference!=metadata['fixed_material_reference']:raise RuntimeError('Different fixed reference')
        with np.load(state.preconditioner) as data:
            data_blocks=state.scale*physical_blocks(data['blocks'],state.material,state.co.edge)
        local_blocks=data_blocks.copy();local_blocks[:3]=state.diagonal;local_blocks[3:,~state.co.mask]=0
        reg_diagonal=np.maximum(state.diagonal-data_blocks[:3],1e-30)
        shift=reg_diagonal[:,state.co.mask].mean(axis=1)
        symbol=np.array(np.load(symbol_path,mmap_mode='r'),copy=True)
        for j in range(3):symbol[...,j,j]+=shift[j]
        inverse_symbol=np.linalg.inv(symbol);del symbol
        spectral_inverse=SpectralDataMetric(inverse_symbol,state.co.mask)
        def coarse_coordinates(value):return np.linalg.solve(state.span.gram,state.span.basis@value.reshape(-1))
        def balanced(rhs,apply):
            coefficient=coarse_coordinates(rhs)
            right=rhs.reshape(-1)-state.span.images.T@coefficient
            value=apply(right.reshape(state.material.shape)).reshape(-1)
            value-=state.span.basis.T@np.linalg.solve(state.span.gram,state.span.images@value)
            return (state.span.basis.T@coefficient+value).reshape(state.material.shape)
        models={'diagonal_inverse':lambda rhs:rhs/state.diagonal,
            'local_block_inverse':lambda rhs:inverse_blocks(local_blocks,rhs)*state.co.mask,
            'spectral_plus_mean_regularizer_inverse':spectral_inverse.apply}
        rows=[]
        for path in paths:
            d=np.load(path/'direction.npy');image=np.load(path/'total_image.npy');details={}
            for name,apply in models.items():
                value=apply(image);corrected=balanced(image,apply)
                details[name]={'base_inverse':quality(value,d,state.diagonal),
                    'true_15_direction_balanced_inverse':quality(corrected,d,state.diagonal)}
            rows.append({'path':str(path),'direction_SHA256':digest(path/'direction.npy'),
                'total_image_SHA256':digest(path/'total_image.npy'),'models':details})
            atomic(a.out/'progress.json',{'completed':len(rows),'total':2})
        result={'pairs':rows,'spectral_regularizer_constant_diagonal':shift.tolist(),
            'spectral_regularizer_chosen_without_test_targets':'spatial mean of the fixed TV+damping diagonal',
            'spectral_inverse_formula':'mask F^-1 [S(k)+diag(mean(R_diagonal))]^-1 F mask',
            'balanced_inverse_formula':'B G^-1 B^T + (I-B G^-1 Y^T) M0^-1 (I-Y G^-1 B^T)',
            'same_frozen_spectral_symbol':metadata['symbol_SHA256'],'no_new_wave_solves_or_GPU_work':True,
            'no_active_algorithm_change':True,'test_of_inverse_action_not_future_convergence':True,
            'joint_imaging_success_claimed':False,'seconds':time.time()-started,
            'peak_CPU_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        atomic(a.out/'metrics.json',result)
        record.update(status='CPU_inverse_action_assessment_finished',finished_unix_s=time.time(),seconds=time.time()-started)
        atomic(a.out/'run.json',record);print(json.dumps(result),flush=True)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time());atomic(a.out/'run.json',record);raise


if __name__=='__main__':main()
