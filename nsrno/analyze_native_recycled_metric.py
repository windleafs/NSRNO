"""CPU Ritz comparison on a snapshot of already completed true HVPs."""
from pathlib import Path
import argparse
import json
import os
import resource
import time
import traceback
import numpy as np
from .native_gn_recycled_state import RecycledState,digest
from .native_gn_coordinates import physical_blocks
from .physical_box_direction import local_apply
from .native_gn_json import atomic


def spectrum(curvature,metric):
    metric=.5*(metric+metric.T);small=np.linalg.eigvalsh(metric)
    if small[0]<=0:raise RuntimeError('Snapshot metric Gram is not positive definite')
    lower=np.linalg.cholesky(metric)
    white=np.linalg.solve(lower,curvature);white=np.linalg.solve(lower,white.T).T
    values=np.linalg.eigvalsh(.5*(white+white.T))
    if values[0]<=0:raise RuntimeError('Snapshot generalized true curvature is not positive definite')
    return {'generalized_eigenvalues':values.tolist(),'Ritz_condition_number':float(values[-1]/values[0]),
        'metric_Gram_eigenvalues':small.tolist(),'metric_Gram_condition_number':float(small[-1]/small[0]),
        'observed_span_only':True,'full_Hessian_condition_number_not_established':True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('out/v5/reconstruction_native_recycled_krylov_v1_train_200'))
    p.add_argument('--out',type=Path,default=Path('out/v5/native_recycled_metric_analysis_v1'))
    a=p.parse_args();native=json.loads((a.root/'run.json').read_text())
    manifest_path=a.root/'step001/fixed_HVP_manifest.json';snapshot=manifest_path.read_bytes();manifest=json.loads(snapshot)
    paths=[Path(p) for p in manifest['paths']]
    if len(paths)<17:raise RuntimeError('Wait for at least two new complete HVPs')
    if a.out.exists():raise RuntimeError('Preserve prior CPU analysis')
    a.out.mkdir();(a.out/'HVP_manifest_snapshot.json').write_bytes(snapshot)
    (a.out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    state_record={'status':'CPU_loading_saved_snapshot','pid':os.getpid(),'created_unix_s':time.time(),
        'snapshot_HVPs':len(paths),'source_root':str(a.root.resolve()),'source_SHA256':digest(__file__),
        'native_source_sha256':native['source_sha256'],
        'new_gradient_HVP_or_forward_wave_solves':0,'GPU_work':False,'truth_read':False,
        'active_native_source_files_unchanged':True,'joint_imaging_success_claimed':False}
    atomic(a.out/'run.json',state_record);started=time.time()
    try:
        state=RecycledState(Path(native['source_root']))
        if digest(state.origin/'old_material.npy')!=native['fixed_reference_SHA256']:raise RuntimeError('Different fixed reference')
        if paths[:15]!=state.paths:raise RuntimeError('Snapshot changed the original true images')
        provenance=[]
        for path in paths[15:]:
            if not (path/'progress.json').exists():raise RuntimeError('Manifest contains an incomplete native HVP')
            state.span.append(np.load(path/'direction.npy'),np.load(path/'total_image.npy'))
            provenance.append({'path':str(path),'direction_SHA256':digest(path/'direction.npy'),
                'total_image_SHA256':digest(path/'total_image.npy')})
        with np.load(state.preconditioner) as data:
            blocks=state.scale*physical_blocks(data['blocks'],state.material,state.co.edge)
        blocks[:3]=state.diagonal;blocks[3:,~state.co.mask]=0
        basis=state.span.basis;gram_diagonal=np.empty_like(state.span.gram);gram_block=np.empty_like(gram_diagonal)
        for j,row in enumerate(basis):
            gram_diagonal[:,j]=basis@(state.diagonal.reshape(-1)*row)
            gram_block[:,j]=basis@local_apply(blocks,row.reshape(state.material.shape)).reshape(-1)
            atomic(a.out/'progress.json',{'stage':'forming_observed_metric_Grams','completed':j+1,'total':len(basis)})
        values={'diagonal':spectrum(state.span.gram,gram_diagonal),'local_three_parameter_block':spectrum(state.span.gram,gram_block)}
        np.savez(a.out/'snapshot_Grams.npz',true_curvature=state.span.gram,
            diagonal_metric=gram_diagonal,local_block_metric=gram_block)
        result={'snapshot_HVPs':len(paths),'metrics':values,'new_HVP_provenance':provenance,
            'same_fixed_native_reference':native['fixed_reference_SHA256'],
            'metric_block_definition':'same physical block off-diagonals; identical diagonal, TV diagonal and damping as current D',
            'comparison_does_not_establish_unseen_direction_spectrum':True,'no_new_wave_solves_or_material_truth':True,
            'no_active_algorithm_change':True,'no_joint_reconstruction_claim':True,
            'seconds':time.time()-started,'peak_CPU_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        atomic(a.out/'metrics.json',result)
        state_record.update(status='CPU_metric_comparison_finished',finished_unix_s=time.time(),seconds=time.time()-started)
        atomic(a.out/'run.json',state_record);print(json.dumps(result),flush=True)
    except BaseException:
        state_record.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time());atomic(a.out/'run.json',state_record);raise


if __name__=='__main__':main()
