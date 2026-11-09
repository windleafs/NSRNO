"""CPU-only fully corrected span of already computed true native GN images."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import time
import traceback
import numpy as np
from .native_gn_coordinates import Coordinates,physical_blocks,dot
from .native_gn_secant import projected_norm
from .native_span_qp import solve
from .native_gn_json import atomic


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('out/v5/reconstruction_native_gn_enriched_v1_train_200'))
    p.add_argument('--out',type=Path,default=Path('out/v5/native_gn_saved_span_v1'))
    a=p.parse_args()
    if a.out.exists():raise RuntimeError('Preserve existing CPU result')
    old=json.loads((a.source/'run.json').read_text())
    if Path(f"/proc/{old['pid']}").exists():raise RuntimeError('Old source workflow still exists')
    reference=json.loads((a.source/'fixed_model_reference.json').read_text());origin=Path(reference['origin_step_folder'])
    if digest(origin/'old_material.npy')!='e49519034c78567e66d93c53eb83e52c4c5e204cb2068bde379d5d882094489b':
        raise RuntimeError('Wrong fixed linearization point')
    manifest=json.loads((a.source/'step001/fixed_HVP_manifest.json').read_text());paths=[Path(v) for v in manifest['paths']]
    if len(paths)!=15:raise ValueError('Reuse all fifteen completed fixed-reference HVPs')
    names=['solve_saved_native_gn_span.py','native_span_qp.py','native_gn_coordinates.py',
        'native_gn_secant.py','physical_box_direction.py','native_gn_json.py']
    a.out.mkdir();record={'status':'starting','pid':os.getpid(),'created_unix_s':time.time(),
        'source_root':str(a.source.resolve()),'fixed_reference':reference,'reused_true_HVPs':15,
        'new_gradient_or_HVP_or_forward_wave_solves':0,'GPU_work':False,'truth_read_by_optimizer':False,
        'full_space_optimality_claimed':False,'joint_imaging_success_claimed':False,
        'source_sha256':{name:digest(Path(__file__).parent/name) for name in names}}
    for name in names:(a.out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    atomic(a.out/'run.json',record);started=time.time()
    try:
        material=np.load(origin/'old_material.npy');co=Coordinates(material,np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'))
        oldrun=json.loads((origin.parent/'run.json').read_text())
        with np.load(oldrun['preconditioner_statistics']) as statistics:blocks=physical_blocks(statistics['blocks'],material,co.edge)
        first=json.loads((paths[0]/'progress.json').read_text());scale=first['local_data_preconditioner_scale']
        damping=reference['damping_relative']*scale*blocks[:3];damping[:,~co.mask]=0
        zero=np.zeros_like(material,dtype=np.float64)
        tv,g_tv,h_tv,tv_diagonal=co.regularization(zero,reference['tv_weight'])
        gradient=np.load(origin/'data_gradient.npy')+g_tv
        diagonal=scale*blocks[:3]+tv_diagonal+damping;diagonal[:,~co.mask]=1.
        diagonal=np.maximum(diagonal,1e-30);low,high=co.bounds()
        basis=[];images=[];norms=[];provenance=[];image_errors=[]
        for i,path in enumerate(paths):
            d=np.load(path/'direction.npy').astype(np.float64);image=np.load(path/'total_image.npy').astype(np.float64)
            data=np.load(path/'data_image.npy');reconstructed=data+h_tv(d)+damping*d
            error=float(np.linalg.norm(image-reconstructed)/max(np.linalg.norm(image),1e-30));image_errors.append(error)
            if error>1e-5:raise RuntimeError('Saved true images do not share the declared fixed metric')
            norm=np.sqrt(max(dot(d,diagonal*d),1e-300));basis.append(d.reshape(-1)/norm);images.append(image.reshape(-1)/norm);norms.append(norm)
            provenance.append({'path':str(path),'direction_SHA256':digest(path/'direction.npy'),
                'total_image_SHA256':digest(path/'total_image.npy'),'normalization':norm,'metric_image_relative_error':error})
            atomic(a.out/'progress.json',{'stage':'loading_saved_HVPs','completed':i+1,'total':len(paths)})
        basis=np.stack(basis);images=np.stack(images);raw=basis@images.T
        asymmetry=float(np.linalg.norm(raw-raw.T)/max(np.linalg.norm(raw),1e-30))
        if asymmetry>1e-4:raise RuntimeError('Saved true Hessian images fail restricted symmetry')
        gram=.5*(raw+raw.T);linear=basis@gradient.reshape(-1)
        values,vectors=np.linalg.eigh(gram);threshold=max(float(values[-1])*1e-10,1e-30)
        if values[0]< -threshold:raise RuntimeError('Restricted true GN metric is not positive semidefinite')
        keep=values>threshold
        if not np.all(keep):raise RuntimeError('Resolve degenerate saved basis before dropping physical directions')
        rotation=vectors/np.sqrt(values)[None,:];white=rotation.T@basis;white_images=rotation.T@images;white_linear=rotation.T@linear
        np.savez(a.out/'restricted_quadratic.npz',gram=gram,linear=linear,eigenvalues=values,normalizations=np.array(norms))
        def progress(item):
            if item['iteration']%5==0 or item['iteration']==1:
                atomic(a.out/'progress.json',{'stage':'solving_restricted_QP',**item})
        y,report=solve(white,white_linear,low.reshape(-1),high.reshape(-1),callback=progress)
        u=(white.T@y).reshape(material.shape);Hu=(white_images.T@y).reshape(material.shape)
        candidate=co.material(u);old_u=np.load(paths[-1]/'cumulative_log_direction.npy');old_Hu=np.load(paths[-1]/'cumulative_total_image.npy')
        old_value=dot(gradient,old_u)+.5*dot(old_u,old_Hu);value=dot(gradient,u)+.5*dot(u,Hu)
        full_relative=projected_norm(u,gradient+Hu,low,high,diagonal)/max(first['initial_projected_KKT_norm'],1e-300)
        if value>old_value+1e-8:raise RuntimeError('Fully corrected span did not improve saved feasible candidate')
        for name,array in [('log_direction',u),('total_image',Hu),('candidate.material',candidate)]:np.save(a.out/(name+'.npy'),array)
        np.save(a.out/'span_coefficients.npy',rotation@y)
        result={'restricted_solver':report,'fixed_reference_material_SHA256':digest(origin/'old_material.npy'),
            'HVP_provenance':provenance,'restricted_Hessian_relative_asymmetry':asymmetry,'restricted_Hessian_eigenvalues':values.tolist(),
            'old_sequential_quadratic_change':old_value,'fully_corrected_quadratic_change':value,
            'additional_predicted_quadratic_reduction':old_value-value,'restricted_value_reconstruction_error':abs(value-report['objective']),
            'full_projected_KKT_relative':full_relative,'full_space_optimality_claimed':False,
            'candidate_update_RMS_c_rho_total_alpha':np.sqrt(np.mean((candidate.astype(np.float64)-material)**2,axis=(1,2))).tolist(),
            'candidate_differs_from_current_physical_incumbent':True,'nonlinear_RF_score_not_computed':True,
            'saved_source_RF_loss':json.loads((origin/'baseline_RF.json').read_text())['loss'],
            'joint_imaging_success_claimed':False,'seconds':time.time()-started,
            'peak_CPU_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        atomic(a.out/'metrics.json',result);record.update(status='saved_span_CPU_solution_finished',finished_unix_s=time.time(),seconds=time.time()-started)
        atomic(a.out/'run.json',record);print(json.dumps({k:v for k,v in result.items() if k not in ['HVP_provenance','restricted_solver']}),flush=True)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time());atomic(a.out/'run.json',record);raise


if __name__=='__main__':main()
