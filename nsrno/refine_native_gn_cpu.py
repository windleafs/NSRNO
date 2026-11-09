"""Restore a saved native-GN quadratic and finish its CPU box solve.

No wave propagation, GPU initialization, target material, or active solver
mutation is involved. The result is a proposal requiring a future true HVP
and physical RF line search before it can become a material update.
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
from scipy.optimize import Bounds,minimize
from .native_gn_coordinates import Coordinates,physical_blocks,dot
from .native_gn_secant import CompactBFGS,projected_norm
from .physical_box_direction import local_apply

def atomic(path,value):
    temp=path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def save_array(path,value):
    temp=path.with_name(path.stem+'.partial.npy')
    np.save(temp,value);temp.replace(path)

def restore(folder,count,tv_weight=.001,damping_relative=.01):
    run=json.loads((folder.parent/'run.json').read_text())
    material=np.load(folder/'old_material.npy')
    coordinates=Coordinates(material,np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'))
    with np.load(run['preconditioner_statistics']) as data:
        blocks=physical_blocks(data['blocks'],material,coordinates.edge)
    first=json.loads((folder/'hvp00/progress.json').read_text())
    scale=first['local_data_preconditioner_scale']
    damping=damping_relative*scale*blocks[:3]
    damping[:,~coordinates.mask]=0
    zero=np.zeros_like(material,dtype=np.float64)
    _,g_tv,h_tv,tv_diagonal=coordinates.regularization(zero,tv_weight)
    def base(x):
        return (scale*local_apply(blocks,x)+h_tv(x)+damping*x)*coordinates.mask
    diagonal=scale*blocks[:3]+tv_diagonal+damping
    diagonal[:,~coordinates.mask]=1.
    model=CompactBFGS(base,diagonal,coordinates.mask)
    for i in range(count):
        checkpoint=folder/f'hvp{i:02d}'
        if not (checkpoint/'progress.json').exists():
            raise RuntimeError(f'Incomplete checkpoint {checkpoint}; no reconstruction')
        direction=np.load(checkpoint/'direction.npy')
        image=np.load(checkpoint/'total_image.npy')
        model.update(direction,image)
    last=folder/f'hvp{count-1:02d}'
    record=json.loads((last/'progress.json').read_text())
    u=np.load(last/'cumulative_log_direction.npy')
    Hu=np.load(last/'cumulative_total_image.npy')
    gradient=np.load(folder/'data_gradient.npy')+g_tv
    value=dot(gradient,u)+.5*dot(u,Hu)
    if abs(value-record['model_change'])>1e-10:
        raise RuntimeError('Saved true quadratic state did not reconstruct')
    if abs(dot(u,damping*u)-record['cumulative_damping_quadratic'])>1e-10:
        raise RuntimeError('Damping differs from the saved fixed-H problem')
    if abs(dot(u,h_tv(u))-record['cumulative_TV_quadratic'])>1e-10:
        raise RuntimeError('TV differs from the saved fixed-H problem')
    low,high=coordinates.bounds()
    return coordinates,model,gradient+Hu,low-u,high-u,u,record

class TargetReached(Exception):
    pass

def solve_accurately(model,gradient,low,high,initial,maximum=1200,
                     tolerance=.01,callback=None):
    started=time.time();shape=gradient.shape
    diagonal=np.maximum(model.diagonal,1e-30)
    metric=np.sqrt(diagonal)
    cold_norm=projected_norm(np.zeros(shape),gradient,low,high,diagonal)
    initial=np.asarray(initial,np.float64)
    if np.any(initial<low-1e-10) or np.any(initial>high+1e-10):
        raise RuntimeError('The saved CPU direction is infeasible')
    original_norm=projected_norm(initial,gradient+model.apply(initial),low,high,diagonal)
    original_value=dot(gradient,initial)+.5*dot(initial,model.apply(initial))
    last_y=(initial*metric).ravel().copy()
    iterations=0;evaluations=0;history=[]
    def objective(y):
        nonlocal evaluations
        evaluations+=1
        d=y.reshape(shape)/metric
        image=model.apply(d)
        return dot(gradient,d)+.5*dot(d,image),((gradient+image)/metric).ravel()
    def report(y,force=False):
        nonlocal last_y
        d=y.reshape(shape)/metric
        image=model.apply(d)
        norm=projected_norm(d,gradient+image,low,high,diagonal)
        row={'additional_iterations':iterations,'function_evaluations':evaluations,
            'model_change':dot(gradient,d)+.5*dot(d,image),
            'relative_KKT':norm/max(cold_norm,1e-300),
            'seconds':time.time()-started,'GPU_used':False,'patient_wave_solves':0}
        history.append(row)
        if callback:callback(row,d,force or iterations%50==0)
        if norm<=tolerance*max(cold_norm,1e-300):
            last_y=y.copy();raise TargetReached()
        return row
    def accepted(y):
        nonlocal iterations
        iterations+=1
        if iterations%10==0:report(y)
    message='initial saved direction already meets target'
    software_success=None
    if original_norm>tolerance*max(cold_norm,1e-300):
        try:
            result=minimize(objective,last_y,method='L-BFGS-B',jac=True,
                bounds=Bounds((low*metric).ravel(),(high*metric).ravel()),callback=accepted,
                options={'maxiter':maximum,'maxcor':10,'maxls':50,'ftol':0.,
                    'gtol':1e-12,'maxfun':maximum*5})
            last_y=result.x.copy();message=str(result.message)
            software_success=bool(result.success)
        except TargetReached:
            message='projected KKT target reached; stopped by explicit accuracy criterion'
    d=last_y.reshape(shape)/metric
    image=model.apply(d)
    value=dot(gradient,d)+.5*dot(d,image)
    norm=projected_norm(d,gradient+image,low,high,diagonal)
    if value>original_value+1e-9 or np.any(d<low-1e-10) or np.any(d>high+1e-10):
        raise RuntimeError('Refined CPU result lost objective decrease or feasibility')
    record={'method':'continue_same_saved_compact_box_QP_to_KKT_target',
        'initial_approximate_model_change':original_value,'approximate_model_change':value,
        'original_relative_KKT':original_norm/max(cold_norm,1e-300),
        'relative_KKT':norm/max(cold_norm,1e-300),'target_relative_KKT':tolerance,
        'tolerance_met':bool(norm<=tolerance*max(cold_norm,1e-300)),
        'additional_iterations':iterations,'function_evaluations':evaluations,
        'message':message,'software_success':software_success,'history':history,
        'seconds':time.time()-started,'GPU_used':False,'patient_wave_solves':0,
        'candidate_has_true_HVP':False,'candidate_is_accepted_material':False}
    return d,record

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--after-hvps',type=int,required=True)
    p.add_argument('--initial-direction',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--maximum',type=int,default=1200)
    p.add_argument('--tolerance',type=float,default=.01)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    co,model,gradient,low,high,u,source_record=restore(args.source.resolve(),args.after_hvps)
    initial=np.load(args.initial_direction)
    expected=json.loads(args.initial_direction.with_name('progress.json').read_text())['CPU_subproblem']
    diagonal=np.maximum(model.diagonal,1e-30)
    cold=projected_norm(np.zeros_like(initial),gradient,low,high,diagonal)
    relative=projected_norm(initial,gradient+model.apply(initial),low,high,diagonal)/max(cold,1e-300)
    if abs(relative-expected['projected_KKT_relative'])>1e-8:
        raise RuntimeError('Saved CPU subproblem residual did not reproduce; do not continue a different QP')
    atomic(args.out/'restored.json',{'source':str(args.source.resolve()),
        'after_HVPs':args.after_hvps,'reproduced_initial_relative_KKT':relative,
        'expected_initial_relative_KKT':expected['projected_KKT_relative'],
        'source_true_model_change':source_record['model_change'],
        'source_material_SHA256':hashlib.sha256((args.source/'old_material.npy').read_bytes()).hexdigest(),
        'truth_read':False,'GPU_used':False,'patient_wave_solves':0})
    def progress(row,d,save):
        atomic(args.out/'progress.json',row)
        if save:save_array(args.out/'direction_checkpoint.npy',d)
    d,record=solve_accurately(model,gradient,low,high,initial,
        maximum=args.maximum,tolerance=args.tolerance,callback=progress)
    save_array(args.out/'refined_increment.npy',d)
    save_array(args.out/'total_candidate_log_direction.npy',u+d)
    save_array(args.out/'candidate_material_unaccepted.npy',co.material(u+d))
    record['source_after_HVPs']=args.after_hvps
    record['source_true_model_change']=source_record['model_change']
    record['original_CPU_iterations']=expected['iterations']
    atomic(args.out/'result.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='history'}),flush=True)

if __name__=='__main__':main()
