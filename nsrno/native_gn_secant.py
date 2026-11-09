"""True-HVP constrained quadratic iteration with compact nonlocal BFGS."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.optimize import Bounds,minimize
from .native_gn_coordinates import dot
from .physical_box_direction import local_apply

def projected_norm(u,gradient,low,high,diagonal):
    p=u-np.clip(u-gradient/diagonal,low,high)
    return np.sqrt(max(dot(p,diagonal*p),0.))

class CompactBFGS:
    """An SPD base operator plus full-vector rank-two BFGS corrections."""
    def __init__(self,base,base_diagonal,mask):
        self.base=base;self.base_diagonal=base_diagonal.copy();self.mask=mask
        self.diagonal=base_diagonal.copy();self.pairs=[]

    def apply(self,x):
        out=self.base(x)
        for v,y,vs,ys in self.pairs:
            out+=(dot(y,x)/ys)*y-(dot(v,x)/vs)*v
        return out*self.mask

    def update(self,s,y):
        v=self.apply(s);vs=dot(s,v);ys=dot(s,y)
        if not np.isfinite(vs+ys) or vs<=0 or ys<=0:
            raise RuntimeError('Nonpositive compact-BFGS curvature; no update accepted')
        self.pairs.append((v.copy(),y.copy(),vs,ys))
        self.diagonal+=y*y/ys-v*v/vs
        negative=float(np.min(self.diagonal[:,self.mask]))
        tolerance=1e-10*max(float(np.max(self.base_diagonal[:,self.mask])),1e-30)
        if negative < -tolerance:
            raise RuntimeError('Compact BFGS diagonal lost positivity')
        self.diagonal=np.maximum(self.diagonal,self.base_diagonal*1e-12)
        self.diagonal[:,~self.mask]=1.
        residual=self.apply(s)-y
        defect=np.sqrt(max(dot(residual,residual),0.))
        norm=np.sqrt(max(dot(y,y),1e-300))
        return {'pairs':len(self.pairs),'sBs':vs,'sHy':ys,
            'last_secant_relative_error':defect/norm}

def solve_box(model,gradient,low,high,maximum=100,tolerance=.01):
    """Solve in the compact metric; all nonlocal rank terms enter the QP."""
    started=time.time();shape=gradient.shape
    diagonal=np.maximum(model.diagonal,1e-30)
    metric=np.sqrt(diagonal)
    initial=projected_norm(np.zeros(shape),gradient,low,high,diagonal)
    evaluations=0
    def value_gradient(y):
        nonlocal evaluations
        evaluations+=1
        d=y.reshape(shape)/metric
        image=model.apply(d)
        return dot(gradient,d)+.5*dot(d,image),((gradient+image)/metric).ravel()
    result=minimize(value_gradient,np.zeros(gradient.size),method='L-BFGS-B',jac=True,
        bounds=Bounds((low*metric).ravel(),(high*metric).ravel()),
        options={'maxiter':maximum,'maxcor':10,'maxls':30,'ftol':1e-13,
            'gtol':1e-11,'maxfun':maximum*5})
    d=result.x.reshape(shape)/metric
    image=model.apply(d)
    value=dot(gradient,d)+.5*dot(d,image)
    norm=projected_norm(d,gradient+image,low,high,diagonal)
    if value>1e-10 or np.any(d<low-1e-10) or np.any(d>high+1e-10):
        raise RuntimeError('Compact constrained QP violated decrease or feasibility')
    return d,{'solver':'CPU diagonally-scaled L-BFGS-B with compact nonlocal BFGS metric',
        'iterations':int(result.nit),'function_evaluations':evaluations,
        'message':str(result.message),'software_success':bool(result.success),
        'approximate_model_change':value,'projected_KKT_relative':norm/max(initial,1e-300),
        'QP_tolerance_met':bool(norm<=tolerance*max(initial,1e-300)),
        'seconds':time.time()-started,'GPU_used':False}

def bounded_direction(coordinates,gradient,hvp,blocks,maximum=8,tv_weight=.001,
                      damping_relative=.01,callback=None,checkpoint_folder=None,
                      cpu_maximum=100):
    """Use compact QP proposals but certify every segment with the actual H."""
    shape=coordinates.m.shape;mask=coordinates.mask
    zero=np.zeros(shape,np.float64)
    tv,g_tv,h_tv,tv_diag=coordinates.regularization(zero,tv_weight)
    g=gradient+g_tv;low,high=coordinates.bounds()
    u=zero.copy();Hu=zero.copy();r=g.copy();history=[]
    data_scale=1.;damping=zero.copy()
    def base_apply(x):
        return (data_scale*local_apply(blocks,x)+h_tv(x)+damping*x)*mask
    def base_diagonal():
        out=data_scale*blocks[:3]+tv_diag+damping
        out[:,~mask]=1.
        return np.maximum(out,1e-30)
    model=CompactBFGS(base_apply,base_diagonal(),mask)
    initial_norm=None;fixed_metric=base_diagonal()
    for iteration in range(maximum):
        current_norm=projected_norm(u,r,low,high,fixed_metric)
        if initial_norm is None:initial_norm=current_norm
        if current_norm<=max(initial_norm*1e-3,1e-16):break
        if callback:callback({'iteration':iteration+1,'phase':'CPU_compact_box_QP',
            'hvp_count':iteration,'model_change':dot(g,u)+.5*dot(u,Hu),
            'projected_KKT_norm':current_norm})
        d,cpu=solve_box(model,r,low-u,high-u,maximum=cpu_maximum)
        slope=dot(r,d)
        kind='compact_BFGS_box_QP'
        if slope>=0:
            d=np.clip(-r/model.diagonal,low-u,high-u)
            slope=dot(r,d);kind='diagonal_feasible_fallback'
        if slope>=0:break
        image_data=hvp(d,iteration)
        data_curvature=dot(d,image_data)
        if data_curvature < -1e-6*abs(slope):
            raise RuntimeError('Native Gauss–Newton data curvature is negative')
        if iteration==0:
            reference=max(dot(d,local_apply(blocks,d)),1e-300)
            data_scale=max(data_curvature/reference,1e-12)
            damping=damping_relative*data_scale*blocks[:3]
            damping[:,~mask]=0
            fixed_metric=base_diagonal()
            initial_norm=projected_norm(zero,g,low,high,fixed_metric)
            model=CompactBFGS(base_apply,base_diagonal(),mask)
        image=image_data+h_tv(d)+damping*d
        curvature=dot(d,image)
        alpha=min(1.,-slope/max(curvature,1e-300))
        old_model=dot(g,u)+.5*dot(u,Hu)
        u+=alpha*d;Hu+=alpha*image;r=g+Hu
        true_model=dot(g,u)+.5*dot(u,Hu)
        if true_model>old_model+1e-10 or np.any(u<low-1e-10) or np.any(u>high+1e-10):
            raise RuntimeError('True compact-GN segment violated decrease or feasibility')
        secant=model.update(d,image)
        final_norm=projected_norm(u,r,low,high,fixed_metric)
        total_quadratic=dot(u,Hu)
        tv_quadratic=dot(u,h_tv(u))
        damping_quadratic=dot(u,damping*u)
        row={'iteration':iteration+1,'kind':kind,'segment_alpha':float(alpha),
            'model_change':true_model,'segment_slope':slope,
            'segment_data_curvature':data_curvature,'segment_total_curvature':curvature,
            'local_data_preconditioner_scale':data_scale,'projected_KKT_norm':final_norm,
            'initial_projected_KKT_norm':initial_norm,
            'projected_KKT_relative':final_norm/max(initial_norm,1e-300),
            'cumulative_data_quadratic':total_quadratic-tv_quadratic-damping_quadratic,
            'cumulative_TV_quadratic':tv_quadratic,
            'cumulative_damping_quadratic':damping_quadratic,
            'hvp_count':iteration+1,'CPU_subproblem':cpu,'compact_secant':secant}
        history.append(row)
        if checkpoint_folder is not None:
            folder=Path(checkpoint_folder)/f'hvp{iteration:02d}'
            folder.mkdir(parents=True,exist_ok=False)
            np.save(folder/'direction.npy',d)
            np.save(folder/'data_image.npy',image_data)
            np.save(folder/'total_image.npy',image)
            np.save(folder/'cumulative_log_direction.npy',u)
            np.save(folder/'cumulative_total_image.npy',Hu)
            (folder/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
        if callback:callback(row)
    final_norm=projected_norm(u,r,low,high,fixed_metric)
    total_quadratic=dot(u,Hu)
    tv_quadratic=dot(u,h_tv(u))
    damping_quadratic=dot(u,damping*u)
    record={'method':'true_native_HVP_GN_with_compact_nonlocal_BFGS_box_QP',
        'history':history,'hvp_count':len(history),'QP_exact_solution_claimed':False,
        'model_change':dot(g,u)+.5*dot(u,Hu),'linear_slope':dot(g,u),
        'total_quadratic':total_quadratic,'projected_KKT_norm':final_norm,
        'data_linear_slope':dot(gradient,u),'TV_linear_slope':dot(g_tv,u),
        'data_quadratic':total_quadratic-tv_quadratic-damping_quadratic,
        'TV_quadratic':tv_quadratic,'damping_quadratic':damping_quadratic,
        'predicted_physical_data_TV_change':dot(g,u)+.5*(total_quadratic-damping_quadratic),
        'initial_projected_KKT_norm':initial_norm,
        'projected_KKT_relative':final_norm/max(initial_norm,1e-300),
        'TV_before':tv,'damping_relative':damping_relative,
        'post_hoc_material_projection':False,'cpu_maximum_per_proposal':cpu_maximum,
        'full_HVP_vectors_saved':checkpoint_folder is not None,'truth_read':False}
    return u,record
