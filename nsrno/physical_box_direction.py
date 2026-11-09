"""Physical log-coordinate box-constrained material quadratic.

Consumes existing receiver curvature and gradients. No target material,
RF simulation, pressure inverse, or GPU is used in this module.
"""
from pathlib import Path
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0) != 0:
    raise RuntimeError('Cannot disable process THP')
import argparse
import hashlib
import json
import time
import numpy as np
from .wri_material import MaterialVariables
from .coupled_coordinates import pcg,dot
from .wri_worker import atomic

PAIRS=((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))

def local_apply(blocks,q):
    a,b,c,ab,ac,bc=blocks
    return np.stack([a*q[0]+ab*q[1]+ac*q[2],
                     ab*q[0]+b*q[1]+bc*q[2],
                     ac*q[0]+bc*q[1]+c*q[2]])

class PhysicalBoxDirection:
    def __init__(self,variables,current,blocks,data_gradient,
                 scales=(.05,.10,.70),log_trust=(.02,.02,.12),
                 tv_weight=.001,damping_relative=.01):
        self.vars=variables
        self.current=np.asarray(current,np.float64)
        self.blocks=np.asarray(blocks,np.float64)
        self.data_gradient=np.asarray(data_gradient,np.float64)
        self.unknown=variables.unknown
        self.scales=np.asarray(scales,np.float64)
        r,beta,t=self.current
        c=1540/np.sqrt(r*beta)
        total=t*np.sqrt(r*beta)
        tissue=total-variables.edge
        if np.any(tissue[self.unknown]<=0):
            raise RuntimeError('Positive independent tissue attenuation is required')
        self.r,self.beta,self.t=r,beta,t
        self.nu=t-variables.edge*c/1540
        sc,sz,sa=self.scales
        self.cr=-sc*r/.1
        self.cb=-sc*beta/.1
        self.ct=sc*t/.1
        self.zr=sz*r/.1
        self.zb=-sz*beta/.1
        self.at=sa*self.nu/.1
        self.TV,self.tv_gradient,self.htv,tvlocal=variables.tv(self.current,tv_weight)
        self.gradient=self.transpose_B(self.data_gradient+self.tv_gradient)
        self.physical_blocks=self.transform_blocks(self.blocks)
        tvphysical=self.transform_blocks(tvlocal)
        diagonal_data=np.maximum(self.physical_blocks[:3],0)
        references=np.array([max(float(np.median(v[self.unknown][v[self.unknown]>0]))
            if np.any(v[self.unknown]>0) else 0.,1e-24) for v in diagonal_data])
        self.damping=damping_relative*np.maximum(diagonal_data,references[:,None,None]*1e-4)
        self.damping[:,~self.unknown]=1.
        self.local_blocks=self.physical_blocks+tvphysical
        self.local_blocks[:3]+=self.damping
        self.diagonal=np.maximum(self.local_blocks[:3],1e-30)
        impedance=1000*r*c
        low=np.stack([np.log(1400/c),-.25-np.log(impedance/1540e3),
                      np.log(variables.alpha_min/np.maximum(tissue,1e-30))])
        high=np.stack([np.log(1680/c),.25-np.log(impedance/1540e3),
                       np.log(variables.alpha_max/np.maximum(tissue,1e-30))])
        # Include the exact saved current state despite sub-ulp physical-bound rounding.
        low=np.minimum(low,0)
        high=np.maximum(high,0)
        radius=np.asarray(log_trust)[:,None,None]
        self.low=np.maximum(low,-radius)/self.scales[:,None,None]
        self.high=np.minimum(high,radius)/self.scales[:,None,None]
        self.low[:,~self.unknown]=0
        self.high[:,~self.unknown]=0
        self.log_trust=list(log_trust)
        self.references=references
        scales_pre=1/np.sqrt(self.diagonal)
        cross=self.local_blocks[3:]
        a=cross[0]*scales_pre[0]*scales_pre[1]/1.00001
        b=cross[1]*scales_pre[0]*scales_pre[2]/1.00001
        c0=cross[2]*scales_pre[1]*scales_pre[2]/1.00001
        det=np.maximum(1+2*a*b*c0-a*a-b*b-c0*c0,1e-12)
        self.pre_scales=scales_pre
        self.pre_inverse=np.stack([1-c0*c0,1-b*b,1-a*a,b*c0-a,a*c0-b,a*b-c0])/det

    def B(self,u):
        return np.stack([self.cr*u[0]+self.zr*u[1],
            self.cb*u[0]+self.zb*u[1],self.ct*u[0]+self.at*u[2]])*self.unknown

    def transpose_B(self,q):
        return np.stack([self.cr*q[0]+self.cb*q[1]+self.ct*q[2],
            self.zr*q[0]+self.zb*q[1],self.at*q[2]])*self.unknown

    def transform_blocks(self,blocks):
        columns=[np.stack([self.cr,self.cb,self.ct]),
                 np.stack([self.zr,self.zb,np.zeros_like(self.zr)]),
                 np.stack([np.zeros_like(self.at),np.zeros_like(self.at),self.at])]
        images=[local_apply(blocks,x) for x in columns]
        return np.stack([np.sum(columns[i]*images[j],axis=0,dtype=np.float64)
                         for i,j in PAIRS])

    def apply(self,u):
        return (local_apply(self.physical_blocks,u)+
            self.transpose_B(self.htv(self.B(u)))+self.damping*u)*self.unknown

    def precondition(self,q):
        y=q*self.pre_scales
        a,b,c,ab,ac,bc=self.pre_inverse
        return np.stack([a*y[0]+ab*y[1]+ac*y[2],
                         ab*y[0]+b*y[1]+bc*y[2],
                         ac*y[0]+bc*y[1]+c*y[2]])*self.pre_scales

    def to_variables(self,u):
        dc,dz,da=self.scales[:,None,None]*u
        proposed=np.stack([self.r*np.exp(dz-dc),self.beta*np.exp(-dz-dc),
                           np.exp(dc)*(self.t+self.nu*np.expm1(da))])
        proposed[:,~self.unknown]=self.current[:,~self.unknown]
        return proposed

    def objective(self,u):
        return dot(self.gradient,u)+.5*dot(u,self.apply(u))

    def projected_norm(self,u,g):
        pg=u-np.clip(u-g/self.diagonal,self.low,self.high)
        return np.sqrt(max(dot(pg,self.diagonal*pg),0.))

    def solve(self,maximum=400,linear_maximum=64,callback=None):
        # Generalized Cauchy points handle many active box faces together.
        # The first custom active-set version stalled on global first-hit lengths.
        from scipy.optimize import minimize,Bounds
        started=time.time()
        history=[]
        initial=self.projected_norm(np.zeros_like(self.current),self.gradient)
        metric=np.sqrt(self.diagonal)
        low=(self.low*metric).ravel()
        high=(self.high*metric).ravel()
        def value_gradient(y):
            u=y.reshape(self.current.shape)/metric
            image=self.apply(u)
            value=dot(self.gradient,u)+.5*dot(u,image)
            grad=(self.gradient+image)/metric
            return float(value),grad.ravel()
        def report(y):
            u=y.reshape(self.current.shape)/metric
            gradient=self.gradient+self.apply(u)
            row={'iteration':len(history)+1,'model_change':float(self.objective(u)),
                'projected_KKT_relative':float(self.projected_norm(u,gradient)/max(initial,1e-300))}
            history.append(row)
            if callback:
                callback(dict(row,phase='bounded_quadratic_iteration'))
        result=minimize(value_gradient,np.zeros(self.current.size,np.float64),
            method='L-BFGS-B',jac=True,bounds=Bounds(low,high),callback=report,
            options={'maxiter':maximum,'maxcor':12,'maxls':30,'ftol':1e-13,
                     'gtol':1e-10,'maxfun':maximum*5})
        u=result.x.reshape(self.current.shape)/metric
        value=float(result.fun)
        proposed=self.to_variables(u)
        q=self.B(u)
        data_curvature=dot(q,local_apply(self.blocks,q))
        tv_curvature=dot(q,self.htv(q))
        damping_curvature=dot(u,self.damping*u)
        model=dot(self.data_gradient+self.tv_gradient,q)+.5*(data_curvature+tv_curvature+damping_curvature)
        if model>1e-12 or np.any(u<self.low-1e-12) or np.any(u>self.high+1e-12):
            raise RuntimeError('Physical box subproblem failed its model/feasibility invariant')
        zero_exact=bool(np.array_equal(self.to_variables(np.zeros_like(u)),self.current))
        change=self.vars.material(proposed).astype(np.float64)-self.vars.material(self.current).astype(np.float64)
        record={'direction_kind':'physical_log_box_QP','history':history,
            'quadratic_solver':'diagonally_scaled_L-BFGS-B','solver_message':str(result.message),
            'solver_success':bool(result.success),'iterations':int(result.nit),
            'model_change':float(model),'data_linear_slope':dot(self.data_gradient,q),
            'TV_linear_slope':dot(self.tv_gradient,q),'data_quadratic':float(data_curvature),
            'TV_quadratic':float(tv_curvature),'damping_quadratic':float(damping_curvature),
            'projected_KKT_relative':float(self.projected_norm(u,self.gradient+self.apply(u))/max(initial,1e-300)),
            'log_trust':self.log_trust,'scales_logc_logZ_logalpha':self.scales.tolist(),
            'zero_mapping_exact':zero_exact,'post_hoc_physical_projection':False,
            'physical_RMS_change_unknown_c_rho_alpha':np.sqrt(np.mean(change[:,self.unknown]**2,axis=1)).tolist(),
            'linear_primitive_RMS':np.sqrt(np.mean(q[:,self.unknown]**2,axis=1)).tolist(),
            'seconds':time.time()-started,'truth_read':False,'patient_wave_solves':0,
            'GPU_used':False,'imaging_success_claimed':False}
        return proposed,u,q,record

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    folder=args.source.resolve()
    root=folder.parent
    variables=MaterialVariables(np.load(root/'initial/material.npy'),np.load(root/'initial/parameters.npy'))
    current=np.load(folder/'old_variables.npy')
    with np.load(folder/'joint_normal.npz') as data:
        blocks=data['blocks']
        gradient=data['gradient']
    args.out.mkdir(parents=True,exist_ok=False)
    system=PhysicalBoxDirection(variables,current,blocks,gradient)
    proposed,u,q,record=system.solve(callback=lambda row:atomic(args.out/'progress.json',row))
    np.save(args.out/'variables.npy',proposed)
    np.save(args.out/'material.npy',variables.material(proposed))
    np.save(args.out/'parameters.npy',variables.parameters(proposed))
    np.save(args.out/'log_step.npy',u)
    np.save(args.out/'linear_primitive_increment.npy',q)
    old=np.load(folder/'candidates/line000/variables.npy')
    oq=(old-current)/.1
    record['old_projected_model_lower_bound']=dot(gradient+system.tv_gradient,oq)+.5*dot(oq,local_apply(blocks,oq))
    record['source_statistics']=str(folder/'joint_normal.npz')
    record['source_statistics_SHA256']=hashlib.sha256((folder/'joint_normal.npz').read_bytes()).hexdigest()
    atomic(args.out/'direction.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='history'}),flush=True)

if __name__=='__main__':
    main()
