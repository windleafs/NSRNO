"""Small CPU implementation check against an independent dense-QP oracle."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.optimize import minimize
from .native_gn_coordinates import Coordinates
from .native_gn_secant import CompactBFGS,bounded_direction

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();started=time.time();rng=np.random.default_rng(704)
    shape=(3,82,3);m=np.empty(shape)
    m[0]=1540;m[1]=1000;m[2]=.5
    co=Coordinates(m,np.zeros(shape[1:]))
    active=np.broadcast_to(co.mask,shape)
    n=int(active.sum())
    def pack(x):
        out=np.zeros(shape);out[active]=x;return out
    low,high=co.bounds();lo=low[active];hi=high[active]
    U=rng.normal(size=(n,6))
    A=np.diag(np.linspace(.5,2.,n))+U@U.T/n*2
    g=rng.normal(size=n)*.3
    dense=lambda x:pack(A@x[active])
    model=CompactBFGS(lambda x:x*co.mask,np.ones(shape),co.mask)
    defects=[];minimum_eigenvalues=[]
    basis=[pack(np.eye(n)[:,j]) for j in range(n)]
    for _ in range(3):
        s=pack(rng.normal(size=n));report=model.update(s,dense(s))
        defects.append(report['last_secant_relative_error'])
        matrix=np.column_stack([model.apply(e)[active] for e in basis])
        assert np.max(abs(matrix-matrix.T))<1e-12
        minimum_eigenvalues.append(float(np.linalg.eigvalsh(matrix).min()))
    blocks=np.zeros((6,*shape[1:]));blocks[:3]=1.
    calls=[]
    def hvp(x,index):
        calls.append(index);return dense(x)
    u,record=bounded_direction(co,pack(g),hvp,blocks,maximum=24,
        tv_weight=0.,damping_relative=.01,cpu_maximum=100)
    calibration=record['history'][0]['local_data_preconditioner_scale']
    H=A+.01*calibration*np.eye(n)
    oracle=minimize(lambda x:(float(g@x+.5*x@H@x),g+H@x),np.zeros(n),
        method='L-BFGS-B',jac=True,bounds=list(zip(lo,hi)),
        options={'maxiter':2000,'ftol':1e-15,'gtol':1e-12,'maxls':40})
    value=float(g@u[active]+.5*u[active]@H@u[active])
    relative_gap=(value-float(oracle.fun))/max(abs(float(oracle.fun)),1e-30)
    assert abs(value-record['model_change'])<1e-10
    assert np.all(u>=low-1e-12) and np.all(u<=high+1e-12)
    changes=[0.]+[r['model_change'] for r in record['history']]
    assert max(np.diff(changes))<=1e-10
    report={'kind':'new_CPU_compact_quadratic_solver_implementation_check',
        'patient_wave_solves':0,'GPU_used':False,'unknown_dimension':n,
        'compact_last_secant_relative_errors':defects,
        'compact_minimum_eigenvalues':minimum_eigenvalues,
        'true_model_change':value,'independent_dense_oracle_model_change':float(oracle.fun),
        'relative_objective_gap':relative_gap,'projected_KKT_relative':record['projected_KKT_relative'],
        'synthetic_HVP_count':len(calls),'seconds':time.time()-started,
        'passed':max(defects)<1e-10 and min(minimum_eigenvalues)>0 and relative_gap<3e-3}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('New compact quadratic solver failed its CPU oracle check')

if __name__=='__main__':main()
