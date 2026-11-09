"""Feasible active-set solution of a small QP with many physical box rows."""
import numpy as np


def solve(physical_basis,linear,low,high,max_iterations=2000,tolerance=1e-9,callback=None):
    """Minimize .5*||y||^2 + linear@y, low <= physical_basis.T@y <= high.

    The caller whitens the actual restricted Hessian. All physical constraints
    are considered on every step; only an independent active set is factored.
    A small active-set residual does not certify full-space stationarity.
    """
    basis=np.asarray(physical_basis,np.float64);linear=np.asarray(linear,np.float64)
    low=np.asarray(low,np.float64).reshape(-1);high=np.asarray(high,np.float64).reshape(-1)
    if basis.shape!=(linear.size,low.size) or high.shape!=low.shape:raise ValueError('Wrong dimensions')
    if np.any(low>0) or np.any(high<0):raise ValueError('Zero must be feasible')
    y=np.zeros_like(linear);active=[];active_rows=[];history=[]
    reference=max(np.linalg.norm(linear),1e-20)
    for iteration in range(max_iterations):
        gradient=y+linear
        if active:
            rows=np.stack(active_rows)
            # SVD is cheap in the at-most-rank-dimensional working set.
            _,singular,right=np.linalg.svd(rows,full_matrices=False)
            if singular[-1]<1e-11*singular[0]:raise RuntimeError('Dependent active constraints')
            projection=right.T@(right@gradient)
            direction=-gradient+projection
            multipliers=np.linalg.lstsq(rows.T,-gradient-direction,rcond=1e-12)[0]
        else:
            direction=-gradient;multipliers=np.empty(0)
        direction_norm=np.linalg.norm(direction)
        if direction_norm<=tolerance*reference:
            if not active or np.min(multipliers)>=-tolerance*reference:
                physical=basis.T@y
                violation=max(float(np.max(low-physical)),float(np.max(physical-high)),0.)
                if violation>1e-8:raise RuntimeError('Final physical feasibility failed')
                report={'status':'restricted_QP_solved','iterations':iteration+1,'active_constraints':len(active),
                    'restricted_stationarity_relative':float(direction_norm/reference),
                    'minimum_normalized_multiplier':float(np.min(multipliers)) if active else None,
                    'maximum_physical_bound_violation':violation,
                    'objective':float(.5*y@y+linear@y),'history':history,
                    'full_space_optimality_claimed':False}
                return y,report
            remove=int(np.argmin(multipliers));active.pop(remove);active_rows.pop(remove)
            continue
        physical=basis.T@y;change=basis.T@direction
        ratios=np.full_like(low,np.inf);positive=change>1e-14;negative=change< -1e-14
        ratios[positive]=(high[positive]-physical[positive])/change[positive]
        ratios[negative]=(low[negative]-physical[negative])/change[negative]
        # Exact tangency makes active ratios undefined; discard floating error.
        for index,_ in active:ratios[index]=np.inf
        blocking=int(np.argmin(ratios));step=min(1.,float(ratios[blocking]))
        if step< -1e-8:raise RuntimeError('Current iterate lost physical feasibility')
        step=max(0.,step);y+=step*direction
        if step<1.-1e-12:
            sign=1. if change[blocking]>0 else -1.
            row=sign*basis[:,blocking];norm=np.linalg.norm(row)
            if norm<=1e-20:raise RuntimeError('Zero blocking constraint')
            active.append((blocking,sign));active_rows.append(row/norm)
        item={'iteration':iteration+1,'step':step,'objective':float(.5*y@y+linear@y),
            'active_constraints':len(active),'direction_norm':float(direction_norm)}
        history.append(item)
        if callback is not None:callback(item)
    raise RuntimeError('Restricted QP did not satisfy its fixed iteration budget')
