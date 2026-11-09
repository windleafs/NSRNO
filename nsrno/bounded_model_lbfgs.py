"""Bounded L-BFGS for a possibly signed, first-order corrected coarse model."""
import numpy as np


def solve(oracle,initial,lower,upper,iterations=8,history=8,maximum_lines=8,callback=None,target=1e-3):
    """oracle(x)->{lower,upper,gradient,...}; every trial must be evaluated.

    Accepted candidate upper bounds must beat the previous lower bound.
    Directions use approximate supplied gradients; exact Armijo conditions
    and global convergence are not asserted. Exceptions leave the last
    completed callback point intact for the caller's physical checkpoint.
    """
    x=np.asarray(initial,np.float64).copy();lo=np.asarray(lower);hi=np.asarray(upper)
    if np.any(x<lo) or np.any(x>hi):raise ValueError('Initial point is outside physical bounds')
    current=oracle(x);pairs=[];initial_norm=None;trace=[]
    for accepted in range(iterations+1):
        g=current['gradient'];pg=x-np.clip(x-g,lo,hi);norm=float(np.linalg.norm(pg))
        if initial_norm is None:initial_norm=max(norm,1e-300)
        point={'accepted_step':accepted,'projected_gradient_relative':norm/initial_norm,
               'initial_projected_gradient_norm':initial_norm,'projected_gradient_norm':norm,
               'lower':current['lower'],'upper':current['upper'],'history_pairs':len(pairs)}
        if callback:callback(x,current,point)
        if target is not None and norm<=target*initial_norm:
            return x,current,{'stop':'projected_gradient_target','trace':trace,'point':point}
        if accepted==iterations:return x,current,{'stop':'accepted_step_budget','trace':trace,'point':point}
        active=((x<=lo)&(g>0))|((x>=hi)&(g<0))|(lo==hi)
        free_gradient=np.where(active,0.,g)
        # Full inverse-Hessian multiplication followed by clipping can feed a
        # nonzero bound-normal gradient into a nearly stationary free variable.
        # Restrict BOTH sides of every history pair to the current free set.
        free_pairs=[]
        for s,y,_ in pairs:
            sf=np.where(active,0.,s);yf=np.where(active,0.,y);curvature=float(np.dot(sf,yf))
            threshold=1e-10*max(float(np.linalg.norm(sf)*np.linalg.norm(yf)),1e-300)
            if curvature>threshold:free_pairs.append((sf,yf,1/curvature))
        q=free_gradient.copy();coefficients=[]
        for s,y,rho in reversed(free_pairs):
            coefficient=rho*float(np.dot(s,q));coefficients.append(coefficient);q-=coefficient*y
        gamma=1.
        if free_pairs:
            s,y,_=free_pairs[-1];gamma=float(np.clip(np.dot(s,y)/max(np.dot(y,y),1e-300),1e-4,1e4))
        inverse=gamma*q
        for (s,y,rho),coefficient in zip(free_pairs,reversed(coefficients)):
            inverse+=s*(coefficient-rho*float(np.dot(y,inverse)))
        direction=np.clip(x-np.where(active,0.,inverse),lo,hi)-x;slope=float(np.dot(g,direction));restarted=False
        if slope>=-1e-12*max(float(np.dot(free_gradient,free_gradient)),1e-300):
            direction=np.clip(x-g,lo,hi)-x;slope=float(np.dot(g,direction));pairs=[];restarted=True
        if not slope<0:return x,current,{'stop':'no_projected_descent_direction','trace':trace,'point':point}
        # Without a curvature history, use the known nonnegative objective
        # scale to avoid a linear model decrease larger than half its value.
        # This is a proposal only; physical objective intervals decide acceptance.
        alpha=min(1.,.5*max(float(current.get('line_search_objective_scale', current['lower'])),1e-12)/(-slope)) if not free_pairs else 1.
        initial_alpha=alpha
        lines=[];found=False
        for line in range(maximum_lines):
            trial_x=x+alpha*direction;trial=oracle(trial_x)
            required=current['lower']+1e-4*alpha*slope
            found=bool(trial['upper']<required and trial['upper']<current['lower'])
            denominator=2*(trial['upper']-current['lower']-alpha*slope)
            interpolated=-slope*alpha*alpha/denominator if denominator>0 and np.isfinite(denominator) else .5*alpha
            next_alpha=float(np.clip(interpolated,.1*alpha,.5*alpha))
            lines.append({'line':line,'alpha':alpha,'next_alpha_if_rejected':next_alpha,
                          'slope_from_approximate_gradient':slope,
                          'current_lower':current['lower'],'trial_upper':trial['upper'],
                          'required_upper':required,'accepted':found,
                          'exact_gradient_Armijo_not_claimed':True})
            if found:
                s=trial_x-x;y=trial['gradient']-g;curvature=float(np.dot(s,y))
                threshold=1e-10*max(float(np.linalg.norm(s)*np.linalg.norm(y)),1e-300)
                if curvature>threshold:
                    pairs.append((s.copy(),y.copy(),1/curvature));pairs=pairs[-history:]
                x=trial_x;current=trial;break
            alpha=next_alpha
        trace.append({'accepted_step_proposed':accepted+1,'L_BFGS_gamma':gamma,
                      'initial_alpha':initial_alpha,'objective_scaled_without_history':not bool(free_pairs),
                      'projected_gradient_restart':restarted,'free_history_pairs':len(free_pairs),
                      'active_variables':int(np.count_nonzero(active)),'candidates':lines})
        if not found:return x,current,{'stop':'no_certified_objective_decrease','trace':trace,'point':point}
    raise RuntimeError('Unreachable optimizer state')
