"""Limited-memory BFGS with a value-filtered strong-Wolfe line search.

Material bounds must be enforced by an invertible physical parameterization.
Steps failing the Armijo value test require no wavefield history or adjoint.
Promising steps get a gradient for the curvature condition; strong Wolfe avoids
the very short steps that plain backtracking can retain on curved valleys.
"""
from types import SimpleNamespace
import numpy as np
from scipy.optimize import line_search

def minimize(evaluate,x0,callback=None,maxiter=20,maxcor=8,maxls=12,ftol=1e-9,gtol=1e-7,initial_step=.25):
    x=np.asarray(x0,np.float64).copy();f,g=evaluate(x,True)
    history=[];nfev=1;njev=1;iterations=0;success=False;message='Iteration limit reached'
    for iteration in range(maxiter):
        if np.max(np.abs(g))<=gtol:
            success=True;message='Gradient tolerance reached';break
        residual=g.copy();alpha=[]
        for s,y,rho in reversed(history):
            a=rho*np.dot(s,residual);alpha.append(a);residual-=a*y
        if history:
            s,y,_=history[-1];gamma=np.clip(np.dot(s,y)/np.dot(y,y),1e-8,1e8)
        else:gamma=1.
        direction=gamma*residual
        for (s,y,rho),a in zip(history,reversed(alpha)):
            direction+=s*(a-rho*np.dot(y,direction))
        direction=-direction;slope=float(np.dot(g,direction))
        if not np.isfinite(slope) or slope>=0:
            history=[];direction=-g;slope=-float(np.dot(g,g))
        if iteration==0:direction=initial_step*direction;slope=initial_step*slope
        cache=[{'x':x.copy(),'f':f,'g':g}]
        def lookup(point,gradient):
            nonlocal nfev,njev
            entry=next((e for e in cache if np.array_equal(e['x'],point)),None)
            if entry is not None and (not gradient or entry['g'] is not None):return entry
            value,derivative=evaluate(point,gradient);nfev+=1;njev+=int(gradient)
            if entry is None:
                entry={'x':point.copy(),'f':value,'g':derivative};cache.append(entry)
            else:entry.update(f=value,g=derivative)
            return entry
        step,_,_,_,_,derivative=line_search(lambda point:lookup(point,False)['f'],lambda point:lookup(point,True)['g'],
            x,direction,gfk=g,old_fval=f,c1=1e-4,c2=.9,maxiter=maxls)
        if step is None or derivative is None:
            norm=float(np.dot(direction,direction))
            candidates=[]
            for entry in cache[1:]:
                trial_step=float(np.dot(entry['x']-x,direction)/norm)
                if np.isfinite(entry['f']) and entry['f']<=f+1e-4*trial_step*slope:candidates.append(entry)
            if not candidates:
                message='Value-filtered line search found no accepted step';break
            chosen=min(candidates,key=lambda e:e['f']);trial=chosen['x'];history=[]
        else:trial=x+step*direction
        entry=lookup(trial,True);next_f,next_g=entry['f'],entry['g']
        if not np.isfinite(next_f) or not np.all(np.isfinite(next_g)):
            message='Non-finite accepted-point gradient';break
        s=trial-x;y=next_g-g;curvature=float(np.dot(s,y))
        if curvature>1e-10*np.linalg.norm(s)*np.linalg.norm(y):
            history.append((s.copy(),y.copy(),1/curvature));history=history[-maxcor:]
        old_f=f;x,f,g=trial,next_f,next_g;iterations+=1
        if callback:callback(x)
        if iterations>=3 and abs(old_f-f)<=ftol*max(1.,abs(old_f),abs(f)):
            success=True;message='Objective tolerance reached';break
    return SimpleNamespace(x=x,fun=f,jac=g,nit=iterations,nfev=nfev,njev=njev,success=success,message=message)
