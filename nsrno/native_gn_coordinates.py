"""Independent native log-physical coordinates and bounded quadratic steps."""
import numpy as np
from .physical_box_direction import local_apply,PAIRS

SCALES=np.array([.05,.10,.70],np.float64)[:,None,None]

def dot(a,b):
    return float(np.einsum('ijk,ijk->',a,b,dtype=np.float64))

class Coordinates:
    def __init__(self,material,edge):
        self.m=np.asarray(material,np.float64)
        self.edge=np.broadcast_to(edge,self.m.shape[1:]).astype(np.float64)
        self.mask=np.broadcast_to(np.arange(self.m.shape[1])[:,None]>=80,self.m.shape[1:])
        self.tissue=self.m[2]-self.edge
        if np.any(self.tissue[self.mask]<=0):
            raise ValueError('Positive tissue attenuation is required')
        self.absolute=np.stack([np.log(self.m[0]/1540)/.05,
            np.log(self.m[0]*self.m[1]/1540e3)/.1,
            np.log(np.maximum(self.tissue,1e-30)/.5)/.7])

    def material(self,u):
        dc,dz,da=SCALES*u
        out=np.stack([self.m[0]*np.exp(dc),self.m[1]*np.exp(dz-dc),
            self.edge+self.tissue*np.exp(da)])
        out[:,~self.mask]=self.m[:,~self.mask]
        return out.astype(np.float32)

    def tangent(self,u):
        dc,dz,da=SCALES*u
        return np.stack([self.m[0]*dc,self.m[1]*(dz-dc),self.tissue*da])*self.mask

    def pullback(self,g):
        return np.stack([.05*(self.m[0]*g[0]-self.m[1]*g[1]),
            .1*self.m[1]*g[1],.7*self.tissue*g[2]])*self.mask

    def bounds(self,trust=(.02,.02,.12)):
        c,rho,_=self.m;Z=c*rho
        low=np.stack([np.log(1400/c),-.25-np.log(Z/1540e3),
            np.log(.5*np.exp(-5.6)/np.maximum(self.tissue,1e-30))])
        high=np.stack([np.log(1680/c),.25-np.log(Z/1540e3),
            np.log(.5*np.exp(1.4)/np.maximum(self.tissue,1e-30))])
        low=np.maximum(np.minimum(low,0),-np.array(trust)[:,None,None])/SCALES
        high=np.minimum(np.maximum(high,0),np.array(trust)[:,None,None])/SCALES
        low[:,~self.mask]=0;high[:,~self.mask]=0
        return low,high

    def regularization(self,u,weight=.001):
        phi=(self.absolute+u)[:,80:]
        dx=phi[:,:-1,1:]-phi[:,:-1,:-1]
        dz=phi[:,1:,:-1]-phi[:,:-1,:-1]
        root=np.sqrt(dx*dx+dz*dz+1e-6)
        value=weight*float(np.mean(root-.001))
        w=weight/root/root.size
        def divergence(dx,dz):
            out=np.zeros_like(self.m)
            body=out[:,80:]
            body[:,:-1,:-1]-=dx+dz
            body[:,:-1,1:]+=dx;body[:,1:,:-1]+=dz
            return out
        gradient=divergence(w*dx,w*dz)
        def metric(q):
            body=q[:,80:]
            return divergence(w*(body[:,:-1,1:]-body[:,:-1,:-1]),
                w*(body[:,1:,:-1]-body[:,:-1,:-1]))
        diagonal=np.zeros_like(self.m)
        body=diagonal[:,80:]
        body[:,:-1,:-1]+=2*w;body[:,:-1,1:]+=w;body[:,1:,:-1]+=w
        return value,gradient,metric,diagonal

def physical_blocks(primitive_blocks,material,edge):
    """Old conditional statistics are used only as a local preconditioner."""
    c,rho,total=np.asarray(material,np.float64)
    r=rho/1000;beta=(1540/c)**2/r;t=total*c/1540;nu=(total-edge)*c/1540
    zero=np.zeros_like(c)
    columns=[np.stack([-.05*r/.1,-.05*beta/.1,.05*t/.1]),
             np.stack([.1*r/.1,-.1*beta/.1,zero]),
             np.stack([zero,zero,.7*nu/.1])]
    images=[local_apply(primitive_blocks,x) for x in columns]
    blocks=np.stack([np.sum(columns[i]*images[j],axis=0,dtype=np.float64)
                     for i,j in PAIRS])
    for j in range(3):
        positive=blocks[j][blocks[j]>0]
        reference=max(float(np.median(positive)) if positive.size else 0,1e-24)
        blocks[j]=np.maximum(blocks[j],reference*1e-4)
    # A diagonal shift prevents near-null local modes from dominating the
    # preconditioner. This is not added to the physical data objective.
    blocks[:3]*=1.05
    return blocks

def inverse_blocks(blocks,rhs):
    diagonal=np.maximum(blocks[:3],1e-30)
    scale=1/np.sqrt(diagonal)
    a=blocks[3]*scale[0]*scale[1]
    b=blocks[4]*scale[0]*scale[2]
    c=blocks[5]*scale[1]*scale[2]
    det=1+2*a*b*c-a*a-b*b-c*c
    if np.any(det<=0):
        raise ValueError('Local preconditioner is not positive definite')
    y=rhs*scale
    return np.stack([(1-c*c)*y[0]+(b*c-a)*y[1]+(a*c-b)*y[2],
        (b*c-a)*y[0]+(1-b*b)*y[1]+(a*b-c)*y[2],
        (a*c-b)*y[0]+(a*b-c)*y[1]+(1-a*a)*y[2]])/det*scale

def bounded_direction(coordinates,gradient,hvp,blocks,maximum=8,tv_weight=.001,
                      damping_relative=.01,callback=None):
    """Truncated projected/conjugate GN; every segment uses a true data HVP.

    Box projection proposes a feasible segment *inside* the quadratic solve.
    Its true quadratic curvature is then computed. There is no clipping of a
    returned material direction. It is not claimed to solve the QP exactly.
    """
    shape=coordinates.m.shape
    zero=np.zeros(shape,np.float64)
    tv,g_tv,h_tv,tv_diag=coordinates.regularization(zero,tv_weight)
    g=gradient+g_tv
    low,high=coordinates.bounds()
    u=zero.copy();Hu=zero.copy();r=g.copy();history=[]
    data_scale=1.;damping=zero.copy()
    old_r=old_z=old_p=None
    initial_norm=None
    for iteration in range(maximum):
        pre=blocks*data_scale
        pre[:3]+=tv_diag+damping
        z=inverse_blocks(pre,r)*coordinates.mask
        kind='block_preconditioned_projected_gradient'
        if old_r is not None:
            beta=max(0.,dot(r,z-old_z)/max(dot(old_r,old_z),1e-300))
            beta=min(beta,10.)
            search=-z+beta*old_p
            kind='projected_conjugate_gradient'
        else:
            search=-z
        p=np.clip(u+search,low,high)-u
        slope=dot(r,p)
        if slope>=-1e-4*max(dot(r,z),0):
            p=np.clip(u-z,low,high)-u
            slope=dot(r,p);kind='restarted_block_projected_gradient'
        if slope>=0:
            p=np.clip(u-r/np.maximum(pre[:3],1e-30),low,high)-u
            slope=dot(r,p);kind='diagonal_feasible_descent'
        pg=u-np.clip(u-r/np.maximum(pre[:3],1e-30),low,high)
        norm=np.sqrt(max(dot(pg,pre[:3]*pg),0))
        if initial_norm is None:initial_norm=norm
        if norm<max(initial_norm*1e-3,1e-16) or slope>=0:
            break
        image_data=hvp(p,iteration)
        data_curvature=dot(p,image_data)
        if data_curvature < -1e-6*abs(slope):
            raise RuntimeError('Native data Gauss–Newton curvature is negative')
        if iteration==0:
            approximate=max(dot(p,local_apply(blocks,p)),1e-300)
            data_scale=max(data_curvature/approximate,1e-12)
            damping=damping_relative*data_scale*blocks[:3]
            damping[:,~coordinates.mask]=0
            calibrated=blocks*data_scale
            calibrated[:3]+=tv_diag+damping
            initial_pg=-np.clip(-g/np.maximum(calibrated[:3],1e-30),low,high)
            initial_norm=np.sqrt(max(dot(initial_pg,calibrated[:3]*initial_pg),0))
        image=image_data+h_tv(p)+damping*p
        curvature=dot(p,image)
        alpha=min(1.,-slope/max(curvature,1e-300))
        model_before=dot(g,u)+.5*dot(u,Hu)
        previous_r=r.copy()
        u+=alpha*p;Hu+=alpha*image;r=g+Hu
        model=dot(g,u)+.5*dot(u,Hu)
        if model>model_before+1e-10 or np.any(u<low-1e-10) or np.any(u>high+1e-10):
            raise RuntimeError('Bounded GN segment violated model decrease or feasibility')
        old_r,old_z,old_p=previous_r,z,p
        if iteration==0:
            # Calibration changed the preconditioner; restart conjugacy once.
            old_r=old_z=old_p=None
        row={'iteration':iteration+1,'kind':kind,'segment_alpha':float(alpha),
            'model_change':model,'segment_slope':slope,'segment_data_curvature':data_curvature,
            'segment_total_curvature':curvature,'projected_KKT_norm_before':norm,
            'local_data_preconditioner_scale':data_scale,'hvp_count':iteration+1}
        history.append(row)
        if callback:callback(row)
    pre=blocks*data_scale
    pre[:3]+=tv_diag+damping
    pg=u-np.clip(u-r/np.maximum(pre[:3],1e-30),low,high)
    final_norm=np.sqrt(max(dot(pg,pre[:3]*pg),0))
    report={'method':'truncated_bound_constrained_full_native_Gauss_Newton',
        'history':history,'hvp_count':len(history),'QP_exact_solution_claimed':False,
        'model_change':dot(g,u)+.5*dot(u,Hu),'linear_slope':dot(g,u),
        'total_quadratic':dot(u,Hu),'projected_KKT_norm':final_norm,
        'initial_projected_KKT_norm':initial_norm,'TV_before':tv,
        'damping_relative':damping_relative,'post_hoc_material_projection':False,
        'truth_read':False}
    return u,report
