"""Native independent material variables and sparse conditional GN updates.

All operations are CPU algebra on fixed-field sufficient statistics. There is
no material truth interface and no wave propagation in the model step.
"""
from pathlib import Path
import numpy as np
from .coupled_coordinates import pcg,dot


def inverse_modulus(frequency,spec,t):
    D=np.exp(2j*np.pi*frequency*spec['native_dt_s']);q=1/D
    e=sum(spec['eta_per_unit_alpha']);A=(1+.5*e*t)**2
    k=sum(v/(1+tau*(D-1)/spec['native_dt_s'])
          for v,tau in zip(spec['eta_per_unit_alpha'],spec['taus_s']))
    h=k*t/A
    return (1-(1-q)*h)/(A*(1-h))


class MaterialVariables:
    def __init__(self,initial_material,initial_parameters):
        self.seed=np.asarray(initial_material,np.float64).copy()
        self.seed_parameters=np.asarray(initial_parameters,np.float64).copy()
        self.shape=self.seed.shape[1:];self.size=int(np.prod(self.shape))
        edge=np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy').astype(np.float64)
        self.edge=np.broadcast_to(edge,self.shape).copy()
        self.unknown=np.broadcast_to(np.arange(self.shape[0])[:,None]>=80,self.shape)
        self.alpha_min=.5*np.exp(-5.6);self.alpha_max=.5*np.exp(1.4)
        self.initial=self.from_material(self.seed)

    @staticmethod
    def from_material(m):
        c,rho,alpha=m
        r=rho/1000.
        return np.stack([r,(1540/c)**2/r,alpha*c/1540])

    def material(self,v):
        r,beta,t=v
        c=1540/np.sqrt(r*beta)
        m=np.stack([c,1000*r,t*1540/c])
        m[:,:80]=self.seed[:,:80]
        return m.astype(np.float32)

    def project(self,v):
        r,beta,t=np.asarray(v,np.float64)
        c=np.clip(1540/np.sqrt(np.maximum(r,1e-8)*np.maximum(beta,1e-8)),1400,1680)
        impedance=np.clip(1000*np.maximum(r,1e-8)*c,1540e3*np.exp(-.25),1540e3*np.exp(.25))
        rho=impedance/c
        alpha=np.clip(t*1540/c-self.edge,self.alpha_min,self.alpha_max)+self.edge
        m=np.stack([c,rho,alpha]);m[:,:80]=self.seed[:,:80]
        return self.from_material(m)

    def parameters(self,v):
        m=self.material(v).astype(np.float64);c,rho,alpha=m
        q=np.stack([(c-1540)/80,np.log(c*rho/1540e3)/.1,
                    np.log(np.maximum(alpha-self.edge,self.alpha_min)/.5)/.7])
        q[:,:80]=self.seed_parameters[:,:80]
        return q.astype(np.float32)

    @staticmethod
    def divergence(dx,dz,shape):
        out=np.zeros(shape,np.float64)
        out[:,:-1,:-1]-=dx+dz;out[:,:-1,1:]+=dx;out[:,1:,:-1]+=dz
        return out

    def tv(self,v,weight):
        r,beta,t=v[:,80:]
        total=t*np.sqrt(r*beta)
        tissue=np.maximum(total-self.edge[80:],self.alpha_min)
        phi=np.stack([(r-1)/.1,(beta-1)/.1,np.log(tissue/.5)/.7])
        dx=phi[:,:-1,1:]-phi[:,:-1,:-1];dz=phi[:,1:,:-1]-phi[:,:-1,:-1]
        root=np.sqrt(dx*dx+dz*dz+1e-6)
        value=weight*float(np.mean(root-.001));w=weight/root/root.size
        jr=.1*total/(2*r*.7*tissue)
        jb=.1*total/(2*beta*.7*tissue)
        jt=.1*np.sqrt(r*beta)/(.7*tissue)
        def transpose(y):
            out=np.zeros((3,*self.shape),np.float64)
            out[:,80:]=np.stack([y[0]+jr*y[2],y[1]+jb*y[2],jt*y[2]])
            return out
        gradient=transpose(self.divergence(w*dx,w*dz,phi.shape))
        def metric(q):
            a=q[:,80:]
            u=np.stack([a[0],a[1],jr*a[0]+jb*a[1]+jt*a[2]])
            dx=u[:,:-1,1:]-u[:,:-1,:-1];dz=u[:,1:,:-1]-u[:,:-1,:-1]
            return transpose(self.divergence(w*dx,w*dz,u.shape))
        diagonal=np.zeros_like(phi)
        diagonal[:,:-1,:-1]+=2*w;diagonal[:,:-1,1:]+=w;diagonal[:,1:,:-1]+=w
        local=np.zeros((6,*self.shape),np.float64)
        local[:,80:]=np.stack([diagonal[0]+jr*jr*diagonal[2],
            diagonal[1]+jb*jb*diagonal[2],jt*jt*diagonal[2],
            jr*jb*diagonal[2],jr*jt*diagonal[2],jb*jt*diagonal[2]])
        return value,gradient,metric,local


class MaterialNormal:
    def __init__(self,variables,geometry_record,normal):
        self.vars=variables;self.normal=normal
        h,w=variables.shape;H,W=geometry_record['padded_shape'];pz,px=geometry_record['offset_zx']
        z=np.clip(np.arange(H)-pz,24,h-25)
        x=np.clip(np.arange(W)-px,24,w-25)
        self.index=(z[:,None]*w+x[None,:]).astype(np.int64)
        self.ix=np.roll(self.index,-1,axis=1);self.iz=np.roll(self.index,-1,axis=0)
        self.rhs=np.stack([normal['br'],normal['bb'],normal['bt']])
        ax,az=normal['arx'],normal['arz']
        diag=self.fold(ax,self.index)+self.fold(ax,self.ix)+self.fold(az,self.index)+self.fold(az,self.iz)
        diag+=self.fold(2*ax*(self.index==self.ix),self.index)
        diag+=self.fold(2*az*(self.index==self.iz),self.index)
        self.diagonal=np.stack([diag,normal['hbb'],normal['htt']])

    def fold(self,value,index=None):
        index=self.index if index is None else index
        return np.bincount(index.ravel(),weights=np.asarray(value).ravel(),
                           minlength=self.vars.size).reshape(self.vars.shape)

    def apply(self,q):
        r=q[0].ravel();ax,az=self.normal['arx'],self.normal['arz']
        dx=ax*(r[self.index]+r[self.ix]);dz=az*(r[self.index]+r[self.iz])
        hr=self.fold(dx,self.index)+self.fold(dx,self.ix)+self.fold(dz,self.index)+self.fold(dz,self.iz)
        out=np.stack([hr,self.normal['hbb']*q[1]+self.normal['hbt']*q[2],
                      self.normal['hbt']*q[1]+self.normal['htt']*q[2]])
        out[:,:80]=0
        return out

    def data_change_density(self,old,new):
        q=np.zeros_like(old);q[0]=(new[0]-old[0])/.1
        return -dot(self.rhs[0],q[0])+.5*dot(q,self.apply(q))

    def pressure_change(self,old,new,moments):
        change=0.
        for item in moments:
            with np.load(item['path']) as data:
                sxx=data['spp'].astype(np.float64);sxe=data['spe'].astype(np.complex128)
            dm=new[1]*inverse_modulus(item['frequency_hz'],item['spec'],new[2])-old[1]*inverse_modulus(item['frequency_hz'],item['spec'],old[2])
            change+=float(np.sum(2*(dm.conj()*sxe).real+abs(dm)**2*sxx,dtype=np.float64))
        return change

    def solve(self,current,moments,tv_weight=.001,damping_relative=.01,maximum=64,
              trust=(.03,.04,.10),callback=None):
        tv,g_tv,h_tv,tv_local=self.vars.tv(current,tv_weight)
        gradient=-self.rhs+g_tv;gradient[:,:80]=0
        data_diag=np.maximum(self.diagonal,0)
        reference=[]
        for a in data_diag:
            body=a[80:];positive=body[body>0]
            reference.append(max(float(np.median(positive)) if positive.size else 0.,1e-18))
        damping=damping_relative*np.maximum(data_diag,np.array(reference)[:,None,None]*1e-4)
        damping[:,:80]=1.
        diagonal=data_diag+tv_local[:3]+damping
        diagonal=np.maximum(diagonal,1e-30)
        scales=1/np.sqrt(diagonal)
        # Positive local 3x3 block approximation, including physical-alpha TV.
        a=tv_local[3]*scales[0]*scales[1]/1.00001
        b=tv_local[4]*scales[0]*scales[2]/1.00001
        c=(self.normal['hbt']+tv_local[5])*scales[1]*scales[2]/1.00001
        determinant=np.maximum(1+2*a*b*c-a*a-b*b-c*c,1e-12)
        inv=np.stack([1-c*c,1-b*b,1-a*a,b*c-a,a*c-b,a*b-c])/determinant
        def precondition(rhs):
            y=rhs*scales
            return np.stack([inv[0]*y[0]+inv[3]*y[1]+inv[4]*y[2],
                             inv[3]*y[0]+inv[1]*y[1]+inv[5]*y[2],
                             inv[4]*y[0]+inv[5]*y[1]+inv[2]*y[2]])*scales
        def operator(q):return self.apply(q)+h_tv(q)+damping*q
        direction,history=pcg(operator,-gradient,precondition,maximum=maximum,tolerance=.01,callback=callback)
        delta=.1*direction
        def limit(value):
            rms=np.sqrt(np.mean(value[:,80:]**2,axis=(-2,-1)))
            return value*np.minimum(1,np.array(trust)/np.maximum(rms,1e-30))[:,None,None]
        delta=limit(delta);kind='conditional_GN'
        if dot(gradient,delta/.1)>=0:
            delta=limit(-.1*gradient/diagonal);kind='diagonal_descent_fallback'
        candidates=[];accepted=current.copy()
        for line in range(9):
            alpha=.5**line
            proposal=self.vars.project(current+alpha*delta)
            slope=dot(gradient,(proposal-current)/.1)
            if slope>=0:
                candidates.append({'alpha':alpha,'slope':slope,'accepted':False});continue
            density=self.data_change_density(current,proposal)
            pressure=self.pressure_change(current,proposal,moments)
            proposed_tv=self.vars.tv(proposal,tv_weight)[0]
            change=density+pressure+proposed_tv-tv
            ok=change<=1e-4*slope
            row={'alpha':alpha,'density_cost_change':density,'pressure_cost_change':pressure,
                 'TV_change':proposed_tv-tv,'fixed_field_objective_change':change,
                 'slope':slope,'accepted':bool(ok)}
            candidates.append(row)
            if ok:accepted=proposal;break
        return accepted,{'direction_kind':kind,'CG_history':history,'line_search':candidates,
                         'accepted':bool(candidates and candidates[-1]['accepted']),
                         'update_RMS_r_beta_eta':np.sqrt(np.mean((accepted-current)[:,80:]**2,axis=(-2,-1))).tolist(),
                         'gradient_norm_r_beta_eta_coordinates':np.sqrt(np.sum(gradient**2,axis=(-2,-1))).tolist(),
                         'TV_before':tv,'damping_reference_per_parameter':reference,
                         'truth_read':False,'propagation_in_material_step':False}
