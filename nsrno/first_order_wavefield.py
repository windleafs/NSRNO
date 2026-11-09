"""First-order form of the existing conditioned native frequency equation.

The three fields are pressure and Z0 times the two particle velocities. The
Schur complement is the existing pressure operator, including its split-PML
update order and two discrete relaxation terms. This is not a ray model.
"""
import torch


class FirstOrderWavefield:
    def __init__(self,system):
        self.system=system;self.g=system.g;s=system;g=self.g
        self.S0=(s.D-1)/(g.dt*1540.)
        self.left_x=s.mx/(s.D-g.arx)
        self.right_x=1+(1-s.mx)/(s.D-g.arx)
        self.left_z=s.mz/(s.D-g.arz)
        self.right_z=1+(1-s.mz)/(s.D-g.arz)
        self.Ix=s.ax*s.svx
        self.Iz=s.az*s.svz
        # Outside the native domain all fields are explicitly fixed to zero.
        self.Rx=torch.where(g.mask>0,1/(self.Ix+1e-30),torch.ones_like(self.Ix))
        self.Rz=torch.where(g.mask>0,1/(self.Iz+1e-30),torch.ones_like(self.Iz))

    def gradient(self,p,axis,adjoint=False):
        symbol=self.g.Dx if axis==0 else self.g.Dz
        multiplier=symbol/self.S0
        if adjoint:multiplier=multiplier.conj()
        return torch.fft.ifft2(multiplier*torch.fft.fft2(p))

    def divergence(self,v,axis,adjoint=False):
        s=self.system;g=self.g
        symbol=g.Dx if axis==0 else g.Dz
        left=self.left_x if axis==0 else self.left_z
        right=self.right_x if axis==0 else self.right_z
        gain=s.gvx if axis==0 else s.gvz
        def derivative(value,multiplier):
            return torch.fft.ifft2(multiplier*torch.fft.fft2(value))
        if adjoint:
            return g.dt*1540.*(derivative(left.conj()*v,-symbol)+
                              gain.conj()*derivative(right.conj()*v,-symbol))
        return g.dt*1540.*(left*derivative(v,-symbol.conj())+
                           right*derivative(gain*v,-symbol.conj()))

    def __call__(self,fields):
        if fields.shape!=(3,self.g.H,self.g.W):
            raise ValueError('Use one three-component field at a time')
        mask=self.g.mask
        p,vx,vz=fields*mask
        result=torch.stack([self.system.mass*p+self.divergence(vx,0)+self.divergence(vz,1),
                            self.gradient(p,0)+self.Rx*vx,
                            self.gradient(p,1)+self.Rz*vz])
        return mask*result+(1-mask)*fields

    def adjoint(self,fields):
        mask=self.g.mask
        yp,yx,yz=fields*mask
        result=torch.stack([self.system.mass.conj()*yp+self.gradient(yx,0,True)+self.gradient(yz,1,True),
                            self.divergence(yp,0,True)+self.Rx.conj()*yx,
                            self.divergence(yp,1,True)+self.Rz.conj()*yz])
        return mask*result+(1-mask)*fields

    def from_pressure(self,p):
        """The velocity equations hold exactly for this constructed field."""
        mask=self.g.mask;p=p*mask
        return torch.stack([p,-self.Ix*self.gradient(p,0),-self.Iz*self.gradient(p,1)])*mask

    def scattering_rhs(self,reference_operator,reference_fields):
        """Stable coefficient differences, with all source cells known gel.

        The source is unchanged. Differential and PML terms cancel exactly.
        No subtraction of two large differentiated incident fields is needed.
        """
        p,vx,vz=reference_fields
        return -torch.stack([(self.system.mass-reference_operator.system.mass)*p,
                             (self.Rx-reference_operator.Rx)*vx,
                             (self.Rz-reference_operator.Rz)*vz])*self.g.mask

    def preconditioners(self,pressure_forward,pressure_adjoint):
        """Block elimination with an approximate neural pressure inverse.

        Returning its analytic complex adjoint keeps a normal-equation CG
        Hermitian. Neither operation stores a differentiation trajectory.
        """
        mask=self.g.mask
        def forward(r):
            rp,rx,rz=r*mask
            p=pressure_forward(rp-self.divergence(self.Ix*rx,0)-self.divergence(self.Iz*rz,1))*mask
            vx=self.Ix*(rx-self.gradient(p,0))
            vz=self.Iz*(rz-self.gradient(p,1))
            return torch.stack([p,vx,vz])*mask
        def adjoint(y):
            yp,yx,yz=y*mask
            pbar=yp-self.gradient(self.Ix.conj()*yx,0,True)-self.gradient(self.Iz.conj()*yz,1,True)
            rhsbar=pressure_adjoint(pbar)*mask
            rxbar=self.Ix.conj()*(yx-self.divergence(rhsbar,0,True))
            rzbar=self.Iz.conj()*(yz-self.divergence(rhsbar,1,True))
            return torch.stack([rhsbar,rxbar,rzbar])*mask
        return forward,adjoint

    def readout(self,fields,response):
        return fields[0,self.g.face,self.g.rx_cols]*response

    def readout_adjoint(self,receivers,response):
        fields=torch.zeros((3,self.g.H,self.g.W),dtype=receivers.dtype,device=receivers.device)
        fields[0,self.g.face,self.g.rx_cols]=receivers*complex(response).conjugate()
        return fields


def relaxation_inverse(frequency,dt,taus,eta_per_unit,eta_scale):
    """M(t)=1/modulus and its exact scalar derivative, for local model updates.

    eta_scale is alpha_total*c/1540. Normalized real compressibility multiplies
    M, while normalized density appears in the two velocity equations.
    """
    import cmath,math
    D=cmath.exp(2j*math.pi*frequency*dt);q=1/D
    e=sum(eta_per_unit);root=1+.5*e*eta_scale
    amplitude=root.square();da=e*root
    k=sum(v/(1+t*(D-1)/dt) for v,t in zip(eta_per_unit,taus))
    h=k*eta_scale/amplitude
    dh=k*(amplitude-eta_scale*da)/amplitude.square()
    s=1-q
    inverse=(1-s*h)/(amplitude*(1-h))
    derivative=inverse*(-da/amplitude+dh*(1/(1-h)-s/(1-s*h)))
    return inverse,derivative
