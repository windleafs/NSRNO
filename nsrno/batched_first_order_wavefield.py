"""Same first-order equation, with an explicit leading transmission axis."""
import torch
from .first_order_wavefield import FirstOrderWavefield


class BatchedFirstOrderWavefield(FirstOrderWavefield):
    def __call__(self,fields):
        p,vx,vz=(fields*self.g.mask).unbind(1)
        value=torch.stack([self.system.mass*p+self.divergence(vx,0)+self.divergence(vz,1),
            self.gradient(p,0)+self.Rx*vx,self.gradient(p,1)+self.Rz*vz],dim=1)
        return value*self.g.mask+fields*(1-self.g.mask)
    def adjoint(self,fields):
        yp,yx,yz=(fields*self.g.mask).unbind(1)
        value=torch.stack([self.system.mass.conj()*yp+self.gradient(yx,0,True)+self.gradient(yz,1,True),
            self.divergence(yp,0,True)+self.Rx.conj()*yx,
            self.divergence(yp,1,True)+self.Rz.conj()*yz],dim=1)
        return value*self.g.mask+fields*(1-self.g.mask)
    def readout(self,fields,response):return fields[:,0,self.g.face,self.g.rx_cols]*response
    def readout_adjoint(self,receivers,response):
        value=torch.zeros((receivers.shape[0],3,self.g.H,self.g.W),dtype=receivers.dtype,device=receivers.device)
        value[:,0,self.g.face,self.g.rx_cols]=receivers*complex(response).conjugate();return value
    def preconditioners(self,pressure_forward,pressure_adjoint):
        def forward(r):
            rp,rx,rz=(r*self.g.mask).unbind(1)
            p=pressure_forward(rp-self.divergence(self.Ix*rx,0)-self.divergence(self.Iz*rz,1))*self.g.mask
            return torch.stack([p,self.Ix*(rx-self.gradient(p,0)),self.Iz*(rz-self.gradient(p,1))],dim=1)*self.g.mask
        def adjoint(y):
            yp,yx,yz=(y*self.g.mask).unbind(1)
            pbar=yp-self.gradient(self.Ix.conj()*yx,0,True)-self.gradient(self.Iz.conj()*yz,1,True)
            w=pressure_adjoint(pbar)*self.g.mask
            return torch.stack([w,self.Ix.conj()*(yx-self.divergence(w,0,True)),
                self.Iz.conj()*(yz-self.divergence(w,1,True))],dim=1)*self.g.mask
        return forward,adjoint
