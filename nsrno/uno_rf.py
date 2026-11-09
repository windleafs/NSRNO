"""Pure neural material-to-channel UNO. No wave solver is imported.

U-shaped spectral blocks follow Rahman et al., arXiv:2204.11127.
The continuous cross-domain readout is an NSRNO adaptation, not a paper replica.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SpectralMap(nn.Module):
    def __init__(self,cin,cout,modes):
        super().__init__(); self.modes=modes
        self.weight=nn.Parameter(torch.randn(2,cin,cout,*modes,dtype=torch.complex64)/math.sqrt(cin))

    def forward(self,x,size):
        # Native non-power-of-two FFTs require FP32, independent of conv AMP.
        with torch.autocast(device_type=x.device.type,enabled=False):
            transform=torch.fft.rfft2(x.float(),norm='forward')
            out=torch.zeros(x.shape[0],self.weight.shape[2],size[0],size[1]//2+1,
                            device=x.device,dtype=torch.complex64)
            m=min(self.modes[0],x.shape[-2]//2,size[0]//2)
            n=min(self.modes[1],transform.shape[-1],out.shape[-1])
            out[:,:,:m,:n]=torch.einsum('bixy,ioxy->boxy',transform[:,:,:m,:n],self.weight[0,:,:,:m,:n])
            out[:,:,-m:,:n]=torch.einsum('bixy,ioxy->boxy',transform[:,:,-m:,:n],self.weight[1,:,:,:m,:n])
            return torch.fft.irfft2(out,s=size,norm='forward')


class OperatorBlock(nn.Module):
    def __init__(self,cin,cout,modes):
        super().__init__(); self.spectral=SpectralMap(cin,cout,modes)
        self.local=nn.Sequential(nn.Conv2d(cin,cout,1),nn.Conv2d(cout,cout,3,padding=1,groups=cout))
        self.norm=nn.GroupNorm(4,cout)

    def forward(self,x,size):
        local=F.interpolate(x,size=size,mode='bilinear',align_corners=False)
        return F.gelu(self.norm(self.spectral(x,size)+self.local(local)))


class UShape(nn.Module):
    def __init__(self,channels,modes,checkpointing=True):
        super().__init__(); self.checkpointing=checkpointing
        self.blocks=nn.ModuleList([OperatorBlock(channels[max(0,i-1)],c,modes[i]) for i,c in enumerate(channels)])
        self.skip=nn.ModuleList([nn.Conv2d(channels[1],channels[3],1),nn.Conv2d(channels[0],channels[4],1)])

    def run(self,block,x,size):
        if self.checkpointing and torch.is_grad_enabled():
            return checkpoint(block,x,size,use_reentrant=False)
        return block(x,size)

    def forward(self,x):
        h,w=x.shape[-2:]; sizes=[(h,w),(h//2,w//2),(h//4,w//4),(h//2,w//2),(h,w)]
        a=self.run(self.blocks[0],x,sizes[0]); b=self.run(self.blocks[1],a,sizes[1])
        c=self.run(self.blocks[2],b,sizes[2]); d=self.run(self.blocks[3],c,sizes[3])+self.skip[0](b)
        return self.run(self.blocks[4],d,sizes[4])+self.skip[1](a)


class CoordinateKernel(nn.Module):
    """Continuous learned integral kernel plus a trainable geometric seed.

    The seed uses fixed acquisition coordinates, never input-dependent rays.
    Signed kernel corrections permit interference; they are not attention weights.
    """
    def __init__(self,heads,out_coord,in_coord,sigma):
        super().__init__(); u,v=torch.meshgrid(out_coord,in_coord,indexing='ij')
        diff=u-v; features=[u,v,diff,diff.square()]
        for freq in (1.,2.,4.,8.,16.,32.,64.,128.):
            for a in (u,v,diff): features.extend((torch.sin(math.pi*freq*a),torch.cos(math.pi*freq*a)))
        self.register_buffer('features',torch.stack(features,-1))
        self.register_buffer('difference',diff)
        self.mlp=nn.Sequential(nn.Linear(len(features)+3,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,heads))
        nn.init.normal_(self.mlp[-1].weight,std=.002); nn.init.zeros_(self.mlp[-1].bias)
        self.log_width=nn.Parameter(torch.full((heads,),math.log(sigma)))
        self.shift=nn.Linear(3,heads); nn.init.zeros_(self.shift.weight); nn.init.zeros_(self.shift.bias)

    def forward(self,condition):
        b=condition.shape[0]; o,n=self.difference.shape
        feature=self.features[None].expand(b,-1,-1,-1)
        cond=condition[:,None,None,:].expand(-1,o,n,-1)
        learned=self.mlp(torch.cat((feature,cond),-1)).permute(0,3,1,2)
        delta=self.difference[None,None]-self.shift(condition)[:,:,None,None]*.1
        seed=torch.softmax(-.5*(delta/self.log_width.exp()[None,:,None,None].clamp_min(.002))**2,dim=-1)
        return seed+.2*learned/n


class ChannelUNO(nn.Module):
    def __init__(self,checkpointing=True):
        super().__init__(); self.config={'architecture':'spatial_UNO_continuous_integral_receiver_UNO_v1',
            'native_shape':[961,1121],'packed_shape':[256,288],'iq_shape':[2,192,640],
            'checkpointing':checkpointing,'sources_mhz':[3.,7.5],'unknown_inputs':['c','rho','alpha_total'],
            'internal_representation':'source_demodulated_retimed_IQ','output_representation':'original_V5_complex_IQ'}
        self.stem=nn.Conv2d(50,32,1)
        self.spatial=UShape([32,48,64,48,32],[(24,24),(20,20),(12,12),(20,20),(24,24)],checkpointing)
        z=(-.003+(torch.arange(256)*4+1.5)*5e-5)/.05
        x=(-.028+(torch.arange(288)*4+1.5)*5e-5)/.03
        rx=(torch.arange(192)*.0002-.0191)/.03
        time=torch.arange(640)*(2401/40e6)/640
        depth=time*1540/2/.05
        self.rx_kernel=CoordinateKernel(4,rx,x,.012)
        self.time_kernel=CoordinateKernel(4,depth,z,.008)
        zz,xx=torch.meshgrid(z,x,indexing='ij'); self.register_buffer('coords',torch.stack((xx,zz))[None])
        self.film=nn.Sequential(nn.Linear(3,64),nn.GELU(),nn.Linear(64,64))
        self.receiver_lift=nn.Conv2d(32,24,1)
        self.receiver=UShape([24,32,48,32,24],[(24,48),(20,40),(12,24),(20,40),(24,48)],checkpointing)
        self.head=nn.Sequential(nn.Conv2d(24,24,1),nn.GELU(),nn.Conv2d(24,2,1))
        nn.init.normal_(self.head[-1].weight,std=.01); nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def condition(angle_deg,source_mhz):
        return torch.stack((angle_deg/8,source_mhz/7.5,torch.ones_like(angle_deg)),-1)

    def encode(self,material):
        if material.ndim!=4 or material.shape[1:]!=(3,961,1121): raise ValueError('Expected [batch,3,961,1121] c/rho/alpha')
        value=torch.stack(((material[:,0]-1540)/100,(material[:,1]-1000)/100,
                           torch.log1p(material[:,2].clamp_min(0))/math.log(3)),1)
        # Same deterministic 24-pixel rim convention as the original simulator.
        value=F.pad(value[:,:,24:-24,24:-24],(24,24,24,24),mode='replicate')
        value=F.pad(value,(0,31,0,63),mode='replicate')
        packed=F.pixel_unshuffle(value,4)
        return self.spatial(self.stem(torch.cat((packed,self.coords.expand(material.shape[0],-1,-1,-1)),1)))

    def observe(self,latent,angle_deg,source_mhz):
        condition=self.condition(angle_deg,source_mhz)
        def readout(h,cond):
            kx=self.rx_kernel(cond); kt=self.time_kernel(cond)
            b,c,z,x=h.shape; heads=4
            h=h.reshape(b,heads,c//heads,z,x)
            value=torch.einsum('bhczx,bhrx->bhczr',h,kx)
            value=torch.einsum('bhczr,bhtz->bhcrt',value,kt).reshape(b,c,192,640)
            scale,bias=self.film(cond).chunk(2,-1)
            return value*(1+.1*scale[:,:,None,None])+.1*bias[:,:,None,None]
        if self.spatial.checkpointing and torch.is_grad_enabled():
            value=checkpoint(readout,latent,condition,use_reentrant=False)
        else: value=readout(latent,condition)
        adapted=self.head(self.receiver(self.receiver_lift(value))).float()
        return self.restore_source(adapted,angle_deg,source_mhz)

    @staticmethod
    def restore_source(adapted,angle_deg,source_mhz,inverse=False):
        """Unitary source-clock change; uses known source, never material rays."""
        with torch.autocast(device_type=adapted.device.type,enabled=False):
            signal=torch.complex(adapted[:,0].float(),adapted[:,1].float())
            duration=2401/40e6
            t=torch.arange(640,device=signal.device)*duration/640
            rx=torch.arange(192,device=signal.device)*.0002-.0191
            sine=torch.sin(angle_deg.float()*math.pi/180)
            delay=4/(source_mhz.float()[:,None]*1e6)+(.0191*sine.abs()[:,None]+rx[None]*sine[:,None])/1540
            frequencies=torch.arange(40,680,device=signal.device)/duration
            rotation=torch.exp(-2j*math.pi*delay[:,:,None]*frequencies[None,None,:])
            carrier=torch.exp(2j*math.pi*(source_mhz.float()*1e6-360/duration)[:,None]*t[None])[:,None,:]
            if inverse:
                spectrum=torch.fft.fftshift(torch.fft.fft(signal,dim=-1,norm='forward'),dim=-1)*rotation.conj()
                value=torch.fft.ifft(torch.fft.ifftshift(spectrum,dim=-1),dim=-1,norm='forward')*carrier.conj()
            else:
                spectrum=torch.fft.fftshift(torch.fft.fft(signal*carrier,dim=-1,norm='forward'),dim=-1)*rotation
                value=torch.fft.ifft(torch.fft.ifftshift(spectrum,dim=-1),dim=-1,norm='forward')
            return torch.stack((value.real,value.imag),1)

    def forward(self,material,angle_deg,source_mhz):
        return self.observe(self.encode(material),angle_deg,source_mhz)
