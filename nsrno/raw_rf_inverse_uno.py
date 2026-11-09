"""RF-domain UNO and learned nonlocal material readout; no fixed-speed focusing."""
import math
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .uno_rf import UShape
from .inverse_uno import HEIGHT,WIDTH,map_loss,pair_loss


@lru_cache(maxsize=16)
def cached_rf(path):
    return np.load(path)


def load_batch(rows,device,scale):
    data=[]
    for row in rows:
        wave=cached_rf(row['raw_rf']);n=row['valid_samples']
        if wave.shape!=(11,2,192,n):raise ValueError('Expected every source and receiver')
        value=np.zeros((11,4,192,544),np.float32)
        value[:,:2,:,:n]=wave/scale
        value[:,2,:,:n]=1
        value[:,3,:,:n]=np.arange(n,dtype=np.float32)[None,None,:]*row['sample_dt_s']/90e-6
        data.append(value)
    return torch.from_numpy(np.stack(data)).to(device),torch.from_numpy(np.stack([np.load(r['label']) for r in rows])).to(device)


def mirror(x,y):
    # Mirroring physical RF reverses receivers AND the signed source angles.
    return x.flip((1,3)),y.flip(-1)


def fourier_positions(position):
    result=[position]
    for frequency in (1.,2.,4.,8.,16.,32.,64.,128.):
        result.extend((torch.sin(math.pi*frequency*position[...,:2]),
                       torch.cos(math.pi*frequency*position[...,:2])))
    return torch.cat(result,-1)


class IntegralAttention(nn.Module):
    def __init__(self,dim=64,heads=2):
        super().__init__();self.heads=heads;self.dim=dim
        self.q=nn.Linear(dim,dim);self.k=nn.Linear(dim,dim);self.v=nn.Linear(dim,dim)
        self.out=nn.Linear(dim,dim)

    def forward(self,query,key,value,bias=None):
        b=query.shape[0];h=self.heads;d=self.dim//h
        q=self.q(query).reshape(b,-1,h,d).transpose(1,2)
        k=self.k(key).reshape(b,-1,h,d).transpose(1,2)
        v=self.v(value).reshape(b,-1,h,d).transpose(1,2)
        attended=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.)
        return self.out(attended.transpose(1,2).reshape(b,-1,self.dim))


class SourceRFEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv2d(69,24,1)
        self.uno=UShape([24,32,48,32,24],[(12,24),(10,16),(6,8),(10,16),(12,24)],False)
        # Retain absolute RF amplitudes; do not normalize each acquisition.
        for block in self.uno.blocks:block.norm=nn.Identity()
        self.key=nn.Linear(24,64);self.value=nn.Linear(24,64)
        self.position=nn.Sequential(nn.Linear(37,64),nn.GELU(),nn.Linear(64,64))
        self.queries=nn.Parameter(torch.randn(1,512,64)*.02)
        xx,tt=torch.meshgrid(torch.linspace(-1,1,16),torch.linspace(0,1,32),indexing='ij')
        self.register_buffer('query_coordinates',torch.stack((xx,tt),-1).reshape(512,2))
        self.readout=IntegralAttention()
        self.refine=nn.Sequential(nn.Linear(64,96),nn.GELU(),nn.Linear(96,64))

    def forward(self,signal,geometry):
        b=signal.shape[0];packed=F.pixel_unshuffle(signal,4)
        h,w=packed.shape[-2:]
        rx=torch.linspace(-1,1,192,device=signal.device).reshape(1,1,192,1).expand(b,1,192,544)
        rx=F.avg_pool2d(rx,4)
        geom=geometry[:,:,None,None].expand(-1,-1,h,w)
        encoded=self.uno(self.stem(torch.cat((packed,rx,geom),1))).flatten(2).transpose(1,2)
        valid=F.avg_pool2d(signal[:,2:3],4).flatten(2).transpose(1,2)
        time=F.avg_pool2d(signal[:,3:4],4).flatten(2).transpose(1,2)/valid.clamp_min(1/16)
        positions=torch.cat((rx.flatten(2).transpose(1,2),time,geometry[:,:3,None].transpose(1,2).expand(-1,h*w,-1)),2)
        query_positions=torch.cat((self.query_coordinates[None].expand(b,-1,-1),
                                   geometry[:,:3,None].transpose(1,2).expand(-1,512,-1)),2)
        queries=self.queries+self.position(fourier_positions(query_positions))
        keys=self.key(encoded)+self.position(fourier_positions(positions))
        # Local RF-grid seed only; this is not a material propagation/focus model.
        distance=query_positions[:,:,:2,None]-positions[:,:,:2].transpose(1,2)[:,None]
        bias=-.5*((distance[:,:,0]/.16)**2+(distance[:,:,1]/.055)**2)
        bias=bias.masked_fill(valid[...,0][:,None,:]==0,float('-inf'))[:,None]
        tokens=self.readout(queries,keys,self.value(encoded),bias)
        tokens=tokens+self.refine(tokens)
        return tokens,self.position(fourier_positions(query_positions))


class RawRFInverseUNO(nn.Module):
    def __init__(self,query_chunk=1024):
        super().__init__();self.query_chunk=query_chunk;self.encoder=SourceRFEncoder()
        angles=torch.linspace(-8,8,11)
        tref=4/3e6+.0191*torch.sin(angles*math.pi/180).abs()/1540
        phase=2*math.pi*3e6*tref
        self.register_buffer('geometry',torch.stack((angles/8,tref/10e-6,torch.sin(phase),torch.cos(phase)),1))
        z,x=torch.meshgrid(torch.linspace(0,1,HEIGHT),torch.linspace(-1,1,WIDTH),indexing='ij')
        self.register_buffer('coordinates',torch.stack((x,z))[None])
        self.register_buffer('query_features',fourier_positions(torch.stack((x,z),-1).reshape(-1,2)))
        self.query=nn.Sequential(nn.Linear(34,64),nn.GELU(),nn.Linear(64,64))
        self.readout=IntegralAttention();self.to_image=nn.Linear(64,24)
        self.stem=nn.Conv2d(26,24,1)
        self.decoder=UShape([24,32,48,32,24],[(16,16),(12,12),(8,8),(12,12),(16,16)],True)
        self.head=nn.Sequential(nn.Conv2d(24,24,1),nn.GELU(),nn.Conv2d(24,3,1))
        nn.init.normal_(self.head[-1].weight,std=.001);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('gel',torch.arange(HEIGHT)[None,None,:,None]<5)
        self.config={'architecture':'raw_complex_RF_UNO_nonlocal_inverse_v1','RF_sources':11,'receivers':192,
            'padded_envelope_samples':544,'RF_pixel_unshuffle':4,'RF_normalization':'one training-only global scale',
            'source_tokens':512,'readout_tokens':11*512,'second_global_compression':False,
            'image_shape':[HEIGHT,WIDTH],'outputs':['c','rho','alpha_tissue'],'fixed_speed_focusing':False,
            'query_chunk':query_chunk,'dtype':'float32 including attention; FFT complex64',
            'checkpointing':'whole source encoder and each material query chunk',
            'known_gel_only':True,'forced_smooth_c':False,'independent_material_outputs':True}

    def forward(self,signal):
        if signal.ndim!=5 or signal.shape[1:]!=(11,4,192,544):raise ValueError('Incomplete raw RF input')
        b=signal.shape[0];values=[];positions=[]
        for shot in range(11):
            geometry=self.geometry[shot][None].expand(b,-1)
            if torch.is_grad_enabled():
                value,position=checkpoint(self.encoder,signal[:,shot],geometry,use_reentrant=False)
            else:value,position=self.encoder(signal[:,shot],geometry)
            values.append(value);positions.append(position)
        values=torch.cat(values,1);keys=values+torch.cat(positions,1)
        def decode_chunk(query_features,k,v):
            query=self.query(query_features)[None].expand(b,-1,-1)
            return self.to_image(self.readout(query,k,v))
        image=[]
        for start in range(0,HEIGHT*WIDTH,self.query_chunk):
            query=self.query_features[start:start+self.query_chunk]
            if torch.is_grad_enabled():image.append(checkpoint(decode_chunk,query,keys,values,use_reentrant=False))
            else:image.append(decode_chunk(query,keys,values))
        image=torch.cat(image,1).transpose(1,2).reshape(b,24,HEIGHT,WIDTH)
        raw=self.head(self.decoder(self.stem(torch.cat((image,self.coordinates.expand(b,-1,-1,-1)),1))))
        c=1540+140*torch.tanh(raw[:,0]);rho=1000+180*torch.tanh(raw[:,1])
        alpha=.002+1.998*torch.sigmoid(raw[:,2]+math.log(.498/1.5))
        result=torch.stack((c,rho,alpha),1)
        fixed=result.new_tensor([1540.,1000.,.002])[None,:,None,None]
        return torch.where(self.gel,fixed,result)
