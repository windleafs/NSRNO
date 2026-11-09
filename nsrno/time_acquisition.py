"""Native V5 source, sparse interpolation, filtering and their exact transposes."""
from pathlib import Path
import json,math,time
import numpy as np
from scipy import signal,fft
import torch
from .native_time import Config,native_wave
from .rf_representation import decode,FIRST_BIN,LAST_BIN
from .v5_observations import plane_observation,canonical_id

ROOT=Path('out/v5/time_triparam_v1')
POINT_INDEX=Path('out/v5/point_tx_3mhz_v1/index.json')

def projection(rf):
    ft=fft.rfft(rf,axis=0)
    # Retain the same physical frequencies when the record duration changes.
    if rf.shape[0]==2401:ft[:FIRST_BIN]=0;ft[LAST_BIN:]=0
    else:
        frequency=fft.rfftfreq(rf.shape[0],1/40e6)
        ft[(frequency<FIRST_BIN/(2401/40e6)) | (frequency>=LAST_BIN/(2401/40e6))]=0
    return fft.irfft(ft,n=rf.shape[0],axis=0)

def prepare_response(source):
    name=f'response_{source}.npy';path=ROOT/name
    if path.exists(): return path
    meta=json.loads((ROOT/'geometry.json').read_text())
    if source not in ['lf','hf','point3']:raise ValueError(source)
    band=[1.5e6,4.5e6] if source in ['lf','point3'] else [4e6,7.5e6]
    sos=signal.butter(4,band,fs=1/meta['dt'],btype='bandpass',output='sos')
    n=36001 if source=='point3' else meta['nt']; samples=(n-1)//10+1
    matrix=np.lib.format.open_memmap(path.with_suffix('.partial.npy'),mode='w+',dtype=np.float32,shape=(samples,n))
    for begin in range(0,n,128):
        end=min(n,begin+128);basis=np.zeros((n,end-begin),np.float64)
        basis[np.arange(begin,end),np.arange(end-begin)]=1
        filtered=signal.sosfiltfilt(sos,basis,axis=0)
        sampled=signal.resample_poly(filtered,1,10,axis=0)
        matrix[:,begin:end]=projection(sampled).astype(np.float32)
    matrix.flush(); del matrix;path.with_suffix('.partial.npy').replace(path)
    print(json.dumps({'prepared':str(path)}),flush=True);return path

def sparse(coordinates,origin,spacing,nx,rec=False):
    # Devito uses division for source indices and a precomputed reciprocal for receivers.
    d=coordinates.astype(np.float32)-origin.astype(np.float32)
    q=d*(np.float32(1)/spacing.astype(np.float32)) if rec else d/spacing.astype(np.float32)
    base=np.floor(q).astype(np.int32);fraction=q-base.astype(np.float32)
    indices=[];weights=[]
    for ox,oz in [(0,0),(0,1),(1,0),(1,1)]:
        indices.append((base[:,1]+oz)*nx+base[:,0]+ox)
        weights.append((fraction[:,0] if ox else 1-fraction[:,0])*(fraction[:,1] if oz else 1-fraction[:,1]))
    return np.stack(indices,1).astype(np.int32),np.stack(weights,1).astype(np.float32)

class TimeAcquisition:
    def __init__(self,source='lf',sample='train_200',device='cuda',chunk=128):
        self.source=source;self.device=device;self.meta=json.loads((ROOT/'geometry.json').read_text())
        geometry=np.load(ROOT/'geometry.npz');self.geometry={k:geometry[k] for k in geometry.files}
        low_index=json.loads(Path('out/v5/low_frequency_3mhz_v1/index.json').read_text())
        if source not in ['lf','hf','point3']:raise ValueError(source)
        fit=low_index['absorption'] if source in ['lf','point3'] else self.meta['high_absorption']
        self.f0=3e6 if source in ['lf','point3'] else 7.5e6
        if source=='point3':
            point=json.loads(POINT_INDEX.read_text());layout=np.load(point['source_layout'])
            self.point_weights=layout['source_weights']
            self.angles=np.arange(len(self.point_weights))
            self.source_positions=layout['transmit_x_m']
            nt=point['acquisition']['native_nt']
            if nt!=36001 or point['acquisition']['native_dt_s']!=self.meta['dt']:raise ValueError('Point acquisition timestep mismatch')
        else:
            self.angles=np.asarray(low_index['acquisition']['angles_deg']);nt=self.meta['nt']
        nz,nx=self.meta['shape'];self.config=Config(nz,nx,nt,len(geometry['source_coordinates']),len(geometry['xe']),self.meta['pml'],chunk,
            self.meta['dx'],self.meta['dt'],*fit['taus_s'],*fit['eta_per_unit_alpha'])
        si,sw=sparse(geometry['source_coordinates'],geometry['origin'],geometry['spacing'],nx)
        ri,rw=sparse(geometry['receiver_coordinates'],geometry['origin'],geometry['spacing'],nx,rec=True)
        self.si=torch.from_numpy(si).to(device);self.sw=torch.from_numpy(sw).to(device)
        self.ri=torch.from_numpy(ri).to(device);self.rw=torch.from_numpy(rw).to(device)
        self.response=torch.from_numpy(np.load(prepare_response(source))).to(device)
        if source=='point3':
            self.reference=np.load(point['reference_native'],mmap_mode='r')
            row=next(r for r in point['records'] if r['id']==canonical_id(sample))
            self.observed=torch.from_numpy(np.load(row['rf'])).to(device)
            return
        ref_path=Path('out/v5/low_frequency_3mhz_v1/reference.shard0.npz') if source=='lf' else Path(self.meta['reference_file'])
        with np.load(ref_path) as a:
            self.reference=a['rf_native'] if source=='lf' else np.moveaxis(a['rf_native'],-1,0)
        row=plane_observation(source,sample)
        if 'absorption' in row:
            for name in ['taus_s','eta_per_unit_alpha']:
                if not np.allclose(row['absorption'][name],fit[name],rtol=1e-12,atol=0):
                    raise ValueError('Paired observation relaxation model differs from the propagation model')
            if not np.allclose(row['angles_deg'],self.angles,rtol=0,atol=1e-10):
                raise ValueError('Paired observation plane-wave angles differ')
        iq=torch.from_numpy(np.load(row['iq']))
        self.observed=decode(iq).to(device)

    def waveform(self,angle_index):
        g=self.geometry;x=g['x'];xe=g['xe'];cols=np.flatnonzero((x>=xe[0])&(x<=xe[-1]))
        if self.source=='point3':delay=np.zeros(len(cols));weights=self.point_weights[angle_index]
        else:
            raw=x[cols]*np.sin(np.deg2rad(self.angles[angle_index]))/1540
            delay=raw-raw.min();weights=g['weights']
        t=np.arange(self.config.nt)*self.meta['dt']
        tau=t[:,None]-(4/self.f0+delay[None,:])
        wave=1e5*weights[None,:]*np.exp(-.5*(tau*self.f0)**2)*np.sin(2*np.pi*self.f0*tau)
        return torch.from_numpy(wave.astype(np.float32)).to(self.device)

    def raw(self,material,angle_index):
        return native_wave(material,self.waveform(angle_index),self.si,self.sw,self.ri,self.rw,self.config)

    def predict(self,material,angle_index):
        raw=self.raw(material,angle_index)
        reference=torch.from_numpy(self.reference[angle_index].copy()).to(self.device)
        return (self.response@(raw-reference)).T

def truth(sample='train_200',device='cuda'):
    material=np.load(f'out/v5/forward_cache_v1/{sample}.medium.npy')
    alpha=np.load(f'out/v5/uno_triparam_v1/alpha/{sample}.npy')
    return torch.from_numpy(np.concatenate([material,alpha[None]],0)).to(device)

def bandlimit(rf,low=0,high=20e6):
    ft=torch.fft.rfft(rf,dim=-1,norm='ortho')
    freq=torch.fft.rfftfreq(rf.shape[-1],1/40e6,device=rf.device)
    return torch.fft.irfft(ft*((freq>=low)&(freq<=high)),n=rf.shape[-1],dim=-1,norm='ortho')
