"""Unbiased all-shot RF source encoding within each V5 acquisition group.

The wave equation is linear in the source, including its material-dependent
attenuation. The material map remains nonlinear and all three fields are free.
Envelope objectives must not use this linearity argument.
"""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
import torch
from .time_acquisition import ROOT,TimeAcquisition
from .native_time import native_wave


def prepare_bank(acquisition):
    """Called sequentially in the parent before GPU workers are created."""
    a=acquisition;h=hashlib.sha256()
    h.update(Path(__file__).with_name('time_acquisition.py').read_bytes())
    h.update(json.dumps([a.source,a.config.nt,a.config.ns,a.f0,a.angles.tolist()]).encode())
    h.update(a.geometry['x'].tobytes());h.update(a.geometry['weights'].tobytes())
    if a.source=='point3':h.update(a.point_weights.tobytes())
    path=ROOT/f'source_bank_{a.source}_{h.hexdigest()[:16]}.npy'
    if path.exists():return path
    if a.device!='cpu':raise ValueError('Build source templates on CPU before workers start')
    partial=path.with_name(path.name+f'.{os.getpid()}.partial.npy')
    bank=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(len(a.angles),a.config.nt,a.config.ns))
    for shot in range(len(a.angles)):bank[shot]=a.waveform(shot).numpy()
    bank.flush();del bank;partial.replace(path)
    return path


class EncodedAcquisition:
    def __init__(self,acquisition,bank):
        self.acquisition=acquisition;self.bank=np.load(bank,mmap_mode='r') if bank else None
        a=acquisition
        if self.bank is not None and self.bank.shape!=(len(a.angles),a.config.nt,a.config.ns):raise ValueError('Source bank does not match acquisition')

    def signs(self,signs):
        signs=np.asarray(signs,np.float32)
        if signs.shape!=(len(self.acquisition.angles),) or not np.all(np.abs(signs)==1):raise ValueError('One Rademacher sign per original shot is required')
        return signs

    def observed(self,signs):
        a=self.acquisition;signs=self.signs(signs)
        return torch.einsum('s,srt->rt',torch.from_numpy(signs).to(a.device),a.observed)

    def predict(self,material,signs):
        a=self.acquisition;signs=self.signs(signs)
        if self.bank is None:raise ValueError('Encoded propagation requires a waveform bank')
        # Host templates avoid storing eleven large waveforms on the GPU.
        wave=np.tensordot(signs,self.bank,axes=(0,0))
        reference=np.tensordot(signs,a.reference,axes=(0,0))
        source=torch.from_numpy(wave).to(a.device)
        raw=native_wave(material,source,a.si,a.sw,a.ri,a.rw,a.config)
        return (a.response@(raw-torch.from_numpy(reference).to(a.device))).T

    def denominators(self):
        a=self.acquisition;windows=[(0,8),(8,18),(18,36),(36,90.025 if a.source=='point3' else 60.025)]
        den=[max(float(a.observed[:,:,round(b*40):round(e*40)].double().square().sum()),1e-20) for b,e in windows]
        return windows,den


def main():
    from argparse import ArgumentParser
    from .time_acquisition import truth
    p=ArgumentParser();p.add_argument('--source',choices=['lf','hf','point3'],default='point3');p.add_argument('--sample',default='train_200')
    p.add_argument('--seed',type=int,default=20261106);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    cpu=TimeAcquisition(a.source,a.sample,device='cpu');bank=prepare_bank(cpu);del cpu
    acquisition=TimeAcquisition(a.source,a.sample);encoded=EncodedAcquisition(acquisition,bank)
    signs=np.random.default_rng(a.seed).choice([-1.,1.],len(acquisition.angles)).astype(np.float32)
    material=truth(a.sample);torch.cuda.reset_peak_memory_stats();start=time.time()
    with torch.no_grad():pred=encoded.predict(material,signs);target=encoded.observed(signs)
    difference=pred-target
    report={'source':a.source,'sample':a.sample,'signs':signs.tolist(),'shots_in_one_propagation':len(signs),
        'relative_rms':float(difference.norm()/target.norm()),'seconds':time.time()-start,
        'memory_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'purpose':'operator consistency only, not reconstruction',
        'truth_used_for_this_check':True,'source_bank':str(bank),'windows':{}}
    windows,_=encoded.denominators()
    for begin,end in windows:
        sl=slice(round(begin*40),round(end*40));report['windows'][f'{begin}-{end}']=float(difference[:,sl].norm()/target[:,sl].norm())
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if report['relative_rms']>1e-3:raise RuntimeError('Encoded source does not match original independent RF')


if __name__=='__main__':main()
