"""CPU data and full signed RF objectives for the independent channel UNO."""
from pathlib import Path
from collections import OrderedDict
import json,math
import numpy as np
import torch
from nsrno.rf_representation import decode

ROOT=Path('out/v5/uno_triparam_v1')
ANGLES=np.linspace(-8,8,11)

class UNOData:
    def __init__(self,root=ROOT,cache_limit=24):
        self.root=Path(root); self.cache=OrderedDict(); self.limit=cache_limit; self.refresh()

    def refresh(self):
        self.index=json.load(open(self.root/'index.json')); self.records=self.index['records']
        extra=self.root/'augmentation_index.json'
        if extra.exists(): self.records+=json.load(open(extra))['records']
        self.lookup={r['id']:r for r in self.records}
        self.groups={key:[r for r in self.records if r['role']==key] for key in ['train','validation']}
        self.scales=self.index['scales']

    def material(self,row):
        key=row['material3'] if row.get('material3') else row['medium']+'|'+row['alpha']
        if key not in self.cache:
            if row.get('material3'): array=np.load(row['material3'])
            else:
                pair=np.load(row['medium']); alpha=np.load(row['alpha'])
                array=np.concatenate((pair,alpha[None]),axis=0)
            self.cache[key]=torch.from_numpy(array.astype(np.float32,copy=False))
            while len(self.cache)>self.limit: self.cache.popitem(last=False)
        else: self.cache.move_to_end(key)
        return self.cache[key]

    def target(self,row,angle):
        return torch.from_numpy(np.array(np.load(row['iq'],mmap_mode='r')[angle]))

    def scale(self,row): return self.scales[str(row['source_mhz'])]['iq_rms']


def rf_objective(predicted,target,floor=.01):
    """Inputs use a fixed source-dependent scale, never a fitted trace gain."""
    predicted_rf=decode(predicted); target_rf=decode(target)
    difference=predicted_rf-target_rf
    energy=target_rf.square().mean().clamp_min(floor)
    full=difference.square().mean()/energy
    window=[]
    for lo,hi in ((8,18),(18,36)):
        a=round(lo*40); b=round(hi*40)
        denom=target_rf[...,a:b].square().mean().clamp_min(floor*.5)
        window.append(difference[...,a:b].square().mean()/denom)
    return full+.1*sum(window),{'full':full.detach(),'mid':window[0].detach(),'late':window[1].detach()}


def rf_statistics(predicted,target):
    p=decode(predicted).double(); t=decode(target).double(); delta=p-t
    stats=[]
    for lo,hi in ((0,60.025),(0,8),(8,18),(18,36),(36,60.025)):
        a=round(lo*40); b=min(2401,round(hi*40))
        stats.append([float(delta[...,a:b].square().sum()),float(t[...,a:b].square().sum())])
    return np.array(stats,np.float64)


def atomic_json(path,value):
    path=Path(path); temporary=path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value,indent=2)); temporary.replace(path)


def atomic_checkpoint(path,value):
    import os
    path=Path(path); temporary=path.with_suffix('.partial.pt')
    with temporary.open('wb') as stream:
        torch.save(value,stream); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
