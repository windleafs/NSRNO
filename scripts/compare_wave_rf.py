"""Matched legacy-model comparison and two band-projected RF illustrations."""
from pathlib import Path
from argparse import ArgumentParser
import json,time
import numpy as np
import torch
from nsrno.wave_operator import WaveResidualOperator
from nsrno.train_wave import sample
from nsrno.forward_model import load_forward_model
from nsrno.rf_representation import decode
from nsrno.dataset import read_json,write_json

p=ArgumentParser(); p.add_argument('--root',type=Path,default=Path('out/v5/forward_round5_wave_v1')); p.add_argument('--iterations',type=int,default=8); p.add_argument('--skip-band',action='store_true'); p.add_argument('--all-records',action='store_true'); args=p.parse_args()
torch.set_num_threads(2); root=args.root; cache=Path('out/v5/forward_cache_v1')
saved=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
wave=WaveResidualOperator(saved['spec'],args.iterations,saved.get('preconditioner','shifted')).cuda(); wave.load_state_dict(saved['model']); wave.eval()
old=load_forward_model(Path('out/v5/forward_round4_material/best.pt'),'cuda'); old.eval()
index=read_json(cache/'index.json'); by_id={r['id']:r for r in index['records']}
ids=read_json(root/'run.json')['validation_probe']
if args.all_records: ids=[r['id'] for r in index['records'] if r['role']=='validation']

def old_rf(sid,angle):
    shared=Path('out/v5/forward_ray_shared_fp32_v1')
    def array(path): return torch.from_numpy(np.load(path)).cuda()
    bg=array(shared/'background'/f'{sid}.npy')[None]
    base=array(shared/'baseline'/f'{sid}.npy')[None,angle:angle+1]
    feature=array(Path('out/v5/forward_ray_material_v1/features')/f'{sid}.npy')[None,angle:angle+1]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        iq=old.predict_cached(bg,torch.tensor([[angle]],device='cuda'),feature,base)
    return decode(iq)[0,0]

with torch.no_grad():
    error=energy=wave_error=0.
    for i,sid in enumerate(ids):
        angle=(0,5,10)[i%3]; bin_index=(270,300,330,360,390,420)[i%6]
        medium,target=sample(by_id[sid],cache,angle,bin_index)
        new,_,_=wave(medium,bin_index*40e6/2401,angle)
        wave_error+=float((new-target).abs().square().sum())
        pred=torch.fft.rfft(old_rf(sid,angle),dim=-1,norm='forward')[:,bin_index]
        error+=float((pred-target).abs().square().sum()); energy+=float(target.abs().square().sum())
    result=dict(records=len(ids),previous_round4_same_single_frequency_rms=(error/energy)**.5,
                wave_same_single_frequency_rms=(wave_error/energy)**.5,
                windowless_spectral_error_energy_reduction=1-wave_error/error,
                selection_probe_rms=saved['validation']['relative_channel_spectrum_rms'],band_rf=[])
    print(json.dumps(result),flush=True)
    for sid in () if args.skip_band else (ids[0],ids[len(ids)//2]):
        row=by_id[sid]; angle=5; medium,_=sample(row,cache,angle,360)
        iq=torch.from_numpy(np.array(np.load(cache/row['iq'])[angle]))[None,None].cuda()
        truth=torch.fft.rfft(decode(iq)[0,0],dim=-1,norm='forward')
        baseline=torch.fft.rfft(old_rf(sid,angle),dim=-1,norm='forward')
        predicted=torch.zeros_like(truth); timer=time.perf_counter()
        for bin_index in range(252,433):
            pred,_,_=wave(medium,bin_index*40e6/2401,angle)
            predicted[:,bin_index]=pred
        mask=torch.zeros_like(truth); mask[:,252:433]=1
        target=torch.fft.irfft(truth*mask,n=2401,dim=-1,norm='forward')
        previous=torch.fft.irfft(baseline*mask,n=2401,dim=-1,norm='forward')
        predicted=torch.fft.irfft(predicted,n=2401,dim=-1,norm='forward')
        def metric(pred):
            values=[]
            for start,end in ((0,320),(320,720),(720,1440),(1440,2401)):
                values.append(float(((pred[:,start:end]-target[:,start:end]).square().sum()/target[:,start:end].square().sum()).sqrt()))
            return dict(overall=float(((pred-target).square().sum()/target.square().sum()).sqrt()),windows=values)
        report=dict(id=sid,angle=angle,iterations=args.iterations,band_mhz=[252*40/2401,432*40/2401],wave=metric(predicted),round4=metric(previous),
                    seconds=time.perf_counter()-timer)
        result['band_rf'].append(report); print(json.dumps(report),flush=True)
    result['note']='Two records, angle 0 deg, both predictions and targets projected onto bins 252:433. Not full-band/full-validation RF.'
    write_json(root/'matched_rf_comparison.json',result)
