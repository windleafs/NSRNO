from pathlib import Path
import json,math
import numpy as np
import torch
from nsrno.rf_representation import decode
from nsrno.reconstruct_material import write_json

out=Path('out/v5/reconstruction_round2_dense_train_200'); rows=[]
for part in range(3): rows.extend(json.load(open(out/f'physical_part{part}.json')))
index=json.load(open('out/v5/forward_cache_v1/index.json')); row=next(r for r in index['records'] if r['id']=='train_200')
iq=np.load(Path('out/v5/forward_cache_v1')/row['iq'])
observed=decode(torch.from_numpy(iq)).numpy(); np.save(out/'observed_rf.npy',observed)
time=np.arange(2401)/40e6*1e6; summary={}; windows=[(0,8),(8,18),(18,36),(36,61)]
for state in ('initial','reconstruction'):
    values=[r for r in rows if r['state']==state]
    if len(values)!=294 or {r['bin'] for r in values}!=set(range(241,535)): raise RuntimeError('Incomplete final dense RF evaluation')
    spectrum=np.zeros((11,192,1201),np.complex64)
    for b in range(241,535): spectrum[:,:,b]=np.load(out/f'prediction_{state}_b{b}.npy')
    predicted=np.fft.irfft(spectrum,n=2401,axis=-1,norm='forward').astype(np.float32)
    np.save(out/(state+'_rf.npy'),predicted)
    error=predicted.astype(np.float64)-observed
    summary[state]={'frequency_count':294,'angle_count':11,'receiver_count':192,
        'band_rf_relative_rms':math.sqrt(sum(r['error_sq'] for r in values)/sum(r['target_sq'] for r in values)),
        'full_rf_relative_rms':float(np.linalg.norm(error)/np.linalg.norm(observed.astype(np.float64))),
        'max_equation_residual':max(r['solver']['residual'] for r in values),'windows':[]}
    for lo,hi in windows:
        selected=(time>=lo)&(time<hi)
        summary[state]['windows'].append({'time_us':[lo,min(hi,60.025)],
            'relative_rms':float(np.linalg.norm(error[:,:,selected])/np.linalg.norm(observed[:,:,selected].astype(np.float64)))})
write_json(out/'physical_evaluation.json',{'summary':summary,'rows':rows})
print(json.dumps(summary,indent=2))
