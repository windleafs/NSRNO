from pathlib import Path
import json,math,shutil
import numpy as np
from scipy import fft
out=Path('out/v5/reconstruction_round2_dense_train_200'); settings=json.load(open(out/'settings.json'))
index=json.load(open('out/v5/forward_cache_v1/index.json')); row=next(r for r in index['records'] if r['id']=='train_200')
iq=np.load(Path('out/v5/forward_cache_v1')/row['iq']); selected=fft.fftshift(fft.fft(iq[:,0]+1j*iq[:,1],axis=-1,norm='forward'),axes=-1)/np.sqrt(2)
spectrum=np.zeros((11,192,1201),np.complex64); spectrum[:,:,40:680]=selected
observed=fft.irfft(spectrum,n=2401,axis=-1,norm='forward'); np.save(out/'observed_rf.npy',observed)
time_us=np.arange(2401)/40e6*1e6; per_cycle=len(settings['frequencies'])//settings['world_size']
states=['initial']+[f'step_{s:03d}' for s in range(per_cycle,settings['steps']+1,per_cycle)]
summary={}
for state in states:
    predicted=np.empty_like(observed); seen=[]; timings=[]
    for part in range(3):
        data=np.load(out/f'time_replay_{state}_part{part}.npz')
        predicted[data['angle_indices']]=data['rf']; seen.extend(data['angle_indices'].tolist()); timings.extend(data['seconds'].tolist())
    if set(seen)!=set(range(11)) or len(seen)!=11: raise RuntimeError('Incomplete full-angle replay')
    np.save(out/(state+'_rf.npy'),predicted); difference=predicted.astype(np.float64)-observed
    error=np.sum(difference**2); target=np.sum(observed.astype(np.float64)**2)
    summary[state]={'full_rf_relative_rms':math.sqrt(error/target),'rf_error_sq':float(error),'rf_target_sq':float(target),
        'sum_angle_seconds':sum(timings),'angles':11,'receivers':192,'rf_samples':2401,'windows':[]}
    for lo,hi in [(0,8),(8,18),(18,36),(36,60.025)]:
        mask=(time_us>=lo)&(time_us<hi)
        window_target=float(np.sum(observed[:,:,mask].astype(np.float64)**2))
        window_error=float(np.sum(difference[:,:,mask]**2))
        summary[state]['windows'].append({'time_us':[lo,hi],
            'relative_rms':math.sqrt(window_error/window_target),
            'target_energy_fraction':window_target/target,
            'error_sq_relative_to_total_target':window_error/target})
candidates=[s for s in states if s!='initial']; chosen=min(candidates,key=lambda s:summary[s]['full_rf_relative_rms']); step=int(chosen.split('_')[1])
shutil.copy2(out/f'medium_step_{step:03d}.npy',out/'reconstructed_medium.npy'); shutil.copy2(out/(chosen+'_rf.npy'),out/'reconstruction_rf.npy')
(out/'selected.json').write_text(json.dumps({'step':step,'state':chosen,
    'selection':'smallest complete RF misfit among updated complete-pass candidates using the original time-domain solver, never material truth',
    'full_rf_relative_rms':summary[chosen]['full_rf_relative_rms'],
    'initial_full_rf_relative_rms':summary['initial']['full_rf_relative_rms'],
    'improved_over_initial':summary[chosen]['full_rf_relative_rms']<summary['initial']['full_rf_relative_rms'],
    'note':'The actual updated candidate is shown even if neither updated candidate beats the initial baseline.'},indent=2))
(out/'physical_evaluation.json').write_text(json.dumps({'summary':summary,'selected':chosen,'method':'original time-domain UltraWave with identicalV5 processing'},indent=2))
print(json.dumps({'summary':summary,'selected':chosen},indent=2))
