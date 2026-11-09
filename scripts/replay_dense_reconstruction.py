"""Independent original time-domain RF replay of inverse candidates.

No PyTorch in this process. It uses candidate c/rho and explicitly known alpha,
never the archived material truth. All11 angles and the complete RF are read.
"""
from pathlib import Path
from argparse import ArgumentParser
import json,time
import numpy as np
from scipy import fft
from nsrno.fullwave import initialize,read_reference

p=ArgumentParser(); p.add_argument('--part',type=int,required=True); args=p.parse_args()
out=Path('out/v5/reconstruction_round2_dense_train_200'); settings=json.load(open(out/'settings.json'))
if not (out/'completion.json').exists(): raise RuntimeError('Only replay completed inverse candidates')
gen=initialize(); manifest=json.load(open('out/v5/manifest.json')); reference=read_reference(gen,manifest)
alpha=np.load(out/'known_alpha.npy')
states=[('initial',out/'initial_medium.npy')]
per_cycle=len(settings['frequencies'])//settings['world_size']
states += [(f'step_{s:03d}',out/f'medium_step_{s:03d}.npy') for s in range(per_cycle,settings['steps']+1,per_cycle)]
for state,path in states:
    result_path=out/f'time_replay_{state}_part{args.part}.npz'
    if result_path.exists(): continue
    medium=np.load(path); case=gen.geometry_case()
    case['maps']={'sound_speed':np.ascontiguousarray(medium[0]),'density':np.ascontiguousarray(medium[1]),
        'alpha_coeff':np.ascontiguousarray(alpha),'BonA':np.zeros_like(alpha)}
    fit=gen.absorption_model(case); solver=gen.solver_for(case,case['maps'],fit)
    angles=[]; waveforms=[]; timings=[]
    for a in range(11):
        if a%3!=args.part: continue
        gen.set_angle(solver,case,float(gen.ANGLES[a])); start=time.perf_counter(); raw,_=solver.run()
        rf,_=gen.sim.analytic_channels(raw-reference[:,:,a],gen.DT,band=manifest['acquisition']['band_hz'])
        angles.append(a); waveforms.append(rf.T); timings.append(time.perf_counter()-start)
        print(json.dumps({'state':state,'angle':a,'seconds':timings[-1]}),flush=True)
    np.savez(result_path,angle_indices=np.array(angles),rf=np.stack(waveforms).astype(np.float32),seconds=np.array(timings))
    del solver
(out/f'time_replay_complete_{args.part}.json').write_text(json.dumps({'completed':True,'states':[s for s,_ in states]}))
