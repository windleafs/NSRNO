"""Post-hoc mechanism diagnosis only. Truth cannot enter initialization."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from .reconstruct_cmpe import PhaseData, rays
from .v5_materials import evaluation_material


def main():
    torch.set_num_threads(4)
    start=time.time()
    out=Path('out/v5/reconstruction_cmpe_v1_train_200')
    run=json.loads((out/'run.json').read_text())
    if run['status']!='complete':
        raise RuntimeError('Diagnosis requires the fixed, finished candidate')
    seed=np.load(out/'stage0_initial_material.npy')
    candidate=np.load(out/'material.npy')
    target=evaluation_material(run['sample'])
    point=json.loads(Path('out/v5/point_tx_3mhz_v1/index.json').read_text())
    acquisition=point['acquisition']
    row=next(r for r in point['records'] if r['id']==run['sample'])
    tx,rx=np.array(acquisition['transmit_x_m']),np.array(acquisition['receiver_x_m'])
    xx,zz=np.meshgrid(np.arange(-.018,.0181,.0015),np.arange(.0025,.0431,.0015))
    foci=np.stack([xx.ravel(),zz.ravel()],1)
    xn,zn=np.arange(-.028,.0281,.001),np.arange(.001,.0451,.001)
    operator,seed_times=rays(foci,np.r_[tx,rx],seed[0],xn,zn)
    correction=np.load(out/'slowness_correction.npy')
    optimized_times=seed_times+(operator@correction.ravel()).reshape(seed_times.shape)
    del operator
    operator,truth_times=rays(foci,np.r_[tx,rx],target[0],xn,zn)
    del operator
    data=PhaseData(np.load(row['rf']),tx,rx,foci,acquisition['fs_hz'],acquisition['source_tref_s'][0])
    data.fix_weights(seed_times)
    values={}
    for name,times in [('RF_seed',seed_times),('CMPE_selected',optimized_times),('truth_c_straight_rays',truth_times)]:
        values[name]=data.value_gradient_times(times,gradient=False)[0]
    mask=(np.arange(961)[:,None]>=80)&(np.arange(961)[:,None]<=920)&(np.arange(1121)[None,:]>=178)&(np.arange(1121)[None,:]<=942)
    statistics={}
    for name,material in [('RF_seed',seed),('CMPE_selected',candidate),('truth',target)]:
        c=material[0][mask].astype(np.float64)
        statistics[name]={'mean_c_m_s':float(c.mean()),'std_c_m_s':float(c.std())}
    report={'posthoc_truth_used_only_for_diagnosis':True,'candidate_chosen_before_truth_access':True,
            'full_native_wave_simulations':0,'GPU_memory_GiB':0,
            'objective':'same fixed initial data weights and straight-ray CMPE; no TV in this comparison',
            'phase_losses':values,'c_statistics':statistics,
            'interpretation_limit':'If true-c cost is not lower than candidate, that refutes cost ranking for this straight-ray surrogate; it does not prove the acoustic three-field inverse problem impossible.',
            'elapsed_seconds':time.time()-start}
    (out/'posthoc_phase_oracle.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
