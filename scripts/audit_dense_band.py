from pathlib import Path
import json
import numpy as np
from scipy import fft
index=json.load(open('out/v5/conditioned_wave_v1/index.json')); root=Path(index['source_cache'])
energy={r:np.zeros(1201,np.float64) for r in ('train','validation')}; rows=[]
for row in index['records']:
    iq=np.load(root/row['iq'],mmap_mode='r')
    spectrum=fft.fftshift(fft.fft(iq[:,0]+1j*iq[:,1],axis=-1,norm='forward'),axes=-1)/np.sqrt(2)
    local=np.zeros(1201,np.float64); local[40:680]=np.sum(abs(spectrum)**2,axis=(0,1),dtype=np.float64)
    energy[row['role']]+=local
    rows.append({'id':row['id'],'role':row['role'],'nominal_fraction':float(local[241:451].sum()/local.sum()),
        'extended_fraction':float(local[241:535].sum()/local.sum())})
summary={role:{'nominal_fraction':float(values[241:451].sum()/values.sum()),
    'extended_fraction':float(values[241:535].sum()/values.sum())} for role,values in energy.items()}
result={'summary':summary,'rows':rows,'bins':list(range(241,535)),'angles':11}
out=Path('out/v5/conditioned_dense_full_v1'); out.mkdir(parents=True,exist_ok=True)
(out/'spectral_energy.json').write_text(json.dumps(result,indent=2)); np.savez(out/'spectral_energy.npz',**energy)
print(json.dumps(summary,indent=2))
