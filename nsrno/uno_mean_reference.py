"""Training-mean RF reference using the same complete acquisition; no new simulations."""
from pathlib import Path
import json,math
import numpy as np
from scipy import fft
from nsrno.dataset import write_json

def decode_np(iq):
    complex_iq=iq[...,0,:,:]+1j*iq[...,1,:,:]
    selected=fft.fftshift(fft.fft(complex_iq,axis=-1,norm='forward'),axes=-1)/np.sqrt(2)
    spectrum=np.zeros((*selected.shape[:-1],1201),np.complex64); spectrum[...,40:680]=selected
    return fft.irfft(spectrum,n=2401,axis=-1,norm='forward')

def main():
    root=Path('out/v5/uno_triparam_v1'); index=json.load(open(root/'index.json')); rows=index['records']
    means={}
    for source in [3.,7.5]:
        selected=[r for r in rows if r['role']=='train' and r['source_mhz']==source]
        path=root/f'training_mean_{source}.npy'
        if path.exists(): means[source]=np.load(path); continue
        value=np.zeros((11,2,192,640),np.float64)
        for r in selected: value+=np.load(r['iq'])
        means[source]=(value/len(selected)).astype(np.float32); np.save(path,means[source])
    extra=root/'augmentation_index.json'
    if extra.exists(): rows+=json.load(open(extra))['records']
    stats={}; records=[]
    for row in rows:
        if row['role']!='validation': continue
        prediction=decode_np(means[row['source_mhz']]); target=decode_np(np.load(row['iq']))
        difference=prediction.astype(np.float64)-target; values=[]
        for lo,hi in [(0,2401),(0,320),(320,720),(720,1440),(1440,2401)]:
            values.append([float(np.sum(difference[...,lo:hi]**2)),float(np.sum(target[...,lo:hi].astype(np.float64)**2))])
        values=np.array(values); key=str(row['source_mhz']); stats.setdefault(key,np.zeros((5,2))); stats[key]+=values
        records.append({'id':row['id'],'relative_rms':np.sqrt(values[:,0]/np.maximum(values[:,1],1e-30)).tolist()})
    total=sum(stats.values()); result={'records':len(records),'method':'training-only mean IQ for each known source and angle',
        'relative_rms':np.sqrt(total[:,0]/total[:,1]).tolist(),'by_source':{k:np.sqrt(v[:,0]/v[:,1]).tolist() for k,v in stats.items()},'per_record':records}
    write_json(root/'training_mean_reference.json',result); print(json.dumps({k:v for k,v in result.items() if k!='per_record'},indent=2))

if __name__=='__main__': main()
