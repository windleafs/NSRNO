"""Data-only separation of absolute RF time and midpoint-equivalent depth."""
from pathlib import Path
import json,os
import numpy as np


def main():
    root=Path('out/v5/point_tx_3mhz_v1');meta=json.loads((root/'index.json').read_text());a=meta['acquisition']
    row=next(r for r in meta['records'] if r['id']=='train_200');rf=np.load(row['rf']);energy=rf.astype(np.float64)**2
    offset=abs(np.asarray(a['transmit_x_m'])[:,None]-np.asarray(a['receiver_x_m'])[None,:])
    t=np.arange(rf.shape[-1])/a['fs_hz'];tref=np.asarray(a['source_tref_s'])[:,None]
    bins=np.zeros(rf.shape,np.uint8)
    for depth in [.005,.012,.025,.043]:
        boundary=tref+np.sqrt(offset**2+(2*depth)**2)/1540.
        bins+=(t[None,None,:]>=boundary[:,:,None])
    names=['earlier_than_midpoint_5mm','midpoint_5_to_12mm','midpoint_12_to_25mm','midpoint_25_to_43mm','later_or_off_midpoint']
    result={'sample':'train_200','uses_material_truth':False,
        'meaning':'Homogeneous midpoint-equivalent arrival classes, NOT true localization; off-midpoint shallow scattering can arrive late and heterogeneous speeds move boundaries.',
        'windows':{}}
    for lo,hi in [(0,8),(8,18),(18,36),(36,60),(60,90.025),(36,90.025)]:
        sl=slice(round(lo*40),round(hi*40));e=energy[:,:,sl];b=bins[:,:,sl]
        total=float(e.sum());result['windows'][f'{lo}-{hi}']={'fraction_of_total_energy':total/float(energy.sum()),
            'nominal_arrival_energy_fraction':{name:float(e[b==k].sum())/max(total,1e-30) for k,name in enumerate(names)}}
    out=Path('out/v5/encoded_rf_v1/point_arrival_energy.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    locations=np.minimum((offset*1e3).astype(int),38);image=np.zeros((39,rf.shape[-1]))
    for i in range(39):
        traces=energy[locations==i]
        if len(traces):image[i]=traces.mean(0)
    db=10*np.log10(np.maximum(image/max(float(image.max()),1e-30),1e-9))
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained');im=ax.imshow(db.T,origin='lower',extent=[0,39,0,90],aspect='auto',vmin=-65,vmax=0,cmap='magma')
    d=np.linspace(0,39,400)*1e-3
    for z in [1,5,12,25,43]:
        arrival=(4/3e6+np.sqrt(d*d+(2*z*1e-3)**2)/1540)*1e6
        ax.plot(d*1e3,arrival,linewidth=1,label=f'midpoint {z} mm')
    ax.set(xlabel='Absolute transmitter-receiver offset (mm)',ylabel='RF time (microseconds)',title='3 MHz diverging RF energy; curves assume homogeneous midpoint reflection')
    ax.legend(fontsize=8);fig.colorbar(im,ax=ax,label='Mean RF energy (dB relative to maximum)');fig.savefig(out.with_suffix('.png'),dpi=150);plt.close(fig)


if __name__=='__main__':main()
