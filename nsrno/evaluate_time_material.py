"""Post-hoc map evaluation; never imported by reconstruction_time."""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from .v5_materials import evaluation_material

def main():
    parser=ArgumentParser();parser.add_argument('material',type=Path);parser.add_argument('--sample',default='train_200')
    parser.add_argument('--out',type=Path);parser.add_argument('--estimate-label',default='Reconstruction');args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    m=np.load(args.material);baseline=args.material.parent/'uniform_initial_material.npy'
    base=np.load(baseline if baseline.exists() else args.material.parent/'stage0_initial_material.npy')
    target=evaluation_material(args.sample)
    z=(np.arange(m.shape[1])-60)*.05;x=(np.arange(m.shape[2])-560)*.05
    mask=(np.abs(x)[None,:]<=19.1)&(z[:,None]>=1)&(z[:,None]<=43)
    def stats(a,b,region):
        av=a[region].astype(np.float64);bv=b[region].astype(np.float64)
        return {'rmse':float(np.sqrt(np.mean((av-bv)**2))),'mae':float(np.mean(np.abs(av-bv))),
                'correlation':float(np.corrcoef(av,bv)[0,1]) if av.std()>1e-10 and bv.std()>1e-10 else None}
    names=['c_m_s','density_kg_m3','alpha_dB_cm_MHz1p1']
    report={'material':str(args.material),'sample':args.sample,'evaluation_only':True,'estimate_label':args.estimate_label,'depth_mm':[1,43],'maps':{}}
    for j,name in enumerate(names):
        report['maps'][name]={'initial':stats(base[j],target[j],mask),'reconstruction':stats(m[j],target[j],mask),'by_depth':{}}
        for lo,hi in [(1,5),(5,12),(12,25),(25,43)]:
            region=mask&(z[:,None]>=lo)&(z[:,None]<hi)
            report['maps'][name]['by_depth'][f'{lo}-{hi}']={'initial':stats(base[j],target[j],region),'reconstruction':stats(m[j],target[j],region)}
    prefix=args.out or args.material.with_suffix('')
    prefix.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(3,3,figsize=(11,12),layout='constrained')
    ranges=[(1450,1610),(930,1140),(0,1.05)];error_range=[100,120,.7];titles=['Sound speed (m/s)','Density (kg/m³)','Attenuation [dB/(cm MHz^1.1)]']
    iz=(z>=0)&(z<=43);ix=abs(x)<=19.1
    for j in range(3):
        for col,image in enumerate([target[j],m[j],m[j]-target[j]]):
            ax=axes[j,col];er=error_range[j]
            im=ax.imshow(image[iz][:,ix],extent=[x[ix][0],x[ix][-1],z[iz][-1],z[iz][0]],aspect='equal',cmap='RdBu_r' if col==2 else 'viridis',vmin=-er if col==2 else ranges[j][0],vmax=er if col==2 else ranges[j][1],interpolation='none')
            ax.set_title((['Truth',args.estimate_label,'Error'][col]+'\n'+titles[j]),fontsize=10);ax.set_xlabel('Lateral (mm)');ax.set_ylabel('Depth (mm)');fig.colorbar(im,ax=ax,shrink=.72)
    fig.suptitle(f'{args.sample}: {args.material.stem}\nMaps are evaluated after optimization; RF fit alone does not establish recovery',fontsize=11)
    fig.savefig(prefix.with_suffix('.png'),dpi=160);plt.close(fig)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
