"""Post-hoc material truth evaluation and unembellished reconstruction plots."""
from pathlib import Path
from argparse import ArgumentParser
import json,math,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from nsrno.reconstruct_material import DEFAULT_OUT,write_json

font_path=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
font_manager.fontManager.addfont(font_path)
font_name=font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({'font.family':font_name,'font.size':11,'axes.unicode_minus':False})

def metric(estimate,true,initial):
    difference=estimate.astype(np.float64)-true
    base=initial.astype(np.float64)-true
    corr=float(np.corrcoef(estimate.ravel(),true.ravel())[0,1]) if np.std(estimate,dtype=np.float64)>1e-5 and np.std(true,dtype=np.float64)>1e-5 else None
    return {'rmse':float(np.sqrt(np.mean(difference**2))),'mae':float(np.mean(abs(difference))),
        'bias':float(np.mean(difference)),'correlation':corr,
        'contrast_relative_error':float(np.linalg.norm(difference)/max(np.linalg.norm(base),1e-30)),
        'mse_improvement_over_initial':float(1-np.sum(difference**2)/max(np.sum(base**2),1e-30)),
        'min':float(estimate.min()),'max':float(estimate.max())}

def main():
    parser=ArgumentParser(); parser.add_argument('--out',type=Path,default=DEFAULT_OUT); args=parser.parse_args(); out=args.out
    if not (out/'completion.json').exists(): raise RuntimeError('Material truth is only evaluated after optimization')
    settings=json.load(open(out/'settings.json')); spec=settings['native_grid']
    index=json.load(open('out/v5/forward_cache_v1/index.json')); row=next(r for r in index['records'] if r['id']==settings['id'])
    truth=np.load(Path('out/v5/forward_cache_v1')/row['medium'])
    initial=np.load(out/'initial_medium.npy'); recon=np.load(out/'reconstructed_medium.npy'); alpha=np.load(out/'known_alpha.npy')
    h,w=truth.shape[-2:]; x=np.arange(w)*spec['dx_m']+spec['x0_m']; z=np.arange(h)*spec['dx_m']+spec['z0_m']
    ix=np.where(abs(x)<=.0191+1e-8)[0]; iz=np.where((z>=-1e-8)&(z<=.043+1e-8))[0]
    def crop(m): return m[...,iz[0]:iz[-1]+1,ix[0]:ix[-1]+1]
    t,i,r=crop(truth),crop(initial),crop(recon)
    dx=spec['dx_m']*1000; xx=x[ix]*1000; zz=z[iz]*1000
    extent=[xx[0]-dx/2,xx[-1]+dx/2,zz[-1]+dx/2,zz[0]-dx/2]
    summary={'id':settings['id'],'fov_mm':{'x':[-19.1,19.1],'z':[0,43]},'parameters':{},'depth_bands':{},
        'known_true_alpha':True,'truth_loaded_only_after_optimization':True}
    for k,name in enumerate(('c','rho')):
        summary['parameters'][name]={'initial':metric(i[k],t[k],i[k]),'reconstruction':metric(r[k],t[k],i[k])}
        bands=[]
        for lo,hi in [(0,10),(10,20),(20,30),(30,43.001)]:
            select=(zz>=lo-1e-6)&(zz<hi)
            bands.append({'depth_mm':[lo,min(hi,43)],'initial':metric(i[k,select],t[k,select],i[k,select]),
                'reconstruction':metric(r[k,select],t[k,select],i[k,select])})
        summary['depth_bands'][name]=bands
    write_json(out/'material_metrics.json',summary)
    selected=json.load(open(out/'selected.json')); history=json.load(open(out/'history.json'))
    fig,axes=plt.subplots(2,4,figsize=(16,10),constrained_layout=True)
    labels=[('声速 c','m/s',(1400,1700),150),('质量密度 ρ','kg/m³',(850,1200),150)]
    for k,(name,unit,limits,err_limit) in enumerate(labels):
        arrays=[t[k],i[k],r[k],r[k]-t[k]]
        titles=['真值（仅评价使用）','均匀初值',f'联合重建（第 {selected["step"]} 次更新）','重建 − 真值']
        for j,(array,title) in enumerate(zip(arrays,titles)):
            ax=axes[k,j]
            if j<3:
                im=ax.imshow(array,extent=extent,cmap='turbo',vmin=limits[0],vmax=limits[1],interpolation='nearest',aspect='equal')
            else:
                im=ax.imshow(array,extent=extent,cmap='RdBu_r',vmin=-err_limit,vmax=err_limit,interpolation='nearest',aspect='equal')
            ax.set_title(title,fontsize=12); ax.set_xlabel('横向位置 (mm)'); ax.set_ylabel('深度 (mm)')
            ax.set_xlim(-19.1,19.1); ax.set_ylim(43,0)
            if j in (1,2):
                value=summary['parameters'][('c','rho')[k]][('initial' if j==1 else 'reconstruction')]['rmse']
                ax.text(.02,.98,f'RMSE = {value:.2f} {unit}',transform=ax.transAxes,va='top',fontsize=10,
                        bbox=dict(facecolor='white',alpha=.88,edgecolor='none'))
            if j==2: fig.colorbar(im,ax=axes[k,:3].tolist(),shrink=.78,pad=.02,label=f'{name} ({unit})')
            if j==3: fig.colorbar(im,ax=ax,shrink=.78,pad=.02,label=f'误差 ({unit})')
    gel_note=f'固定 {settings.get("fixed_gel_mm",0):g} mm 已知耦合层'
    fig.suptitle('首次声速 / 密度联合反演：Neg_35_Left / train_200\n已知真实衰减图；3 个频率 × 3 个角度；'+gel_note+'；未按材料真值选择结果',fontsize=16)
    fig.savefig(out/'reconstruction.png',dpi=180); fig.savefig(out/'reconstruction.pdf'); plt.close(fig)
    # Preserve full float32 arrays; no display smoothing or anatomy overlays.
    np.savez_compressed(out/'reconstruction_arrays.npz',c_true=t[0],rho_true=t[1],c_initial=i[0],rho_initial=i[1],
        c_reconstruction=r[0],rho_reconstruction=r[1],known_alpha=crop(alpha),x_mm=xx,z_mm=zz)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    steps=[v['step'] for v in history]
    axes[0].semilogy(steps,[v['fit_rms'] for v in history],label='用于优化的 168 个阵元')
    axes[0].semilogy(steps,[v['heldout_rms'] for v in history],label='留出的 24 个阵元')
    axes[0].set(xlabel='材料更新次数',ylabel='相对频谱 RMS 误差',title='冻结神经正演模型下的数据拟合'); axes[0].legend(); axes[0].grid(alpha=.25)
    im=axes[1].imshow(crop(alpha),extent=extent,cmap='magma',vmin=0,vmax=1.,interpolation='nearest',aspect='equal')
    axes[1].set(xlabel='横向位置 (mm)',ylabel='深度 (mm)',title='已知条件：真实衰减图')
    fig.colorbar(im,ax=axes[1],label='dB/(cm MHz^1.1)')
    for k,name in enumerate(('c','rho')):
        band=summary['depth_bands'][name]
        axes[2].plot([(v['depth_mm'][0]+v['depth_mm'][1])/2 for v in band],
                     [v['reconstruction']['contrast_relative_error'] for v in band],marker='o',label=('声速' if k==0 else '密度'))
    axes[2].axhline(1,color='grey',ls='--',label='均匀初值')
    axes[2].set(xlabel='深度 (mm)',ylabel='误差 / 均匀初值误差',title='材料恢复是否优于初值',ylim=(.8,1.2)); axes[2].legend(); axes[2].grid(alpha=.25)
    fig.savefig(out/'diagnostics.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,5.7),constrained_layout=True)
    for k,ax in enumerate(axes):
        im=ax.imshow(r[k]-i[k],extent=extent,cmap='RdBu_r',vmin=-15,vmax=15,interpolation='nearest',aspect='equal')
        unit=('m/s','kg/m³')[k]
        ax.set(xlabel='横向位置 (mm)',ylabel='深度 (mm)',title=('声速增量 Δc','密度增量 Δρ')[k])
        fig.colorbar(im,ax=ax,label=unit,shrink=.85)
    fig.suptitle('重建相对均匀初值的变化（独立色标；不是材料真值）',fontsize=13)
    fig.savefig(out/'updates.png',dpi=180); plt.close(fig)
    print(json.dumps(summary['parameters'],indent=2),flush=True)

if __name__=='__main__': main()
