"""Final material figures; true maps are loaded only after reconstruction."""
from pathlib import Path
import json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.plot_reconstruction import metric
from nsrno.reconstruct_material import write_json

out=Path('out/v5/reconstruction_round2_dense_train_200')
if not (out/'completion.json').exists(): raise RuntimeError('Material truth cannot be used before reconstruction completes')
settings=json.load(open(out/'settings.json')); spec=settings['native_grid']; completion=json.load(open(out/'completion.json')); selection=json.load(open(out/'selected.json'))
index=json.load(open('out/v5/forward_cache_v1/index.json')); row=next(r for r in index['records'] if r['id']=='train_200')
truth=np.load(Path('out/v5/forward_cache_v1')/row['medium']); initial=np.load(out/'initial_medium.npy'); result=np.load(out/'reconstructed_medium.npy')
h,w=truth.shape[-2:]; x=np.arange(w)*spec['dx_m']+spec['x0_m']; z=np.arange(h)*spec['dx_m']+spec['z0_m']
ix=np.where(abs(x)<=.0191+1e-8)[0]; iz=np.where((z>=-1e-8)&(z<=.043+1e-8))[0]
def crop(m): return m[...,iz[0]:iz[-1]+1,ix[0]:ix[-1]+1]
t,i,r=crop(truth),crop(initial),crop(result); xx=x[ix]*1000; zz=z[iz]*1000; dx=spec['dx_m']*1000
extent=[xx[0]-dx/2,xx[-1]+dx/2,zz[-1]+dx/2,zz[0]-dx/2]
metrics={'id':'train_200','known_true_alpha':True,'truth_loaded_only_after_optimization':True,'parameters':{},'depth_bands':{}}
for k,name in enumerate(('c','rho')):
    metrics['parameters'][name]={'initial':metric(i[k],t[k],i[k]),'reconstruction':metric(r[k],t[k],i[k])}
    metrics['depth_bands'][name]=[]
    for lo,hi in [(0,10),(10,20),(20,30),(30,43.001)]:
        mask=(zz>=lo-1e-6)&(zz<hi)
        metrics['depth_bands'][name].append({'depth_mm':[lo,min(hi,43)],'initial':metric(i[k,mask],t[k,mask],i[k,mask]),'reconstruction':metric(r[k,mask],t[k,mask],i[k,mask])})
write_json(out/'material_metrics.json',metrics)
fig,axes=plt.subplots(2,4,figsize=(16,10),constrained_layout=True)
for k,(name,unit,limits) in enumerate([('声速 c','m/s',(1400,1700)),('质量密度 ρ','kg/m³',(850,1200))]):
    for j,array in enumerate((t[k],i[k],r[k],r[k]-t[k])):
        ax=axes[k,j]; cmap='turbo' if j<3 else 'RdBu_r'; vmin,vmax=limits if j<3 else (-150,150)
        im=ax.imshow(array,extent=extent,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest',aspect='equal')
        ax.set(xlabel='横向位置 (mm)',ylabel='深度 (mm)',xlim=(-19.1,19.1),ylim=(43,0))
        ax.set_title(['真值（仅评价使用）','均匀初值',f'联合重建（第 {selection["step"]} 次更新）','重建 − 真值'][j])
        if j in (1,2):
            rms=metrics['parameters'][('c','rho')[k]][('initial' if j==1 else 'reconstruction')]['rmse']
            ax.text(.02,.98,f'RMSE = {rms:.2f} {unit}',transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
        if j==2: fig.colorbar(im,ax=axes[k,:3].tolist(),label=f'{name} ({unit})',shrink=.8,pad=.02)
        if j==3: fig.colorbar(im,ax=ax,label=f'误差 ({unit})',shrink=.8,pad=.02)
fig.suptitle('全角度、密集频带的声速 / 密度联合重建：train_200\n294 个相邻频点 × 11 个角度；收敛物理正演与隐式伴随；已知真实衰减图、1 mm 耦合层',fontsize=15)
fig.savefig(out/'reconstruction.png',dpi=180); fig.savefig(out/'reconstruction.pdf'); plt.close(fig)
np.savez_compressed(out/'reconstruction_arrays.npz',c_true=t[0],rho_true=t[1],c_initial=i[0],rho_initial=i[1],c_reconstruction=r[0],rho_reconstruction=r[1],x_mm=xx,z_mm=zz,known_alpha=crop(np.load(out/'known_alpha.npy')))
fig,axes=plt.subplots(1,2,figsize=(9,5.7),constrained_layout=True)
for k,ax in enumerate(axes):
    limit=50 if k==0 else 40
    im=ax.imshow(r[k]-i[k],extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest',aspect='equal')
    ax.set(xlabel='横向位置 (mm)',ylabel='深度 (mm)',title=('声速增量 Δc','密度增量 Δρ')[k]); fig.colorbar(im,ax=ax,label=('m/s','kg/m³')[k])
fig.suptitle('相对均匀初值的材料更新（独立色标：±50 m/s、±40 kg/m³）'); fig.savefig(out/'updates.png',dpi=180); plt.close(fig)
history=json.load(open(out/'history.json')); fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
axes[0].semilogy([v['step'] for v in history],[v['sampled_full_band_loss'] for v in history]); axes[0].set(xlabel='材料更新次数',ylabel='按频率批次计算的数据误差平方',title='频率逐步增加，批次之间不代表同一损失'); axes[0].grid(alpha=.25)
for name,label in [('c','声速'),('rho','密度')]:
    values=metrics['depth_bands'][name]
    axes[1].plot([(a['depth_mm'][0]+a['depth_mm'][1])/2 for a in values],[a['reconstruction']['contrast_relative_error'] for a in values],marker='o',label=label)
axes[1].axhline(1,color='grey',ls='--',label='均匀初值'); axes[1].set(xlabel='深度 (mm)',ylabel='材料误差 / 均匀初值误差',ylim=(0,max(1.2,*[1.1*a['reconstruction']['contrast_relative_error'] for values in metrics['depth_bands'].values() for a in values])),title='材料恢复随深度的变化'); axes[1].legend(); axes[1].grid(alpha=.25)
fig.savefig(out/'diagnostics.png',dpi=180); plt.close(fig)
print(json.dumps(metrics['parameters'],indent=2),flush=True)
