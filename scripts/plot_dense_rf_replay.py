"""Plots from the already computed independent full-RF replay; no new solves."""
from pathlib import Path
import json
import numpy as np
from scipy import fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.plot_reconstruction import metric  # registers the Chinese font

out=Path('out/v5/reconstruction_round2_dense_train_200')
observed=np.load(out/'observed_rf.npy')
initial=np.load(out/'initial_rf.npy')
result=np.load(out/'reconstruction_rf.npy')
selection=json.load(open(out/'selected.json'))
fs=40e6; time=np.arange(observed.shape[-1])/fs*1e6
f=fft.rfftfreq(observed.shape[-1],1/fs)/1e6
spectra=[fft.rfft(a,axis=-1) for a in (observed,initial,result)]
power=np.sum(abs(spectra[0].astype(np.complex128))**2,axis=(0,1))
band=(f>=4)&(f<=9)
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
axes[0].semilogy(f[band],power[band]/power.sum())
axes[0].set(xlabel='频率 (MHz)',ylabel='每个频点占总频谱能量的比例',title='观测 RF 的频谱能量')
for spectrum,label in zip(spectra[1:],('均匀初值','联合重建')):
    error=np.sum(abs(spectrum.astype(np.complex128)-spectra[0])**2,axis=(0,1))
    axes[1].plot(f[band],np.sqrt(error[band]/power[band]),label=label)
axes[1].set(xlabel='频率 (MHz)',ylabel='相对 RF RMS',title='全部 11 角、192 阵元的独立时域复算')
axes[1].legend()
for ax in axes: ax.grid(alpha=.25)
fig.savefig(out/'rf_spectrum.png',dpi=180); plt.close(fig)

# A fixed central angle is shown for visualization only; metrics use all angles.
a=5; arrays=(observed[a],initial[a],result[a],result[a]-observed[a])
fig,axes=plt.subplots(1,4,figsize=(14,6),constrained_layout=True)
limit=float(np.max(abs(observed[a])))
for ax,array,title in zip(axes,arrays,('观测 RF','均匀初值','联合重建 RF','重建 RF − 观测 RF')):
    im=ax.imshow(array.T,extent=[-.5,191.5,time[-1],time[0]],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
    ax.axhline(18,color='black',ls='--',lw=.6)
    ax.axhline(36,color='black',ls='--',lw=.6)
    ax.set(xlabel='接收阵元',ylabel='时间 (μs)',title=title)
fig.colorbar(im,ax=axes.tolist(),label='RF（原始幅值）',shrink=.7)
fig.suptitle(f'0° 角 RF 示例（显示范围 0–60 μs）；第 {selection["step"]} 次材料更新\n所有定量误差均使用全部 11 角；水平虚线为 18 和 36 μs')
fig.savefig(out/'rf_replay.png',dpi=180); plt.close(fig)

fig,axes=plt.subplots(2,1,figsize=(11,5),constrained_layout=True)
for ax,(lo,hi) in zip(axes,((0,18),(18,36))):
    mask=(time>=lo)&(time<=hi)
    for data,label,lw in ((observed,'观测',1.3),(initial,'均匀初值',.8),(result,'联合重建',.8)):
        ax.plot(time[mask],data[a,96,mask],label=label,lw=lw)
    ax.set(xlabel='时间 (μs)',ylabel='RF（原始幅值）',xlim=(lo,hi)); ax.grid(alpha=.25)
axes[0].legend(ncol=3); fig.suptitle('固定 0° / 第 96 接收阵元的波形示例（两幅使用各自纵轴）')
fig.savefig(out/'rf_traces.png',dpi=180); plt.close(fig)
