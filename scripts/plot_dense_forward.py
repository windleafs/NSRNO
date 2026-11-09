from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.plot_reconstruction import metric

out=Path('out/v5/forward_round8_dense_full')
if not (out/'forward_summary.json').exists(): raise RuntimeError('Complete full-band summary first')
rows=[]
for part in range(3): rows.extend(json.load(open(out/f'forward_part{part}.json')))
fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
labels={'before':'训练前，32 步','after':'训练后，32 步','after64':'训练后，64 步','converged':'继续求解至收敛'}
for ax,sid in zip(axes,('train_000','train_200')):
    methods=('before','after') if sid=='train_000' else tuple(labels)
    for name in methods:
        values=sorted([r for r in rows if r['id']==sid and r['method']==name],key=lambda r:r['bin'])
        ax.semilogy([r['frequency_hz']/1e6 for r in values],[r['rf_relative_rms'] for r in values],label=labels[name])
    ax.axvline(7.5,color='grey',ls=':',lw=1)
    ax.set(xlabel='RF 频率 (MHz)',ylabel='11 角合并的通道频谱相对 RMS',title=('训练病例固定样本' if sid=='train_000' else '第二开发病例固定样本')+' / '+sid)
    ax.legend(); ax.grid(alpha=.25)
fig.suptitle('全部 11 个角度、294 个相邻频点的正演误差：已知真实衰减图',fontsize=14)
fig.savefig(out/'forward_error.png',dpi=180); fig.savefig(out/'forward_error.pdf'); plt.close(fig)

index=json.load(open('out/v5/forward_cache_v1/index.json')); row=next(r for r in index['records'] if r['id']=='train_200')
iq=np.load(Path('out/v5/forward_cache_v1')/row['iq'])
spectrum=np.zeros((11,192,1201),np.complex64)
spectrum[:,:,40:680]=np.fft.fftshift(np.fft.fft(iq[:,0]+1j*iq[:,1],axis=-1,norm='forward'),axes=-1)/np.sqrt(2)
observed=np.fft.irfft(spectrum,n=2401,axis=-1,norm='forward')
predictions={}
for name in labels:
    values=np.zeros_like(spectrum)
    for b in range(241,535): values[:,:,b]=np.load(out/f'prediction_train_200_{name}_b{b}.npy')
    predictions[name]=np.fft.irfft(values,n=2401,axis=-1,norm='forward')
time=np.arange(2401)/40e6*1e6
fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
for ax,(a,r) in zip(axes.ravel(),[(0,48),(5,95),(10,143),(5,95)]):
    ax.plot(time,observed[a,r],color='black',lw=1.2,label='V5 观测 RF')
    for name in ('after','after64','converged'):
        ax.plot(time,predictions[name][a,r],lw=.8,alpha=.85,label=labels[name])
    ax.set(xlabel='时间 (μs)',ylabel='RF 幅值',title=f'角度索引 {a} / 阵元 {r}',xlim=(18,36) if ax is axes[1,1] else (0,36))
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=9)
fig.suptitle('固定阵元的完整 RF 对照（预测保留 294 个频点，其余频率置零）',fontsize=14)
fig.savefig(out/'forward_rf_examples.png',dpi=180); plt.close(fig)
