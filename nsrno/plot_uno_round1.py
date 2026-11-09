"""Scientific figures of the actual pure-UNO forward results, not inverse images."""
from pathlib import Path
import json,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from nsrno.uno_mean_reference import decode_np

def main():
    font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True)
    font_manager.fontManager.addfont(font); plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    root=Path('out/v5/uno_triparam_v1'); out=Path('out/v5/uno_round1_triparam')
    evaluation=json.load(open(out/'forward_evaluation.json')); validations=json.load(open(out/'validations.json'))
    profile=json.load(open(root/'memory_profile.json')); done=json.load(open(out/'training_complete.json'))
    mean=json.load(open(root/'training_mean_reference.json'))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for source,label in [('7.5','原 7.5 MHz 源'),('3.0','3 MHz 源')]:
        values=[]
        fixed_ids={'hf_train_200','hf_train_250','hf_train_300','lf_train_200'}
        for v in validations:
            fixed=[r for r in v['per_record'] if r['id'] in fixed_ids and str(r['source_mhz'])==source]
            if fixed:
                rms=np.sqrt(sum(r['error_sq'][0] for r in fixed)/sum(r['target_sq'][0] for r in fixed))
                values.append((v['step'],rms))
        if values: axes[0].plot(*zip(*values),marker='o',label=label)
    axes[0].axvline(1500,color='grey',ls=':',label='已知源相位表示变换')
    axes[0].axvline(2000,color='grey',ls='--',label='加入独立三参数数据')
    axes[0].set(xlabel='累计优化步',ylabel='完整 RF 相对 RMS',title='固定开发观测的正演误差'); axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)
    labels=['完整 RF','0–8 μs','8–18 μs','18–36 μs','36–60 μs']; x=np.arange(5)
    axes[1].bar(x-.18,evaluation['relative_rms'],width=.36,label='纯 UNO')
    axes[1].bar(x+.18,mean['relative_rms'],width=.36,label='仅训练集平均波形')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='相对 RF RMS',title=f'全部 {evaluation["records"]} 条开发记录 × 11 角'); axes[1].tick_params(axis='x',rotation=25)
    axes[1].legend(); axes[1].grid(axis='y',alpha=.25)
    fig.suptitle('纯 UNO 通道正演：实际训练和完整观测评价')
    fig.savefig(out/'forward_results.png',dpi=180); fig.savefig(out/'forward_results.pdf'); plt.close(fig)
    phases=profile['phases']; names=list(phases)
    fig,ax=plt.subplots(figsize=(9,4.7),constrained_layout=True); x=np.arange(len(names)); width=.24
    for j,(key,label) in enumerate([('allocated_gib','PyTorch 分配峰值'),('reserved_gib','PyTorch 保留峰值'),('sampled_process_gib','进程显存采样峰值')]):
        vals=[phases[name][key] for name in names]; bars=ax.bar(x+(j-1)*width,vals,width,label=label)
        ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
    ax.set(xticks=x,xticklabels=['单角纯正演','单角训练步','全部 11 角三参数反传'],ylabel='GiB',ylim=(0,max(1.5,*[v['reserved_gib']*1.35 for v in phases.values()])))
    ax.legend(fontsize=9); ax.grid(axis='y',alpha=.25)
    ax.set_title(f'独立显存测量；实际训练全程 PyTorch 保留峰值 {done["reserved_peak_gib"]:.2f} GiB\n进程采样可能漏掉短暂峰值；三参数图为原生 961×1121')
    fig.savefig(out/'memory.png',dpi=180); plt.close(fig)
    rows=json.load(open(root/'index.json'))['records']+json.load(open(root/'augmentation_index.json'))['records']
    lookup={r['id']:r for r in rows}; time=np.arange(2401)/40
    for identity in ['hf_train_200','lf_train_200']:
        prediction_path=out/'predictions'/(identity+'.prediction.npy')
        if identity not in lookup or not prediction_path.exists(): continue
        row=lookup[identity]; observed=decode_np(np.load(row['iq'])); predicted=decode_np(np.load(prediction_path))
        average=decode_np(np.load(root/f'training_mean_{row["source_mhz"]}.npy')); a=5
        fig,axes=plt.subplots(1,4,figsize=(13,5.2),constrained_layout=True); limit=float(abs(observed[a]).max())
        for ax,values,title in zip(axes,[observed[a],average[a],predicted[a],predicted[a]-observed[a]],['观测 RF','训练集平均 RF','纯 UNO 预测','预测 − 观测']):
            im=ax.imshow(values.T,extent=[-.5,191.5,time[-1],0],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
            ax.set(xlabel='接收阵元',ylabel='时间 (μs)',title=title,ylim=(36,0)); ax.axhline(18,color='black',ls='--',lw=.5)
        fig.colorbar(im,ax=axes.tolist(),label='RF 原始幅值',shrink=.7)
        fig.suptitle(f'{row["source_mhz"]} MHz 源、0° 的正演示例；输入真实 c / ρ / α\n这是正演预测，不是三参数重建；定量误差使用全部角度')
        fig.savefig(out/(identity+'_rf.png'),dpi=180); plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(10,4.8),constrained_layout=True)
        for ax,(lo,hi) in zip(axes,[(0,8),(18,36)]):
            mask=(time>=lo)&(time<hi)
            for data,label in [(observed,'观测'),(predicted,'纯 UNO'),(average,'训练集平均')]: ax.plot(time[mask],data[a,96,mask],label=label,lw=.8)
            ax.set(xlabel='时间 (μs)',ylabel='原始 RF',xlim=(lo,hi)); ax.grid(alpha=.25)
        axes[0].legend(ncol=3); fig.suptitle(f'{row["source_mhz"]} MHz 源：0° / 第 96 阵元，两个时间窗独立纵轴')
        fig.savefig(out/(identity+'_traces.png'),dpi=180); plt.close(fig)
    example=next((p for p in evaluation['alpha_pairs'] if p['source_mhz']==7.5),None)
    if example:
        matched=sorted([r for r in rows if r.get('pair_group')==example['group'] and r['source_mhz']==7.5],key=lambda r:r['pair_sign'])
        observed=decode_np(np.load(matched[1]['iq'])-np.load(matched[0]['iq']))
        predicted=decode_np(np.load(out/'predictions'/(matched[1]['id']+'.prediction.npy'))-np.load(out/'predictions'/(matched[0]['id']+'.prediction.npy')))
        fig,axes=plt.subplots(1,3,figsize=(10.5,5.2),constrained_layout=True); limit=float(abs(observed[5]).max())
        for ax,values,title in zip(axes,[observed[5],predicted[5],predicted[5]-observed[5]],['仅衰减变化：真实 ΔRF','纯 UNO 的 ΔRF','预测差分 − 真实差分']):
            im=ax.imshow(values.T,extent=[-.5,191.5,time[-1],0],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
            ax.set(xlabel='接收阵元',ylabel='时间 (μs)',title=title,ylim=(36,0))
        fig.colorbar(im,ax=axes.tolist(),shrink=.7,label='RF 差分原始幅值')
        fig.suptitle(f'独立开发材料，c / ρ 完全相同；只改变组织衰减\n7.5 MHz 源的 0° 示例；该配对全部 11 角差分相对误差 {example["relative_rms"]:.3f}')
        fig.savefig(out/'alpha_response.png',dpi=180); plt.close(fig)
    (out/'plots_complete.json').write_text(json.dumps({'completed':True}))

if __name__=='__main__': main()
