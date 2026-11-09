"""Summarize the late-RF experiment without choosing a new checkpoint criterion."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/home_nvme/zhuangyang/nsrno/out/v5')
out=root/'forward_round4_summary'; out.mkdir(exist_ok=True)
data={}
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for mode,color,label in [('reference','#9b6333','Fixed-speed ray delays'),('material','#177f72','Input-dependent ray delays')]:
    folder=root/('forward_round4_'+mode)
    run={key:json.loads((folder/file).read_text()) for key,file in
         [('settings','run.json'),('completion','completion.json'),('validation','validation_all_angles.json'),
          ('training_best','training_best_all_angles.json'),('training_last','training_last_all_angles.json')]}
    history=json.loads((folder/'history.json').read_text())
    rows=[r for r in history if 'validation' in r]
    run['history_monitor']=rows
    run['training_and_monitor_seconds']=sum(r['seconds'] for r in history)
    data[mode]=run
    if (folder/'correction_alignment.json').exists():
        run['correction_alignment']=json.loads((folder/'correction_alignment.json').read_text())
    for col,role in enumerate(('training_monitor','validation')):
        for row,window in enumerate((2,3)):
            axes[row,col].plot([r['epoch'] for r in rows],[r[role]['model']['window_relative_rf_rms'][window] for r in rows],color=color,label=label)
            if mode=='reference':
                axes[row,col].axhline(rows[0][role]['frozen_baseline']['window_relative_rf_rms'][window],color='gray',ls='--',label='Frozen round-3 model (fp32)')
    print(json.dumps({'mode':mode,'completion':run['completion'],'validation':run['validation']['model'],
                      'baseline':run['validation']['frozen_baseline']}))
for col,title in enumerate(('Fixed training subset (64 records)','Development case (270 records)')):
    axes[0,col].set_title(title)
    for row,window in enumerate(('18-36 us','36 us to record end')):
        axes[row,col].set_ylabel('Relative RF RMS: '+window)
        axes[row,col].set_xlabel('Epoch'); axes[row,col].grid(alpha=.2); axes[row,col].legend(fontsize=8)
fig.suptitle('NSRNO: scatterer-to-receiver travel times')
fig.savefig(out/'learning_curves.png',dpi=160)
for mode in ('material','reference'):
    data[mode]['feature_cache']=json.loads((root/('forward_ray_'+mode+'_v1/completion.json')).read_text())
(out/'metrics.json').write_text(json.dumps(data,indent=2)+'\n')
