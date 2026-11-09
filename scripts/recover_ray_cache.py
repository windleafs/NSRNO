"""After an observed server reboot, retain valid arrays and quarantine incomplete writes."""
from pathlib import Path
import json
import time
import numpy as np

root=Path('/home_nvme/zhuangyang/nsrno')
index=json.loads((root/'out/v5/forward_cache_v1/index.json').read_text())
boot=int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
stamp=str(int(time.time()))
report={'boot_unix':boot,'checked_features':{},'quarantined':[],'recent_shared_recomputed':0}

def quarantine(path,reason):
    if path.exists():
        target=path.with_name(path.name+'.interrupted_'+stamp)
        path.rename(target)
        report['quarantined'].append({'path':str(path),'reason':reason})

for mode in ('material','reference'):
    folder=root/('out/v5/forward_ray_'+mode+'_v1')
    count=0
    for row in index['records']:
        stat=folder/'stats'/(row['id']+'.json')
        array=folder/'features'/(row['id']+'.npy')
        if not stat.exists(): continue
        try:
            old=json.loads(stat.read_text())
            value=np.load(array,mmap_mode='r',allow_pickle=False)
            if value.shape!=(11,2,2,192,640) or value.dtype!=np.float16:
                raise ValueError('wrong feature shape/dtype')
            converted=value.astype(np.float32)
            if not np.isfinite(converted).all(): raise ValueError('non-finite feature')
            power=np.square(converted).sum((0,2,3,4),dtype=np.float64)
            expected=np.asarray(old['power'])
            difference=np.abs(power-expected)/np.maximum(expected,1e-12)
            if difference.max()>.002: raise ValueError('feature power differs from recorded power')
            count+=1
        except (OSError,ValueError,EOFError,KeyError) as exc:
            quarantine(array,str(exc)); quarantine(stat,str(exc))
    report['checked_features'][mode]=count
    if (folder/'completion.json').exists():
        quarantine(folder/'completion.json','completion must be rebuilt after recovery')

shared=root/'out/v5/forward_ray_shared_fp32_v1'
for kind,shape in [('baseline',(11,2,192,640)),('background',(3,121,141))]:
    for row in index['records']:
        path=shared/kind/(row['id']+'.npy')
        if not path.exists(): continue
        reason=None
        try:
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
                raise ValueError('invalid shared array')
            # Recent files may have reached the page cache without durable contents.
            if path.stat().st_mtime>=boot-120:
                reason='shared output written within 120 seconds before reboot'
                report['recent_shared_recomputed']+=1
        except (OSError,ValueError,EOFError) as exc:
            reason=str(exc)
        if reason: quarantine(path,reason)

destination=root/'out/v5/forward_round4_checks'/('recovery_'+stamp+'.json')
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(destination),'checked_features':report['checked_features'],
                  'quarantined_files':len(report['quarantined']),'recent_shared_recomputed':report['recent_shared_recomputed']}),flush=True)
