from pathlib import Path
import json,math
import numpy as np
from nsrno.conditioned_data import ROOT,key,field_path,receiver_response

index=json.load(open(ROOT/'index.json')); spec=index['spec']; cache=Path(index['source_cache'])
face=round(-spec['z0_m']/spec['dx_m']); cols=np.rint((np.arange(192)*.0002-.0191-spec['x0_m'])/spec['dx_m']).astype(int)
results=[]
for row in index['records']:
    for angle in spec['angle_indices']:
        iq=np.load(cache/row['iq'],mmap_mode='r')[angle]
        spectrum=np.fft.fftshift(np.fft.fft(iq[0]+1j*iq[1],axis=-1,norm='forward'),axes=-1)/np.sqrt(2)
        for bin_index in spec['bins']:
            path=field_path(row,bin_index,angle)
            if not path.exists(): continue
            receipt=json.load(open(path.with_suffix('.json')))
            field=np.load(path,mmap_mode='r'); response=receiver_response(bin_index*40e6/2401)
            prediction=field[face,cols]*response; target=spectrum[:,bin_index-40]
            results.append({'id':row['id'],'role':row['role'],'bin':bin_index,'angle':angle,
                'receiver_response':response,'rf_error_sq':float(np.sum(abs(prediction-target)**2)),
                'rf_target_sq':float(np.sum(abs(target)**2)),
                'steps':receipt['last']['steps'],'residual':receipt['last']['residual'],'seconds':receipt['seconds']})
summary={}
for role in ('train','validation'):
    for bin_index in (*spec['bins'],'all'):
        rows=[r for r in results if r['role']==role and (r['bin']==bin_index or bin_index=='all')]
        if not rows: continue
        summary[f'{role}_{bin_index}']={'observations':len(rows),
            'rf_relative_rms':math.sqrt(sum(r['rf_error_sq'] for r in rows)/sum(r['rf_target_sq'] for r in rows)),
            'max_residual':max(r['residual'] for r in rows),'steps_mean':float(np.mean([r['steps'] for r in rows])),
            'seconds_mean':float(np.mean([r['seconds'] for r in rows]))}
(ROOT/'teacher_summary.json').write_text(json.dumps({'summary':summary,'rows':results},indent=2))
print(json.dumps(summary,indent=2))
