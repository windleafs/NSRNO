"""Resolve one SAVED fixed-field material problem, without propagation/truth.

Writes a separate candidate. Live wavefield banks and the running trajectory
are untouched. This is an actual material solve, not a derivative probe.
"""
from pathlib import Path
import ctypes,os
if os.name=='posix' and ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
import argparse,json,time,hashlib
import numpy as np
from .wri_material import MaterialVariables
from .wri_material_v2 import MaterialNormal


def main():
    p=argparse.ArgumentParser();p.add_argument('--step',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    root=a.step.parent;initial=root/'initial';start=time.time()
    v=MaterialVariables(np.load(initial/'material.npy'),np.load(initial/'parameters.npy'))
    current=np.load(a.step/'old_variables.npy');normal={};moments=[];geometry=None
    summaries=sorted(a.step.glob('field_rank[0-9]*.json'))
    if len(summaries)!=3:raise RuntimeError('Require all three completed frequency sweeps')
    for file in summaries:
        row=json.loads(file.read_text())
        if row['status']!='complete':raise RuntimeError('Incomplete saved frequency sweep')
        with np.load(row['normal_path']) as data:
            for key in data.files:normal[key]=normal.get(key,0)+data[key]
        moments.extend(row['moments']);geometry=row['geometry']
    if len(moments)!=450:raise RuntimeError('Require all450 saved frequency moments')
    normal_system=MaterialNormal(v,geometry,normal)
    def progress(row):
        if row['iteration']%8==0:print(json.dumps({'CPU_CG':row}),flush=True)
    candidate,record=normal_system.solve(current,moments,callback=progress)
    np.save(a.out/'variables.npy',candidate);np.save(a.out/'material.npy',v.material(candidate))
    np.save(a.out/'parameters.npy',v.parameters(candidate))
    record.update(pid=os.getpid(),source_step=str(a.step.resolve()),seconds=time.time()-start,
        solver_sha256=hashlib.sha256(Path(__file__).with_name('wri_material_v2.py').read_bytes()).hexdigest(),
        live_banks_changed=False,optimizer_trajectory_changed=False,
        interpretation='Fixed-field material candidate only; physical RF not yet scored; no recovery claim')
    (a.out/'material_update.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
