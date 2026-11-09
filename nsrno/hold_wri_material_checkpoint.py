"""Reversibly hold our coordinator only after a complete field sweep.

Every field worker must have exited and flushed its native banks. This does
not interrupt GPU computation. A failed phase check immediately resumes the
coordinator; the successful checkpoint explicitly labels Y as Y_base.
"""
from pathlib import Path
import argparse,json,os,signal,time,hashlib


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--timeout-seconds',type=float,default=1800);a=p.parse_args()
    root=a.root.resolve();folder=root/f'step{a.step:03d}'
    marker=root/f'material_pending_step{a.step:03d}.json'
    if marker.exists():raise FileExistsError('Checkpoint already exists')
    expected=f'-m nsrno.reconstruct_wri --out {a.root}'
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cmdpath=Path(f'/proc/{a.pid}/cmdline')
    def identity():
        if not cmdpath.exists():raise RuntimeError('Original coordinator no longer exists')
        cmd=cmdpath.read_bytes().replace(b'\0',b' ').decode()
        if expected not in cmd:raise RuntimeError('Coordinator identity differs')
        return cmd
    def validate(run):
        if run['pid']!=a.pid or run['step']!=a.step or run['status']!='material_update':raise RuntimeError('Not the expected material phase')
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=boot:raise RuntimeError('Boot changed')
        if (root/f'step{a.step+1:03d}').exists() or (folder/'dual_processes.json').exists():raise RuntimeError('A later phase exists')
        processes=json.loads((folder/'field_processes.json').read_text())
        for pid in processes['pids']:
            path=Path(f'/proc/{pid}/cmdline')
            if path.exists() and path.read_bytes():raise RuntimeError('A field worker is still live')
        summaries=[json.loads((folder/f'field_rank{r}.json').read_text()) for r in range(3)]
        if any(s['status']!='complete' or s['frequencies']!=150 for s in summaries):raise RuntimeError('Incomplete field sweep')
        moments=[m for s in summaries for m in s['moments']]
        if len(moments)!=450 or len({(m['source'],m['position']) for m in moments})!=450:raise RuntimeError('Incomplete frequency moments')
        if any(not Path(m['path']).is_file() for m in moments):raise RuntimeError('Missing saved moments')
        return processes,summaries
    start=time.time();identity()
    while time.time()-start<a.timeout_seconds:
        run=json.loads((root/'run.json').read_text())
        if run.get('step',0)>a.step or (run.get('step')==a.step and run['status'] not in ['field_sweep','material_update']):
            print(json.dumps({'held':False,'reason':'safe material boundary missed; no signal sent','run_status':run['status']}),flush=True);return
        if run.get('step')==a.step and run['status']=='material_update':
            cmd=identity();validate(run)
            os.kill(a.pid,signal.SIGSTOP)
            try:
                for _ in range(100):
                    state=Path(f'/proc/{a.pid}/status').read_text()
                    if '\nState:\tT' in state:break
                    time.sleep(.02)
                else:raise RuntimeError('Coordinator did not enter the stopped state')
                frozen=json.loads((root/'run.json').read_text());processes,summaries=validate(frozen)
                names=['old_variables.npy','old_material.npy','field_processes.json']+[f'field_rank{r}.json' for r in range(3)]+[f'normals_rank{r}.npz' for r in range(3)]
                files={name:{'bytes':(folder/name).stat().st_size,'sha256':hashlib.sha256((folder/name).read_bytes()).hexdigest()} for name in names}
                record={'type':'fixed_field_material_pending','status':'held_reversibly','step':a.step,'source_pid':a.pid,
                    'source_cmdline':cmd,'boot_id':boot,'unix_s':time.time(),'root':str(root),
                    'field_workers_all_exited':True,'frequencies':450,'all_transmissions':11,
                    'X_state':'complete fixed-old-model field sweep','Y_state':'Y_base, not completed equation multipliers',
                    'Ydata_state':'updated data multipliers','files':files,'signals_sent':['SIGSTOP'],
                    'no_GPU_iteration_interrupted':True,'no_bank_write_by_checkpoint':True}
                tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(marker)
                print(json.dumps(record),flush=True)
                # A failed replacement must not leave a paused coordinator
                # indefinitely. The takeover marks the checkpoint before it
                # writes any new bank state.
                deadline=time.time()+1200
                while time.time()<deadline:
                    latest=json.loads(marker.read_text())
                    if latest['status']!='held_reversibly':return
                    if not cmdpath.exists() or not cmdpath.read_bytes():return
                    time.sleep(1)
                identity();latest=json.loads(marker.read_text())
                if latest['status']=='held_reversibly':
                    os.kill(a.pid,signal.SIGCONT)
                    latest.update(status='released_without_takeover',release_unix_s=time.time(),reason='20minute safety deadline')
                    tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(latest,indent=2));tmp.replace(marker)
                    print(json.dumps({'released':True,'reason':'20minute safety deadline'}),flush=True)
                return
            except BaseException:
                os.kill(a.pid,signal.SIGCONT);raise
        time.sleep(.25)
    print(json.dumps({'held':False,'reason':'timeout; no signal sent'}),flush=True)


if __name__=='__main__':main()
