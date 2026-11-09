"""Preserve evaluated RF-selected candidates before deliberate resource change.

This is a one-off research decision, not an automatic convergence detector.
No material is selected by truth; the already RF-selected snapshot is retained.
"""
from pathlib import Path
import hashlib, json, os, signal, subprocess, time


ROOT = Path('/home_nvme/zhuangyang/nsrno')
SPECS = [
    ('reconstruction_geometry_v1_train_200',1003839,[1004186,1004187],1003840,1097482),
    ('reconstruction_travel_geometry_v1_train_200',1075517,[1075566,1075567],1075518,1097483),
]


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if stat[0] == 'Z':
            return None
        return {'pid':pid,'ppid':int(stat[1]),'start_tick':stat[19],'cmdline':cmd}
    except FileNotFoundError:
        return None


def atomic(path, value):
    partial = path.with_suffix('.partial.json')
    partial.write_text(json.dumps(value,indent=2)+'\n')
    partial.replace(path)


def same(handle):
    now = identity(handle['pid'])
    return now is not None and now['start_tick'] == handle['start_tick'] and now['cmdline'] == handle['cmdline']


def main():
    decisions = []
    registered = json.loads((ROOT/'out/v5/reconstruction_coupled_v1_train_200/launch.json').read_text())
    expected = {v['pid']:v for v in registered['registered_control_handles']}
    # Validate every targeted identity and every stable snapshot before signals.
    for name,main_pid,worker_pids,post_pid,watch_pid in SPECS:
        p = ROOT/'out/v5'/name
        if (p/'resource_decision.json').exists():
            raise FileExistsError('Inspect existing retirement decision; do not duplicate signals')
        snapshot = p/'snapshots/stage1_step048'
        choice = json.loads((snapshot/'rf_selection.json').read_text())
        current = json.loads((p/'stage1_best.json').read_text())
        if choice['step'] != 48 or choice['stage'] != 1 or current['step'] != choice['step']:
            raise RuntimeError('The current RF-selected candidate changed; reassess before stopping')
        files = ['material.npy','parameters.npy','rf_selection.json','material.metrics.json','material.png']
        stamps = {f:(snapshot/f).stat().st_mtime_ns for f in files}
        digests = {f:hashlib.sha256((snapshot/f).read_bytes()).hexdigest() for f in files}
        if any((snapshot/f).stat().st_mtime_ns!=stamps[f] for f in files):
            raise RuntimeError('An evaluated snapshot changed during preservation')
        handles = {}
        for pid in [main_pid,*worker_pids,post_pid,watch_pid]:
            h = identity(pid)
            if h is None:
                raise RuntimeError(f'Expected live control handle {pid} is missing; inspect current state')
            if pid in expected and h['start_tick'] != expected[pid]['start_tick']:
                raise RuntimeError('A control PID was reused')
            if pid == main_pid and ('nsrno.reconstruct_encoded' not in h['cmdline'] or name not in h['cmdline']):
                raise RuntimeError('Control main identity does not match')
            if pid in worker_pids and (h['ppid'] != main_pid or 'multiprocessing.spawn' not in h['cmdline']):
                raise RuntimeError('Control worker parent does not match')
            if pid == post_pid and ('nsrno.run_time_postprocess' not in h['cmdline'] or name not in h['cmdline']):
                raise RuntimeError('Control postprocess identity does not match')
            if pid == watch_pid and ('nsrno.watch_material_snapshots' not in h['cmdline'] or name not in h['cmdline']):
                raise RuntimeError('Control watcher identity does not match')
            handles[pid] = h
        children = subprocess.check_output(['ps','-o','pid=','--ppid',str(main_pid)],text=True)
        for item in children.split():
            pid = int(item)
            if pid not in handles:
                h = identity(pid)
                if h and h['ppid']==main_pid and 'multiprocessing.resource_tracker' in h['cmdline']:
                    handles[pid] = h
                elif h:
                    raise RuntimeError('Unexpected control child; inspect before stopping')
        evaluations = [json.loads(line) for line in (p/'run.log').read_text().splitlines() if line.startswith('{')]
        last = evaluations[-1]
        metrics = json.loads((snapshot/'material.metrics.json').read_text())
        decision = {'status':'validated_before_stop','reason':
            'Full HF48 RF decreased, while density correlation stayed about -0.10 and alpha RMSE worsened versus HF32. Allocate remaining first-order budget to a coupled-curvature method.',
            'original_HF_updates_planned':128,'last_completed_record':last,
            'retained_RF_selected_snapshot':str(snapshot),'RF_selection':choice,
            'snapshot_sha256':digests,'evaluation_only_metrics':{k:v['reconstruction'] for k,v in metrics['maps'].items()},
            'truth_used_for_checkpoint_selection':False,
            'truth_used_for_method_resource_decision':True,
            'not_convergence_or_full_budget_completion':True,
            'handles':list(handles.values()),'unix_s':time.time()}
        decisions.append((p,decision,handles,main_pid,worker_pids,post_pid,watch_pid))
    for p,decision,handles,main_pid,worker_pids,post_pid,watch_pid in decisions:
        atomic(p/'resource_decision.json',decision)
        (p/'run_before_retirement.json').write_bytes((p/'run.json').read_bytes())
        for pid in [post_pid,watch_pid,main_pid,*worker_pids,*[v for v in handles if v not in [post_pid,watch_pid,main_pid,*worker_pids]]]:
            h = handles[pid]
            if same(h):
                os.kill(pid,signal.SIGTERM)
            elif identity(pid) is not None:
                raise RuntimeError('Process identity changed during controlled stop; no signal sent to it')
        for _ in range(30):
            remaining = [pid for pid,h in handles.items() if same(h)]
            if not remaining:
                break
            time.sleep(.5)
        else:
            raise RuntimeError(f'Stop not complete; surviving exact handles {remaining}')
        run = json.loads((p/'run.json').read_text())
        run.update(status='stopped_for_resource_reallocation',
                   stopped_after_completed_record=decision['last_completed_record'],
                   preserved_snapshot=decision['retained_RF_selected_snapshot'],
                   stopping_does_not_mean_convergence=True)
        atomic(p/'run.json',run)
        decision.update(status='stopped_and_preserved',unix_s=time.time(),remaining_exact_handles=[])
        atomic(p/'resource_decision.json',decision)
        print(json.dumps({'control':p.name,'status':decision['status'],
            'last_completed_stage':decision['last_completed_record']['stage'],
            'last_completed_update':decision['last_completed_record']['step'],
            'retained_RF_selected_step':48}),flush=True)


if __name__ == '__main__':
    main()
