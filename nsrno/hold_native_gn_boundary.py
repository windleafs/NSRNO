"""Hold only the coordinator at a completed material/post-hoc boundary.

GPU jobs are already complete. The independent CPU map child is allowed to
finish. This is a reversible operator hold, not a change to the goal status.
"""
from pathlib import Path
import argparse
import json
import os
import signal
import time

def atomic(path,data):
    tmp=path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def identity(pid):
    root=Path('/proc')/str(pid)
    stat=(root/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'start_ticks':int(stat[19]),'state':stat[0],
        'command':(root/'cmdline').read_bytes().replace(b'\0',b' ').decode()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--step',type=int,default=1);args=p.parse_args()
    root=args.root.resolve();marker=root/'operator_boundary_hold.json'
    if marker.exists():raise RuntimeError('An operator hold record already exists')
    initial=json.loads((root/'run.json').read_text());owner=identity(initial['pid'])
    if ' -m nsrno.reconstruct_native_gn ' not in owner['command']:
        raise RuntimeError('The target is not the expected native GN coordinator')
    atomic(root/'boundary_watcher.json',{'watcher_pid':os.getpid(),'owner':owner,
        'step':args.step,'started_unix_s':time.time(),'status':'watching'})
    while True:
        try:
            now=identity(owner['pid'])
        except FileNotFoundError:
            atomic(root/'boundary_watcher.json',{'watcher_pid':os.getpid(),
                'owner':owner,'step':args.step,'status':'owner_exited_without_hold'})
            return
        if now['start_ticks']!=owner['start_ticks']:
            raise RuntimeError('Coordinator PID was reused; no signal sent')
        run=json.loads((root/'run.json').read_text())
        if run.get('step',0)>args.step:
            atomic(root/'boundary_watcher.json',{'watcher_pid':os.getpid(),
                'owner':owner,'step':args.step,'status':'boundary_missed_no_signal_sent'})
            return
        folder=root/f'step{args.step:03d}'
        if ((folder/'maps.metrics.json').exists()
            and (folder/'material_update.json').exists()
            and run.get('status')=='post_hoc_maps_only'
            and run.get('completed_step')==args.step):
            job=json.loads((root/'active_job.json').read_text())
            progress=[json.loads(p.read_text()) for p in (root/'workers').glob('rank*/progress.json')]
            if not progress or any(d['token']!=job['token'] or d['state']!='completed' for d in progress):
                raise RuntimeError('The completed material boundary still has an active GPU job; no signal sent')
            os.kill(owner['pid'],signal.SIGSTOP)
            atomic(marker,{'watcher_pid':os.getpid(),'owner':owner,'step':args.step,
                'held_unix_s':time.time(),'signal':'SIGSTOP','goal_status_changed':False,
                'run_source_edited':False,'GPU_jobs_complete':True,
                'reason':'Review actual RF/material maps before spending the next three outer sweeps',
                'release':'SIGCONT to continue; caught SIGINT then SIGCONT to close workers via coordinator finally',
                'map_child_allowed_to_finish':True})
            print(json.dumps({'status':'coordinator_held_at_completed_material_boundary',
                'pid':owner['pid'],'step':args.step}),flush=True)
            return
        time.sleep(.1)

if __name__=='__main__':main()
