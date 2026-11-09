"""Retire only a fully checkpointed, held CPU coordinator; retain all fields."""
from pathlib import Path
import argparse,json,os,signal,time,hashlib,shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);a=p.parse_args()
    checkpoint=json.loads(a.checkpoint.read_text());root=Path(checkpoint['root']);step=checkpoint['step']
    folder=root/f'step{step:03d}';pid=checkpoint['source_pid'];run=json.loads((root/'run.json').read_text())
    if checkpoint['type']!='fixed_field_material_pending' or checkpoint['status'] not in ['held_reversibly','held_reversibly_extended']:
        raise RuntimeError('Require the complete, explicitly typed held field checkpoint')
    if run['pid']!=pid or run['step']!=step or run['status']!='material_update':raise RuntimeError('Source phase changed')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=checkpoint['boot_id']:raise RuntimeError('Boot changed')
    cmdpath=Path(f'/proc/{pid}/cmdline')
    if cmdpath.read_bytes().replace(b'\0',b' ').decode()!=checkpoint['source_cmdline']:raise RuntimeError('Source identity changed')
    if '\nState:\tT' not in Path(f'/proc/{pid}/status').read_text():raise RuntimeError('Source not held')
    if (folder/'dual_processes.json').exists() or (root/f'step{step+1:03d}').exists():raise RuntimeError('Later phase exists')
    for file,value in checkpoint['files'].items():
        path=folder/file
        if path.stat().st_size!=value['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=value['sha256']:raise RuntimeError('Saved checkpoint changed: '+file)
    for worker in json.loads((folder/'field_processes.json').read_text())['pids']:
        proc=Path(f'/proc/{worker}/cmdline')
        if proc.exists() and proc.read_bytes():raise RuntimeError('A field worker is still live')
    archive=root/f'retired_coordinator_step{step:03d}';archive.mkdir(exist_ok=False)
    shutil.copy2(root/'run.json',archive/'run_before_retirement.json');shutil.copy2(a.checkpoint,archive/a.checkpoint.name)
    shutil.copy2(Path(__file__),archive/Path(__file__).name)
    for name in ['variables.npy','material.npy','parameters.npy','material_update.json','material_CG_progress.json']:
        if (folder/name).exists():shutil.copy2(folder/name,archive/name)
    intent={'source_pid':pid,'unix_s':time.time(),'CPU_only':True,'all450_frequencies_all11_fields_saved':True,
        'no_live_GPU_worker':True,'X_Ybase_Ydata_preserved':True,'GPU_iterations_lost':0,'reason':'Known failed old material scaling; hold exact field checkpoint while fixing field solver'}
    (archive/'retirement_intent.json').write_text(json.dumps(intent,indent=2))
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    for _ in range(200):
        if not cmdpath.exists() or not cmdpath.read_bytes():break
        time.sleep(.05)
    else:raise RuntimeError('CPU coordinator did not exit; no bank files changed')
    checkpoint.update(status='retired_at_material_pending',source_coordinator_retired=True,
        retired_unix_s=time.time(),retirement_archive=str(archive.resolve()),GPU_iterations_lost=0,
        signals_sent=checkpoint['signals_sent']+['SIGTERM','SIGCONT'])
    tmp=a.checkpoint.with_suffix('.retirement.tmp');tmp.write_text(json.dumps(checkpoint,indent=2));tmp.replace(a.checkpoint)
    run.update(status='solver_upgrade_checkpoint',source_coordinator_retired=True,checkpoint=str(a.checkpoint.resolve()),
        no_GPU_iteration_lost=True,solver_upgrade_reason='Preserved complete field sweep; material and field numerical fixes pending qualification')
    tmp=(root/'run.json').with_suffix('.retirement.tmp');tmp.write_text(json.dumps(run,indent=2));tmp.replace(root/'run.json')
    print(json.dumps({'retired_CPU_pid':pid,'checkpoint':str(a.checkpoint),'banks_changed':False,'GPU_iterations_lost':0,'goal_work_continues':True}),flush=True)


if __name__=='__main__':main()
