"""Retire only the held, flushed CPU continuation after immutable RHS exists."""
from pathlib import Path
import argparse,json,os,signal,time,hashlib


def atomic(path,value):
    tmp=path.with_suffix('.retire.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);a=p.parse_args();cp=json.loads(a.checkpoint.read_text())
    if cp['type']!='fixed_model_field_continue_partial_boundary' or cp['status']!='fixed_rhs_ready':raise RuntimeError('Require held boundaries AND complete immutable RHS')
    root=Path(cp['root']);folder=root/f"step{cp['step']:03d}";run=json.loads((root/'run.json').read_text())
    if run['pid']!=cp['source_pid'] or run['status']!='field_continue_held_for_gpu_upgrade':raise RuntimeError('Source phase changed')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=cp['boot_id']:raise RuntimeError('Boot changed')
    for name,sha in cp['material_sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen material changed')
    ledger=Path(cp['fixed_rhs_directory']);index=json.loads((ledger/'index.json').read_text())
    if hashlib.sha256((ledger/'index.json').read_bytes()).hexdigest()!=cp['fixed_rhs_index_sha256'] or index['status']!='complete' or len(index['frequencies'])!=450:
        raise RuntimeError('Immutable augmented RHS incomplete or changed')
    for value in index['files'].values():
        if Path(value['path']).stat().st_size!=value['bytes']:raise RuntimeError('Immutable RHS file size changed')
    identities=[*cp['source_worker_identities'].values(),cp['source_main_identity']]
    for identity in identities:
        proc=Path(f"/proc/{identity['pid']}")
        if '\nState:\tT' not in (proc/'status').read_text() or (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()!=identity['cmdline']:
            raise RuntimeError('Source is not the exact held process')
        if (proc/'stat').read_text().rsplit(')',1)[1].split()[19]!=identity['start_ticks']:raise RuntimeError('Source process start time changed')
    intent={'unix_s':time.time(),'pids':[x['pid'] for x in identities],'all_workers_at_flushed_frequency_boundary':True,
        'immutable_rhs_complete':450,'GPU_field_iterations_lost':0,'CPU_aggregate_normals':'recomputed for all450 before material update',
        'existing_X_Y_banks_changed':False,'materials_updated':False}
    atomic(Path(cp['archive'])/'retirement_intent.json',intent)
    for identity in identities:
        os.kill(identity['pid'],signal.SIGTERM);os.kill(identity['pid'],signal.SIGCONT)
    for _ in range(200):
        if all(not (p:=Path(f"/proc/{x['pid']}/cmdline")).exists() or not p.read_bytes() for x in identities):break
        time.sleep(.05)
    else:raise RuntimeError('Some held source handles did not exit; no existing bank writes performed')
    cp.update(status='retired_with_fixed_rhs',retired_unix_s=time.time(),GPU_field_iterations_lost=0)
    atomic(a.checkpoint,cp)
    run.update(status='field_continue_checkpoint',source_coordinator_retired=True,upgrade_checkpoint=str(a.checkpoint.resolve()),
        fixed_rhs_directory=cp['fixed_rhs_directory'],needs_full450_statistics_recompute=True)
    atomic(root/'run.json',run);print(json.dumps({'retired_pids':intent['pids'],'immutable_frequencies':450,'GPU_field_iterations_lost':0,'checkpoint':str(a.checkpoint)}),flush=True)


if __name__=='__main__':main()
