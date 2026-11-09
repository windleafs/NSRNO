"""One explicit bounded extension of a preserved CPU-only material hold."""
from pathlib import Path
import argparse,json,os,signal,time


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);a=p.parse_args();path=a.checkpoint
    record=json.loads(path.read_text());pid=record['source_pid'];cmdpath=Path(f'/proc/{pid}/cmdline')
    if record['status']!='held_reversibly' or record.get('extension_deadline_unix_s'):raise RuntimeError('Only one explicit bounded extension')
    def identity():
        if not cmdpath.exists() or cmdpath.read_bytes().replace(b'\0',b' ').decode()!=record['source_cmdline']:raise RuntimeError('Source identity changed')
        if '\nState:\tT' not in Path(f'/proc/{pid}/status').read_text():raise RuntimeError('Source is not held')
    identity();deadline=time.time()+1200
    record.update(status='held_reversibly_extended',extension_deadline_unix_s=deadline,
        extension_reason='Qualify full11 block solver after an entry-point bug; all4950 fields and moments remain saved; no GPU iteration held')
    tmp=path.with_suffix('.extension.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(path)
    print(json.dumps(record),flush=True)
    while time.time()<deadline:
        latest=json.loads(path.read_text())
        if latest['status']!='held_reversibly_extended' or not cmdpath.exists() or not cmdpath.read_bytes():return
        time.sleep(1)
    identity();latest=json.loads(path.read_text())
    if latest['status']=='held_reversibly_extended':
        os.kill(pid,signal.SIGCONT);latest.update(status='released_without_takeover',release_unix_s=time.time(),reason='bounded extension expired')
        tmp=path.with_suffix('.extension.tmp');tmp.write_text(json.dumps(latest,indent=2));tmp.replace(path)
        print(json.dumps({'released':True}),flush=True)


if __name__=='__main__':main()
