"""Read-only sampling of whole-card memory and host pressure during this run.

No process is stopped, signaled, rescheduled, or restarted by this monitor.
The sampled maximum is explicitly not an instantaneous memory certificate.
"""
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime, timezone
import fcntl, json, os, subprocess, time


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':
            return None
        return fields[19],Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return None


def atomic(path,value):
    temp=path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def main():
    p=ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--interval',type=float,default=15.)
    a=p.parse_args()
    lock=(a.out/'resource_monitor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=a.out/'resource_monitor.json'
    if status.exists():
        raise FileExistsError('Inspect the existing resource monitor before starting another')
    start_identity=identity(a.pid)
    if start_identity is None or b'nsrno.reconstruct_coupled' not in start_identity[1] or a.out.name.encode() not in start_identity[1]:
        raise RuntimeError('The specific coupled optimizer is not live')
    record={'status':'monitoring_live_process','pid':os.getpid(),'target_pid':a.pid,
            'target_start_tick':start_identity[0],'interval_seconds':a.interval,
            'method':'nvidia-smi whole-card memory sampling; /proc/meminfo and memory pressure',
            'sampled_maximum_only':True,'signals_or_process_modifications':False,
            'samples':0,'maximum_gpu_used_MiB':{},'minimum_host_available_GiB':None,
            'maximum_memory_pressure_avg10':0.,'budget_exceeded_observed':False}
    atomic(status,record)
    while True:
        current=identity(a.pid)
        if current is None or current[0]!=start_identity[0] or current[1]!=start_identity[1]:
            record['status']='finished_target_exited'
            record['finished_utc']=datetime.now(timezone.utc).isoformat()
            atomic(status,record)
            break
        sample={'unix_s':time.time(),'utc':datetime.now(timezone.utc).isoformat()}
        try:
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                       '--format=csv,noheader,nounits'],text=True,timeout=8)
            used,util={},{}
            for line in output.splitlines():
                index,memory,load=[int(s.strip()) for s in line.split(',')]
                used[index]=memory
                util[index]=load
                key=str(index)
                record['maximum_gpu_used_MiB'][key]=max(record['maximum_gpu_used_MiB'].get(key,0),memory)
                record['budget_exceeded_observed'] |= memory>16*1024
            sample.update(GPU_used_MiB=used,GPU_utilization_percent=util)
            info=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            available=int(info['MemAvailable'].split()[0])/1024**2
            pressure=Path('/proc/pressure/memory').read_text().splitlines()
            avg10=float(pressure[0].split('avg10=')[1].split()[0])
            sample.update(host_available_GiB=available,memory_pressure_some_avg10=avg10)
            old=record['minimum_host_available_GiB']
            record['minimum_host_available_GiB']=available if old is None else min(old,available)
            record['maximum_memory_pressure_avg10']=max(record['maximum_memory_pressure_avg10'],avg10)
            workers=json.loads((a.out/'workers.json').read_text())['workers']
            rss={}
            for w in workers:
                try:
                    text=Path(f"/proc/{w['pid']}/status").read_text()
                    line=next(s for s in text.splitlines() if s.startswith('VmRSS:'))
                    rss[w['pid']]=int(line.split()[1])/1024**2
                except (FileNotFoundError,StopIteration):
                    pass
            sample['worker_RSS_GiB']=rss
            sample['optimizer_status']=json.loads((a.out/'run.json').read_text())['status']
            record['samples']+=1
        except (subprocess.SubprocessError,OSError,ValueError) as error:
            sample['observation_error']=str(error)
        with (a.out/'resource_history.jsonl').open('a') as f:
            f.write(json.dumps(sample)+'\n')
        record['last_sample']=sample
        atomic(status,record)
        time.sleep(a.interval)


if __name__=='__main__':
    main()
