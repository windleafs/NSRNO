"""Durably materialize the original augmented RHS from a held continuation.

This is an algebraic snapshot of existing fields, not a new propagation or
material experiment. It makes partial-frequency continuation recoverable
without depending on the subsequent in-place pairing of X and Y_base.
"""
from pathlib import Path
import argparse,ctypes,json,time,os,hashlib,subprocess
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
import numpy as np
import torch
from .conditioned_wave import FrequencyGeometry,ConditionedFrequencySystem
from .batched_first_order_wavefield import BatchedFirstOrderWavefield
from .frequency_group_data import receiver_response


def atomic(path,value):
    tmp=path.with_suffix('.rhs.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def validate(checkpoint):
    if checkpoint['status']!='held_for_fixed_rhs':raise RuntimeError('Require the typed held continuation')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=checkpoint['boot_id']:raise RuntimeError('Boot changed')
    for identity in [checkpoint['source_main_identity'],*checkpoint['source_worker_identities'].values()]:
        proc=Path(f"/proc/{identity['pid']}")
        if not proc.exists() or '\nState:\tT' not in (proc/'status').read_text():raise RuntimeError('A source writer is not held')
        if (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()!=identity['cmdline']:raise RuntimeError('Held identity changed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',type=int,default=2);p.add_argument('--cap-gib',type=float,default=3.)
    a=p.parse_args();start=time.time();cp=json.loads(a.checkpoint.read_text());validate(cp)
    root=Path(cp['root']);folder=root/f"step{cp['step']:03d}";out=folder/'fixed_augmented_rhs';out.mkdir(exist_ok=False)
    # X/Y+RHS need about119GiB additional native storage, not additional GPU history.
    import shutil
    required=450*11*3*961*1121*8+450*11*192*8
    if shutil.disk_usage(out).free<required+24*2**30:raise RuntimeError('Insufficient disk for immutable native RHS snapshot')
    cards={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
    if cards[a.device]+a.cap_gib*1024+768>16384:raise RuntimeError('Whole-card16GiB admission failed')
    torch.set_num_threads(1);torch.cuda.set_device(a.device)
    torch.cuda.set_per_process_memory_fraction(a.cap_gib*2**30/torch.cuda.get_device_properties(a.device).total_memory,a.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=f'cuda:{a.device}';paths={'lf':Path('out/v5/conditioned_lf_full_v1/index.json'),
        'point3':Path('out/v5/conditioned_point_full_v1/index.json')}
    specs={s:json.loads(p.read_text())['spec'] for s,p in paths.items()}
    g=FrequencyGeometry(specs['lf']).to(device)
    old=torch.from_numpy(np.load(folder/'old_material.npy')).to(device);rows=[];files={}
    with torch.no_grad():
        for source,spec in specs.items():
            n=len(spec['bins']);banks={k:np.load(root/'banks'/f'{source}_{k}.npy',mmap_mode='r') for k in ['X','Y','Ydata']}
            rhs_path=out/f'{source}_rhs.npy';data_path=out/f'{source}_data.npy'
            R=np.lib.format.open_memmap(rhs_path,mode='w+',dtype=np.complex64,shape=(n,11,3,g.h,g.w))
            D=np.lib.format.open_memmap(data_path,mode='w+',dtype=np.complex64,shape=(n,11,192))
            for position,bin in enumerate(spec['bins']):
                validate(json.loads(a.checkpoint.read_text()))
                if time.time()+120>cp['lease_deadline_unix_s']:raise RuntimeError('Insufficient safe hold lease remaining')
                frequency=bin*40e6/spec['rf_samples']
                system=ConditionedFrequencySystem(g,old[:2],old[2],frequency);op=BatchedFirstOrderWavefield(system)
                x=g.pad_field(torch.from_numpy(np.array(banks['X'][position])).to(device))
                y=g.pad_field(torch.from_numpy(np.array(banks['Y'][position])).to(device))
                data=torch.from_numpy(np.array(banks['Ydata'][position])).to(device)
                b=g.native(y+op(x)).cpu().numpy();d=(data+op.readout(x,receiver_response(frequency))).cpu().numpy()
                b=np.ascontiguousarray(b);d=np.ascontiguousarray(d)
                if not np.isfinite(b).all() or not np.isfinite(d).all():raise RuntimeError('Nonfinite source checkpoint')
                R[position]=b;D[position]=d;R.flush();D.flush()
                rows.append({'source':source,'position':position,'frequency_hz':frequency,
                    'rhs_sha256':hashlib.sha256(b.tobytes()).hexdigest(),'data_sha256':hashlib.sha256(d.tobytes()).hexdigest()})
                if len(rows)%15==0:print(json.dumps({'fixed_frequencies':len(rows),'total':450,'seconds':time.time()-start}),flush=True)
                del system,op,x,y,data,b,d
            del R,D,banks
            for name,path in [('rhs',rhs_path),('data',data_path)]:
                with path.open('r+b') as stream:os.fsync(stream.fileno())
                files[source+'_'+name]={'path':str(path.resolve()),'bytes':path.stat().st_size}
    validate(json.loads(a.checkpoint.read_text()))
    if len(rows)!=450:raise RuntimeError('Incomplete immutable RHS')
    index={'status':'complete','step':cp['step'],'sample':cp['sample'],'source_checkpoint':str(a.checkpoint.resolve()),
        'native_shape':[3,g.h,g.w],'groups':{s:len(v['bins']) for s,v in specs.items()},
        'transmissions':11,'receivers':192,'dtype':'complex64','shared_geometry':'LF geometry, same as production worker',
        'material_sha256':cp['material_sha256'],'files':files,'frequencies':rows,'new_propagation_runs':0,
        'material_truth_read':False,'existing_banks_changed':False,'GPU_peak_GiB':torch.cuda.max_memory_allocated()/2**30,
        'seconds':time.time()-start,'definition':'b_shift=Y_base+B_old X; d_shift=Ydata_base+C X, reconstructed to complex64 precision'}
    atomic(out/'index.json',index)
    cp.update(status='fixed_rhs_ready',fixed_rhs_directory=str(out.resolve()),fixed_rhs_index_sha256=hashlib.sha256((out/'index.json').read_bytes()).hexdigest(),
        fixed_rhs_complete_frequencies=450,fixed_rhs_saved_unix_s=time.time())
    atomic(a.checkpoint,cp);print(json.dumps({'complete':True,'out':str(out),'seconds':index['seconds'],'GPU_peak_GiB':index['GPU_peak_GiB']}),flush=True)


if __name__=='__main__':main()
