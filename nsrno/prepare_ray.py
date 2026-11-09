"""Cache deterministic straight-ray features; RF targets remain unchanged."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import time
import numpy as np
import torch
from .dataset import read_json,write_json as atomic_write_json
from .forward_model import load_forward_model
from .ray_propagation import RayPropagation
from .ray_model import make_background


def durable(path):
    with path.open('rb') as handle: os.fsync(handle.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def write_json(path,value):
    atomic_write_json(path,value)
    durable(path)


def save_array(path,array):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as handle:
        np.save(handle,array,allow_pickle=False)
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)
    durable(path)


def main():
    p=ArgumentParser()
    p.add_argument('--delay',choices=('material','reference'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('out/v5/forward_cache_v1'))
    p.add_argument('--warm-start',type=Path,default=Path('out/v5/forward_round3_slowness/best.pt'))
    p.add_argument('--shared',type=Path,default=Path('out/v5/forward_ray_shared_fp32_v1'))
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--shard-id',type=int,default=0)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=True
    index=read_json(args.cache/'index.json')
    old=torch.load(args.warm_start,map_location='cpu',weights_only=True)
    spec={**old['spec'],'ray_fan_count':513,'ray_time_oversample':4,'ray_receiver_chunk':8}
    settings={'version':1,'delay':args.delay,'spec':spec,
              'index_sha256':sha256((args.cache/'index.json').read_bytes()).hexdigest(),
              'warm_start_sha256':sha256(args.warm_start.read_bytes()).hexdigest(),
              'propagation_sha256':sha256(Path(__file__).with_name('ray_propagation.py').read_bytes()).hexdigest()}
    for folder in (args.out,args.out/'features',args.out/'stats',args.shared,args.shared/'baseline',args.shared/'background'):
        folder.mkdir(parents=True,exist_ok=True)
    if (args.out/'settings.json').exists() and read_json(args.out/'settings.json')!=settings:
        raise ValueError('Existing feature cache uses different code or data')
    write_json(args.out/'settings.json',settings)
    source=args.out/'source'; source.mkdir(exist_ok=True)
    for name in ('ray_propagation.py','ray_model.py','prepare_ray.py'):
        shutil.copy2(Path(__file__).with_name(name),source/name)
    operator=RayPropagation(spec).cuda()
    baseline=load_forward_model(args.warm_start,'cuda') if args.delay=='material' else None
    if not 0 <= args.shard_id < args.shards:
        raise ValueError('Invalid cache shard')
    all_rows=index['records'][:args.limit] if args.limit else index['records']
    rows=all_rows[args.shard_id::args.shards]
    started=time.perf_counter()
    for i,row in enumerate(rows):
        output=args.out/'features'/(row['id']+'.npy')
        stat=args.out/'stats'/(row['id']+'.json')
        base_path=args.shared/'baseline'/(row['id']+'.npy')
        bg_path=args.shared/'background'/(row['id']+'.npy')
        complete=output.exists() and stat.exists()
        if args.delay=='material': complete=complete and base_path.exists() and bg_path.exists()
        if complete: continue
        timer=time.perf_counter()
        medium=torch.from_numpy(np.load(args.cache/row['medium'],allow_pickle=False)).cuda()[None]
        with torch.no_grad():
            existing=read_json(stat) if output.exists() and stat.exists() else None
            if existing is None:
                features=operator(medium,material=args.delay=='material')[0]
                quantized=features.half()
                if not torch.isfinite(quantized).all():
                    raise ValueError('Non-finite ray feature or float16 overflow')
                powers=features.double().square().sum((0,2,3,4)).cpu().tolist()
                count=features.shape[0]*features.shape[2]*features.shape[3]*features.shape[4]
                quant_error=float((features-quantized.float()).square().sum()/features.square().sum().clamp_min(1e-20))**.5
                save_array(output,quantized.cpu().numpy())
            else:
                powers,count,quant_error=existing['power'],existing['count'],existing['quantization_relative_rms']
            if baseline is not None:
                predicted=baseline.predict_iq(medium)[0]
                save_array(base_path,predicted.float().cpu().numpy())
                save_array(bg_path,make_background(medium,spec)[0].cpu().numpy())
        elapsed=time.perf_counter()-timer
        write_json(stat,{'id':row['id'],'role':row['role'],'power':powers,'count':count,
                         'quantization_relative_rms':quant_error,'seconds':elapsed+(existing['seconds'] if existing else 0.)})
        progress={'completed':i+1,'total':len(rows),'last_id':row['id'],'record_seconds':elapsed,
                  'elapsed_this_process_seconds':time.perf_counter()-started}
        write_json(args.out/('progress_%d.json'%args.shard_id),progress)
        if i%10==0 or i+1==len(rows): print(json.dumps(progress),flush=True)
    write_json(args.out/('shard_%d_done.json'%args.shard_id),{'records':len(rows),'shards':args.shards})
    if not args.limit and all((args.out/'stats'/(r['id']+'.json')).exists() for r in index['records']):
        stats=[read_json(args.out/'stats'/(r['id']+'.json')) for r in index['records']]
        training=[s for s in stats if s['role']=='train']
        power=np.sum([s['power'] for s in training],0)
        count=sum(s['count'] for s in training)
        write_json(args.out/'completion.json',{'records':len(index['records']),'feature_scales':np.sqrt(power/count).tolist(),
                    'max_quantization_relative_rms':max(s['quantization_relative_rms'] for s in stats),
                    'record_seconds_sum':sum(s['seconds'] for s in stats)})
        print(json.dumps(read_json(args.out/'completion.json')),flush=True)


if __name__=='__main__': main()
