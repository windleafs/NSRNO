"""Paired lower-frequency UltraWave acquisition; independent NSRNO outputs."""
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256
import json, os, time, shutil
import numpy as np
from scipy.optimize import nnls
from .dataset import read_json, write_json
from .fullwave import initialize, build_case

def durable_array(path, **arrays):
    temporary=path.with_suffix('.partial.npz')
    with temporary.open('wb') as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)
    fd=os.open(path.parent,os.O_DIRECTORY); os.fsync(fd); os.close(fd)

def absorption(case, band):
    frequency=np.linspace(*band,800); omega=2*np.pi*frequency
    taus=np.array([.5,2.])/(2*np.pi*case['f0'])
    basis=np.stack([omega**2*t/(2*case['c0']*(1+(omega*t)**2)) for t in taus],1)
    target=100*np.log(10)/20*(frequency/1e6)**1.1
    eta,_=nnls(basis/target[:,None],np.ones(len(target)))
    error=float(np.max(np.abs(basis@eta/target-1)))
    if error>.10: raise ValueError(f'Absorption fit error {error}')
    if case['dt']>min(taus)/8: raise ValueError('Relaxation timestep too large')
    return dict(taus_s=taus.tolist(),eta_per_unit_alpha=eta.tolist(),
                fit_band_hz=list(band),max_relative_error=error,
                target_units='alpha_coeff dB/(cm MHz^1.1), fitted in Np/m',nonlinearity_supported=False)

def pick(rows,n):
    rows=sorted(rows,key=lambda r:(r.get('z_index',0),r['id']))
    return [rows[i] for i in np.linspace(0,len(rows)-1,min(n,len(rows)),dtype=int)]

def main():
    p=ArgumentParser(); p.add_argument('--f0-mhz',type=float,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--shard-id',type=int,default=0)
    p.add_argument('--shards',type=int,default=1); args=p.parse_args()
    root=Path('out/v5'); manifest_path=root/'manifest.json'; cache=root/'forward_cache_v1'
    manifest=read_json(manifest_path); index=read_json(cache/'index.json')
    by_id={r['id']:r for r in manifest['records']}
    counts={'train':{'oa_breast_original':24,'high_echo_circles_v1':2,'cswap':4,'reseed':2},
            'validation':{'oa_breast_original':14,'high_echo_circles_v1':2}}
    selected=[]
    for role,variants in counts.items():
        for variant,count in variants.items():
            selected+=pick([r for r in index['records'] if r['role']==role and r['variant']==variant],count)
    # Alternate train/validation records so both are usable early in generation.
    train=[r for r in selected if r['role']=='train']; val=[r for r in selected if r['role']=='validation']
    selected=[r for i in range(max(len(train),len(val))) for r in train[i:i+1]+val[i:i+1]]
    args.out.mkdir(parents=True,exist_ok=True)
    generator=initialize(); case=generator.geometry_case(); case['f0']=args.f0_mhz*1e6
    case['tref']=4/case['f0']; band=[.5*case['f0'],1.5*case['f0']]
    fit=absorption(case,band)
    settings=dict(version=1,source_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        source_cache_sha256=sha256((cache/'index.json').read_bytes()).hexdigest(),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        acquisition={**manifest['acquisition'],'source_f0_hz':case['f0'],'band_hz':band},
        absorption=fit,source='1e5 Pa Gaussian sine; sigma=1/f0; same Hann and delays',
        comparison_note='Same media and peak source pressure; pulse energy and retained bandwidth differ from V5.',
        records=[{**r,'paired_high_frequency_id':r['id'],'rf':r['id']+'.npz'} for r in selected])
    settings_path=args.out/'index.json'
    if settings_path.exists() and read_json(settings_path)!=settings: raise ValueError('Dataset configuration changed')
    # Each shard writes its own manifest first; canonical file is identical.
    shard_settings=args.out/f'index.shard{args.shard_id}.json'; write_json(shard_settings,settings)
    if args.shard_id==0: write_json(settings_path,settings)
    shutil.copy2(__file__,args.out/f'generate_source.shard{args.shard_id}.py')
    ref_path=args.out/f'reference.shard{args.shard_id}.npz'
    fingerprint=sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    if ref_path.exists():
        with np.load(ref_path) as a:
            if a['fingerprint'].item()!=fingerprint: raise ValueError('Reference mismatch')
            refs=a['rf_native']; trefs=a['trefs']
    else:
        solver=generator.solver_for(case,case['maps'],fit); refs=[]; trefs=[]
        for angle in settings['acquisition']['angles_deg']:
            trefs.append(generator.set_angle(solver,case,angle)); rf,timing=solver.run(); refs.append(rf)
            print(json.dumps({'reference_angle':angle,'seconds':timing['solve_readback_s']}),flush=True)
        refs=np.stack(refs); trefs=np.asarray(trefs)
        durable_array(ref_path,rf_native=refs,trefs=trefs,fingerprint=np.array(fingerprint))
        del solver
    for row in selected[args.shard_id::args.shards]:
        path=args.out/row['id']; receipt=path.with_suffix('.json'); output=path.with_suffix('.npz')
        if receipt.exists() and output.exists():
            if read_json(receipt)['settings_sha256']!=fingerprint: raise ValueError('Record mismatch')
            continue
        started=time.perf_counter(); case=build_case(generator,by_id[row['id']]); case['f0']=args.f0_mhz*1e6
        medium=np.stack([case['maps'][k] for k in ('sound_speed','density')]).astype(np.float32)
        cached=np.load(cache/row['medium'],mmap_mode='r')
        if not np.array_equal(medium,cached): raise ValueError('Low/high frequency material mismatch')
        solver=generator.solver_for(case,case['maps'],fit); values=[]; timings=[]
        for ai,angle in enumerate(settings['acquisition']['angles_deg']):
            generator.set_angle(solver,case,angle); total,timing=solver.run()
            filtered,_=generator.sim.analytic_channels(total-refs[ai],case['dt'],band=band)
            values.append(filtered.T.astype(np.float32)); timings.append(float(timing['solve_readback_s']))
        values=np.stack(values)
        if values.shape!=(11,192,2401) or not np.isfinite(values).all(): raise ValueError('Invalid RF')
        durable_array(output,rf=values,source_tref_s=trefs)
        meta=dict(id=row['id'],role=row['role'],case=row['case'],variant=row['variant'],
            settings_sha256=fingerprint,medium_exactly_matches_v5=True,
            medium_sha256={k:sha256(v.tobytes()).hexdigest() for k,v in case['maps'].items()},
            rf_sha256=sha256(values.tobytes()).hexdigest(),seconds_per_angle=timings,
            wall_seconds=time.perf_counter()-started,source_tref_s=trefs.tolist())
        write_json(receipt,meta); print(json.dumps(meta),flush=True)
        del solver
    write_json(args.out/f'completion.shard{args.shard_id}.json',dict(records=len(selected[args.shard_id::args.shards]),settings_sha256=fingerprint))

if __name__=='__main__': main()
