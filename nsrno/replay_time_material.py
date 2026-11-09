"""Independent original-UltraWave replay of a recovered three-parameter map."""
from argparse import ArgumentParser
from pathlib import Path
import json,time
import numpy as np
from scipy import fft
from .fullwave import initialize
from .v5_observations import plane_observation,canonical_id

def project(rf):
    spectrum=fft.rfft(rf,axis=-1);spectrum[...,:40]=0;spectrum[...,680:]=0
    return fft.irfft(spectrum,n=2401,axis=-1)

def decode_numpy(iq):
    spectrum=fft.fftshift(fft.fft(iq[:,0]+1j*iq[:,1],axis=-1,norm='forward'),axes=-1)/np.sqrt(2)
    return fft.irfft(np.pad(spectrum,((0,0),(0,0),(40,521))),n=2401,axis=-1,norm='forward')

def main():
    p=ArgumentParser();p.add_argument('material',type=Path);p.add_argument('--source',choices=['lf','hf','point3'],default='lf');p.add_argument('--sample',default='train_200');p.add_argument('--out',type=Path)
    args=p.parse_args();m=np.load(args.material);g=initialize()
    maps=dict(sound_speed=m[0],density=m[1],alpha_coeff=m[2],BonA=np.zeros_like(m[0]))
    case=g.geometry_case(maps);lf=json.loads(Path('out/v5/low_frequency_3mhz_v1/index.json').read_text())
    if args.source=='point3':
        from .generate_point_tx import configure_case,project as point_project
        point=json.loads(Path('out/v5/point_tx_3mhz_v1/index.json').read_text())
        case=configure_case(g,maps);fit=point['absorption'];band=point['acquisition']['band_hz']
        ref=np.load(point['reference_native'],mmap_mode='r');weights=np.load(point['source_layout'])['source_weights']
        row=next(r for r in point['records'] if r['id']==canonical_id(args.sample));obs=np.load(row['rf'])
    elif args.source=='lf':
        case['f0']=3e6;fit=lf['absorption'];ref=np.load('out/v5/low_frequency_3mhz_v1/reference.shard0.npz')['rf_native'];band=[1.5e6,4.5e6];cache=Path('out/v5/forward_cache_3mhz_v1')
    else:
        fit=g.absorption_model(case);manifest=json.loads(Path('out/v5/manifest.json').read_text());ref=np.moveaxis(np.load(manifest['reference_file'])['rf_native'],-1,0);band=[4e6,7.5e6];cache=Path('out/v5/forward_cache_v1')
    if args.source!='point3':obs=decode_numpy(np.load(plane_observation(args.source,args.sample)['iq']))
    solver=g.solver_for(case,maps,fit);values=[];start=time.time()
    for i,angle in enumerate(lf['acquisition']['angles_deg']):
        if args.source=='point3':
            case['weights']=weights[i];_,_,wave=g.bench.source_arrays(case);solver.source.data[:]=wave.T
        else:g.set_angle(solver,case,angle)
        raw,timing=solver.run();processed,_=g.sim.analytic_channels(raw-ref[i],g.DT,band=band)
        values.append(point_project(processed.T) if args.source=='point3' else project(processed.T))
        print(json.dumps({'source':args.source,'angle':i,'seconds':timing['solve_readback_s']}),flush=True)
    prediction=np.stack(values);difference=prediction-obs
    r={'material':str(args.material),'source':args.source,'sample':args.sample,'solver':'original UltraWave; independent of new CUDA implementation','relative_rms':float(np.linalg.norm(difference)/np.linalg.norm(obs)),'windows':{},'seconds':time.time()-start}
    for lo,hi in [(0,8),(8,18),(18,36),(36,90.025 if args.source=='point3' else 60.025)]:
        sl=slice(round(lo*40),round(hi*40));r['windows'][f'{lo}-{hi}']=float(np.linalg.norm(difference[:,:,sl])/np.linalg.norm(obs[:,:,sl]))
    out=args.out or args.material.parent/f'{args.material.stem}_original_{args.source}'
    np.savez(out.with_suffix('.npz'),prediction=prediction,target=obs)
    out.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)

if __name__=='__main__':main()
