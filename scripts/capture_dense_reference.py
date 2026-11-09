"""Capture all dense homogeneous frequency fields in one time solve per angle.

Runs without PyTorch. Keeps exact original pressure update and end-step timing.
"""
from argparse import ArgumentParser
from pathlib import Path
import inspect,json,time,hashlib,gc
import numpy as np
from scipy import fft
from nsrno.fullwave import initialize

def main():
    p=ArgumentParser(); p.add_argument('--angle',type=int,required=True)
    p.add_argument('--out',type=Path,default=Path('out/v5/conditioned_dense_v1'))
    p.add_argument('--last-bin',type=int,default=450); args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); (args.out/'reference').mkdir(exist_ok=True)
    path=args.out/'reference'/f'a{args.angle:02d}.npy'
    if path.exists() and path.with_suffix('.json').exists(): print('Reference already complete'); return
    gen=initialize(); import ultrawave
    original=ultrawave.Acoustic2DOperator; code=inspect.getsource(original)
    code=code.replace('    dt = model.critical_dt', '''    dt = model.critical_dt
    t_capture = ConditionalDimension(name='t_capture', parent=model.grid.time_dim, factor=10)
    pressure_snapshots = TimeFunction(name='pressure_snapshots', grid=model.grid,
        time_order=0, space_order=0, save=2401, time_dim=t_capture)
    capture_terms = [Eq(pressure_snapshots, p.forward)]
''')
    if code.count('+ rec_term)')!=2: raise ValueError('Unexpected physical generator')
    code=code.replace('+ rec_term)', '+ rec_term + capture_terms)')
    namespace={**original.__globals__}; exec(code,namespace)
    ultrawave.Acoustic2DOperator=namespace['Acoustic2DOperator']
    (args.out/f'instrumented_operator_a{args.angle:02d}.py').write_text(code)
    case=gen.geometry_case(); fit=gen.absorption_model(case)
    solver=gen.solver_for(case,case['maps'],fit)
    angle=float(gen.ANGLES[args.angle]); tref=gen.set_angle(solver,case,angle)
    start=time.perf_counter(); rf,timing=solver.run()
    parameters={v.name:v for v in solver.op.parameters}
    snapshots=parameters['pressure_snapshots'].data
    if snapshots.shape!=(2401,len(case['x']),len(case['z'])): raise ValueError(str(snapshots.shape))
    # Receiver coordinates lie exactly at native columns; verify sampling/timing.
    cols=np.rint((case['xe']-case['x'][0])/case['dx']).astype(int)
    observed=snapshots[:,cols,case['face']]
    sampled=rf[::10]
    sampling_error=float(np.linalg.norm(observed-sampled)/np.linalg.norm(sampled))
    if sampling_error>2e-4: raise RuntimeError('Snapshot receiver mismatch '+str(sampling_error))
    bins=np.arange(241,args.last_bin+1); target=np.lib.format.open_memmap(path.with_suffix('.partial.npy'),mode='w+',dtype=np.complex64,
        shape=(len(bins),len(case['z']),len(case['x'])))
    # FFT in lateral strips avoids a second20GB allocation and keeps workers
    # below the host-memory budget. Source/field clocks retain the RF convention.
    for left in range(0,len(case['x']),32):
        block=fft.rfft(snapshots[:,left:left+32,:],axis=0,norm='forward',workers=1)
        target[:,:,left:left+32]=block[bins].transpose(0,2,1)
    target.flush(); del target
    path.with_suffix('.partial.npy').replace(path)
    meta={'angle_index':args.angle,'angle_deg':angle,'bins':bins.tolist(),'shape':[len(bins),len(case['z']),len(case['x'])],
        'source_tref_s':tref,'sampling_receiver_error':sampling_error,'seconds':time.perf_counter()-start,
        'native_dt_s':gen.DT,'native_nt':gen.NT,'capture_fs_hz':40e6,'fit':fit,
        'original_operator_sha256':hashlib.sha256(inspect.getsource(original).encode()).hexdigest()}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta),flush=True)

if __name__=='__main__': main()
