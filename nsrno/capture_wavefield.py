"""Capture one V5 full-wave pressure Fourier field without changing its update."""
from argparse import ArgumentParser
from pathlib import Path
import inspect,json,time,hashlib
import numpy as np
from .fullwave import initialize,build_case
from .dataset import read_json,write_json

def main():
    parser=ArgumentParser(); parser.add_argument('--id',default='train_200'); parser.add_argument('--bin',type=int,default=360)
    parser.add_argument('--angle',type=int,default=5); parser.add_argument('--out',type=Path,default=Path('out/v5/wave_diagnosis_v1'))
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    gen=initialize(); manifest=read_json(Path('out/v5/manifest.json'))
    row=next(r for r in manifest['records'] if r['id']==args.id)
    frequency=args.bin*40e6/2401; duration=2401/40e6
    import ultrawave
    original=ultrawave.Acoustic2DOperator
    code=inspect.getsource(original)
    code=code.replace('    dt = model.critical_dt','''    dt = model.critical_dt
    fourier_real = Function(name='fourier_real', grid=model.grid, space_order=0)
    fourier_imag = Function(name='fourier_imag', grid=model.grid, space_order=0)
    frequency_cos = Function(name='frequency_cos', dimensions=(model.grid.time_dim,), shape=(CAPTURE_NT,), space_order=0)
    frequency_sin = Function(name='frequency_sin', dimensions=(model.grid.time_dim,), shape=(CAPTURE_NT,), space_order=0)
    phase=2*np.pi*CAPTURE_FREQUENCY*np.arange(CAPTURE_NT)*CAPTURE_DT
    frequency_cos.data[:]=np.cos(phase)*CAPTURE_DT/CAPTURE_DURATION
    frequency_sin.data[:]=-np.sin(phase)*CAPTURE_DT/CAPTURE_DURATION
    fourier_terms=[Eq(fourier_real,fourier_real+p.forward*frequency_cos), Eq(fourier_imag,fourier_imag+p.forward*frequency_sin)]
''')
    if code.count('+ rec_term)')!=2: raise ValueError('Unexpected physical generator source; inspect before adapting')
    code=code.replace('+ rec_term)', '+ rec_term + fourier_terms)')
    namespace={**original.__globals__,'CAPTURE_NT':gen.NT,'CAPTURE_DT':gen.DT,
               'CAPTURE_FREQUENCY':frequency,'CAPTURE_DURATION':duration}
    exec(code,namespace); ultrawave.Acoustic2DOperator=namespace['Acoustic2DOperator']
    (args.out/'instrumented_operator.py').write_text(code)
    outputs={}
    for kind in ('reference','material'):
        path=args.out/f'{args.id}_{kind}_field.npz'
        if path.exists(): raise FileExistsError(path)
        case=gen.geometry_case() if kind=='reference' else build_case(gen,row)
        fit=gen.absorption_model(case); solver=gen.solver_for(case,case['maps'],fit)
        tref=gen.set_angle(solver,case,manifest['acquisition']['angles_deg'][args.angle])
        timer=time.perf_counter(); rf,timing=solver.run()
        parameters={v.name:v for v in solver.op.parameters}
        pressure=np.array(parameters['fourier_real'].data.T+1j*parameters['fourier_imag'].data.T)
        receivers=np.rint((case['xe']-case['x'][0])/case['dx']).astype(int)
        weight=np.exp(-2j*np.pi*frequency*np.arange(case['nt'])*case['dt'])*case['dt']/duration
        receiver_dft=weight@rf
        captured=pressure[case['face'],receivers]
        error=float(np.linalg.norm(captured-receiver_dft)/np.linalg.norm(receiver_dft))
        if error>2e-4: raise ValueError(f'Captured field and receiver DFT disagree: {error}')
        arrays=dict(pressure=pressure,rf_native=rf,receiver_dft=receiver_dft,
                    c=np.array(parameters['vp'].data.T),rho=np.array(parameters['rho'].data.T),
                    eta1=np.array(parameters['eta1'].data.T),eta2=np.array(parameters['eta2'].data.T),
                    original_alpha=case['maps']['alpha_coeff'],original_c=case['maps']['sound_speed'],
                    original_rho=case['maps']['density'])
        for name in ('p','vx','vy','rhox','rhoy','S1','S2'):
            arrays[name+'_final']=np.array(parameters[name].data[case['nt']%2].T)
        np.savez(path,**arrays)
        meta=dict(id=args.id,kind=kind,frequency_hz=frequency,angle_index=args.angle,
             source_tref_s=tref,dt_s=case['dt'],nt=case['nt'],dx_m=case['dx'],face=case['face'],
             x0_m=float(case['x'][0]),z0_m=float(case['z'][0]),pml=case['pml'],fit=fit,
             captured_receiver_relative_error=error,seconds=time.perf_counter()-timer,
             original_operator_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest())
        write_json(path.with_suffix('.json'),meta); outputs[kind]=rf
        print(json.dumps(meta),flush=True); del solver
    rf,_=gen.sim.analytic_channels(outputs['material']-outputs['reference'],gen.DT,band=manifest['acquisition']['band_hz'])
    np.save(args.out/f'{args.id}_processed_rf.npy',rf.T)
    write_json(args.out/'capture_completion.json',dict(id=args.id,frequency_hz=frequency,angle=args.angle))

if __name__=='__main__': main()
