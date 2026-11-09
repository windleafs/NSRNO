"""Original UltraWave targets for independent three-parameter states. No torch."""
from pathlib import Path
from argparse import ArgumentParser
import json,time,hashlib,os
import numpy as np
from nsrno.fullwave import initialize,read_reference
from nsrno.generate_low_frequency import durable_array
from nsrno.rf_representation import encode
from nsrno.dataset import write_json

ROOT=Path('out/v5/uno_triparam_v1')

def main():
    p=ArgumentParser(); p.add_argument('--part',type=int,required=True); p.add_argument('--parts',type=int,default=3); args=p.parse_args()
    start=time.monotonic()
    while not (ROOT/'preparation_complete.json').exists():
        if time.monotonic()-start>3600: raise TimeoutError('UNO preparation did not complete')
        time.sleep(10)
    plan=json.load(open(ROOT/'augmentation_plan.json'))['materials']; selected=plan[args.part::args.parts]
    dest=ROOT/'paired'; dest.mkdir(exist_ok=True); gen=initialize()
    manifest=json.load(open('out/v5/manifest.json')); low=json.load(open('out/v5/low_frequency_3mhz_v1/index.json'))
    high_reference=read_reference(gen,manifest).transpose(2,0,1)
    with np.load('out/v5/low_frequency_3mhz_v1/reference.shard0.npz') as archive: low_reference=archive['rf_native']
    refs={7.5:high_reference,3.:low_reference}; records=[]
    for item in selected:
        material=np.load(item['material3'])
        if hashlib.sha256(material.tobytes()).hexdigest()!=item['sha256']: raise ValueError('Material hash mismatch')
        for source in item['source_mhz']:
            name=item['id']+('_hf' if source==7.5 else '_lf'); receipt=dest/(name+'.json')
            if receipt.exists(): records.append(json.load(open(receipt))); continue
            case=gen.geometry_case(); case['f0']=source*1e6; case['tref']=4/case['f0']
            case['maps']={'sound_speed':np.ascontiguousarray(material[0]),'density':np.ascontiguousarray(material[1]),
                'alpha_coeff':np.ascontiguousarray(material[2]),'BonA':np.zeros_like(material[2])}
            if source==7.5: band=manifest['acquisition']['band_hz']; fit=gen.absorption_model(case)
            else: band=low['acquisition']['band_hz']; fit=low['absorption']
            solver=gen.solver_for(case,case['maps'],fit); wave=[]; trefs=[]; timings=[]
            for a,angle in enumerate(manifest['acquisition']['angles_deg']):
                trefs.append(gen.set_angle(solver,case,angle)); tic=time.perf_counter(); raw,_=solver.run()
                rf,_=gen.sim.analytic_channels(raw-refs[source][a],case['dt'],band=band)
                wave.append(rf.T.astype(np.float32)); timings.append(time.perf_counter()-tic)
                print(json.dumps({'id':name,'angle':a,'seconds':timings[-1]}),flush=True)
            wave=np.stack(wave); iq,error=encode(wave); del solver
            raw_path=dest/(name+'.npz'); durable_array(raw_path,rf=wave)
            iq_path=dest/(name+'.iq.npy'); tmp=iq_path.with_suffix('.partial.npy')
            with tmp.open('wb') as stream: np.save(stream,iq); stream.flush(); os.fsync(stream.fileno())
            tmp.replace(iq_path)
            row={**item,'id':name,'source_mhz':source,'iq':str(iq_path.resolve()),'raw_rf':str(raw_path.resolve()),
                'source_tref_s':trefs,'seconds_per_angle':timings,'rf_representation_relative_rms':error,
                'band_hz':band,'absorption':fit,'original_time_solver':True,'angles_deg':manifest['acquisition']['angles_deg']}
            write_json(receipt,row); records.append(row)
            write_json(ROOT/f'augmentation_part{args.part}.json',{'records':records})
            print(json.dumps({'finished_source_record':name,'records':len(records)}),flush=True)
    write_json(ROOT/f'generation_complete_{args.part}.json',{'completed':True,'records':len(records)})

if __name__=='__main__': main()
