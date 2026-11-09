"""One original-UltraWave worker; matched 60/90 us records, no PyTorch."""
from pathlib import Path
import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
import time
import traceback
if os.name=='posix':
    if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
        raise OSError('Cannot disable process transparent huge pages')
import numpy as np
from .fullwave import initialize
from .rf_representation import encode


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block=stream.read(8*2**20)
            if not block:break
            h.update(block)
    return h.hexdigest()


def atomic(path,value):
    temporary=path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def chunk_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if 'torch' in sys.modules:raise RuntimeError('No torch in the UltraWave worker')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise RuntimeError('This bounded acquisition uses only physical GPU 0')
    plan=json.loads((a.root/'material_plan.json').read_text())
    if plan['status']!='materials_prepared':raise RuntimeError('Require fixed material plan')
    if a.out.exists():raise RuntimeError('Inspect the existing generation worker before resuming')
    a.out.mkdir(parents=True)
    data=a.root/'rf';data.mkdir(exist_ok=True)
    index_path=a.root/'generated_index.json'
    if index_path.exists():raise RuntimeError('A complete/partial generation index already exists')
    low_path=Path('out/v5/low_frequency_3mhz_v1/index.json')
    low=json.loads(low_path.read_text());fit=low['absorption']
    acquisition=plan['acquisition'];angles=acquisition['angles_deg']
    if len(angles)!=11 or acquisition['native_nt']!=36001 or acquisition['native_dt_s']!=2.5e-9:
        raise ValueError('Unexpected acquisition')
    state={'status':'initializing','pid':os.getpid(),'device':0,
           'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
           'material_plan_SHA256':digest(a.root/'material_plan.json'),
           'source_SHA256':digest(Path(__file__)),'low_acquisition_SHA256':digest(low_path),
           'source_f0_hz':3e6,'native_nt':36001,'native_dt_s':2.5e-9,
           'records_planned':len(plan['records']),'completed_records':0,'completed_material_shots':0,
           'all_angles':11,'receivers':192,'new_centre_frequencies_hz':[3e6],
           'short_record_from_same_native_propagation':True,
           'gradient_or_HVP_evaluations':0,'created_unix_s':time.time()}
    atomic(a.out/'run.json',state)
    try:
        g=initialize()
        if 'torch' in sys.modules:raise RuntimeError('Original generator imported torch')
        state['original_generator_SHA256']=digest(Path(g.__file__))
        atomic(a.out/'run.json',state)

        def case_for(material=None):
            maps=None if material is None else {
                'sound_speed':np.ascontiguousarray(material[0]),
                'density':np.ascontiguousarray(material[1]),
                'alpha_coeff':np.ascontiguousarray(material[2]),
                'BonA':np.zeros_like(material[2])}
            case=g.geometry_case(maps)
            case.update(f0=3e6,nt=36001,end=90e-6,tref=4/3e6)
            return case

        reference_path=a.root/'reference_native.npy'
        if reference_path.exists():raise RuntimeError('Do not overwrite an existing reference')
        temporary=reference_path.with_suffix('.partial.npy')
        if temporary.exists():raise RuntimeError('Inspect partial reference ledger before resuming')
        reference=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(11,36001,192))
        case=case_for();solver=g.solver_for(case,case['maps'],fit)
        reference_shots=[]
        for angle_index,angle in enumerate(angles):
            tref=g.set_angle(solver,case,angle)
            raw,timing=solver.run()
            if raw.shape!=(36001,192) or not np.isfinite(raw).all():raise ValueError('Invalid reference RF')
            reference[angle_index]=raw;reference.flush()
            reference_shots.append({'angle_index':angle_index,'angle_deg':angle,'tref_s':tref,
                                    'SHA256':chunk_hash(reference[angle_index]),
                                    'seconds':float(timing['solve_readback_s'])})
            atomic(a.root/'reference_progress.json',{'completed_shots':reference_shots,
                   'material_plan_SHA256':state['material_plan_SHA256']})
            atomic(a.out/'progress.json',{'phase':'reference','completed_reference_shots':len(reference_shots),
                   'planned_reference_shots':11,'unix_s':time.time()})
            print(json.dumps({'phase':'reference','angle_index':angle_index,
                              'seconds':reference_shots[-1]['seconds']}),flush=True)
        del raw,solver,reference
        gc.collect();temporary.replace(reference_path)
        reference=np.load(reference_path,mmap_mode='r')
        state['reference_SHA256']=digest(reference_path)
        state['status']='generating';atomic(a.out/'run.json',state)
        generated=[];started=time.time()
        training=[r for r in plan['records'] if r['role']=='train']
        development=[r for r in plan['records'] if r['role']!='train']
        # Interleave only scheduling; development observations never enter training.
        ordered=[]
        for i,row in enumerate(training):
            ordered.append(row)
            if i%6==5 and development:ordered.append(development.pop(0))
        ordered.extend(development)
        for row in ordered:
            if digest(row['material3'])!=row['material_file_SHA256']:
                raise ValueError('Fixed material was modified')
            material=np.load(row['material3'])
            if chunk_hash(material)!=row['material_array_SHA256']:
                raise ValueError('Material contents do not match the fixed plan')
            case=case_for(material);solver=g.solver_for(case,case['maps'],fit)
            long_path=data/(row['id']+'.rf90.npy');short_path=data/(row['id']+'.iq60.npy')
            receipt=data/(row['id']+'.json')
            if any(v.exists() for v in [long_path,short_path,receipt,
                                        long_path.with_suffix('.partial.npy'),short_path.with_suffix('.partial.npy')]):
                raise RuntimeError('Inspect existing record/shot ledger; never replay implicitly')
            long_temp=long_path.with_suffix('.partial.npy')
            short_temp=short_path.with_suffix('.partial.npy')
            long=np.lib.format.open_memmap(long_temp,mode='w+',dtype=np.float32,shape=(11,192,3601))
            short=np.lib.format.open_memmap(short_temp,mode='w+',dtype=np.float32,shape=(11,2,192,640))
            shots=[]
            for angle_index,angle in enumerate(angles):
                tref=g.set_angle(solver,case,angle)
                raw,timing=solver.run()
                if raw.shape!=(36001,192) or not np.isfinite(raw).all():raise ValueError('Invalid material RF')
                scattered=raw-reference[angle_index]
                full,_=g.sim.analytic_channels(scattered,2.5e-9,band=[1.5e6,4.5e6])
                prefix,_=g.sim.analytic_channels(scattered[:24001],2.5e-9,band=[1.5e6,4.5e6])
                if full.shape!=(3601,192) or prefix.shape!=(2401,192):raise ValueError('Unexpected processed RF clock')
                encoded,error=encode(prefix.T[None])
                if not np.isfinite(full).all() or not np.isfinite(encoded).all() or error>1e-3:
                    raise ValueError('RF compression or signal is invalid')
                long[angle_index]=full.T;short[angle_index]=encoded[0]
                long.flush();short.flush()
                shots.append({'angle_index':angle_index,'angle_deg':angle,'source_tref_s':tref,
                              'seconds':float(timing['solve_readback_s']),
                              'RF90_SHA256':chunk_hash(long[angle_index]),
                              'IQ60_SHA256':chunk_hash(short[angle_index]),
                              'IQ60_relative_rms_error':error})
                atomic(data/(row['id']+'.shots.json'),{'material_file_SHA256':row['material_file_SHA256'],
                       'material_plan_SHA256':state['material_plan_SHA256'],'completed_shots':shots,
                       'long_partial':str(long_temp.resolve()),'short_partial':str(short_temp.resolve())})
                state['completed_material_shots']+=1
                progress={'phase':'materials','id':row['id'],'role':row['role'],
                          'completed_records':len(generated),'planned_records':len(ordered),
                          'record_completed_shots':len(shots),'completed_material_shots':state['completed_material_shots'],
                          'last_solve_seconds':shots[-1]['seconds'],'seconds':time.time()-started,'unix_s':time.time()}
                atomic(a.out/'progress.json',progress)
                print(json.dumps(progress),flush=True)
                del raw,scattered,full,prefix,encoded
            del solver,long,short,material
            gc.collect();long_temp.replace(long_path);short_temp.replace(short_path)
            record={**row,'rf90':str(long_path.resolve()),'iq60':str(short_path.resolve()),
                    'RF90_file_SHA256':digest(long_path),'IQ60_file_SHA256':digest(short_path),
                    'source_mhz':3.,'shots':shots,'original_time_solver':True,
                    'reference_SHA256':state['reference_SHA256'],'same_material_for_both_record_windows':True}
            atomic(receipt,record);generated.append(record)
            atomic(index_path,{'status':'generating','records':generated,'acquisition':acquisition,
                   'material_plan_SHA256':state['material_plan_SHA256'],'completed':len(generated),
                   'planned_records':len(ordered),'only_new_centre_frequency_hz':3e6})
            state['completed_records']=len(generated);atomic(a.out/'run.json',state)
        atomic(index_path,{'status':'complete','records':generated,'acquisition':acquisition,
               'material_plan_SHA256':state['material_plan_SHA256'],'completed':len(generated),
               'planned_records':len(ordered),'only_new_centre_frequency_hz':3e6})
        state.update(status='generation_finished',finished_unix_s=time.time(),seconds=time.time()-started)
        atomic(a.out/'run.json',state)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),failed_unix_s=time.time())
        atomic(a.out/'run.json',state);raise


if __name__=='__main__':
    main()
