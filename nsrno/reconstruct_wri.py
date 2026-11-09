"""Full native three-material first-order wavefield reconstruction.

Wavefield and multiplier banks are streamed from NVMe. All material variables
are independent native-grid fields. Observation-only full RF chooses a saved
candidate; it does not roll back the augmented trajectory.
"""
from pathlib import Path
import ctypes,os
if os.name=='posix' and ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
import argparse,json,time,subprocess,sys,hashlib,shutil,traceback
import multiprocessing as mp
import numpy as np
from .wri_material import MaterialVariables,MaterialNormal
from .wri_worker import run_worker,atomic


def gpu_memory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'],text=True)
    return {int(s.split(',')[0]):int(s.split(',')[1]) for s in rows.splitlines()}


class Resources:
    def __init__(self,out,resume=False):
        self.out=out;self.maximum={};self.minimum_available=None
        self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if resume and (out/'resource_summary.json').exists():
            previous=json.loads((out/'resource_summary.json').read_text())
            self.maximum={int(k):v for k,v in previous['sampled_maximum_card_MiB'].items()}
            self.minimum_available=previous['minimum_host_available_GiB']
        self.log=open(out/(f'resources_{self.boot[:8]}.jsonl' if resume else 'resources.jsonl'),'a',buffering=1)

    def sample(self,phase):
        used=gpu_memory()
        for device,value in used.items():self.maximum[device]=max(self.maximum.get(device,0),value)
        info=Path('/proc/meminfo').read_text().splitlines()
        available=int(next(s.split()[1] for s in info if s.startswith('MemAvailable:')))/2**20
        self.minimum_available=available if self.minimum_available is None else min(self.minimum_available,available)
        row={'unix_s':time.time(),'boot_id':self.boot,'phase':phase,'card_MiB':used,'host_available_GiB':available,
             'disk_free_GiB':shutil.disk_usage(self.out).free/2**30}
        self.log.write(json.dumps(row)+'\n')
        atomic(self.out/'resource_summary.json',{'sampled_maximum_card_MiB':self.maximum,
            'observed_above_16_GiB':any(v>16384 for v in self.maximum.values()),
            'minimum_host_available_GiB':self.minimum_available,'latest':row,
            'sampling_interval_seconds':20,'sampled_not_continuous':True})
        return used

    def wait_budget(self,devices,cap,phase):
        while True:
            used=self.sample(phase)
            if all(used[d]+cap*1024+768<=16384 for d in devices):return
            print(json.dumps({'waiting':'whole_card_budget','phase':phase,'used_MiB':used,
                              'per_worker_cap_GiB':cap}),flush=True)
            time.sleep(20)


def command(args,log,monitor,phase):
    with open(log,'w',buffering=1) as stream:
        process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        while process.poll() is None:
            monitor.sample(phase);time.sleep(20)
        if process.returncode:raise RuntimeError(f'{phase} exited {process.returncode}; see {log}')


def score(material,sample,out,devices,cap,monitor):
    used=monitor.sample('native_RF_admission');device=min(devices,key=lambda d:used[d])
    monitor.wait_budget([device],cap,'native_RF_admission')
    command([sys.executable,'-m','nsrno.score_wri_material',str(material),'--sample',sample,
             '--out',str(out),'--device',str(device),'--cap-gib',str(cap)],
            out.with_suffix('.log'),monitor,'native_full_RF')
    return json.loads(out.read_text())


def evaluate(material,sample,seed,monitor):
    # This separate, post-hoc process is the only reader of material truth.
    initial=material.parent/'stage0_initial_material.npy'
    if not initial.exists():shutil.copy2(seed,initial)
    command([sys.executable,'-m','nsrno.evaluate_time_material',str(material),'--sample',sample,
             '--out',str(material.parent/'maps'),'--estimate-label','First-order WRI'],
            material.parent/'map_evaluation.log',monitor,'post_hoc_maps_only')


def workers(out,sweep,mode,assignments,args,monitor,field_iterations,tolerance):
    monitor.wait_budget(args.devices,args.cap_gib,mode+'_admission')
    context=mp.get_context('spawn');results=context.Queue();processes=[]
    for rank,device in enumerate(args.devices):
        process=context.Process(target=run_worker,args=(rank,device,assignments[rank],out,
            sweep,mode,args.cap_gib,results,field_iterations,tolerance))
        process.start();processes.append(process)
    atomic(out/(mode+'_processes.json'),{'mode':mode,'pids':[p.pid for p in processes],
        'devices':args.devices,'per_worker_cap_GiB':args.cap_gib,'truth_read':False})
    while any(p.is_alive() for p in processes):
        monitor.sample(mode);time.sleep(20)
    for p in processes:p.join()
    codes=[p.exitcode for p in processes]
    if any(c!=0 for c in codes):raise RuntimeError(f'{mode} worker exit codes {codes}; partial banks retained, do not resume as a complete sweep')
    summaries=[json.loads((out/f'{mode}_rank{r}.json').read_text()) for r in range(len(processes))]
    for r,summary in enumerate(summaries):
        if summary['status']!='complete' or summary['frequencies']!=len(assignments[r]):
            raise RuntimeError('Incomplete frequency sweep')
    results.close()
    return summaries


def create_banks(out,groups,shape,warm_allowed=False):
    folder=out/'banks';folder.mkdir()
    required=sum(len(bins)*11*3*int(np.prod(shape))*8*2 for bins in groups.values())
    required+=sum(len(bins)*int(np.prod(shape))*12 for bins in groups.values())
    if shutil.disk_usage(out).free<required+32*2**30:raise RuntimeError('Insufficient disk for native field banks and moments')
    for source,bins in groups.items():
        for name in ['X','Y']:
            bank=np.lib.format.open_memmap(folder/f'{source}_{name}.npy',mode='w+',
                dtype=np.complex64,shape=(len(bins),11,3,*shape))
            # New sparse files read as zeros; touching the entire bank here is unnecessary.
            bank.flush();del bank
        for name,dtype,bank_shape in [('Ydata',np.complex64,(len(bins),11,192)),
                                      ('penalty',np.float64,(len(bins),))]:
            bank=np.lib.format.open_memmap(folder/f'{source}_{name}.npy',mode='w+',dtype=dtype,shape=bank_shape)
            bank[:]=0;bank.flush();del bank
    warm=Path('out/v5/first_wavefield_update_v1_train_200/scattered_field.npy')
    if warm_allowed and warm.exists() and 'lf' in groups and 180 in groups['lf']:
        bank=np.load(folder/'lf_X.npy',mmap_mode='r+')
        bank[groups['lf'].index(180),0]=np.load(warm);bank.flush();del bank


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--sample',default='train_200');parser.add_argument('--devices',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--cap-gib',type=float,default=1.1);parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--field-cg',type=int,default=64);parser.add_argument('--material-cg',type=int,default=64)
    parser.add_argument('--tv',type=float,default=.001);parser.add_argument('--damping',type=float,default=.01)
    parser.add_argument('--resume-wait',action='store_true',help='Explicit recovery only before banks exist, after a confirmed process exit')
    parser.add_argument('--seed',type=Path,default=Path('out/v5/reconstruction_geometry_v1_train_200/snapshots/stage0_step064'))
    args=parser.parse_args();out=args.out.resolve()
    resumed=None
    if (out/'run.json').exists():
        if not args.resume_wait:raise RuntimeError('Output already contains a run; automatic partial-sweep resumption is disabled')
        resumed=json.loads((out/'run.json').read_text())
        if Path('/proc').joinpath(str(resumed['pid']),'cmdline').exists():
            raise RuntimeError('The recorded process handle still exists; no duplicate recovery')
        if resumed['status']!='waiting_complete_references' or (out/'banks').exists() or any(out.glob('step[0-9]*')):
            raise RuntimeError('Only a saved pre-bank reference wait can be recovered here')
        expected={'sample':args.sample,'devices':args.devices,'cap_GiB':args.cap_gib,'iterations':args.iterations,
                  'initial_field_CG':args.field_cg,'material_CG':args.material_cg,'TV_weight':args.tv,'damping_relative':args.damping}
        if any(resumed[k]!=v for k,v in expected.items()) or Path(resumed['seed']).resolve()!=args.seed.resolve():
            raise RuntimeError('Recovery arguments differ from the recorded numerical experiment')
        for name,sha in resumed['source_sha256'].items():
            if name!='reconstruct_wri.py' and hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=sha:
                raise RuntimeError('Numerical source changed before recovery: '+name)
        for name in ['material.npy','parameters.npy','variables.npy','rf_score.json']:
            if not (out/'initial'/name).exists():raise RuntimeError('Missing saved initial '+name)
        if not (out/'best_RF.json').exists():raise RuntimeError('Missing RF-selected initial candidate')
    elif args.resume_wait:raise RuntimeError('No saved wait phase to recover')
    out.mkdir(parents=True,exist_ok=True)
    start=resumed['created_unix_s'] if resumed else time.time();monitor=Resources(out,resume=bool(resumed))
    source_files=['reconstruct_wri.py','wri_worker.py','wri_material.py','frequency_group_data.py',
        'score_wri_material.py','first_order_wavefield.py','wavefield_step.py','conditioned_wave.py',
        'dense_wave.py','native_time.py','native_time.cu','coupled_coordinates.py','lf_frequency_data.py']
    archive=out/'source';archive.mkdir(exist_ok=bool(resumed));hashes={}
    for name in source_files:
        path=Path(__file__).parent/name;hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if not resumed:shutil.copy2(path,archive/name)
    run={'status':'initial_RF_score','pid':os.getpid(),'sample':args.sample,'created_unix_s':start,
        'sources':['lf','point3'],'all_transmissions_per_group':11,'receivers':192,
        'native_material_shape':[3,961,1121],'unknowns':['density','compressibility','attenuation_scale'],
        'physical_outputs':['sound_speed','mass_density','tissue_attenuation'],
        'truth_read':False,'post_hoc_truth_process':True,'devices':args.devices,'cap_GiB':args.cap_gib,
        'whole_card_limit_GiB':16,'iterations':args.iterations,'initial_field_CG':args.field_cg,
        'material_CG':args.material_cg,'TV_weight':args.tv,'damping_relative':args.damping,
        'seed':str(args.seed),'source_sha256':hashes,
        'selection':'mean full point+LF RF energy-normalized error; HF diagnostic until final refinement',
        'model_acceptance':'exact fixed-field half-objective change plus independent physical TV',
        'trajectory':'continues independently of RF candidate ranking; no RF rollback',
        'known_approximation':'frequency-domain omitted finite-record endpoint terms; native time score required',
        'scope':'includes additional existing point transmissions; does not establish plane-only sufficiency'}
    if resumed:
        shutil.copy2(out/'run.json',out/f'run_before_recovery_{os.getpid()}.json')
        shutil.copy2(Path(__file__),archive/f'reconstruct_wri_recovery_{os.getpid()}.py')
        run=resumed.copy()
        run.setdefault('recovery_history',[]).append({'previous_pid':resumed['pid'],'previous_handle_missing':True,
            'pid':os.getpid(),'boot_id':monitor.boot,'unix_s':time.time(),'initial_RF_reused':True,
            'banks_preexisting':False,'signals_sent':False,'controller_sha256':hashes['reconstruct_wri.py']})
        run['pid']=os.getpid()
    run['boot_id']=monitor.boot
    atomic(out/'run.json',run)
    try:
        initial=out/'initial';initial.mkdir(exist_ok=bool(resumed))
        if not resumed:
            for name in ['material.npy','parameters.npy']:shutil.copy2(args.seed/name,initial/name)
        variable=MaterialVariables(np.load(initial/'material.npy'),np.load(initial/'parameters.npy'))
        if resumed:
            current=np.load(initial/'variables.npy');best=json.loads((out/'best_RF.json').read_text())
            print(json.dumps({'recovered_pre_bank_wait':True,'pid':os.getpid(),'initial_RF_reused':True}),flush=True)
        else:
            current=variable.initial.copy();np.save(initial/'variables.npy',current)
            initial_score=score(initial/'material.npy',args.sample,initial/'rf_score.json',args.devices,args.cap_gib,monitor)
            best={'step':0,'material':str(initial/'material.npy'),'primary_loss':initial_score['primary_loss']}
            atomic(out/'best_RF.json',best)
            evaluate(initial/'material.npy',args.sample,initial/'material.npy',monitor)
        roots={'lf':Path('out/v5/conditioned_lf_full_v1'),'point3':Path('out/v5/conditioned_point_full_v1')}
        run['status']='waiting_complete_references';atomic(out/'run.json',run)
        while not all((p/'index.json').exists() for p in roots.values()):
            monitor.sample('waiting_complete_references');time.sleep(20)
        specs={source:json.loads((path/'index.json').read_text())['spec'] for source,path in roots.items()}
        groups={source:spec['bins'] for source,spec in specs.items()}
        if any(spec['medium_shape'][-2:]!=list(variable.shape) for spec in specs.values()):
            raise RuntimeError('Native reference shape differs')
        run['frequency_counts']={s:len(b) for s,b in groups.items()};atomic(out/'run.json',run)
        warm_allowed=(args.sample=='train_200' and args.seed.resolve()==Path(
            'out/v5/reconstruction_geometry_v1_train_200/snapshots/stage0_step064').resolve())
        create_banks(out,groups,variable.shape,warm_allowed)
        tasks=[(s,i) for s,bins in groups.items() for i in range(len(bins))]
        assignments=[tasks[r::len(args.devices)] for r in range(len(args.devices))]
        for step in range(1,args.iterations+1):
            folder=out/f'step{step:03d}';folder.mkdir();(folder/'moments').mkdir()
            np.save(folder/'old_variables.npy',current);np.save(folder/'old_material.npy',variable.material(current))
            run.update(status='field_sweep',step=step,phase_started_unix_s=time.time());atomic(out/'run.json',run)
            iterations=args.field_cg if step<=2 else (96 if step<=4 else 128)
            tolerance=.01 if step<=2 else .003
            records=workers(folder,step,'field',assignments,args,monitor,iterations,tolerance)
            normal={};moments=[]
            for record in records:
                with np.load(record['normal_path']) as data:
                    for key in data.files:
                        if key in normal:normal[key]+=data[key]
                        else:normal[key]=data[key].copy()
                moments.extend(record['moments'])
            if len(moments)!=len(tasks):raise RuntimeError('Missing fixed-field frequency moments')
            run['status']='material_update';atomic(out/'run.json',run)
            normal_system=MaterialNormal(variable,records[0]['geometry'],normal)
            def progress(row):
                atomic(folder/'material_CG_progress.json',row)
                if row['iteration']%8==0:print(json.dumps({'step':step,'material_CG':row}),flush=True)
            proposed,material_record=normal_system.solve(current,moments,args.tv,args.damping,
                maximum=args.material_cg,callback=progress)
            np.save(folder/'variables.npy',proposed);np.save(folder/'material.npy',variable.material(proposed))
            np.save(folder/'parameters.npy',variable.parameters(proposed));atomic(folder/'material_update.json',material_record)
            del normal_system,normal,records,moments
            run['status']='dual_update';atomic(out/'run.json',run)
            workers(folder,step,'dual',assignments,args,monitor,iterations,tolerance)
            current=proposed;atomic(folder/'complete_sweep.json',{'step':step,'status':'complete','unix_s':time.time(),'truth_read':False})
            run['status']='full_native_RF_score';atomic(out/'run.json',run)
            rf=score(folder/'material.npy',args.sample,folder/'rf_score.json',args.devices,args.cap_gib,monitor)
            if rf['primary_loss']<best['primary_loss']:
                best={'step':step,'material':str(folder/'material.npy'),'primary_loss':rf['primary_loss']}
                atomic(out/'best_RF.json',best)
            evaluate(folder/'material.npy',args.sample,initial/'material.npy',monitor)
            row={'step':step,'material_accepted':material_record['accepted'],'primary_full_RF_loss':rf['primary_loss'],
                 'best_RF':best,'elapsed_seconds':time.time()-start,'truth_read_by_optimizer':False}
            with open(out/'history.jsonl','a') as stream:stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        run.update(status='planned_sweeps_finished',elapsed_seconds=time.time()-start,best_RF=best,
                   joint_reconstruction_success_claimed=False)
        atomic(out/'run.json',run)
    except BaseException:
        run.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.time()-start)
        atomic(out/'run.json',run);raise
    finally:monitor.log.close()


if __name__=='__main__':main()
