"""One native-RF comparison of physical-c and fixed-impedance c paths.

Material truth is read ONLY for an explicitly marked counterfactual: replace
seed c with true c at fixed rho, or at fixed Z=c*rho, keeping seed alpha. No optimizer runs and no
oracle material is saved or passed into an inversion. All original shots/RX
are used; there are no material-derivative or artificial-shift probes.
"""
from pathlib import Path
import argparse,json,os,time,subprocess,hashlib,gc
import numpy as np
import torch
from .time_acquisition import TimeAcquisition
from .rf_depth_weights import DepthWeights,WEIGHTS
from .kinematic_rf_loss import KinematicRFLoss


def atomic(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',choices=['lf','point3'],required=True)
    p.add_argument('--sample',default='train_200');p.add_argument('--device',type=int,required=True)
    p.add_argument('--cap-gib',type=float,default=1.1)
    p.add_argument('--seed-material',type=Path,default=Path('out/v5/reconstruction_wri_v1_train_200/initial/material.npy'))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(s.split(',')[0]):int(s.split(',')[1]) for s in used.splitlines()}
    if used[args.device]+args.cap_gib*1024+768>16384:raise RuntimeError('Whole-card16GiB budget does not admit this forward comparison')
    torch.set_num_threads(1);torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(args.cap_gib*2**30/torch.cuda.get_device_properties(args.device).total_memory,args.device)
    start=time.time();device=f'cuda:{args.device}'
    record={'status':'native_RF_prediction','pid':os.getpid(),'sample':args.sample,'source':args.source,
        'device':args.device,'cap_GiB':args.cap_gib,'whole_card_limit_GiB':16,'card_before_MiB':used,
        'truth_read':True,'truth_scope':'diagnostic true-c only; compare fixed seed rho versus fixed seed Z, alpha unchanged',
        'optimization_executed':False,'dataset_measurements_generated':False,'oracle_material_saved':False,
        'new_center_frequency_generated':False,'all_transmissions':11,'receivers':192,
        'candidate_objectives':['AWI_geometry','frequency_difference_geometry'],
        'prospective_priority':['frequency_difference_geometry','AWI_geometry'],
        'comparison_mechanism':'same existing targets and parameters; true c on fixed-rho versus fixed-Z paths',
        'functional_parameters':{'implementation':'existing AdaptiveMatchingLoss and DifferentialPhaseLoss unchanged',
            'AWI_lag_gaussian_sigma_s':1.5e-6,'AWI_ridge_fraction_of_trace_max_power':.1,
            'frequency_difference_lag_bins':16,'spectral_floor':1e-3,'taper_samples':17,'filter_energy_floor':1e-3,
            'frequency_band_hz':[1.5e6,4.5e6],
            'depth_weights':np.asarray(WEIGHTS).tolist()},
        'decision_criterion':'oracle-c cost must be lower than seed and close at least10percent of seed-to-self gap in BOTH existing3MHz groups; necessary, not sufficient for c-stage inversion',
        'seed_material':str(args.seed_material),'seed_sha256':hashlib.sha256(args.seed_material.read_bytes()).hexdigest(),
        'source_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in ['compare_kinematic_rf_losses.py','kinematic_rf_loss.py','adaptive_rf_loss.py',
                      'differential_phase_loss.py','rf_depth_weights.py','native_time.py','native_time.cu','time_acquisition.py']},
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    atomic(args.out/'run.json',record)
    try:
        acquisition=TimeAcquisition(args.source,args.sample,device=device,chunk=96)
        observed=acquisition.observed.cpu().numpy();depth=DepthWeights(acquisition)
        bins=depth.bins.cpu().numpy().astype(np.int8);del depth
        seed=np.load(args.seed_material).astype(np.float32)
        from .v5_materials import evaluation_material
        truth=evaluation_material(args.sample);oracle=seed.copy();oracle[0]=truth[0];del truth
        if not np.array_equal(seed[1:],oracle[1:]):raise RuntimeError('Counterfactual changed nuisance material')
        fixed_Z=oracle.copy();fixed_Z[1]=seed[1]*seed[0]/oracle[0]
        if not np.allclose(fixed_Z[0]*fixed_Z[1],seed[0]*seed[1],rtol=2e-7,atol=.5):
            raise RuntimeError('The fixed-Z path changed impedance')
        cases=[('seed',seed),('oracle_c_fixed_seed_rho_alpha',oracle),('oracle_c_fixed_seed_Z_alpha',fixed_Z)]
        for name,material in cases:
            target=args.out/f'{name}.RF.npy'
            predictions=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=observed.shape)
            m=torch.from_numpy(material).to(device)
            with torch.no_grad():
                for shot in range(11):
                    atomic(args.out/'progress.json',{'case':name,'shot':shot,'count':11,
                        'pid':os.getpid(),'unix_s':time.time()})
                    value=acquisition.predict(m,shot);predictions[shot]=value.cpu().numpy();del value
            predictions.flush();del predictions,m
            print(json.dumps({'case':name,'all_shots_complete':True,'seconds':time.time()-start}),flush=True)
        peak=torch.cuda.max_memory_allocated()/2**30
        np.save(args.out/'observed.RF.npy',observed);np.save(args.out/'depth_bins.npy',bins)
        del acquisition,seed,oracle,fixed_Z,cases;gc.collect();torch.cuda.empty_cache()
        record['status']='CPU_objectives';atomic(args.out/'run.json',record)
        functional=KinematicRFLoss(observed,bins,WEIGHTS,args.source)
        scores={'self':functional.evaluate(observed)}
        for name in ['seed','oracle_c_fixed_seed_rho_alpha','oracle_c_fixed_seed_Z_alpha']:
            predictions=np.load(args.out/f'{name}.RF.npy',mmap_mode='r')
            scores[name]=functional.evaluate(predictions);del predictions
        improvement={}
        for name in ['AWI_geometry','frequency_difference_geometry']:
            initial=scores['seed'][name];counter=scores['oracle_c_fixed_seed_rho_alpha'][name];floor=scores['self'][name]
            ratio=(initial-counter)/max(initial-floor,1e-30)
            zcounter=scores['oracle_c_fixed_seed_Z_alpha'][name]
            improvement[name]={'oracle_is_lower':bool(counter<initial),'fraction_seed_to_self_gap_removed':ratio,
                'fixed_rho_oracle_better_than_fixed_Z_oracle':bool(counter<zcounter),
                'fixed_Z_oracle_is_lower_than_seed':bool(zcounter<initial),
                'passes_this_group':bool(counter<initial and ratio>=.1)}
        result={'scores':scores,'decision':improvement,'all_transmissions':11,'receivers':192,
            'source':args.source,'actual_frequency_difference_hz':functional.actual_difference_hz,
            'frequency_products_used':'same transform on predicted and observed RF; no fictitious low-frequency PDE data',
            'GPU_peak_GiB':peak,'elapsed_seconds':time.time()-start,'optimizer_executed':False,
            'interpretation':'A positive comparison only supports a prospective c-stage objective, not joint material recovery'}
        atomic(args.out/'scores.json',result)
        record.update(status='complete',seconds=time.time()-start,GPU_peak_GiB=peak);atomic(args.out/'run.json',record)
        print(json.dumps(result),flush=True)
    except BaseException:
        import traceback
        record.update(status='failed',error=traceback.format_exc(),seconds=time.time()-start)
        atomic(args.out/'run.json',record);raise


if __name__=='__main__':main()
