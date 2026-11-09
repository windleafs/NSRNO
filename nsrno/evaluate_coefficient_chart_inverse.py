"""Fixed-final EMA reconstructions; development truth is read only by later CPU evaluation."""
import ctypes
if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import traceback
import numpy as np
import torch
from .coefficient_chart_angular_inverse_uno import CoefficientChartAngularInverseUNO
from .central_secant_data import TrainingPairs, array, tissue_label
from .evaluate_amplitude_angular_inverse import predict, native_proposal
from .native_gn_json import atomic
from .pressure_resources import PressureResources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=600.)
    args = parser.parse_args()
    args.train_root, args.out = args.train_root.resolve(), args.out.resolve()
    if (not 30 <= args.max_seconds <= 600 or not all(p.is_relative_to(Path.cwd().resolve()/'out/v5')
                                                for p in [args.train_root, args.out])):
        raise ValueError('Bounded same-project fixed inference only')
    training = json.loads((args.train_root/'run.json').read_text())
    if training['status'] != 'coefficient_chart_training_finished' or Path('/proc', str(training['pid'])).exists():
        raise RuntimeError('Wait for the same finished actual training process')
    checkpoint = Path(training['checkpoint_path'])
    if digest(checkpoint) != training['checkpoint_SHA256']:
        raise RuntimeError('Fixed final complete checkpoint changed')
    for name, sha in training['source_SHA256'].items():
        if digest(Path(__file__).parent/name) != sha:
            raise RuntimeError('Frozen trained source changed: '+name)
    index_path = Path(training['index'])
    if digest(index_path) != training['index_SHA256']:
        raise RuntimeError('Training/inference index changed')
    index = json.loads(index_path.read_text())
    if args.out.exists():
        raise RuntimeError('Preserve same actual prediction outputs; do not repeat implicitly')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                               '--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Another GPU experiment is active')
    if shutil.disk_usage('.').free < 2*2**30+400*2**20+40*2**20:
        raise RuntimeError('Reserve complete native predictions and CPU plots')
    args.out.mkdir()
    started = time.time()
    deadline = started+args.max_seconds
    run = {'status':'initializing_fixed_chart_inference','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'created_unix_s':started,'deadline_unix_s':deadline,
        'training_root':str(args.train_root),'checkpoint_SHA256':training['checkpoint_SHA256'],
        'weights_selected':'fixed final EMA from complete checkpoint',
        'source_SHA256':digest(__file__),'device':training['device'],
        'allocator_cap_GiB':8,'per_card_process_tree_limit_GiB':16,
        'development_truth_arrays_read':False,'reserved_case_unread':'Neg_47_Left',
        'new_wave_solves':0,'new_acoustic_material_derivatives':0,
        'joint_imaging_success_claimed':False}
    atomic(args.out/'run.json',run)
    monitor = PressureResources(args.out)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    try:
        monitor.wait_budget([training['device']],8,'fixed_chart_inference')
        torch.set_num_threads(1)
        torch.cuda.set_device(training['device'])
        torch.cuda.set_per_process_memory_fraction(8*2**30 /
            torch.cuda.get_device_properties(training['device']).total_memory,training['device'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        device=f"cuda:{training['device']}"
        model=CoefficientChartAngularInverseUNO().to(device)
        weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if weights['step']!=training['completed_steps']:
            raise RuntimeError('Checkpoint is not the final completed step')
        model.load_state_dict(weights['ema'],strict=True)
        del weights
        model.eval()
        edge=np.broadcast_to(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'),(961,1121))
        uniform_path=Path('out/v5/reconstruction_single_case_0p5mhz_stage1_v1/initial_material.npy')
        if digest(uniform_path)!='7d8d47327648a6313fc8f3914a8429a00fa4b6efc6916274add97eb45225f459':
            raise RuntimeError('Known homogeneous comparison changed')
        scale=index['training_RF_component_RMS_scale']
        outputs=[]
        def save(name,prediction,base_roi,base_native,base_path,information):
            folder=args.out/name;folder.mkdir()
            material,projection=native_proposal(prediction,base_roi,base_native,edge)
            np.save(folder/'material.npy',material)
            np.save(folder/'prediction.npy',prediction.astype(np.float32))
            os.link(uniform_path,folder/'uniform_initial_material.npy')
            if base_path is not None:
                os.link(base_path,folder/'stage0_initial_material.npy')
            record={'name':name,'sample':'train_200','development_truth_arrays_read':False,
                'material_SHA256':digest(folder/'material.npy'),
                'checkpoint_SHA256':training['checkpoint_SHA256'],**information,**projection}
            atomic(folder/'prediction.json',record)
            outputs.append({'folder':str(folder),**record})
            run.update(status='saving_actual_native_chart_predictions',actual_native_predictions=len(outputs))
            atomic(args.out/'run.json',run)
            monitor.sample('actual_native_chart_prediction_saved')

        row=next(r for r in index['records'] if r['id']=='train_200' and r['role']!='train' and r['valid_samples']==541)
        if row['case']!='Neg_35_Left':
            raise RuntimeError('Only the authorized development case')
        measured=np.load(row['raw_rf'])
        reference_roi=np.load(index['reference_ROI'])
        reference_context=np.load(index['reference_context'])
        reference_native=np.empty((3,961,1121),np.float32)
        reference_native[0]=1540.;reference_native[1]=1000.;reference_native[2]=.002+edge
        prediction,info=predict(model,np.zeros_like(measured),measured,reference_roi,reference_context,
            row['valid_samples'],row['sample_dt_s'],scale,device)
        save('train_200_reference',prediction,reference_roi,reference_native,None,
             {**info,'baseline':'known water scattered RF reference; no patient material input'})

        current=json.loads(Path(index['native_current_inference']).read_text())
        if current['sample']!='train_200' or digest(current['baseline_material'])!=current['baseline_material_SHA256']:
            raise RuntimeError('Prepared actual inverse baseline changed')
        base_roi=np.load(current['baseline_ROI']);context=np.load(current['baseline_context'])
        base_native=np.load(current['baseline_material'])
        prediction,info=predict(model,np.load(current['baseline_RF']),np.load(current['observed_RF']),
            base_roi,context,current['valid_samples'],current['sample_dt_s'],scale,device)
        save('train_200_native_current',prediction,base_roi,base_native,Path(current['baseline_material']),
             {**info,'baseline_material_SHA256':current['baseline_material_SHA256'],
              'same_input_baseline_as_closed_amplitude_parent':True})
        atomic(args.out/'actual_predictions.json',{'outputs':outputs,'development_truth_arrays_read':False})

        run['status']='fixed_training_response_assessment'
        atomic(args.out/'run.json',run)
        pairs=TrainingPairs(index)
        details=[];assessment_complete=True
        for kind,baseline,target in pairs.evaluation_pairs():
            if time.time()>=deadline:
                assessment_complete=False;break
            if baseline['role']!='train' or target['role']!='train':
                raise RuntimeError('Only Neg_07 training labels in response assessment')
            base_roi=tissue_label(baseline,pairs.edge)
            exact_target=tissue_label(target,pairs.edge)
            prediction,_=predict(model,array(baseline['raw_rf']),array(target['raw_rf']),base_roi,
                array(baseline['context']),target['valid_samples'],target['sample_dt_s'],scale,device)
            changed={'pure_c':0,'pure_rho':1,'alpha':2}[kind]
            wanted=(exact_target-base_roi)[changed,5:].astype(np.float64).ravel()
            update=(prediction-base_roi)[changed,5:].astype(np.float64).ravel()
            denominator=max(float(np.linalg.norm(wanted)),1e-30)
            norm=max(float(np.linalg.norm(update)),1e-30)
            details.append({'kind':kind,'pair_group':target['pair_group'],'sign':target['pair_sign'],
                'valid_samples':target['valid_samples'],
                'relative_RMS_error':float(np.linalg.norm(update-wanted)/denominator),
                'RMS_gain':norm/denominator,'direction_cosine':float(update@wanted/(norm*denominator))})
        groups={kind:{key:float(np.mean([r[key] for r in details if r['kind']==kind]))
                      for key in ['relative_RMS_error','RMS_gain','direction_cosine']}
                for kind in pairs.groups if any(r['kind']==kind for r in details)}
        atomic(args.out/'training_responses.json',{'training_only':True,'common_unperturbed_baseline':True,
            'all_existing_pairs_completed':assessment_complete,'details':details,'groups':groups,
            'does_not_establish_independent_patient_joint_reconstruction':True})
        run.update(status='coefficient_chart_actual_predictions_finished',finished_unix_s=time.time(),
            seconds=time.time()-started,actual_native_predictions=len(outputs),
            assessed_training_pairs=len(details),training_pair_assessment_complete=assessment_complete,
            training_response_groups=groups,CPU_post_hoc_map_evaluation_still_required=True,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        monitor.sample('fixed_chart_inference_finished')
        atomic(args.out/'run.json',run)
        print(json.dumps({k:run[k] for k in ['status','seconds','actual_native_predictions',
            'assessed_training_pairs','training_pair_assessment_complete','peak_reserved_GiB']}),flush=True)
    except BaseException:
        run.update(status='coefficient_chart_inference_failed',error=traceback.format_exc(),
                   failed_unix_s=time.time())
        atomic(args.out/'run.json',run)
        raise
    finally:
        monitor.log.close()


if __name__=='__main__':
    main()
