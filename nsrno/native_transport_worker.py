"""Native, source-streamed three-field gradient with a CPU RF transport adjoint."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import gc
import time
import traceback
import numpy as np
import torch
from .time_acquisition import TimeAcquisition
from .rf_depth_weights import DepthWeights,WEIGHTS
from .rf_transport_objective import RFTransportObjective
from .native_time import native_wave
from .native_gn_worker import physical_pullback
from .wri_worker import atomic


def saved_array(path,value):
    temporary=path.with_name(path.stem+'.partial.npy')
    np.save(temporary,value);temporary.replace(path)


def worker(rank,device,jobs,results,sample,chunk,cap_gib,folder,workers,options):
    token=-1
    try:
        torch.set_num_threads(1);torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(
            cap_gib*2**30/torch.cuda.get_device_properties(device).total_memory,device)
        device=f'cuda:{device}';folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        cache={};edge=torch.from_numpy(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy')).to(device)
        for source in ['lf','point3']:
            a=TimeAcquisition(source,sample,device=device,chunk=chunk)
            if len(a.angles)!=11 or a.observed.shape[1]!=192:raise ValueError('Complete acquisition required')
            depth=DepthWeights(a)
            objective=RFTransportObjective(a.observed.cpu().numpy(),depth.bins.cpu().numpy(),WEIGHTS,.025,
                options['lambda'],tolerance=options['tolerance'],maximum=options['maximum'],
                smoothing_fraction=options['smoothing_fraction'])
            cache[source]=(a,depth,objective)
        atomic(folder/'ready.json',{'rank':rank,'device':device,'complete_acquisition':True,
            'unknowns':'independent full-native c/rho/tissue alpha','objective':'CPU weighted analytic-lift vector transport',
            'RF_objectives':{s:cache[s][2].metadata() for s in cache},'truth_read':False})
        while True:
            job=jobs.get()
            if job is None:break
            token=job['token'];kind=job['kind'];started=time.time()
            if kind not in ['forward','gradient']:raise ValueError('No quadratic/HVP reuse for the new RF objective')
            torch.cuda.reset_peak_memory_stats();need_gradient=kind=='gradient'
            m=torch.from_numpy(job['material']).to(device);m.requires_grad_(need_gradient)
            mask=(torch.arange(m.shape[1],device=device)[:,None]>=80).expand(m.shape[1:])
            lower=upper=0.;raw_loss=0.;groups={};shot_rows=[]
            inner_seconds=0.;all_converged=True;maximum_gap=0.;RF_error_bound_squared=0.
            for source,(a,depth,objective) in cache.items():
                bins_error=np.zeros(5,np.float64);error=0.
                for shot in range(rank,11,workers):
                    atomic(folder/'progress.json',{'token':token,'kind':kind,'source':source,'shot':shot,
                        'state':'native_forward','started_unix_s':time.time()})
                    waveform=a.waveform(shot)
                    with torch.set_grad_enabled(need_gradient):
                        raw=native_wave(m,waveform,a.si,a.sw,a.ri,a.rw,a.config)
                        reference=torch.from_numpy(a.reference[shot].copy()).to(device)
                        prediction=(a.response@(raw-reference)).T
                    atomic(folder/'progress.json',{'token':token,'kind':kind,'source':source,'shot':shot,
                        'state':'CPU_RF_transport','started_unix_s':time.time()})
                    prediction_cpu=prediction.detach().cpu().numpy()
                    if token==1:
                        archive=folder/'initial_RF';archive.mkdir(exist_ok=True)
                        saved_array(archive/f'{source}_{shot:02d}.npy',prediction_cpu)
                    record,adjoint=objective.evaluate(prediction_cpu,shot,gradient=need_gradient)
                    shot_rows.append({'source':source,'shot':shot,**record})
                    atomic(folder/f'token{token:04d}_RF.json',{'token':token,'kind':kind,'shots':shot_rows})
                    if not record['all_inner_converged']:
                        failed=np.array([row['receiver'] for row in record['receivers'] if not row['converged']],dtype=np.int64)
                        failure=folder/f'token{token:04d}_{source}_{shot:02d}_inner_failure.npz'
                        states=[objective.warm.get((shot,int(receiver))) for receiver in failed]
                        warm={}
                        if states and all(state is not None for state in states):
                            warm={'warm_z':np.stack([state['z'] for state in states]),
                                  'warm_scaled_multiplier':np.stack([state['scaled_multiplier'] for state in states])}
                        temporary=failure.with_suffix('.partial.npz')
                        np.savez_compressed(temporary,receiver_indices=failed,
                            predicted_RF=prediction_cpu[failed],observed_RF=objective.observed[shot,failed],
                            node_weight=objective.window_node_weights[objective.bins[shot,failed]],
                            fixed_amplitude_unit=np.array(objective.amplitude),dt_us=np.array(objective.h),
                            transport_lambda=np.array(objective.lam),smoothing=np.array(objective.smoothing),
                            external_factor=np.array(objective.factor),**warm)
                        temporary.replace(failure)
                        raise RuntimeError(f'RF inner budget reached at {source} shot {shot}; inspect saved feasible bounds')
                    lower+=record['lower'];upper+=record['upper'];raw_loss+=record['raw_RF_loss']
                    inner_seconds+=record['CPU_inner_seconds'];all_converged &= record['all_inner_converged']
                    maximum_gap=max(maximum_gap,record['maximum_channel_relative_gap'])
                    RF_error_bound_squared+=record['RF_adjoint_L2_error_upper_bound']**2
                    if need_gradient:
                        atomic(folder/'progress.json',{'token':token,'kind':kind,'source':source,'shot':shot,
                            'state':'native_backward','started_unix_s':time.time()})
                        prediction.backward(torch.from_numpy(adjoint).to(device=device,dtype=prediction.dtype))
                    diff=(prediction.detach()-a.observed[shot]).double();error+=float(diff.square().sum())
                    for k in range(5):bins_error[k]+=float(diff[depth.bins[shot]==k].square().sum())
                    del raw,reference,prediction,prediction_cpu,diff,waveform,adjoint
                    atomic(folder/'progress.json',{'token':token,'kind':kind,'source':source,'shot':shot,
                        'state':'completed','elapsed_seconds':time.time()-started,
                        'peak_Torch_GiB':torch.cuda.max_memory_allocated()/2**30})
                groups[source]={'error_squared':error,'energy':float(a.observed.double().square().sum()),
                    'window_error_squared':bins_error.tolist(),'window_energy':depth.den}
            gradient=None
            if need_gradient:
                gradient=(physical_pullback(m.detach(),edge,m.grad).double()*mask).cpu().numpy()
            results.put({'rank':rank,'token':token,'kind':kind,'loss':.5*(lower+upper),'lower':lower,'upper':upper,
                'raw_RF_loss':raw_loss,'gradient':gradient,'groups':groups,'CPU_inner_seconds':inner_seconds,
                'maximum_channel_relative_gap':maximum_gap,'all_inner_converged':bool(all_converged),
                'RF_adjoint_L2_error_upper_bound':float(np.sqrt(RF_error_bound_squared)),
                'seconds':time.time()-started,'peak_Torch_GiB':torch.cuda.max_memory_allocated()/2**30,'error':None})
            del m,job,gradient;gc.collect()
    except BaseException:
        message=traceback.format_exc();atomic(Path(folder)/'error.json',{'rank':rank,'token':token,'error':message})
        results.put({'rank':rank,'token':token,'error':message});raise
