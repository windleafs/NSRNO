"""Full-frequency WRI worker with all11 transmissions in each block solve."""
from pathlib import Path
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('Cannot disable process THP')
import json,os,time,sys,traceback
import numpy as np
import torch
from .conditioned_wave import ConditionedFrequencySystem,LearnedPreconditioner
from .dense_wave import learned_apply
from .first_order_wavefield import FirstOrderWavefield,relaxation_inverse
from .batched_first_order_wavefield import BatchedFirstOrderWavefield
from .frequency_group_data import FrequencyGroupData,receiver_response
from .native_compact_weighted_pressure import two_level_pressure
from .receiver_normal_preconditioner_batch import ReceiverNormalPreconditioner
from .block_wavefield_step import update_block
from .wri_worker import atomic


def run_worker(rank,device,assignments,out,sweep,mode,cap,results,field_iterations=32,field_tolerance=.003):
    log=open(out/f'{mode}_rank{rank}.log','w',buffering=1);sys.stdout=log;sys.stderr=log
    try:
        torch.set_num_threads(1);torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(cap*2**30/torch.cuda.get_device_properties(device).total_memory,device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        name=f'cuda:{device}';start=time.time();root=out.parent;field_mode=mode in ['field','field_continue']
        run=json.loads((root/'run.json').read_text());groups={};geometry=None
        for source in run['sources']:
            groups[source]=FrequencyGroupData(source,run['sample'],name,geometry);geometry=groups[source].geometry
        g=geometry;h,w=g.h,g.w;N=h*w
        zz=np.clip(np.arange(g.H)-g.pz,24,h-25);xx=np.clip(np.arange(g.W)-g.px,24,w-25)
        index=torch.from_numpy((zz[:,None]*w+xx[None,:]).astype(np.int64)).to(name)
        ix=torch.roll(index,-1,-1);iz=torch.roll(index,-1,-2)
        def fold(value,indices=index):
            result=torch.zeros(N,dtype=value.dtype,device=name);result.scatter_add_(0,indices.flatten(),value.flatten())
            return result.reshape(h,w)
        def fold_complex(value):return torch.complex(fold(value.real),fold(value.imag))
        old_material=torch.from_numpy(np.load(out/'old_material.npy')).to(name)
        material=torch.from_numpy(np.load(out/('old_material.npy' if field_mode else 'material.npy'))).to(name)
        uniform=material.clone();uniform[0].fill_(1540);uniform[1].fill_(1000);uniform[2].fill_(.002)
        network=LearnedPreconditioner().to(name)
        network.load_state_dict(torch.load('out/v5/forward_round8_dense_full/last.pt',map_location='cpu',weights_only=True)['model'])
        network.requires_grad_(False);network.eval()
        banks={};yd={};penalties={}
        for source in groups:
            banks[source]=tuple(np.load(root/'banks'/f'{source}_{kind}.npy',mmap_mode='r+') for kind in ['X','Y'])
            yd[source]=np.load(root/'banks'/f'{source}_Ydata.npy',mmap_mode='r+')
            penalties[source]=np.load(root/'banks'/f'{source}_penalty.npy',mmap_mode='r')
        normals={k:np.zeros((h,w),np.float64) for k in ['br','hbb','hbt','htt','bb','bt']}
        normals.update(arx=np.zeros((g.H,g.W),np.float64),arz=np.zeros((g.H,g.W),np.float64))
        moments=[];frequency_rows=[]
        for number,(source,position) in enumerate(assignments):
            data=groups[source];frequency=data.frequency(position);response=receiver_response(frequency);torch.cuda.reset_peak_memory_stats()
            atomic(out/f'progress_rank{rank}.json',{'mode':mode,'rank':rank,'device':device,'pid':os.getpid(),
                'source':source,'frequency_position':position,'frequency_hz':frequency,'angle_index':-1,
                'all11_transmissions_together':field_mode,'complete_frequencies':number,
                'assigned_frequencies':len(assignments),'unix_s':time.time()})
            with torch.no_grad():
                system=ConditionedFrequencySystem(g,material[:2],material[2],frequency)
                refsystem=ConditionedFrequencySystem(g,uniform[:2],uniform[2],frequency)
                op,refop=FirstOrderWavefield(system),FirstOrderWavefield(refsystem);X,Y=banks[source]
                if field_mode:
                    penalty=float(penalties[source][position])
                    if penalty<=0:raise RuntimeError('Continuation requires the original frozen positive penalty')
                    pressure,pressure_adjoint,pressure_meta=two_level_pressure(system,
                        learned_apply(network,system),learned_apply(network,system,adjoint=True))
                    normal_pre=ReceiverNormalPreconditioner(op,pressure_adjoint,response,penalty)
                    receiver_meta=normal_pre.metadata;batchop=BatchedFirstOrderWavefield(system)
                    forward,adjoint=batchop.preconditioners(pressure,pressure_adjoint)
                    current=g.pad_field(torch.from_numpy(np.array(X[position])).to(name))
                    dual=g.pad_field(torch.from_numpy(np.array(Y[position])).to(name))
                    data_dual=torch.from_numpy(np.array(yd[source][position])).to(name)
                    references=torch.stack([refop.from_pressure(data.reference(position,a)) for a in range(11)])
                    rhs=torch.stack([op.scattering_rhs(refop,references[a]) for a in range(11)])
                    target=torch.stack([data.target(position,a) for a in range(11)])
                    if mode=='field_continue':
                        shifted_rhs=dual+batchop(current);shifted_target=data_dual+batchop.readout(current,response)
                    else:shifted_rhs=rhs+dual;shifted_target=target+data_dual
                    del dual,data_dual
                    values,block_record,bases,data_bases=update_block(batchop,shifted_rhs,shifted_target,response,
                        current,penalty,forward,adjoint,normal_pre,maximum=field_iterations,tolerance=field_tolerance)
                    data_errors=(target-batchop.readout(values,response)).abs().square().sum(-1,dtype=torch.float64).tolist()
                    equation_errors=(rhs-batchop(values)).abs().square().flatten(1).sum(-1,dtype=torch.float64).tolist()
                    del current,shifted_rhs,shifted_target,rhs,target,forward,adjoint,normal_pre,pressure,pressure_adjoint,batchop
                    scale=penalty/(data.energy*len(groups));t=system.alpha*system.c/1540
                    beta=system.a*(1540/system.c).square()
                    M,Mprime=relaxation_inverse(frequency,g.dt,data.spec['taus_s'],data.spec['eta_per_unit_alpha'],t)
                    stats=torch.zeros((9,g.H,g.W),dtype=torch.float64,device=name)
                    spp=torch.zeros((g.H,g.W),dtype=torch.float64,device=name);spe=torch.zeros((g.H,g.W),dtype=torch.complex128,device=name)
                else:
                    oldsystem=ConditionedFrequencySystem(g,old_material[:2],old_material[2],frequency)
                    oldop=FirstOrderWavefield(oldsystem)
                angle_rows=[]
                for angle in range(11):
                    if field_mode:
                        value=values[angle];base=bases[angle];total=references[angle]+value;e=-base
                        jb=.1*M*total[0];jt=.1*beta*Mprime*total[0]
                        stats[0].add_(jb.abs().square().double()*scale);stats[1].add_((jb.conj()*jt).real.double()*scale)
                        stats[2].add_(jt.abs().square().double()*scale);stats[3].add_(-(jb.conj()*e[0]).real.double()*scale)
                        stats[4].add_(-(jt.conj()*e[0]).real.double()*scale)
                        for axis,field,sv in [(0,total[1],system.svx),(1,total[2],system.svz)]:
                            j=.05*field/sv;stats[5+axis].add_(j.abs().square().double()*g.mask*scale)
                            stats[7+axis].add_(-(j.conj()*e[1+axis]).real.double()*g.mask*scale)
                        spp.add_(.5*scale*total[0].abs().square().double()*g.mask)
                        spe.add_(.5*scale*(total[0].conj()*e[0]).to(torch.complex128)*g.mask)
                        X[position,angle]=g.native(value).cpu().numpy();Y[position,angle]=g.native(base).cpu().numpy()
                        yd[source][position,angle]=data_bases[angle].cpu().numpy()
                        physical=block_record['physical_global_normal_residual'][angle];pre=block_record['preconditioned_global_normal_residual'][angle]
                        angle_rows.append({'angle_index':angle,'data_loss':.5*data_errors[angle]/(data.energy*len(groups)),
                            'PDE_loss':.5*penalty*equation_errors[angle]/(data.energy*len(groups)),
                            'normal_residual':pre,'physical_global_normal_residual':physical,
                            'preconditioned_global_normal_residual':pre,'converged':physical<=field_tolerance and pre<=field_tolerance,
                            'CG_iterations':block_record['iterations'],'peak_GiB':torch.cuda.max_memory_allocated()/2**30})
                        del value,base,total,e,jb,jt,j,field
                    else:
                        reference=refop.from_pressure(data.reference(position,angle))
                        current=g.pad_field(torch.from_numpy(np.array(X[position,angle])).to(name))
                        dual=g.pad_field(torch.from_numpy(np.array(Y[position,angle])).to(name));total=reference+current
                        change=torch.stack([(system.mass-oldsystem.mass)*total[0],
                            (op.Rx-oldop.Rx)*total[1],(op.Rz-oldop.Rz)*total[2]])*g.mask
                        Y[position,angle]=g.native(dual-change).cpu().numpy();del reference,current,dual,total,change
                X.flush();Y.flush();yd[source].flush()
                if field_mode:
                    for idx,key in enumerate(['hbb','hbt','htt','bb','bt']):normals[key]+=fold(stats[idx]*g.mask).cpu().numpy()
                    normals['arx']+=stats[5].cpu().numpy();normals['arz']+=stats[6].cpu().numpy()
                    normals['br']+=(fold(stats[7],index)+fold(stats[7],ix)+fold(stats[8],index)+fold(stats[8],iz)).cpu().numpy()
                    path=out/'moments'/f'{source}_{position:03d}.npz'
                    np.savez(path,spp=fold(spp).cpu().numpy().astype(np.float32),spe=fold_complex(spe).cpu().numpy().astype(np.complex64))
                    moments.append({'path':str(path.resolve()),'source':source,'position':position,'frequency_hz':frequency,
                        'spec':{k:data.spec[k] for k in ['native_dt_s','eta_per_unit_alpha','taus_s']}})
                    row={'source':source,'frequency_position':position,'frequency_hz':frequency,'angle_records':angle_rows,
                        'block_record':block_record,'pressure_preconditioner':pressure_meta,'receiver_preconditioner':receiver_meta,
                        'elapsed_seconds':time.time()-start};frequency_rows.append(row);print(json.dumps(row),flush=True)
                    del stats,spp,spe,M,Mprime,t,beta,values,bases,data_bases,references
                else:del oldsystem,oldop
                del system,refsystem,op,refop
            if number%8==7:torch.cuda.empty_cache()
        normal_path=out/f'normals_rank{rank}.npz'
        if field_mode:np.savez(normal_path,**normals)
        summary={'status':'complete','rank':rank,'pid':os.getpid(),'mode':mode,'frequencies':len(assignments),
            'seconds':time.time()-start,'normal_path':str(normal_path.resolve()),'moments':moments,
            'geometry':{'padded_shape':[g.H,g.W],'offset_zx':[g.pz,g.px]},
            'data_loss':sum(a['data_loss'] for r in frequency_rows for a in r['angle_records']),
            'PDE_loss':sum(a['PDE_loss'] for r in frequency_rows for a in r['angle_records']),
            'max_physical_global_normal_residual':max((a['physical_global_normal_residual'] for r in frequency_rows for a in r['angle_records']),default=0),
            'max_preconditioned_global_normal_residual':max((a['preconditioned_global_normal_residual'] for r in frequency_rows for a in r['angle_records']),default=0),
            'unconverged_shots':sum(not a['converged'] for r in frequency_rows for a in r['angle_records']),
            'peak_GiB':max((a['peak_GiB'] for r in frequency_rows for a in r['angle_records']),default=0),'truth_read':False}
        atomic(out/f'{mode}_rank{rank}.json',summary);results.put({'rank':rank,'status':'complete'})
    except BaseException:
        error=traceback.format_exc();print(error,flush=True);atomic(out/f'{mode}_rank{rank}.failed.json',{'rank':rank,'pid':os.getpid(),'error':error})
        results.put({'rank':rank,'status':'failed','error':error});raise
    finally:log.close()
