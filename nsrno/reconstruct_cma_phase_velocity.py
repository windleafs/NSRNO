"""CPU-only common-mid-angle phase velocity seed; not a joint reconstruction."""
import ctypes
if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:
    raise RuntimeError('Cannot disable process THP')
from pathlib import Path
import argparse
import json
import os
import resource
import signal
import sys
import time
import traceback
import numpy as np
from scipy import ndimage,optimize,sparse
from .cma_phase_geometry import (analytic_channels,beamform,common_midangle_images,
    tracked_phase,geometric_coverage,ray_matrix,effective_speed,nominal_physical_speed)
from .low_frequency_0p5 import atomic,digest,SAMPLE,CASE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,default=Path('out/v5/matched_existing_3mhz_full90_v1/index.json'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=600.)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not 60<=a.max_seconds<=900:
        raise ValueError('Require explicit CPU-only execution and a bounded wall budget')
    a.index,a.out=a.index.resolve(),a.out.resolve()
    if not all(path.is_relative_to(Path.cwd().resolve()) for path in [a.index,a.out]):
        raise ValueError('Paths must remain inside this independent nsrno project')
    if a.out.exists():
        raise RuntimeError('Inspect the existing CPU job; never overwrite')
    index=json.loads(a.index.read_text())
    if index['status']!='matched_existing_3MHz_full90_ready' or index['sample']!=SAMPLE or index['case']!=CASE:
        raise ValueError('Only the existing authorized development RF')
    record=index['families']['lf']
    if digest(record['observed_coefficients'])!=record['observed_coefficients_SHA256']:
        raise ValueError('Protected measured RF changed')
    reserve=2*2**30+400*2**20
    other=Path('out/v5/reconstruction_native_frequency_incremental_full3mhz_v1/run.json')
    upcoming=0
    if other.exists():
        s=json.loads(other.read_text())
        if s['status']!='native_frequency_incremental_stage_finished' and Path('/proc',str(s['pid'])).exists():
            upcoming=90*2**20
    import shutil
    if shutil.disk_usage('.').free<reserve+upcoming+20*2**20:
        raise RuntimeError('Wait for reserved native output writes before CPU seed storage')
    a.out.mkdir()
    started=time.time()
    deadline=started+a.max_seconds
    compute_deadline=deadline-10
    run={'status':'loading_existing_RF','pid':os.getpid(),
         'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
         'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         'created_unix_s':started,'deadline_unix_s':deadline,'sample':SAMPLE,'case':CASE,
         'GPU_work':False,'new_acoustic_wave_solves':0,'new_acoustic_material_derivatives':0,
         'new_patient_data':False,'patient_truth_read_by_optimizer':False,
         'joint_reconstruction_claimed':False,'index_SHA256':digest(a.index),
         'observed_coefficients_SHA256':record['observed_coefficients_SHA256'],
         'source_SHA256':{name:digest(Path(__file__).parent/name) for name in
             [Path(__file__).name,'cma_phase_geometry.py']}}
    atomic(a.out/'run.json',run)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    latest={'x':np.zeros(42*39,np.float64)}
    try:
        with np.load('out/v5/time_triparam_v1/geometry.npz') as data:
            g={k:data[k] for k in data.files}
        meta=json.loads(Path('out/v5/time_triparam_v1/geometry.json').read_text())
        rx=np.asarray(g['xe'],np.float64)
        source_x=g['x'][(g['x']>=rx[0])&(g['x']<=rx[-1])]
        z_rx=float(np.mean(g['receiver_coordinates'][:,1]))
        z_tx=float(np.mean(g['source_coordinates'][:,1]))
        x_translation=float(np.mean(g['receiver_coordinates'][:,0]-rx))
        if len(rx)!=192 or len(source_x)!=765 or tuple(meta['shape'])!=(961,1121):
            raise ValueError('Expected original flat-array native geometry')
        if np.ptp(g['receiver_coordinates'][:,1])>1e-8 or np.ptp(g['source_coordinates'][:,1])>1e-8:
            raise ValueError('This implementation uses planar source/receive surfaces')
        angles=np.asarray(index['angles_deg'],np.float64)
        if len(angles)!=11 or not np.allclose(np.diff(angles),1.6):
            raise ValueError('Use the acquired eleven plane-wave angles')
        coefficients=np.load(record['observed_coefficients'])
        if coefficients.shape!=(11,192,270):
            raise ValueError('Use all existing dense valid-band coefficients')
        rf=analytic_channels(coefficients,np.asarray(index['DFT_mask'],bool))
        x=np.linspace(-.0191,.0191,192)
        z=np.linspace(0.,.043,861)
        c0,f0=1540.,3e6
        run.update(status='CPU_DAS_beamforming',known_x_translation_m=x_translation,
                   source_depth_relative_receive_m=z_tx-z_rx,RF_source_shots=11,
                   RF_receivers=192,RF_dense_frequency_count=270,
                   beamformed_shape=[861,192])
        atomic(a.out/'run.json',run)
        def progress(n):
            run['beamformed_transmits']=n
            atomic(a.out/'run.json',run)
        images=beamform(rf,angles,x,z,source_x,z_tx-z_rx,rx,c0,f0,compute_deadline,progress)
        del rf,coefficients
        kernel=np.exp(-.5*np.arange(-2,3)**2);kernel/=kernel.sum()
        centres=np.arange(2,9)
        run.update(reference_speed_m_s=c0,source_f0_hz=f0,
                   target_transmit_angles_deg=angles[centres].tolist(),
                   common_midangles_deg=[-6.4,-3.2,0.,3.2,6.4],
                   compound_transmit_weights=kernel.tolist(),
                   angular_filter_halfwidth_deg=1.6,tracking_sigma_m=.0007,
                   normalized_slowness_unit_fraction=.05)
        compound=np.stack([np.tensordot(kernel,images[n-2:n+3],axes=(0,0)) for n in centres])
        del images
        xedges=np.linspace(-.0191,.0191,40)
        zedges=np.linspace(.001,.043,43)
        xc=(xedges[:-1]+xedges[1:])/2
        zc=(zedges[:-1]+zedges[1:])/2
        shape=(len(zc),len(xc))
        cache={}
        def ray(theta):
            key=round(float(theta),12)
            if key not in cache:
                cache[key]=ray_matrix(theta,xc,zedges)
            return cache[key]
        blocks,measurements,weights=[],[],[]
        phase_maps,confidence_maps=[],[]
        coverage=np.zeros(shape,np.float64)
        scale=2*np.pi*f0/c0*.05
        for gamma_deg in [-6.4,-3.2,0.,3.2,6.4]:
            if time.time()>compute_deadline:
                raise TimeoutError('CPU phase budget ended before inversion')
            gamma=np.deg2rad(gamma_deg)
            run.update(status='CPU_common_midangle_phase',active_midangle_deg=gamma_deg)
            atomic(a.out/'run.json',run)
            filtered=common_midangle_images(compound,gamma,x[1]-x[0],z[1]-z[0],f0,c0)
            for k in range(len(centres)-1):
                ia,ib=int(centres[k]),int(centres[k+1])
                phi_a,phi_b=np.deg2rad(angles[[ia,ib]])
                if abs(phi_a+phi_b-2*gamma)<1e-9:
                    continue
                phase,weight,coherence=tracked_phase(filtered[k],filtered[k+1],x,z,xc,zc)
                valid=(geometric_coverage(xc,zc,angles[ia-2:ia+3],gamma,
                          [source_x.min(),source_x.max()],[rx.min(),rx.max()],z_tx-z_rx)&
                       geometric_coverage(xc,zc,angles[ib-2:ib+3],gamma,
                          [source_x.min(),source_x.max()],[rx.min(),rx.max()],z_tx-z_rx))
                weight*=valid
                use=(weight.ravel()>1e-6)&np.isfinite(phase.ravel())
                psi_a,psi_b=2*gamma-phi_a,2*gamma-phi_b
                block=scale*((ray(phi_b)+ray(psi_b))/np.cos(phi_b-gamma)-
                             (ray(phi_a)+ray(psi_a))/np.cos(phi_a-gamma))
                if np.any(use):
                    blocks.append(block[use])
                    measurements.append(phase.ravel()[use])
                    weights.append(weight.ravel()[use])
                coverage+=weight
                phase_maps.append(phase.astype(np.float32))
                confidence_maps.append(weight.astype(np.float32))
            del filtered
        matrix=sparse.vstack(blocks,format='csr')
        data=np.concatenate(measurements);weight=np.concatenate(weights)
        normalization=max(float(np.sum(weight*data*data)),1e-20)
        sqrtw=np.sqrt(weight/normalization)
        A=matrix.multiply(sqrtw[:,None]).tocsr()
        b=sqrtw*data
        del matrix,blocks,cache
        run.update(status='CPU_bounded_TV_slowness_inverse',phase_rows=len(data),
                   ray_nonzeros=int(A.nnz),unknown_slowness_cells=int(np.prod(shape)),
                   mean_phase_coherence_weight=float(weight.mean()),
                   control_shape=list(shape),TV_weight=.02,mean_prior_weight=.001,
                   phase_normalization=normalization)
        atomic(a.out/'run.json',run)
        counts=[0]
        def objective(v):
            if time.time()>compute_deadline:
                raise TimeoutError('CPU phase inversion wall budget')
            field=v.reshape(shape)
            residual=A@v-b
            dx=field[:-1,1:]-field[:-1,:-1]
            dz=field[1:,:-1]-field[:-1,:-1]
            norm=np.sqrt(dx*dx+dz*dz+.02**2)
            tv=.02*float(np.mean(norm-.02))
            grad=np.zeros_like(field)
            wx,wz=.02*dx/norm/norm.size,.02*dz/norm/norm.size
            grad[:-1,:-1]-=wx+wz
            grad[:-1,1:]+=wx;grad[1:,:-1]+=wz
            prior=.001*float(np.mean(v*v))
            value=.5*float(residual@residual)+tv+prior
            return value,np.asarray(A.T@residual)+grad.ravel()+.002*v/len(v)
        def callback(v):
            latest['x']=v.copy()
            counts[0]+=1
            if counts[0]%10==0:
                value,_=objective(v)
                run.update(optimization_iterations=counts[0],current_phase_objective=float(value))
                atomic(a.out/'run.json',run)
        lo=float((c0/effective_speed(1680.,index['absorption'])-1)/.05)
        hi=float((c0/effective_speed(1400.,index['absorption'])-1)/.05)
        initial_value=objective(latest['x'])[0]
        terminated_by_budget=False
        try:
            fit=optimize.minimize(objective,latest['x'],jac=True,method='L-BFGS-B',
                bounds=[(lo,hi)]*len(latest['x']),callback=callback,
                options={'maxiter':300,'maxls':20,'gtol':1e-8,'ftol':1e-10})
            value=fit.x;termination=str(fit.message);final_value=float(fit.fun)
        except TimeoutError:
            value=latest['x'];termination='CPU_budget_at_last_completed_iteration'
            terminated_by_budget=True
            final_value=None
        effective=c0/(1+.05*value.reshape(shape))
        physical=nominal_physical_speed(effective,index['absorption'])
        if physical.min()<1400-1e-3 or physical.max()>1680+1e-3:
            raise RuntimeError('Nominal physical seed exceeds its declared box')
        nz,nx=meta['shape']
        native_z=(g['origin'][1]+np.arange(nz)*g['spacing'][1]-z_rx)
        coordinates=np.stack(np.meshgrid((native_z-zc[0])/(zc[1]-zc[0]),
                              (g['x']-xc[0])/(xc[1]-xc[0]),indexing='ij'))
        speed=ndimage.map_coordinates(physical,coordinates,order=1,mode='nearest',prefilter=False)
        edge=np.broadcast_to(np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'),(nz,nx))
        material=np.stack((speed,np.full((nz,nx),1000.),.5+edge)).astype(np.float32)
        material[0,:80]=1540.;material[1,:80]=1000.;material[2,:80]=.002+edge[:80]
        np.save(a.out/'material.npy',material)
        np.savez_compressed(a.out/'phase_seed.npz',c_nominal=physical.astype(np.float32),
            c_effective=effective.astype(np.float32),x=xc,z=zc,coverage=coverage.astype(np.float32),
            phase=np.stack(phase_maps),confidence=np.stack(confidence_maps))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(9,5),constrained_layout=True)
        im=axes[0].imshow(physical,extent=[-19.1,19.1,43,1],vmin=1400,vmax=1680,cmap='viridis',aspect='equal')
        axes[0].set_title('CMA nominal sound-speed seed (m/s)');fig.colorbar(im,ax=axes[0])
        im=axes[1].imshow(coverage,extent=[-19.1,19.1,43,1],aspect='equal')
        axes[1].set_title('Accumulated phase reliability');fig.colorbar(im,ax=axes[1])
        for ax in axes:
            ax.set_xlabel('Lateral (mm)');ax.set_ylabel('Depth (mm)')
        fig.savefig(a.out/'phase_velocity.png',dpi=160);plt.close(fig)
        run.update(status='CPU_CMA_phase_velocity_seed_finished',finished_unix_s=time.time(),
            seconds=time.time()-started,material_SHA256=digest(a.out/'material.npy'),
            physical_c_range=[float(physical.min()),float(physical.max())],
            optimization_iterations=counts[0],initial_phase_objective=initial_value,
            final_phase_objective=final_value,
            optimization_termination=termination,nominal_alpha_assumed=.5,
            rho_and_alpha_are_uniform_seeds_not_estimates=True,
            RSS_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            native_initialization_shape=list(material.shape),
            declared_FOV_mm={'x':[-19.1,19.1],'z':[1,43]},torch_imported='torch' in sys.modules,
            joint_reconstruction_claimed=False)
        atomic(a.out/'run.json',run)
        try:
            print(json.dumps({k:run[k] for k in ['status','seconds','physical_c_range',
                'phase_rows','ray_nonzeros','initial_phase_objective','final_phase_objective',
                'optimization_iterations','RSS_peak_GiB','GPU_work','joint_reconstruction_claimed']}),flush=True)
        except BrokenPipeError:
            pass
    except BaseException:
        run.update(status='CPU_CMA_phase_initialization_interrupted_or_failed',
                   error=traceback.format_exc(),stopped_unix_s=time.time())
        atomic(a.out/'run.json',run)
        raise


if __name__=='__main__':
    main()

