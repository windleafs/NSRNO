"""Spatially allocated exact-image Krylov proposals and saved native state."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .native_gn_galerkin import TrueImageSpan


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EmptySpatialBlock(RuntimeError):
    pass


class SpatialSpan(TrueImageSpan):
    def projected_norm_on(self,mask):
        selected=np.broadcast_to(mask,self.shape).reshape(-1)
        residual=self.g+self.Hu
        change=self.u-np.clip(self.u-residual/self.diagonal,self.low,self.high)
        return float(np.sqrt(np.dot(change[selected],self.diagonal[selected]*change[selected])))

    def next_spatial_direction(self,mask,label):
        selected=np.broadcast_to(mask,self.shape).reshape(-1)
        residual=self.g+self.Hu;v=-residual/self.diagonal
        v[~selected]=0
        low=self.u<=self.low+1e-10;high=self.u>=self.high-1e-10
        v[(low&(v<0))|(high&(v>0))]=0
        slope=float(residual@v)
        if slope>=0:
            raise EmptySpatialBlock('No feasible residual descent in '+label)
        if len(self.basis):
            coefficient=np.linalg.solve(self.gram,self.images@v)
            direction=v-self.basis.T@coefficient
        else:
            direction=v.copy()
        norm=np.sqrt(np.dot(direction,self.diagonal*direction))
        if not np.isfinite(norm) or norm<=1e-20:
            raise EmptySpatialBlock('No independent direction in '+label)
        direction/=norm
        conjugacy=self.images@direction
        return direction.reshape(self.shape),{
            'proposal_block':label,'feasible_preconditioned_residual_slope':slope,
            'slope_at_current_subspace_solution':float(residual@direction),
            'normalization':float(norm),
            'known_H_conjugacy_absolute_max':float(np.max(abs(conjugacy))) if len(conjugacy) else 0.,
            'unrestricted_three_material_coordinates':True,
            'spatial_mask_applies_to_proposal_before_H_deflation':True,
            'physical_boxes_solved_in_entire_span':True,
            'pixelwise_magnitude_clipping':False}


class NativeSpatialState:
    def __init__(self,source,callback=None):
        from .native_gn_coordinates import Coordinates,physical_blocks,dot
        self.source=Path(source).resolve();self.origin=self.source/'step001'
        run=json.loads((self.source/'run.json').read_text())
        if run['status']!='native_block_GN_stage_finished' or run['block']!='impedance':
            raise RuntimeError('Require the completed impedance stage')
        workers=json.loads((self.source/'worker_processes.json').read_text())
        pids=[run['pid'],run.get('child_pid'),*workers['worker_pids']]
        if any(pid and Path('/proc',str(pid)).exists() for pid in pids):
            raise RuntimeError('Previous parent or child still exists')
        for name,expected in run['source_sha256'].items():
            if digest(Path(__file__).parent/name)!=expected:
                raise RuntimeError('Saved native source changed: '+name)
        self.source_run=run;self.sample=run['sample']
        self.material=np.load(self.origin/'stage0_initial_material.npy')
        self.fixed_SHA256=digest(self.origin/'stage0_initial_material.npy')
        if self.fixed_SHA256!=run['source_material_SHA256']:
            raise RuntimeError('Exact HVP reference differs from saved initial material')
        if self.material.shape!=(3,961,1121) or self.material.dtype!=np.float32:
            raise ValueError('Full native reference required')
        if not np.array_equal(self.material,np.load(self.origin/'old_material.npy')):
            raise RuntimeError('Reference copies disagree')
        self.incumbent=np.load(self.origin/'material.npy')
        self.incumbent_SHA256=digest(self.origin/'material.npy')
        update=json.loads((self.origin/'material_update.json').read_text())
        if self.incumbent_SHA256!=run['material_SHA256'] or self.incumbent_SHA256!=update['material_SHA256']:
            raise RuntimeError('Accepted physical material changed')
        self.baseline_rf=json.loads((self.origin/'baseline_RF.json').read_text())
        self.incumbent_rf=json.loads((self.origin/'incumbent_RF.json').read_text())
        if self.incumbent_rf['loss']!=update['native_data_after']:
            raise RuntimeError('Accepted physical RF evidence disagrees')
        self.preconditioner=Path(run['preconditioner_statistics'])
        if digest(self.preconditioner)!='acd29954a613f156e95530c9dd67d413e6f1ea9a1b1492ec51eb58a354bfa961':
            raise RuntimeError('Metric statistics changed')
        self.co=Coordinates(self.material,np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'))
        self.reference={'origin_step_folder':str(self.origin),'tv_weight':.001,'damping_relative':.01,
            'fixed_reference_material':str(self.origin/'stage0_initial_material.npy'),
            'full_joint_coordinates':[0,1,2]}
        self.scale=run['preconditioner_scale']
        with np.load(self.preconditioner) as data:
            blocks=physical_blocks(data['blocks'],self.material,self.co.edge)
        self.damping=.01*self.scale*blocks[:3];self.damping[:,~self.co.mask]=0
        zero=np.zeros_like(self.material,dtype=np.float64)
        self.tv,self.g_tv,self.h_tv,tv_diagonal=self.co.regularization(zero,.001)
        self.gradient=np.load(self.origin/'data_gradient.npy')+self.g_tv
        if not np.isfinite(self.gradient).all():
            raise RuntimeError('Saved full gradient is not finite')
        self.diagonal=np.maximum(self.scale*blocks[:3]+tv_diagonal+self.damping,1e-30)
        self.diagonal[:,~self.co.mask]=1
        low,high=self.co.bounds(trust=(1.,1.,10.))
        self.span=SpatialSpan(self.gradient,self.diagonal,low,high)
        z=(np.arange(961)-60)*.05;x=(np.arange(1121)-560)*.05
        self.roi=(abs(x)[None,:]<=19.1)&(z[:,None]>=1)&(z[:,None]<=43)
        self.initial_norm=self.span.projected_norm()
        self.initial_roi_norm=self.span.projected_norm_on(self.roi)
        self.paths=[Path(v) for v in json.loads((self.origin/'fixed_HVP_manifest.json').read_text())['paths']]
        if len(self.paths)!=6 or run['new_HVPs']!=6:
            raise RuntimeError('Use the six completed exact reference images')
        jobs={json.loads(path.read_text())['tag']:json.loads(path.read_text())
            for path in (self.source/'jobs').glob('*.json')}
        self.provenance=[]
        for index,path in enumerate(self.paths):
            direction=np.load(path/'direction.npy');data_image=np.load(path/'data_image.npy')
            image=np.load(path/'total_image.npy')
            if any(value.shape!=self.material.shape or not np.isfinite(value).all()
                   for value in [direction,data_image,image]):
                raise RuntimeError('Incomplete saved native pair')
            if np.any(direction[[0,2]]):
                raise RuntimeError('Unexpected source direction coordinates')
            expected=data_image+self.h_tv(direction)+self.damping*direction
            error=float(np.linalg.norm(expected-image)/max(np.linalg.norm(image),1e-300))
            progress=json.loads((path/'progress.json').read_text())
            job=jobs[progress['true_HVP_job']]
            curvature=dot(direction,data_image)
            curvature_error=abs(curvature-job['adjoint_quadratic'])/max(abs(curvature),1e-300)
            if max(error,curvature_error)>1e-10:
                raise RuntimeError('Saved exact curvature does not reconstruct')
            self.span.append(direction,image)
            self.provenance.append({'path':str(path),'direction_SHA256':digest(path/'direction.npy'),
                'total_image_SHA256':digest(path/'total_image.npy'),'total_image_error':error,
                'data_curvature_error':curvature_error})
            if callback:callback({'stage':'restoring_six_exact_images','completed':index+1,'total':6})
        self.initial_report=self.span.solve()
        last=self.paths[-1]
        u_error=float(np.linalg.norm(self.span.u-np.load(last/'cumulative_log_direction.npy').reshape(-1))
            /max(np.linalg.norm(self.span.u),1e-300))
        h_error=float(np.linalg.norm(self.span.Hu-np.load(last/'cumulative_total_image.npy').reshape(-1))
            /max(np.linalg.norm(self.span.Hu),1e-300))
        if max(u_error,h_error)>1e-7:
            raise RuntimeError('Widened coordinate boxes did not preserve the old span optimum')
        self.initial_report.update(self.residual_report(),restored_cumulative_direction_error=u_error,
            restored_cumulative_image_error=h_error,
            source_conditional_residual_not_comparable_to_full_residual=True)
        if update['accepted']:
            accepted=next(row for row in reversed(json.loads((self.origin/'line_search.json').read_text())['candidates']) if row['accepted'])
            self.anchor_u=np.load(self.origin/'accepted_log_direction.npy')
            self.anchor_Hu=accepted['alpha']*np.load(last/'cumulative_total_image.npy')
            if not np.allclose(self.anchor_u,accepted['alpha']*np.load(last/'cumulative_log_direction.npy'),rtol=1e-12,atol=1e-14):
                raise RuntimeError('Accepted anchor is not in the known true-image span')
        else:
            self.anchor_u=np.zeros_like(self.material,dtype=np.float64)
            self.anchor_Hu=np.zeros_like(self.anchor_u)
        if not np.array_equal(self.co.material(self.anchor_u),self.incumbent):
            raise RuntimeError('Known-image anchor does not reproduce the physical incumbent')
        self.incumbent_tv=self.co.regularization(self.anchor_u,.001)[0]

    def residual_report(self):
        return {'full_projected_KKT_relative':self.span.projected_norm()/max(self.initial_norm,1e-300),
            'ROI_projected_KKT_relative':self.span.projected_norm_on(self.roi)/max(self.initial_roi_norm,1e-300),
            'initial_full_projected_norm':self.initial_norm,'initial_ROI_projected_norm':self.initial_roi_norm}

    def propose(self,index):
        requested='global' if index%3==2 else 'ROI'
        try:
            direction,record=self.span.next_spatial_direction(
                self.roi if requested=='ROI' else self.co.mask,requested)
        except EmptySpatialBlock as error:
            if requested=='global':
                raise
            direction,record=self.span.next_spatial_direction(self.co.mask,'global')
            record['ROI_to_global_fallback_reason']=str(error)
        record['requested_spatial_block']=requested
        return direction,record

    def coordinate(self,material):
        m=np.asarray(material,np.float64);base=self.co.m;edge=self.co.edge
        u=np.stack([np.log(m[0]/base[0])/.05,np.log((m[0]*m[1])/(base[0]*base[1]))/.1,
            np.log(np.maximum(m[2]-edge,1e-30)/np.maximum(base[2]-edge,1e-30))/.7])
        u[:,~self.co.mask]=0
        return u

    def current_u(self):
        return self.span.u.reshape(self.material.shape)

    def current_Hu(self):
        return self.span.Hu.reshape(self.material.shape)
