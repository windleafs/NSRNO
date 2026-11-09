"""Restore an interrupted native global solve without relabeling its old images."""
from pathlib import Path
import json
import numpy as np
from .native_gn_spatial_state import NativeSpatialState,SpatialSpan,digest
from .native_gn_coordinates import Coordinates,physical_blocks,dot


class NativeRecoveredGlobalState(NativeSpatialState):
    def __init__(self,source):
        source=Path(source).resolve();folder=source/'step001'
        run=json.loads((source/'run.json').read_text())
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if run['status'] not in ['native_global_HVP','global_true_subspace_QP',
                'native_material_line_search','post_hoc_maps_only','restored_global_continuation_state']:
            raise RuntimeError('Require an interrupted active global solve')
        if boot==run['boot_id']:
            raise RuntimeError('This recovery requires positive evidence of a changed server boot')
        # Old processes cannot survive a changed boot; reused numeric PIDs are irrelevant.
        for name,expected in run['source_sha256'].items():
            if digest(Path(__file__).parent/name)!=expected:
                raise RuntimeError('Saved exact native source changed: '+name)
        self.source=source;self.origin=folder;self.source_run=run;self.sample=run['sample']
        self.reference_source_root=Path(run['reference_source_root']).resolve()
        reference_run=json.loads((self.reference_source_root/'run.json').read_text())
        self.material=np.load(folder/'linear_reference_material.npy')
        self.fixed_reference_path=folder/'linear_reference_material.npy'
        self.fixed_SHA256=digest(self.fixed_reference_path)
        if self.fixed_SHA256!=run['fixed_reference_SHA256']:
            raise RuntimeError('Fixed linear reference changed')
        if digest(self.reference_source_root/'step001/stage0_initial_material.npy')!=self.fixed_SHA256:
            raise RuntimeError('The original exact reference no longer agrees')
        if self.material.shape!=(3,961,1121) or self.material.dtype!=np.float32:
            raise ValueError('Full native float32 reference required')
        if not np.array_equal(self.material,np.load(folder/'old_material.npy')):
            raise RuntimeError('Reference material copies disagree')
        if digest(folder/'data_gradient.npy')!=digest(self.reference_source_root/'step001/data_gradient.npy'):
            raise RuntimeError('Saved exact data gradient changed')
        self.preconditioner=Path(reference_run['preconditioner_statistics'])
        if digest(self.preconditioner)!='acd29954a613f156e95530c9dd67d413e6f1ea9a1b1492ec51eb58a354bfa961':
            raise RuntimeError('Preconditioner statistics changed')
        self.scale=reference_run['preconditioner_scale']
        self.co=Coordinates(self.material,np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'))
        with np.load(self.preconditioner) as data:
            blocks=physical_blocks(data['blocks'],self.material,self.co.edge)
        self.damping=.01*self.scale*blocks[:3];self.damping[:,~self.co.mask]=0
        zero=np.zeros_like(self.material,dtype=np.float64)
        self.tv,self.g_tv,self.h_tv,tv_diagonal=self.co.regularization(zero,.001)
        self.gradient=np.load(folder/'data_gradient.npy')+self.g_tv
        if not np.isfinite(self.gradient).all():raise RuntimeError('Nonfinite saved full gradient')
        self.diagonal=np.maximum(self.scale*blocks[:3]+tv_diagonal+self.damping,1e-30)
        self.diagonal[:,~self.co.mask]=1
        low,high=self.co.bounds(trust=(1.,1.,10.))
        self.span=SpatialSpan(self.gradient,self.diagonal,low,high)
        z=(np.arange(961)-60)*.05;x=(np.arange(1121)-560)*.05
        self.roi=(abs(x)[None,:]<=19.1)&(z[:,None]>=1)&(z[:,None]<=43)
        self.initial_norm=self.span.projected_norm();self.initial_roi_norm=self.span.projected_norm_on(self.roi)
        self.paths=[Path(v) for v in json.loads((folder/'fixed_HVP_manifest.json').read_text())['paths']]
        if len(self.paths)!=run['reused_complete_HVPs']+run['new_HVPs']:
            raise RuntimeError('Complete manifest does not match recorded work')
        self.provenance=[]
        for path in self.paths:
            direction=np.load(path/'direction.npy');data_image=np.load(path/'data_image.npy')
            image=np.load(path/'total_image.npy')
            if any(v.shape!=self.material.shape or not np.isfinite(v).all() for v in [direction,data_image,image]):
                raise RuntimeError('Incomplete saved image')
            expected=data_image+self.h_tv(direction)+self.damping*direction
            error=float(np.linalg.norm(expected-image)/max(np.linalg.norm(image),1e-300))
            progress=json.loads((path/'progress.json').read_text())
            jobs=list((path.parent.parent/'jobs').glob('*_'+progress['true_HVP_job']+'.json'))
            if len(jobs)!=1:raise RuntimeError('Ambiguous original native job')
            job=json.loads(jobs[0].read_text());curvature=dot(direction,data_image)
            curvature_error=abs(curvature-job['adjoint_quadratic'])/max(abs(curvature),1e-300)
            if max(error,curvature_error)>1e-10:raise RuntimeError('Saved image fails its original operator evidence')
            self.span.append(direction,image)
            self.provenance.append({'path':str(path),'direction_SHA256':digest(path/'direction.npy'),
                'total_image_SHA256':digest(path/'total_image.npy'),'total_image_error':error,
                'data_curvature_error':curvature_error,'original_native_job':str(jobs[0])})
        self.initial_report=self.span.solve()
        expected_u=np.load(folder/'log_direction.npy').reshape(-1)
        expected_Hu=np.load(folder/'cumulative_total_image.npy').reshape(-1)
        u_error=float(np.linalg.norm(self.span.u-expected_u)/max(np.linalg.norm(expected_u),1e-300))
        h_error=float(np.linalg.norm(self.span.Hu-expected_Hu)/max(np.linalg.norm(expected_Hu),1e-300))
        if max(u_error,h_error)>1e-7:raise RuntimeError('Restored cumulative model does not reproduce')
        self.incumbent=np.load(folder/'material.npy');self.incumbent_SHA256=digest(folder/'material.npy')
        update=json.loads((folder/'material_update.json').read_text())
        if self.incumbent_SHA256!=run['material_SHA256'] or update['material_SHA256']!=self.incumbent_SHA256:
            raise RuntimeError('Protected physical material changed')
        self.incumbent_rf=json.loads((folder/'incumbent_RF.json').read_text())
        if self.incumbent_rf['loss']!=update['native_data_after']:
            raise RuntimeError('Protected physical RF changed')
        self.anchor_u=np.load(folder/'accepted_log_direction.npy')
        self.anchor_Hu=np.load(folder/'accepted_total_image.npy')
        if not np.array_equal(self.co.material(self.anchor_u),self.incumbent):
            raise RuntimeError('Known-image anchor does not reproduce physical material')
        coefficient=np.linalg.solve(self.span.gram,self.span.images@self.anchor_u.reshape(-1))
        anchor_error=float(np.linalg.norm(self.span.basis.T@coefficient-self.anchor_u.reshape(-1))
            /max(np.linalg.norm(self.anchor_u),1e-300))
        anchor_image_error=float(np.linalg.norm(self.span.images.T@coefficient-self.anchor_Hu.reshape(-1))
            /max(np.linalg.norm(self.anchor_Hu),1e-300))
        if max(anchor_error,anchor_image_error)>1e-7:raise RuntimeError('Known-image physical anchor did not reconstruct')
        self.incumbent_tv=self.co.regularization(self.anchor_u,.001)[0]
        self.baseline_rf=json.loads((folder/'baseline_RF.json').read_text())
        if self.baseline_rf['loss']!=run['fixed_reference_RF_loss']:raise RuntimeError('Reference RF evidence changed')
        self.reference={'origin_step_folder':str(folder),'tv_weight':.001,'damping_relative':.01,
            'fixed_reference_material':str(self.fixed_reference_path),'full_joint_coordinates':[0,1,2]}
        self.original_new_offset=run.get('original_new_HVPs',run['new_HVPs'])
        self.original_total_budget=run.get('original_total_new_HVP_budget',run['extra_HVP_budget'])
        self.original_budget_start=run.get('original_budget_start_unix_s',run['created_unix_s'])
        self.original_deadline=run.get('original_deadline_unix_s',self.original_budget_start+run['walltime_budget_seconds'])
        self.original_initial_root=run.get('original_initial_root',str(source))
        self.original_source_physical_root=run.get('original_source_physical_root',run['source_root'])
        self.recovery_evidence={'previous_boot_id':run['boot_id'],'current_boot_id':boot,
            'source_status_is_stale_after_reboot':run['status'],'restored_complete_HVPs':len(self.paths),
            'original_new_HVPs_completed':self.original_new_offset,'original_total_HVP_budget':self.original_total_budget,
            'original_deadline_unix_s':self.original_deadline,'new_gradient_wave_solves':0,
            'new_complete_HVP_wave_solves_for_restoration':0,
            'partial_direction_without_image_preserved':str(folder/f'hvp{len(self.paths):03d}'),
            'partial_image_not_relabelled_as_complete':True}
        self.initial_report.update(self.residual_report(),restored_cumulative_direction_error=u_error,
            restored_cumulative_image_error=h_error,anchor_direction_error=anchor_error,
            anchor_image_error=anchor_image_error,recovery_evidence=self.recovery_evidence)

    def propose_global(self):
        direction,record=self.span.next_direction()
        record.update(proposal_block='global',no_new_hard_FOV_mask=True,all_three_coordinates_free=True)
        return direction,record
