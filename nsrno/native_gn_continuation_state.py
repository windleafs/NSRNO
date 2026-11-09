"""Restore saved native GN images at one reference for a longer global solve."""
from pathlib import Path
import json
import numpy as np
from .native_gn_spatial_state import NativeSpatialState,digest
from .native_gn_coordinates import dot


class NativeContinuationState(NativeSpatialState):
    def __init__(self,source):
        source=Path(source).resolve();folder=source/'step001'
        run=json.loads((source/'run.json').read_text())
        if run['status'] not in ['native_spatial_Krylov_finished','native_long_GN_finished']:
            raise RuntimeError('Require a finished physical source')
        workers=json.loads((source/'worker_processes.json').read_text())
        pids=[run['pid'],run.get('child_pid'),*workers['worker_pids']]
        if any(pid and Path('/proc',str(pid)).exists() for pid in pids):
            raise RuntimeError('Source parent or child still exists')
        for name,expected in run['source_sha256'].items():
            if digest(Path(__file__).parent/name)!=expected:
                raise RuntimeError('Native source changed: '+name)
        reference_source=Path(run.get('reference_source_root',run['source_root'])).resolve()
        super().__init__(reference_source)
        self.reference_source_root=reference_source
        if digest(folder/'linear_reference_material.npy')!=self.fixed_SHA256:
            raise RuntimeError('Different exact linear reference')
        if digest(folder/'data_gradient.npy')!=digest(self.origin/'data_gradient.npy'):
            raise RuntimeError('Different saved gradient')
        paths=[Path(v) for v in json.loads((folder/'fixed_HVP_manifest.json').read_text())['paths']]
        if paths[:6]!=self.paths or len(paths)<12:
            raise RuntimeError('Expected the same six initial and subsequent exact images')
        for path in paths[6:]:
            progress=json.loads((path/'progress.json').read_text())
            direction=np.load(path/'direction.npy');data_image=np.load(path/'data_image.npy')
            image=np.load(path/'total_image.npy')
            if any(value.shape!=self.material.shape or not np.isfinite(value).all()
                   for value in [direction,data_image,image]):
                raise RuntimeError('Incomplete saved reference image')
            expected=data_image+self.h_tv(direction)+self.damping*direction
            error=float(np.linalg.norm(expected-image)/max(np.linalg.norm(image),1e-300))
            job_files=list((path.parent.parent/'jobs').glob('*_'+progress['true_HVP_job']+'.json'))
            if len(job_files)!=1:
                raise RuntimeError('Ambiguous original HVP job')
            job=json.loads(job_files[0].read_text())
            curvature=dot(direction,data_image)
            curvature_error=abs(curvature-job['adjoint_quadratic'])/max(abs(curvature),1e-300)
            if max(error,curvature_error)>1e-10:
                raise RuntimeError('Saved HVP does not match its own reference operator')
            self.span.append(direction,image)
            self.provenance.append({'path':str(path),'direction_SHA256':digest(path/'direction.npy'),
                'total_image_SHA256':digest(path/'total_image.npy'),'total_image_error':error,
                'data_curvature_error':curvature_error})
        self.paths=paths
        self.initial_report=self.span.solve()
        expected_u=np.load(folder/'log_direction.npy').reshape(-1)
        expected_Hu=np.load(folder/'cumulative_total_image.npy').reshape(-1)
        u_error=float(np.linalg.norm(self.span.u-expected_u)/max(np.linalg.norm(expected_u),1e-300))
        h_error=float(np.linalg.norm(self.span.Hu-expected_Hu)/max(np.linalg.norm(expected_Hu),1e-300))
        if max(u_error,h_error)>1e-7:
            raise RuntimeError('Full saved span did not reconstruct')
        self.incumbent=np.load(folder/'material.npy');self.incumbent_SHA256=digest(folder/'material.npy')
        update=json.loads((folder/'material_update.json').read_text())
        if self.incumbent_SHA256!=run['material_SHA256'] or update['material_SHA256']!=self.incumbent_SHA256:
            raise RuntimeError('Protected material evidence changed')
        self.incumbent_rf=json.loads((folder/'incumbent_RF.json').read_text())
        if self.incumbent_rf['loss']!=update['native_data_after']:
            raise RuntimeError('Protected RF record changed')
        accepted=(folder/'accepted_log_direction.npy').exists()
        self.anchor_u=np.load(folder/('accepted_log_direction.npy' if accepted else 'anchor_log_direction.npy'))
        self.anchor_Hu=np.load(folder/('accepted_total_image.npy' if accepted else 'anchor_total_image.npy'))
        if not np.array_equal(self.co.material(self.anchor_u),self.incumbent):
            raise RuntimeError('Saved anchor does not reproduce accepted physical material')
        coefficient=np.linalg.solve(self.span.gram,self.span.images@self.anchor_u.reshape(-1))
        anchor_error=float(np.linalg.norm(self.span.basis.T@coefficient-self.anchor_u.reshape(-1))
            /max(np.linalg.norm(self.anchor_u),1e-300))
        anchor_image_error=float(np.linalg.norm(self.span.images.T@coefficient-self.anchor_Hu.reshape(-1))
            /max(np.linalg.norm(self.anchor_Hu),1e-300))
        if max(anchor_error,anchor_image_error)>1e-7:
            raise RuntimeError('Physical anchor is not represented by the known exact images')
        self.anchor_coefficients=coefficient
        self.incumbent_tv=self.co.regularization(self.anchor_u,.001)[0]
        self.baseline_rf=json.loads((folder/'baseline_RF.json').read_text())
        if self.baseline_rf['loss']!=run['fixed_reference_RF_loss']:
            raise RuntimeError('Fixed reference RF evidence disagrees')
        self.initial_report.update(self.residual_report(),restored_cumulative_direction_error=u_error,
            restored_cumulative_image_error=h_error,anchor_direction_error=anchor_error,
            anchor_image_error=anchor_image_error,restored_complete_HVPs=len(paths))
        self.source=source;self.origin=folder;self.source_run=run
        self.reference.update(origin_step_folder=str(folder),
            fixed_reference_material=str(folder/'linear_reference_material.npy'))
        self.fixed_reference_path=folder/'linear_reference_material.npy'

    def propose_global(self):
        direction,record=self.span.next_direction()
        record.update(proposal_block='global',no_new_hard_FOV_mask=True,
            all_three_coordinates_free=True)
        return direction,record
