"""Restore only saved true native images, without an approximate QP model."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .native_gn_coordinates import Coordinates,physical_blocks,dot
from .native_gn_galerkin import TrueImageSpan


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecycledState:
    def __init__(self,source,prepared_span=Path('out/v5/native_gn_saved_span_v1'),callback=None):
        self.source=Path(source).resolve();self.prepared_span=Path(prepared_span).resolve()
        run=json.loads((self.source/'run.json').read_text())
        self.reference=json.loads((self.source/'fixed_model_reference.json').read_text())
        self.origin=Path(self.reference['origin_step_folder'])
        previous=json.loads((self.prepared_span/'run.json').read_text())
        if previous['status']!='saved_span_CPU_solution_finished':raise RuntimeError('Complete CPU span solution first')
        provenance=json.loads((self.prepared_span/'metrics.json').read_text())
        if digest(self.origin/'old_material.npy')!=provenance['fixed_reference_material_SHA256']:
            raise RuntimeError('Different fixed native reference')
        oldrun=json.loads((self.origin.parent/'run.json').read_text());self.sample=oldrun['sample']
        self.preconditioner=Path(oldrun['preconditioner_statistics'])
        if digest(self.preconditioner)!=oldrun['preconditioner_SHA256']:raise RuntimeError('Native metric statistics changed')
        self.material=np.load(self.origin/'old_material.npy')
        self.co=Coordinates(self.material,np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'))
        with np.load(self.preconditioner) as data:blocks=physical_blocks(data['blocks'],self.material,self.co.edge)
        self.paths=[Path(p) for p in json.loads((self.source/'step001/fixed_HVP_manifest.json').read_text())['paths']]
        if len(self.paths)!=15 or [str(p) for p in self.paths]!=[r['path'] for r in provenance['HVP_provenance']]:
            raise RuntimeError('Reuse the same fifteen true HVPs')
        first=json.loads((self.paths[0]/'progress.json').read_text());self.initial_norm=first['initial_projected_KKT_norm']
        self.scale=first['local_data_preconditioner_scale'];self.damping=self.reference['damping_relative']*self.scale*blocks[:3]
        self.damping[:,~self.co.mask]=0
        zero=np.zeros_like(self.material,dtype=np.float64)
        self.tv,self.g_tv,self.h_tv,tv_diagonal=self.co.regularization(zero,self.reference['tv_weight'])
        self.gradient=np.load(self.origin/'data_gradient.npy')+self.g_tv
        self.diagonal=self.scale*blocks[:3]+tv_diagonal+self.damping;self.diagonal[:,~self.co.mask]=1
        self.diagonal=np.maximum(self.diagonal,1e-30);self.low,self.high=self.co.bounds()
        self.span=TrueImageSpan(self.gradient,self.diagonal,self.low,self.high)
        for i,(path,row) in enumerate(zip(self.paths,provenance['HVP_provenance'])):
            if digest(path/'direction.npy')!=row['direction_SHA256'] or digest(path/'total_image.npy')!=row['total_image_SHA256']:
                raise RuntimeError('Saved native pair changed')
            self.span.append(np.load(path/'direction.npy'),np.load(path/'total_image.npy'))
            if callback is not None:callback({'stage':'restoring_known_true_images','completed':i+1,'total':15})
        self.initial_report=self.span.solve()
        if not np.allclose(self.span.u.reshape(self.material.shape),np.load(self.prepared_span/'log_direction.npy'),rtol=1e-8,atol=1e-10):
            raise RuntimeError('Independent saved CPU optimum did not reconstruct')
        self.baseline_rf=json.loads((self.origin/'baseline_RF.json').read_text())

    def append(self,direction,data_image):
        image=data_image+self.h_tv(direction)+self.damping*direction
        report=self.span.append(direction,image);report.update(self.span.solve())
        report['full_projected_KKT_relative']=self.span.projected_norm()/self.initial_norm
        return image,report

    def coordinate(self,material):
        m=np.asarray(material,np.float64);base=self.co.m;edge=self.co.edge
        u=np.stack([np.log(m[0]/base[0])/.05,
            np.log((m[0]*m[1])/(base[0]*base[1]))/.1,
            np.log(np.maximum(m[2]-edge,1e-30)/np.maximum(base[2]-edge,1e-30))/.7])
        u[:,~self.co.mask]=0
        return u

    def current_u(self):return self.span.u.reshape(self.material.shape)

    def current_Hu(self):return self.span.Hu.reshape(self.material.shape)
