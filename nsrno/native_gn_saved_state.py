"""Restore/extend one fixed native Gauss–Newton quadratic without replays."""
from pathlib import Path
import json
import numpy as np
from .native_gn_coordinates import Coordinates,physical_blocks,dot
from .native_gn_secant import CompactBFGS,projected_norm
from .physical_box_direction import local_apply

class SavedQuadratic:
    def __init__(self,source_root,step=1):
        self.source_root=Path(source_root).resolve()
        self.source_folder=self.source_root/f'step{step:03d}'
        reference=self.source_root/'fixed_model_reference.json'
        if reference.exists():
            self.reference=json.loads(reference.read_text())
            origin=Path(self.reference['origin_step_folder'])
        else:
            origin=self.source_folder
            self.reference={'origin_step_folder':str(origin),'tv_weight':.001,
                'damping_relative':.01,'true_QP_target_relative':.001}
        self.origin=origin
        original_run=json.loads((origin.parent/'run.json').read_text())
        self.sample=original_run['sample']
        self.preconditioner=Path(original_run['preconditioner_statistics'])
        self.material=np.load(origin/'old_material.npy')
        self.co=Coordinates(self.material,np.load('out/v5/uno_triparam_v1/known_edge_alpha.npy'))
        with np.load(self.preconditioner) as data:
            self.blocks=physical_blocks(data['blocks'],self.material,self.co.edge)
        manifest=self.source_folder/'fixed_HVP_manifest.json'
        if manifest.exists():
            self.paths=[Path(p) for p in json.loads(manifest.read_text())['paths']]
        else:
            self.paths=sorted(p.parent for p in self.source_folder.glob('hvp*/progress.json'))
        if not self.paths:raise RuntimeError('No complete true-HVP checkpoint')
        rows=[json.loads((p/'progress.json').read_text()) for p in self.paths]
        self.scale=rows[0]['local_data_preconditioner_scale']
        self.damping=self.reference['damping_relative']*self.scale*self.blocks[:3]
        self.damping[:,~self.co.mask]=0
        zero=np.zeros_like(self.material,dtype=np.float64)
        self.tv,self.g_tv,self.h_tv,self.tv_diagonal=self.co.regularization(
            zero,self.reference['tv_weight'])
        self.gradient0=np.load(origin/'data_gradient.npy')+self.g_tv
        self.fixed_diagonal=self.scale*self.blocks[:3]+self.tv_diagonal+self.damping
        self.fixed_diagonal[:,~self.co.mask]=1.
        self.fixed_diagonal=np.maximum(self.fixed_diagonal,1e-30)
        self.model=CompactBFGS(self.base,self.fixed_diagonal,self.co.mask)
        for p in self.paths:
            self.model.update(np.load(p/'direction.npy'),np.load(p/'total_image.npy'))
        self.u=np.load(self.paths[-1]/'cumulative_log_direction.npy')
        self.Hu=np.load(self.paths[-1]/'cumulative_total_image.npy')
        self.gradient=self.gradient0+self.Hu
        self.low,self.high=self.co.bounds()
        self.initial_norm=rows[0]['initial_projected_KKT_norm']
        self.last_record=rows[-1]
        if abs(self.value()-self.last_record['model_change'])>1e-10:
            raise RuntimeError('True quadratic checkpoint value failed to reconstruct')
        if abs(dot(self.u,self.damping*self.u)-self.last_record['cumulative_damping_quadratic'])>1e-10:
            raise RuntimeError('Fixed damping did not reconstruct')
        if abs(dot(self.u,self.h_tv(self.u))-self.last_record['cumulative_TV_quadratic'])>1e-10:
            raise RuntimeError('Fixed TV metric did not reconstruct')
        self.best_material=np.load(self.source_folder/'material.npy')
        if (self.source_folder/'best_log_direction.npy').exists():
            self.best_u=np.load(self.source_folder/'best_log_direction.npy')
            self.best_rf=json.loads((self.source_folder/'incumbent_RF.json').read_text())
        else:
            update=json.loads((self.source_folder/'material_update.json').read_text())
            if update['accepted']:
                self.best_u=update['accepted_alpha']*np.load(self.source_folder/'log_direction.npy')
                self.best_rf=json.loads((self.source_folder/'line_search.json').read_text())['candidates'][-1]['native_RF']
            else:
                self.best_u=zero
                self.best_rf=json.loads((origin/'baseline_RF.json').read_text())
        if not np.array_equal(self.co.material(self.best_u),self.best_material):
            raise RuntimeError('Incumbent material does not match the saved fixed-reference coordinates')
        self.baseline_rf=json.loads((origin/'baseline_RF.json').read_text())
        self.best_cost=self.best_rf['loss']+self.co.regularization(
            self.best_u,self.reference['tv_weight'])[0]

    def base(self,x):
        return (self.scale*local_apply(self.blocks,x)+self.h_tv(x)+self.damping*x)*self.co.mask

    def value(self):
        return dot(self.gradient0,self.u)+.5*dot(self.u,self.Hu)

    def norm(self):
        return projected_norm(self.u,self.gradient,self.low,self.high,self.fixed_diagonal)

    def append(self,d,data_image,folder,cpu_record):
        image=data_image+self.h_tv(d)+self.damping*d
        slope=dot(self.gradient,d);curvature=dot(d,image)
        if slope>=0 or curvature<=0:
            raise RuntimeError('Accurate-QP proposal has no true quadratic descent')
        alpha=min(1.,-slope/curvature)
        old_value=self.value()
        self.u+=alpha*d;self.Hu+=alpha*image
        self.gradient=self.gradient0+self.Hu
        if self.value()>old_value+1e-10 or np.any(self.u<self.low-1e-10) or np.any(self.u>self.high+1e-10):
            raise RuntimeError('Resumed true quadratic lost decrease or feasibility')
        secant=self.model.update(d,image)
        total=dot(self.u,self.Hu);tvq=dot(self.u,self.h_tv(self.u));dampingq=dot(self.u,self.damping*self.u)
        record={'iteration':len(self.paths)+1,'kind':'accurate_compact_BFGS_box_QP',
            'segment_alpha':alpha,'model_change':self.value(),'segment_slope':slope,
            'segment_data_curvature':dot(d,data_image),'segment_total_curvature':curvature,
            'local_data_preconditioner_scale':self.scale,'projected_KKT_norm':self.norm(),
            'initial_projected_KKT_norm':self.initial_norm,
            'projected_KKT_relative':self.norm()/max(self.initial_norm,1e-300),
            'cumulative_data_quadratic':total-tvq-dampingq,'cumulative_TV_quadratic':tvq,
            'cumulative_damping_quadratic':dampingq,'hvp_count':len(self.paths)+1,
            'CPU_subproblem':cpu_record,'compact_secant':secant}
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
        for name,value in [('direction',d),('data_image',data_image),('total_image',image),
            ('cumulative_log_direction',self.u),('cumulative_total_image',self.Hu)]:
            np.save(folder/f'{name}.npy',value)
        (folder/'progress.json').write_text(json.dumps(record,indent=2)+'\n')
        self.paths.append(folder.resolve());self.last_record=record
        return record
