"""Continue the identical fixed quadratic with a source-resolved CPU metric."""
from pathlib import Path
import json
import numpy as np
from .native_gn_saved_state import SavedQuadratic
from .native_gn_source_metric import SourceResolvedBFGS
from .native_gn_coordinates import dot
from .refine_native_gn_cpu import atomic,save_array


def load_components(rows,d):
    components=[]
    actual=[]
    for row in rows:
        vector=np.load(row['image'],mmap_mode='r')
        if vector.dtype!=np.float32 or vector.shape!=d.shape:
            raise RuntimeError('Per-shot curvature has the wrong grid or dtype')
        energy=dot(d,vector)
        tangent=row['weighted_tangent_energy']
        defect=abs(energy-tangent)/max(abs(energy),abs(tangent),1e-300)
        if defect>2e-3:raise RuntimeError(f'Per-shot native energy identity failed: {row["name"]}')
        components.append({'name':row['name'],'vector':vector})
        actual.append(dict(row,adjoint_quadratic=energy,
            routine_curvature_identity_relative_error=defect))
    return components,actual


class SourceResolvedQuadratic(SavedQuadratic):
    def __init__(self,source_root,step=1):
        super().__init__(source_root,step)
        self.model=SourceResolvedBFGS(self.base,self.fixed_diagonal,self.co.mask)
        for p in self.paths:
            d=np.load(p/'direction.npy')
            y=np.load(p/'total_image.npy')
            metadata=p/'source_curvature.json'
            if metadata.exists():
                rows=json.loads(metadata.read_text())['shots']
                components,_=load_components(rows,d)
                components.append({'name':'fixed_TV_and_damping',
                    'vector':np.load(p/'regularizer_image.npy')})
                self.model.update(d,y,components)
            else:self.model.update(d,y)

    def append_source(self,d,data_image,folder,cpu_record,rows):
        if len(rows)!=22:
            raise RuntimeError('All 22 shot images are required')
        components,actual=load_components(rows,d)
        prior=self.h_tv(d)+self.damping*d
        components.append({'name':'fixed_TV_and_damping','vector':prior})
        self.model.pending_components=components
        row=super().append(d,data_image,folder,cpu_record)
        row['kind']='source_resolved_SPD_BFGS_box_QP'
        row['per_shot_curvature_vectors']=22
        row['extra_native_wave_solves']=0
        folder=Path(folder)
        save_array(folder/'regularizer_image.npy',prior)
        atomic(folder/'source_curvature.json',{'shots':actual,
            'regularizer_image':str((folder/'regularizer_image.npy').resolve()),
            'additional_native_wave_solves':0})
        atomic(folder/'progress.json',row)
        self.last_record=row
        return row
