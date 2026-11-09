"""Independent full-grid material quadratic at each new nonlinear iterate."""
from pathlib import Path
import numpy as np
from .native_gn_coordinates import Coordinates,physical_blocks,dot
from .native_gn_secant import projected_norm
from .native_gn_spectral import SpectralDataMetric
from .native_gn_block_metric import SourceBlockMetric
from .native_gn_json import atomic
from .refine_native_gn_cpu import save_array


class FreshSourceQuadratic:
    def __init__(self,material,edge,gradient,baseline,symbol,primitive_blocks,
                 damping_scale,tv_weight=.001,damping_relative=.01,
                 trust=(.02,.10,.35)):
        self.co=Coordinates(material,edge)
        self.material=np.asarray(material,np.float32)
        self.gradient_data=np.asarray(gradient,np.float64)
        self.baseline=baseline
        self.trust=tuple(trust)
        self.u=np.zeros_like(material,dtype=np.float64)
        self.Hu=self.u.copy()
        self.tv,self.g_tv,self.h_tv,self.tv_diagonal=self.co.regularization(self.u,tv_weight)
        self.gradient0=self.gradient_data+self.g_tv
        self.gradient=self.gradient0.copy()
        blocks=physical_blocks(primitive_blocks,material,self.co.edge)
        self.damping=damping_relative*damping_scale*blocks[:3]
        self.damping[:,~self.co.mask]=0
        self.low,self.high=self.co.bounds(trust=self.trust)
        self.model=SourceBlockMetric(SpectralDataMetric(symbol,self.co.mask),
            self.co.mask,lambda x:self.h_tv(x)+self.damping*x,
            self.tv_diagonal+self.damping,baseline['residual_features'])
        self.fixed_diagonal=self.model.diagonal.copy()
        self.initial_norm=self.norm()
        self.history=[]
        self.tv_weight=tv_weight

    def norm(self):
        return projected_norm(self.u,self.gradient,self.low,self.high,self.fixed_diagonal)

    def value(self):
        return dot(self.gradient0,self.u)+.5*dot(self.u,self.Hu)

    def append(self,d,data_image,source_rows,folder,cpu):
        folder=Path(folder)
        raw=folder.parent/(folder.name+'_native_raw')
        raw.mkdir(parents=True,exist_ok=False)
        save_array(raw/'direction.npy',d)
        save_array(raw/'data_image.npy',data_image)
        atomic(raw/'source_curvature.json',{'shots':source_rows})
        image=data_image+self.h_tv(d)+self.damping*d
        slope=dot(self.gradient,d);curvature=dot(d,image)
        if slope>=0 or curvature<=0:raise RuntimeError('Fresh source direction has no native descent')
        alpha=min(1.,-slope/curvature)
        old=self.value()
        self.u+=alpha*d;self.Hu+=alpha*image
        self.gradient=self.gradient0+self.Hu
        if self.value()>old+1e-10 or np.any(self.u<self.low-1e-10) or np.any(self.u>self.high+1e-10):
            raise RuntimeError('Fresh native segment lost decrease or feasibility')
        self.model.pending_partials=source_rows
        model=self.model.update(d,image)
        row={'iteration':len(self.history)+1,'kind':'fresh_source_block_native_GN',
            'segment_alpha':alpha,'segment_slope':slope,'segment_data_curvature':dot(d,data_image),
            'segment_total_curvature':curvature,'model_change':self.value(),
            'projected_KKT_relative':self.norm()/max(self.initial_norm,1e-300),
            'CPU_subproblem':cpu,'metric':model,'trust_log_c_log_Z_log_alpha':self.trust,
            'reference_is_current_nonlinear_material':True,'truth_read':False}
        folder.mkdir(parents=True,exist_ok=False)
        for name,value in [('direction',d),('data_image',data_image),('total_image',image),
            ('cumulative_log_direction',self.u),('cumulative_total_image',self.Hu)]:
            save_array(folder/f'{name}.npy',value)
        atomic(folder/'source_curvature.json',{'shots':source_rows})
        atomic(folder/'progress.json',row)
        self.history.append(row)
        return row

    def predicted_physical_change(self,alpha):
        return alpha*dot(self.gradient0,self.u)+.5*alpha**2*(
            dot(self.u,self.Hu)-dot(self.u,self.damping*self.u))
