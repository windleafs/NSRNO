"""Full native fixed quadratic using all retained source and residual data."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .native_gn_saved_state import SavedQuadratic
from .native_gn_spectral import SpectralDataMetric
from .native_gn_enriched_metric import SourceEnrichedMetric
from .native_gn_json import atomic
from .refine_native_gn_cpu import save_array
from .native_gn_coordinates import dot


class SourceEnrichedState(SavedQuadratic):
    def __init__(self,source_root,step,symbol_snapshot):
        super().__init__(source_root,step)
        self.symbol_snapshot=Path(symbol_snapshot).resolve()
        metadata=json.loads((self.symbol_snapshot.parent/'report.json').read_text())
        expected=metadata['symbol_SHA256']
        if hashlib.sha256(self.symbol_snapshot.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Spectral snapshot changed')
        if metadata['fixed_material_reference']!=self.reference:
            raise RuntimeError('Metric and source have different fixed references')
        features=[]
        for index,path in enumerate(self.paths):
            record=path/'source_curvature.json'
            if record.exists():
                for row in json.loads(record.read_text())['shots']:
                    features.append(dict(row,direction_index=index))
        if not features:raise RuntimeError('No complete native per-shot features')
        base=SpectralDataMetric(np.load(self.symbol_snapshot,mmap_mode='r'),self.co.mask)
        pairs=[(np.load(p/'direction.npy',mmap_mode='r'),np.load(p/'data_image.npy',mmap_mode='r'))
            for p in self.paths]
        self.model=SourceEnrichedMetric(base,self.co.mask,
            lambda x:self.h_tv(x)+self.damping*x,self.tv_diagonal+self.damping,
            pairs,np.load(self.origin/'data_gradient.npy'),2*self.baseline_rf['loss'],features)
        error=self.model.apply(self.u)-self.Hu
        defect=np.sqrt(max(dot(error,error),0)/max(dot(self.Hu,self.Hu),1e-300))
        if defect>2e-5:raise RuntimeError(f'Enriched metric lost the saved cumulative state: {defect}')
        self.spectral_report=dict(self.model.report,symbol_file=str(self.symbol_snapshot),
            symbol_SHA256=expected,symbol_training_count=len(metadata['training_paths']),
            known_state_model_Hu_relative_error=defect)

    def append(self,d,data_image,folder,cpu_record):
        folder=Path(folder)
        raw=folder.parent/(folder.name+'_native_raw')
        raw.mkdir(parents=True,exist_ok=False)
        save_array(raw/'direction.npy',d)
        save_array(raw/'data_image.npy',data_image)
        atomic(raw/'native_raw.json',{'native_HVP_number':len(self.paths)+1,
            'data_curvature':dot(d,data_image),'metric_update_completed':False,
            'native_result_saved_before_CPU_metric_update':True})
        self.model.pending_data=data_image
        row=super().append(d,data_image,folder,cpu_record)
        row['kind']='all_saved_native_source_and_residual_feature_QP'
        atomic(folder/'progress.json',row)
        atomic(raw/'native_raw.json',{'native_HVP_number':len(self.paths),
            'data_curvature':dot(d,data_image),'metric_update_completed':True,
            'native_result_saved_before_CPU_metric_update':True})
        self.last_record=row
        return row
