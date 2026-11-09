"""Restore the identical native GN problem with a full-rank spectral base."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .native_gn_saved_state import SavedQuadratic
from .native_gn_spectral import SpectralDataMetric
from .native_gn_multisecant import MultisecantMetric
from .native_gn_json import atomic


class SpectralSavedQuadratic(SavedQuadratic):
    def __init__(self,source_root,step,symbol_snapshot):
        super().__init__(source_root,step)
        self.symbol_snapshot=Path(symbol_snapshot).resolve()
        metadata=json.loads((self.symbol_snapshot.parent/'report.json').read_text())
        expected=metadata['symbol_SHA256']
        if hashlib.sha256(self.symbol_snapshot.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Spectral metric snapshot changed')
        if metadata['fixed_material_reference']!=self.reference:
            raise RuntimeError('Spectral fit and saved HVPs use different fixed quadratics')
        base=SpectralDataMetric(np.load(self.symbol_snapshot,mmap_mode='r'),self.co.mask)
        pairs=[(np.load(p/'direction.npy',mmap_mode='r'),np.load(p/'data_image.npy',mmap_mode='r'))
            for p in self.paths]
        self.model=MultisecantMetric(base,self.co.mask,
            lambda x:self.h_tv(x)+self.damping*x,self.tv_diagonal+self.damping,pairs)
        self.spectral_report=dict(self.model.report,symbol_file=str(self.symbol_snapshot),
            symbol_SHA256=expected,symbol_training_count=len(metadata['training_paths']))

    def append(self,d,data_image,folder,cpu_record):
        self.model.pending_data=data_image
        row=super().append(d,data_image,folder,cpu_record)
        row['kind']='spectral_base_all_saved_native_multisecant_box_QP'
        atomic(Path(folder)/'progress.json',row)
        self.last_record=row
        return row
