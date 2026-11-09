"""Dense CPU oracle for the fresh source-block algebra; no patient waves."""
from pathlib import Path
import json
import tempfile
import numpy as np
from .native_gn_spectral import SpectralDataMetric
from .native_gn_block_metric import SourceBlockMetric


def main():
    rng=np.random.default_rng(20261107)
    shape=(3,3,4);size=int(np.prod(shape));mask=np.ones(shape[-2:],bool)
    symbol=np.broadcast_to(np.diag([.8,1.3,1.9]),(3,3,3,3)).copy()
    regularizer=.2
    shots={};rows=[];H=np.zeros((size,size));g=np.zeros(size)
    with tempfile.TemporaryDirectory(prefix='fresh_source_block_') as temporary:
        root=Path(temporary)
        for index in range(22):
            name=f'source_{index:02d}'
            J=rng.normal(size=(45,size))/np.sqrt(45)
            r=rng.normal(size=45)
            hs=J.T@J;gs=J.T@r
            path=root/f'{name}_g.npy';np.save(path,gs.reshape(shape))
            rows.append({'name':name,'image':str(path),'weighted_residual_energy':float(r@r)})
            shots[name]=hs;H+=hs;g+=gs
        metric=SourceBlockMetric(SpectralDataMetric(symbol,mask),mask,
            lambda x:regularizer*x,np.full(shape,regularizer),rows)
        errors=[]
        directions=[]
        for index in range(2):
            d=rng.normal(size=shape);directions.append(d)
            partials=[]
            for name,hs in shots.items():
                path=root/f'{name}_h{index}.npy';np.save(path,(hs@d.ravel()).reshape(shape))
                partials.append({'name':name,'image':str(path)})
            metric.pending_partials=partials
            metric.update(d,(H@d.ravel()).reshape(shape)+regularizer*d)
            for saved in directions:
                target=(H@saved.ravel()).reshape(shape)+regularizer*saved
                errors.append(float(np.linalg.norm(metric.apply(saved)-target)/np.linalg.norm(target)))
        P=sum((np.outer(v.ravel(),v.ravel()) for v in metric.positive),np.zeros((size,size)))
        E=sum(row['weighted_residual_energy'] for row in rows)
        B=np.column_stack([metric.apply(np.eye(size)[:,i].reshape(shape)).ravel() for i in range(size)])
        report={'CPU_only':True,'patient_wave_solves':0,
            'maximum_retained_secant_relative_error':max(errors),
            'source_sketch_upper_bound_margin':float(np.linalg.eigvalsh(H-P).min()),
            'source_residual_lower_bound_margin':float(np.linalg.eigvalsh(P-np.outer(g,g)/E).min()),
            'full_metric_minimum_eigenvalue':float(np.linalg.eigvalsh(B).min()),
            'metric_symmetry_relative_error':float(np.linalg.norm(B-B.T)/np.linalg.norm(B)),
            'metric_diagonal_relative_error':float(np.linalg.norm(B.diagonal()-metric.diagonal.ravel())/np.linalg.norm(B.diagonal()))}
        assert max(errors)<1e-10,report
        assert report['source_sketch_upper_bound_margin']>-1e-10,report
        assert report['source_residual_lower_bound_margin']>-1e-10,report
        assert report['full_metric_minimum_eigenvalue']>0,report
        assert report['metric_symmetry_relative_error']<1e-10,report
        assert report['metric_diagonal_relative_error']<1e-10,report
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
