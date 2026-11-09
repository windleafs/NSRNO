"""Build uncertain CDF envelopes from existing TRAIN quantiles only."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from .global_cdf_prior import make_grid,kernel,histogram_cdf,smooth_cdf


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fit-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise RuntimeError('Do not overwrite a prepared template')
    run=json.loads((a.fit_root/'run.json').read_text())
    source=a.fit_root/'products/prior.npz'
    if run['status']!='independent_TRAIN_global_quantile_fit_finished' or digest(source)!=run['prior_SHA256']:
        raise RuntimeError('Frozen completed TRAIN quantiles required')
    started=time.time()
    with np.load(source,allow_pickle=False) as archive:
        q,u,w=archive['training_quantiles'],archive['probability'],archive['training_weights']
    if q.shape!=(542,3,2049) or w.shape!=(542,) or abs(w.sum()-1)>1e-12:
        raise RuntimeError('Unexpected TRAIN quantile model')
    k=kernel()
    state={'kernel':k,'quantile_fit_SHA256':np.array(run['prior_SHA256'])}
    mid=(np.arange(2048)+.5)/2048
    summaries={}
    for field in range(3):
        grid=make_grid(field)
        curves=[]
        for quantile in q[:,field]:
            values=np.interp(mid,u,quantile)
            cdf,_=histogram_cdf(values,grid)
            curves.append(smooth_cdf(cdf,k))
        curves=np.asarray(curves)
        order=np.argsort(curves,axis=0,kind='stable')
        sorted_cdf=np.take_along_axis(curves,order,axis=0)
        cumulative=np.cumsum(w[order],axis=0)
        lo=sorted_cdf[np.argmax(cumulative>=.05,axis=0),np.arange(len(grid))]
        hi=sorted_cdf[np.argmax(cumulative>=.95,axis=0),np.arange(len(grid))]
        lo,hi=np.maximum(0.,lo-.10),np.minimum(1.,hi+.10)
        state.update({f'grid_{field}':grid,f'lower_{field}':lo,f'upper_{field}':hi})
        summaries[f'f{field}']={'grid_size':len(grid),'feature_range':[float(grid[0]),float(grid[-1])],
            'minimum_lower':float(lo.min()),'maximum_upper':float(hi.max())}
    a.out.mkdir(parents=True)
    np.savez(a.out/'prior.npz',**state)
    result={'status':'independent_TRAIN_CDF_envelopes_prepared','seconds':time.time()-started,
        'prior_SHA256':digest(a.out/'prior.npz'),'source_quantiles_SHA256':run['prior_SHA256'],
        'global_cdf_source_SHA256':digest(Path(__file__).with_name('global_cdf_prior.py')),
        'TRAIN_patients':1,'base_records':510,'unique_augmentation_materials':32,
        'empirical_CDF_percentiles':[.05,.95],'additional_CDF_tolerance':.10,
        'feature_bin_width':.01,'feature_kernel_sigma_bins':2.,
        'material_or_RF_arrays_read':False,'development_truth_read':False,'reserved_arrays_read':False,
        'new_wave_solves':0,'new_material_derivative_probes':0,'CUDA_used':False,
        'not_a_reconstruction':True,'per_field':summaries}
    (a.out/'metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
