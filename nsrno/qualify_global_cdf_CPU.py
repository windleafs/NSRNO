"""One synthetic full-chain derivative check of the new independent CDF energy."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.special import ndtr
from .global_cdf_prior import GlobalCDFPrior,make_grid,kernel
from .independent_physical_coordinates import IndependentPhysicalCoordinates


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise RuntimeError('Do not repeat a passed CDF-energy derivative check')
    started=time.time()
    state={'kernel':kernel()}
    for field in range(3):
        grid=make_grid(field)
        cdf=.6*ndtr((grid+.65)/.12)+.4*ndtr((grid-.45)/.15)
        state.update({f'grid_{field}':grid,f'lower_{field}':np.maximum(0.,cdf-.03),
                      f'upper_{field}':np.minimum(1.,cdf+.03)})
    prior=GlobalCDFPrior(state=state)
    rng=np.random.default_rng(61109031)
    shape=(91,19)
    medium=np.stack([1540.*np.ones(shape),1000.*np.ones(shape),.5*np.ones(shape)])
    coordinates=IndependentPhysicalCoordinates(medium,np.zeros(shape))
    control=rng.normal(0,.07,size=medium.shape)
    control[:,:80]=0.
    region=(slice(80,None),slice(None))
    value,gradient,_=prior.evaluate(coordinates,control,region=region)
    checks=[]
    for field in range(3):
        direction=np.zeros_like(control)
        direction[field,80:]=rng.normal(size=direction[field,80:].shape)
        eps=1e-6
        plus=prior.evaluate(coordinates,control+eps*direction,gradient=False,region=region)[0]
        minus=prior.evaluate(coordinates,control-eps*direction,gradient=False,region=region)[0]
        finite=(plus-minus)/(2*eps)
        analytic=float(np.sum(gradient*direction))
        error=abs(analytic-finite)/max(abs(analytic),abs(finite),1e-12)
        if error>2e-5:
            raise RuntimeError('New CDF control gradient does not match its energy')
        checks.append({'field':field,'analytic':analytic,'finite_difference':finite,'relative_error':error})
    assert np.isfinite(gradient).all() and value>0
    result={'status':'independent_CDF_envelope_full_physical_chain_synthetic_CPU_passed',
        'source_SHA256':hashlib.sha256(Path(__file__).with_name('global_cdf_prior.py').read_bytes()).hexdigest(),
        'seconds':time.time()-started,'checks':checks,'maximum_relative_error':max(r['relative_error'] for r in checks),
        'new_wave_solves':0,'new_material_derivative_probes':0,'patient_arrays_read':False,'CUDA_used':False}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
