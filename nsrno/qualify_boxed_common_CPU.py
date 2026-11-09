"""One CPU synthetic qualification of the new per-pixel trust-domain projection."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.optimize import minimize
from .boxed_common_descent import project_box, common_box_direction, CAPS
from .independent_physical_coordinates import IndependentPhysicalCoordinates, SCALES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('Do not repeat a passed projection qualification')
    started = time.time()
    rng = np.random.default_rng(6110913)
    shape = (83, 9)
    material = np.stack([1540.*np.ones(shape),1000.*np.ones(shape),.5*np.ones(shape)])
    coordinates = IndependentPhysicalCoordinates(material, np.zeros(shape))
    active = coordinates.mask
    center = np.zeros_like(material)
    # Include active physical boundaries while keeping the trust domain feasible.
    center[0, 80, 0] = coordinates.c_lower
    center[1, 80, 0] = (coordinates.logZ_upper-.05*coordinates.c_lower)/.1
    center[2, 80, 0] = coordinates.alpha_upper
    target = center+rng.normal(size=material.shape)*3
    metric = np.exp(rng.normal(size=material.shape))
    actual = project_box(coordinates, target, metric, active, center)
    comparisons = []
    for z, x in np.argwhere(active)[::2]:
        t, b, c = target[:2,z,x],metric[:2,z,x],center[:2,z,x]
        lo = np.array([max(coordinates.c_lower,c[0]-CAPS[0,0,0]),c[1]-CAPS[1,0,0]])
        hi = np.array([min(coordinates.c_upper,c[0]+CAPS[0,0,0]),c[1]+CAPS[1,0,0]])
        n = np.array([.05,.1])
        result = minimize(lambda v:.5*np.sum((v-t)**2/b),c,jac=lambda v:(v-t)/b,
            bounds=list(zip(lo,hi)),constraints=[
                {'type':'ineq','fun':lambda v:n@v-coordinates.logZ_lower,'jac':lambda v:n},
                {'type':'ineq','fun':lambda v:coordinates.logZ_upper-n@v,'jac':lambda v:-n}],
            method='SLSQP',options={'ftol':1e-12,'maxiter':100})
        if not result.success:
            raise RuntimeError('Independent reference QP failed; inspect without patient work')
        expected = result.x
        gap = float(np.max(abs(actual[:2,z,x]-expected)))
        if gap > 2e-5:
            raise RuntimeError('Six-boundary projection differs from independent convex QP')
        comparisons.append(gap)
    absfields = coordinates.absolute+actual
    Z = .05*absfields[0]+.1*absfields[1]
    assert np.all(abs(actual-center)[:,active] <= np.broadcast_to(CAPS,actual.shape)[:,active]+1e-10)
    assert np.min(absfields[0,active]) >= coordinates.c_lower-1e-10
    assert np.max(absfields[0,active]) <= coordinates.c_upper+1e-10
    assert np.min(Z[active]) >= coordinates.logZ_lower-1e-10
    assert np.max(Z[active]) <= coordinates.logZ_upper+1e-10
    cases = []
    for kind in ['equal','mixed','opposite']:
        g = rng.normal(size=material.shape)*active
        h = g if kind == 'equal' else (-g if kind == 'opposite' else .3*g+rng.normal(size=material.shape)*active)
        direction, details = common_box_direction(coordinates,center,g,h,active,np.ones(shape[1],bool))
        if kind == 'opposite':
            assert not details['actual_common_step_available']
        else:
            assert details['actual_common_step_available'] and details['RF_slope'] < 0 and details['prior_slope'] < 0
        cases.append({'case':kind,**details})
    result = {'status':'physical_step_box_projection_and_common_descent_synthetic_CPU_passed',
        'source_SHA256':hashlib.sha256(Path(__file__).with_name('boxed_common_descent.py').read_bytes()).hexdigest(),
        'seconds':time.time()-started,'maximum_control_difference_from_independent_QP':max(comparisons),
        'cases':cases,'patient_arrays_read':False,'CUDA_used':False,
        'new_acoustic_solves':0,'new_material_derivative_probes':0}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
