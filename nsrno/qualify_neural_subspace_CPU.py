"""One necessary synthetic CPU check for the new flat neural JVP and TV Hessian."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import time
import numpy as np
import torch
from torch.func import jvp, vjp
from .independent_physical_coordinates import IndependentPhysicalCoordinates
from .reference_neural_material_fields import ReferenceNeuralMaterialFields
from .neural_subspace_coordinates import NeuralParameterChart, physical_TV_hessian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('One CPU-only qualification; never reset an existing receipt')
    torch.set_num_threads(8)
    started = time.time()
    material = np.empty((3, 961, 1121), np.float64)
    material[:] = np.array([1540., 1000., .5])[:, None, None]
    coordinates = IndependentPhysicalCoordinates(material, np.zeros((961, 1121)))
    z, x = np.ogrid[:961, :1121]
    mask = (z >= 80) & (z < 937) & (x >= 24) & (x < 1097)
    model = ReferenceNeuralMaterialFields(coordinates, mask, seed=8039)
    model.depth_scale.copy_(torch.from_numpy(np.broadcast_to(
        1.+.2*np.sin(np.arange(961)[None, :, None]/137.), (3, 961, 1)).copy()))
    model.depth_frozen = True
    chart = NeuralParameterChart(model)
    base = chart.vector()
    generator = torch.Generator().manual_seed(88137)
    direction = torch.randn(base.shape, generator=generator)
    direction /= direction.norm()
    gradient = torch.randn((3, 961, 1121), dtype=torch.float64, generator=generator)/1000
    _, image = jvp(chart.control, (base,), (direction,))
    _, pull = vjp(chart.control, base)
    parameter_gradient, = pull(gradient)
    left = float((image.detach()*gradient).sum())
    right = float((direction.double()*parameter_gradient.detach().double()).sum())
    error = abs(left-right)/max(abs(left), abs(right), 1e-20)
    del image, pull, parameter_gradient, gradient
    rng = np.random.default_rng(9913)
    control = rng.normal(0., .01, material.shape)
    vector = rng.normal(0., .01, material.shape)
    hessian_vector = physical_TV_hessian(coordinates, control, vector)
    epsilon = 1e-4
    plus = coordinates.regularization(control+epsilon*vector, .001)[1]
    minus = coordinates.regularization(control-epsilon*vector, .001)[1]
    independent_difference = (plus-minus)/(2*epsilon)
    hessian_error = float(np.linalg.norm(hessian_vector-independent_difference)/
                          np.linalg.norm(independent_difference))
    quadratic = float(np.sum(vector*hessian_vector))
    if not np.isfinite([error, hessian_error, quadratic]).all() or error > 2e-5 or hessian_error > 2e-5 or quadratic < 0:
        raise RuntimeError(f'New CPU chain failed: {error}, {hessian_error}, {quadratic}')
    result = {'status': 'CPU_flat_neural_JVP_VJP_and_independent_TV_Hessian_passed',
        'seconds': time.time()-started, 'neural_parameter_count': chart.size,
        'neural_JVP_VJP_relative_error': error, 'TV_Hessian_central_difference_relative_error': hessian_error,
        'TV_Hessian_positive_quadratic': quadratic,
        'peak_host_RSS_GiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        'synthetic_fields_only': True, 'patient_arrays_read': False,
        'new_acoustic_solves': 0, 'new_patient_material_derivative_probes': 0,
        'source_SHA256': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['neural_subspace_coordinates.py', Path(__file__).name]}}
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
