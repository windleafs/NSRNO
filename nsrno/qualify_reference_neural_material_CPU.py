"""One CPU-only neural shape/projection qualification; no patient or acoustic data."""
from pathlib import Path
import argparse
import json
import os
import resource
import time
import numpy as np
import torch
from .independent_physical_coordinates import IndependentPhysicalCoordinates
from .reference_neural_material_fields import ReferenceNeuralMaterialFields as IndependentNeuralMaterialFields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Explicit CPU-only qualification')
    args.out = args.out.resolve()
    if not args.out.is_relative_to(Path.cwd().resolve()/'out/v5') or args.out.exists():
        raise RuntimeError('New bounded qualification record only; no repetition')
    args.out.mkdir()
    started = time.time()
    torch.set_num_threads(8)
    toy = np.empty((3, 84, 8), np.float64)
    toy[0], toy[1], toy[2] = 1540., 1000., .5
    coordinates = IndependentPhysicalCoordinates(toy, np.zeros((84, 8)))
    mask = coordinates.mask.copy()
    model = IndependentNeuralMaterialFields(coordinates, mask)
    rng = np.random.default_rng(8039)
    candidate = rng.normal(0, 10, toy.shape)
    actual = model.project(torch.from_numpy(candidate)).detach().numpy()
    expected = coordinates.project(candidate, mask)
    projection_error = float(np.max(abs(actual-expected)))
    if projection_error>1e-12:
        raise RuntimeError('Torch projection disagrees with independently qualified numpy projection')
    # A single active logZ face has exact derivative I - n n^T / ||n||虏.
    point = np.zeros_like(toy)
    point[0, 81, 3], point[1, 81, 3] = 1., 2.5
    value = torch.tensor(point, requires_grad=True)
    projected = model.project(value)
    (2*projected[0, 81, 3]-3*projected[1, 81, 3]).backward()
    normal = np.array([.05, .10])
    exact = (np.eye(2)-np.outer(normal, normal)/(normal@normal))@np.array([2., -3.])
    face_chain_error = float(np.max(abs(value.grad[:2, 81, 3].numpy()-exact)))
    if face_chain_error>1e-12:
        raise RuntimeError('Constraint-face chain rule disagrees with its analytic affine map')
    del model, coordinates, projected, value
    material = np.empty((3, 961, 1121), np.float64)
    material[0], material[1], material[2] = 1540., 1000., .5
    coordinates = IndependentPhysicalCoordinates(material, np.zeros((961, 1121)))
    z, x = np.ogrid[:961, :1121]
    mask = (z>=80)&(z<937)&(x>=24)&(x<1097)
    model = IndependentNeuralMaterialFields(coordinates, mask)
    lateral = np.abs((np.arange(1121)-560)*50e-6)<=.0191+1e-12
    synthetic_gradient = np.stack([np.sin(.011*x+.3*j)*np.cos(.013*z+.7*j)
                                   for j in range(3)])*1e-6
    metric = model.freeze_depth_metric(synthetic_gradient, lateral)
    with torch.no_grad():
        initial = model()
        if initial.shape!=(3, 961, 1121) or float(initial.abs().max())!=0:
            raise RuntimeError('Initial full native control must be exactly zero')
        model.fields[0].output.bias.fill_(.02)
        changed = model()
        if float(changed[0].abs().max())==0 or float(changed[1:].abs().max())!=0:
            raise RuntimeError('Independent c head changed another material without an active constraint')
        model.fields[0].output.bias.zero_()
    before = time.time()
    model.zero_grad(set_to_none=True)
    control = model()
    (control*torch.from_numpy(synthetic_gradient)).sum().backward()
    gradient_norms = [float(torch.sqrt(sum((p.grad.double().square().sum()
        for p in field.parameters() if p.grad is not None), torch.tensor(0., dtype=torch.float64))))
        for field in model.fields]
    hidden_norms = [float(torch.sqrt(sum((p.grad.double().square().sum()
        for name,p in field.named_parameters() if not name.startswith('output.') and p.grad is not None),
        torch.tensor(0., dtype=torch.float64)))) for field in model.fields]
    if min(hidden_norms)<=1e-12:
        raise RuntimeError('The reference parameterization did not activate hidden layers at the first gradient')
    feature_fractions = []
    with torch.no_grad():
        for field in model.fields:
            h = field.features()[0, :, 80:921, 178:943].double()
            mean = h.mean((1,2))
            variance = (h-mean[:,None,None]).square().mean((1,2)).sum()
            feature_fractions.append(float(variance/(1+h.square().mean((1,2)).sum())))
    if min(feature_fractions)<.5:
        raise RuntimeError('The initial spatial feature kernel remains nearly constant')
    if not np.isfinite(gradient_norms).all() or min(gradient_norms)<=0:
        raise RuntimeError('A full native material generator does not carry a finite CPU gradient')
    seconds = time.time()-started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    if seconds>60 or peak>4:
        raise RuntimeError('CPU generator cost exceeds the fixed preflight resource allowance')
    result = {'status':'CPU_reference_neural_material_shape_projection_hidden_chain_passed',
        'seconds':seconds, 'one_native_forward_backward_seconds':time.time()-before,
        'peak_host_RSS_GiB':peak, 'max_projection_difference':projection_error,
        'max_active_constraint_analytic_chain_error':face_chain_error,
        'separate_c_head_leaves_rho_alpha_unchanged_in_constraint_interior':True,
        'native_gradient_norms':gradient_norms, 'hidden_layer_gradient_norms':hidden_norms,
        'initial_last_feature_spatial_kernel_fractions':feature_fractions, 'config':model.config(), 'metric':metric,
        'patient_material_or_RF_arrays_read':False, 'new_acoustic_solves':0,
        'new_patient_material_derivative_or_identifiability_probes':0,
        'GPU_work':False, 'patient_reconstruction_claimed':False}
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__=='__main__':
    main()
