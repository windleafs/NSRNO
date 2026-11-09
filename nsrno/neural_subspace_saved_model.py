"""Identity-checked reuse of actual inverse derivatives and neural directions."""
from pathlib import Path
import json
import hashlib
import io
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matches_saved_state(folder, material, parameters):
    folder = Path(folder)
    return ((folder/'base_neural_parameters.npy').exists() and
        np.array_equal(np.asarray(material), np.load(folder/'start_material.npy')) and
        np.array_equal(parameters.detach().numpy(), np.load(folder/'base_neural_parameters.npy')))


def load_saved_step(source, step, chart, coordinates, require_matrix=False):
    source = Path(source)
    run = json.loads((source/'run.json').read_text())
    module_root = Path(__file__).parent
    for name, expected in run['source_SHA256'].items():
        if digest(module_root/name) != expected:
            raise RuntimeError('Saved inverse operator changed: '+name)
    initial_checkpoint = Path(run['parent_root'])/'checkpoint.pt'
    if digest(initial_checkpoint) != run['resumed_checkpoint_SHA256']:
        raise RuntimeError('Original neural reference checkpoint changed')
    reference = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)['neural_state']
    parameter_names = {name for name, _ in chart.model.named_parameters()}
    for name, value in chart.model.state_dict().items():
        if name in reference and name not in parameter_names and not torch.equal(value, reference[name]):
            raise RuntimeError('Saved derivative noise/reference/depth buffers differ')
    folder = source/'candidates'/f'step{step:02d}'
    base = torch.from_numpy(np.load(folder/'base_neural_parameters.npy')).clone()
    control = np.load(folder/'start_control.npy')
    directions = torch.from_numpy(np.load(folder/'neural_parameter_basis.npy')).clone()
    images = np.load(folder/'control_basis.npy')
    gradient = np.load(folder/'actual_objective_gradient.npy')
    if (base.shape != (chart.size,) or directions.shape != (len(images), chart.size) or
            images.shape[1:] != control.shape or gradient.shape != control.shape):
        raise RuntimeError('Saved actual inverse direction shapes differ')
    chart.install(base)
    with torch.no_grad():
        restored = chart.model().numpy()
    if float(np.max(abs(restored-control))) > 2e-7:
        raise RuntimeError('Saved neural derivative base control changed')
    buffer = io.BytesIO()
    np.save(buffer, coordinates.material(restored))
    if hashlib.sha256(buffer.getvalue()).hexdigest() != digest(folder/'start_material.npy'):
        raise RuntimeError('Saved inverse physical material identity changed')
    jobs = list((source/'jobs').glob(f'*_step{step:02d}_actual_gradient.json'))
    if len(jobs) == 1:
        receipt = jobs[0]
    elif len(jobs) == 0 and (folder/'source_actual_gradient_receipt.json').exists():
        receipt = folder/'source_actual_gradient_receipt.json'
    else:
        raise RuntimeError('Missing or duplicate original actual gradient receipt')
    job = json.loads(receipt.read_text())
    measured = {'loss': job['loss'], 'token': job.get('token'), 'reused_from': str(receipt)}
    meta = source/'steps'/f'step{step:02d}.json'
    saved = json.loads(meta.read_text()) if meta.exists() else None
    if require_matrix and saved is None:
        raise RuntimeError('Complete saved native subspace matrix missing')
    basis = saved['basis'] if saved else {'dimension': len(images),
        'labels': [f'field{j}_{kind}' for j in range(3)
                   for kind in ['nonconstant_parameter_gradient', 'output_bias']],
        'neural_parameter_count': chart.size, 'reused_saved_gradient_only_basis': True,
        'not_a_full_parameter_space_Newton_solve': True}
    product_file = folder/'subspace_products.json'
    products = json.loads(product_file.read_text()).get('product_summaries', []) if product_file.exists() else []
    if require_matrix and len(products) != len(images):
        raise RuntimeError('Saved native inverse products are incomplete')
    return {'control': control.copy(), 'base': base, 'directions': directions,
        'images': images, 'gradient': gradient, 'slopes': np.einsum('kizx,izx->k', images, gradient, optimize=True),
        'codes': job['encoding'], 'measured': measured, 'basis': basis,
        'matrix': np.asarray(saved['coupled_matrix']) if require_matrix else None,
        'RF_matrix': np.asarray(saved['RF_matrix']) if require_matrix else None,
        'TV_matrix': np.asarray(saved['TV_matrix']) if require_matrix else None,
        'products': products, 'source_root': str(source), 'source_step': step,
        'source_material_SHA256': digest(folder/'start_material.npy'),
        'source_file_SHA256': {name: digest(folder/name) for name in [
            'base_neural_parameters.npy', 'control_basis.npy', 'actual_objective_gradient.npy',
            'neural_parameter_basis.npy', 'start_control.npy', 'start_material.npy']}}
