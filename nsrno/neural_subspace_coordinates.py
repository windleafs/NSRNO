"""Flat neural coordinates and actual six-direction inverse update subspaces."""
import numpy as np
import torch
from torch.func import functional_call, jvp


class NeuralParameterChart:
    def __init__(self, model):
        self.model = model
        self.entries = []
        start = 0
        for name, parameter in model.named_parameters():
            self.entries.append((name, tuple(parameter.shape), slice(start, start+parameter.numel())))
            start += parameter.numel()
        self.size = start

    def vector(self):
        return torch.cat([p.detach().reshape(-1) for p in self.model.parameters()]).clone()

    def control(self, vector):
        if vector.shape != (self.size,) or vector.dtype != torch.float32:
            raise ValueError('Original CPU float32 neural weights required')
        parameters = {name: vector[section].reshape(shape)
                      for name, shape, section in self.entries}
        return functional_call(self.model, parameters, ())

    @torch.no_grad()
    def install(self, vector):
        for parameter, (_, shape, section) in zip(self.model.parameters(), self.entries):
            parameter.copy_(vector[section].reshape(shape))

    def actual_gradient_subspace(self, gradient):
        base = self.vector()
        variable = base.detach().requires_grad_(True)
        control = self.control(variable)
        g = torch.as_tensor(gradient, dtype=torch.float64)
        parameter_gradient, = torch.autograd.grad((control*g).sum(), variable)
        directions, labels = [], []
        for field in range(3):
            fine = torch.zeros_like(base)
            bias = None
            for name, _, section in self.entries:
                if name == f'fields.{field}.output.bias':
                    bias = section
                elif name.startswith(f'fields.{field}.'):
                    fine[section] = -parameter_gradient.detach()[section]
            norm = float(fine.double().norm())
            if norm:
                directions.append(fine/norm)
                labels.append(f'field{field}_nonconstant_parameter_gradient')
            if bias is None or bias.stop-bias.start != 1:
                raise RuntimeError('Independent scalar output bias missing')
            mean = torch.zeros_like(base)
            mean[bias] = 1.
            directions.append(mean)
            labels.append(f'field{field}_output_bias')
        images, slopes, errors = [], [], []
        for direction in directions:
            _, image = jvp(self.control, (base,), (direction,))
            image = image.detach().numpy().copy()
            slope = float(np.sum(image*np.asarray(gradient), dtype=np.float64))
            parameter_slope = float((parameter_gradient.detach().double()*direction.double()).sum())
            errors.append(abs(slope-parameter_slope)/max(abs(slope), abs(parameter_slope), 1e-20))
            images.append(image)
            slopes.append(slope)
        if not all(np.isfinite(v).all() for v in images) or max(errors, default=0.) > 1e-3:
            raise RuntimeError('Actual neural update JVP/VJP chains differ')
        return base, torch.stack(directions), np.stack(images), np.asarray(slopes), {
            'labels': labels, 'neural_parameter_count': self.size,
            'dimension': len(labels), 'same_gradient_chain_relative_errors': errors,
            'gradient_parameter_L2': float(parameter_gradient.detach().double().norm()),
            'not_a_full_parameter_space_Newton_solve': True}


def physical_TV_hessian(coordinates, control, vector, weight=.001):
    """Exact positive TV Hessian in independent physical scaled-log controls."""
    physical = (coordinates.absolute+control)[:, 80:]
    dx = physical[:, :-1, 1:]-physical[:, :-1, :-1]
    dz = physical[:, 1:, :-1]-physical[:, :-1, :-1]
    value = np.asarray(vector, np.float64)[:, 80:]
    vx = value[:, :-1, 1:]-value[:, :-1, :-1]
    vz = value[:, 1:, :-1]-value[:, :-1, :-1]
    radius_squared = dx*dx+dz*dz+1e-6
    scale = weight/np.sqrt(radius_squared)/radius_squared.size
    dot = (dx*vx+dz*vz)/radius_squared
    qx, qz = scale*(vx-dx*dot), scale*(vz-dz*dot)
    result = np.zeros_like(control, dtype=np.float64)
    body = result[:, 80:]
    body[:, :-1, :-1] -= qx+qz
    body[:, :-1, 1:] += qx
    body[:, 1:, :-1] += qz
    return result
