"""Positive layer metric shared by independent material generators."""
import math
import torch


class SharedLayerRMS:
    def __init__(self, model, decay=.95):
        self.decay = float(decay)
        self.moments = [0.] * len(list(model.fields[0].parameters()))
        self.steps = 0

    def state_dict(self):
        return {'decay': self.decay, 'moments': list(self.moments), 'steps': self.steps}

    def load_state_dict(self, state):
        if state['decay'] != self.decay or len(state['moments']) != len(self.moments):
            raise ValueError('Different positive parameter metric')
        self.moments = list(state['moments'])
        self.steps = int(state['steps'])

    def direction(self, model):
        parameters = [list(field.parameters()) for field in model.fields]
        if any([tuple(p.shape) for p in field] !=
               [tuple(p.shape) for p in parameters[0]] for field in parameters):
            raise ValueError('Corresponding independent layers must have the same shapes')
        for layer in range(len(self.moments)):
            squared, count = 0., 0
            for field in parameters:
                p = field[layer]
                if p.grad is not None:
                    squared += float(p.grad.detach().double().square().sum())
                count += p.numel()
            self.moments[layer] = (self.decay*self.moments[layer] +
                                   (1-self.decay)*squared/count)
        self.steps += 1
        floor = 1e-6*max(sum(self.moments)/len(self.moments), 1e-30)
        scales = [1/math.sqrt(value+floor) for value in self.moments]
        directions = [[torch.zeros_like(p) if p.grad is None else
                       -p.grad.detach()*scales[layer]
                       for layer, p in enumerate(field)] for field in parameters]
        norm = math.sqrt(sum(float(d.double().square().sum())
                             for field in directions for d in field))
        if not math.isfinite(norm):
            raise RuntimeError('Nonfinite neural parameter direction')
        if norm:
            directions = [[d/norm for d in field] for field in directions]
        return directions, {'common_parameter_direction_norm_before_normalization': norm,
            'shared_layer_positive_scales': scales, 'common_layer_RMS_floor': floor,
            'separate_field_parameter_gradient_L2': [math.sqrt(sum(
                float(p.grad.detach().double().square().sum()) for p in field
                if p.grad is not None)) for field in parameters],
            'separate_field_direction_L2': [math.sqrt(sum(float(d.double().square().sum())
                for d in field)) for field in directions],
            'separate_field_normalization': False}


@torch.no_grad()
def restore_parameters(model, snapshot):
    for field, saved in zip(model.fields, snapshot):
        for parameter, value in zip(field.parameters(), saved):
            parameter.copy_(value)
