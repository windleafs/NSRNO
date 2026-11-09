"""Independent neural fields with full initial network derivatives and fixed reference.

The initial physical control is exactly zero without a zero output head:
N(theta)-stop_gradient(N(theta_initial)). No patient anatomy enters the reference.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .independent_neural_material_fields import FieldGenerator as OriginalGenerator
from .independent_neural_material_fields import IndependentNeuralMaterialFields as OriginalFields


class ReferenceFieldGenerator(OriginalGenerator):
    def __init__(self, seed, shape):
        super().__init__(seed, shape)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+2000)
            for layer in [*self.layers, self.native]:
                nn.init.kaiming_normal_(layer.weight, a=.2, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(layer.bias)
            nn.init.normal_(self.output.weight, std=1/(8**.5))
            nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            reference = self.unanchored().detach().clone()
        self.register_buffer('initial_reference', reference)

    def features(self):
        value = self.noise
        for size, layer in zip(self.sizes, self.layers):
            value = F.leaky_relu(layer(F.interpolate(value, size=size, mode='nearest')),
                                 negative_slope=.2)
        value = F.leaky_relu(self.native(value), negative_slope=.2)
        # Independent per-field, per-feature spatial statistics; no learned or
        # observed anatomical template and no cross-material normalization.
        mean = value.mean((-2, -1), keepdim=True)
        variance = (value-mean).square().mean((-2, -1), keepdim=True)
        return (value-mean)/torch.sqrt(variance+1e-5)

    def unanchored(self):
        return self.output(self.features())[0, 0]

    def forward(self):
        return self.unanchored()-self.initial_reference


class ReferenceNeuralMaterialFields(OriginalFields):
    def __init__(self, coordinates, active_mask, seed=8039):
        super().__init__(coordinates, active_mask, seed)
        self.fields = nn.ModuleList([ReferenceFieldGenerator(seed+100*j, coordinates.m.shape[1:])
                                    for j in range(3)])

    def config(self):
        result = super().config()
        result.update(last_layer_initially_zero=False,
            fixed_initial_output_subtracted=True,
            fixed_reference_is_random_network_not_anatomy=True,
            all_hidden_layers_can_receive_first_inverse_gradient=True,
            per_field_feature_spatial_normalization=True,
            network_variance_initialization='Kaiming for leakyReLU, zero biases; nonzero output weights')
        return result
