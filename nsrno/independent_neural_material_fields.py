"""Three CPU neural material generators; no shared field parameters or RF solver."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter1d


class FieldGenerator(nn.Module):
    def __init__(self, seed, shape):
        super().__init__()
        sizes = [(int(np.ceil((shape[0]-1)/factor))+1,
                  int(np.ceil((shape[1]-1)/factor))+1) for factor in [16, 8, 4, 2, 1]]
        self.sizes = sizes
        generator = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn((1, 8, 31, 36), generator=generator)*.1
        self.register_buffer('noise', noise)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1000)
            widths = [8, 32, 32, 24, 16, 8]
            self.layers = nn.ModuleList([nn.Conv2d(a, b, 3, padding=1)
                                        for a, b in zip(widths[:-1], widths[1:])])
            self.native = nn.Conv2d(8, 8, 3, padding=1)
            self.output = nn.Conv2d(8, 1, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self):
        value = self.noise
        for size, layer in zip(self.sizes, self.layers):
            value = F.leaky_relu(layer(F.interpolate(value, size=size, mode='nearest')),
                                 negative_slope=.2)
        value = F.leaky_relu(self.native(value), negative_slope=.2)
        return self.output(value)[0, 0]


class IndependentNeuralMaterialFields(nn.Module):
    def __init__(self, coordinates, active_mask, seed=8039):
        super().__init__()
        shape = coordinates.m.shape[1:]
        self.fields = nn.ModuleList([FieldGenerator(seed+100*j, shape) for j in range(3)])
        self.register_buffer('absolute', torch.from_numpy(coordinates.absolute.copy()))
        self.register_buffer('active', torch.from_numpy(
            (np.asarray(active_mask, bool)&coordinates.mask).copy()))
        self.register_buffer('depth_scale', torch.ones((3, shape[0], 1), dtype=torch.float64))
        self.c_lower, self.c_upper = coordinates.c_lower, coordinates.c_upper
        self.z_lower, self.z_upper = coordinates.logZ_lower, coordinates.logZ_upper
        self.a_lower, self.a_upper = coordinates.alpha_lower, coordinates.alpha_upper
        self.depth_frozen = False

    def raw_control(self):
        return torch.stack([field() for field in self.fields]).double()*self.depth_scale

    def project(self, control):
        """Differentiable piecewise-affine projection onto the original convex domain."""
        absolute = self.absolute+control
        c, rho = absolute[:2]
        logZ = .05*c+.10*rho
        valid = ((c>=self.c_lower)&(c<=self.c_upper)&
                 (logZ>=self.z_lower)&(logZ<=self.z_upper))
        best_c, best_rho = c, rho
        distance = torch.where(valid, 0., torch.inf)
        def consider(new_c, new_rho):
            nonlocal best_c, best_rho, distance
            cost = (new_c-c).square()+(new_rho-rho).square()
            choose = cost<distance
            best_c, best_rho = torch.where(choose, new_c, best_c), torch.where(choose, new_rho, best_rho)
            distance = torch.minimum(distance, cost)
        for boundary in [self.c_lower, self.c_upper]:
            projected_c = torch.full_like(c, boundary)
            projected_rho = rho.clamp((self.z_lower-.05*boundary)/.10,
                                      (self.z_upper-.05*boundary)/.10)
            consider(projected_c, projected_rho)
        for boundary in [self.z_lower, self.z_upper]:
            projected_c = (c-.05*(logZ-boundary)/(.05**2+.10**2)).clamp(
                self.c_lower, self.c_upper)
            consider(projected_c, (boundary-.05*projected_c)/.10)
        result = torch.stack([best_c, best_rho, absolute[2].clamp(self.a_lower, self.a_upper)])
        return (result-self.absolute)*self.active[None]

    def forward(self):
        return self.project(self.raw_control())

    def freeze_depth_metric(self, gradient, lateral_roi, sigma_rows=40., floor=.02):
        if self.depth_frozen:
            raise RuntimeError('The positive neural depth metric is fixed from the first actual gradient')
        gradient = np.asarray(gradient, np.float64)
        mask = self.active.numpy()&np.asarray(lateral_roi, bool)[None]
        count = mask.sum(-1).astype(np.float64)
        squared = np.sum(gradient**2*mask[None], axis=-1)
        row = np.sqrt(gaussian_filter1d(squared, sigma_rows, axis=-1, mode='nearest')/
            np.maximum(gaussian_filter1d(count, sigma_rows, mode='nearest')[None], 1e-30))
        global_rms = np.sqrt(squared.sum(-1)/count.sum())
        inverse = 1/np.maximum(row, np.maximum(floor*global_rms[:, None], 1e-24))
        active_rows = count>0
        median = np.median(inverse[:, active_rows], axis=1)
        diagonal = np.clip(inverse/median[:, None], .1, 10.)
        self.depth_scale.copy_(torch.from_numpy(np.sqrt(diagonal)[:, :, None]))
        self.depth_frozen = True
        return {'metric':'symmetric sqrt(D) J_N J_N^T sqrt(D)',
            'D_min':diagonal[:, active_rows].min(-1).tolist(),
            'D_max':diagonal[:, active_rows].max(-1).tolist(),
            'metric_sigma_rows':sigma_rows, 'RMS_relative_floor':floor,
            'material_fields_smoothed':False, 'new_acoustic_probes':0}

    def parameter_snapshot(self):
        return [[p.detach().clone() for p in field.parameters()] for field in self.fields]

    def unit_negative_parameter_gradient(self):
        directions = []
        for field in self.fields:
            parameters = list(field.parameters())
            norm = torch.sqrt(sum((p.grad.double().square().sum() for p in parameters
                                   if p.grad is not None), torch.tensor(0., dtype=torch.float64)))
            directions.append([torch.zeros_like(p) if p.grad is None or float(norm)==0 else
                               -p.grad.detach()/float(norm) for p in parameters])
        return directions

    @torch.no_grad()
    def set_parameter_step(self, snapshot, direction, field_steps, fraction=1.):
        for j, field in enumerate(self.fields):
            for parameter, old, delta in zip(field.parameters(), snapshot[j], direction[j]):
                parameter.copy_(old+float(field_steps[j])*fraction*delta)

    def config(self):
        return {'coordinate_fields':['log c', 'log rho', 'log positive tissue alpha'],
            'shared_field_weights':False, 'shared_field_noise':False, 'device':'CPU',
            'native_shape':list(self.absolute.shape), 'hard_material_lowpass':False,
            'network_parameter_counts':[sum(p.numel() for p in f.parameters()) for f in self.fields],
            'projection':'exact piecewise-affine original c/Z/alpha domain; autograd retained',
            'last_layer_initially_zero':True, 'supervised_material_truth_used':False,
            'not_a_trained_neural_RF_forward_surrogate':True}
