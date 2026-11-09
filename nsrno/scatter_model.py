"""Fine-scale material features with learned local or nonlocal receive mixing."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .forward_model import C_REF, RHO_REF, ForwardOperator
from .rf_representation import IQ_SAMPLES, RF_SAMPLES, RF_FS, decode


class MaterialMix(nn.Module):
    """Low-rank channel mixing with either full spectral or local spatial support."""
    def __init__(self, width, rank, shape, aggregation):
        super().__init__()
        self.aggregation = aggregation
        self.shape = tuple(shape)
        self.padded = tuple(32 * math.ceil((n + 16) / 32) for n in shape)
        self.project_in = nn.Conv2d(width, rank, 1)
        self.project_out = nn.Conv2d(rank, width, 1)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.fuse = nn.Sequential(nn.SiLU(), nn.Conv2d(width, width, 1))
        if aggregation == "global":
            h, w = self.padded
            z = torch.zeros(rank, h, 2)
            x = torch.zeros(rank, w // 2 + 1, 2)
            z[..., 0] = 1.
            x[..., 0] = 1.
            self.filter_z = nn.Parameter(z)
            self.filter_x = nn.Parameter(x)
        elif aggregation == "local":
            kernel = torch.zeros(rank, 1, 5, 5)
            kernel[:, :, 2, 2] = 1.
            self.kernel = nn.Parameter(kernel)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def forward(self, feature):
        if tuple(feature.shape[-2:]) != self.shape:
            raise ValueError("The spectral grid is fixed by the acquisition")
        reduced = self.project_in(feature)
        if self.aggregation == "global":
            spectrum = torch.fft.rfft2(reduced.float(), s=self.padded, norm="ortho")
            wz = torch.view_as_complex(self.filter_z.contiguous())
            wx = torch.view_as_complex(self.filter_x.contiguous())
            # Retain every spatial Fourier mode, including fine-scale modes.
            multiplier = wz[:, :, None] * wx[:, None, :]
            mixed = torch.fft.irfft2(spectrum * multiplier[None], s=self.padded, norm="ortho")
            mixed = mixed[..., :self.shape[0], :self.shape[1]].to(reduced.dtype)
        else:
            mixed = F.conv2d(reduced, self.kernel, padding=2, groups=reduced.shape[1])
        return feature + self.fuse(self.project_out(mixed) + self.pointwise(feature))


class FineScatterForwardOperator(nn.Module):
    """Frozen previous forward predictor plus a trainable fine-scattering branch."""
    def __init__(self, spec, template=None, scale=1.):
        super().__init__()
        if spec.get("residual_frame") != "analytic":
            raise ValueError("The scattering branch uses the established analytic frame")
        self.spec = spec
        self.base = ForwardOperator(spec, template, scale)
        self.base.requires_grad_(False)
        h, w = spec["medium_shape"][-2:]
        self.native_shape = (h, w)
        packed_shape = ((h + 3) // 4, (w + 3) // 4)
        width, rank = 48, 12
        self.stem = nn.Conv2d(37, width, 1)
        self.angle_condition = nn.Sequential(nn.Linear(3, 48), nn.SiLU(), nn.Linear(48, 2 * width))
        nn.init.zeros_(self.angle_condition[-1].weight)
        nn.init.zeros_(self.angle_condition[-1].bias)
        mode = spec["scatter_aggregation"]
        self.material_mix = nn.ModuleList([
            MaterialMix(width, rank, packed_shape, mode),
            MaterialMix(width, rank, packed_shape, mode),
        ])
        if spec.get("scatter_readout", "coarse") == "native":
            self.native_readout = nn.Conv2d(width // 16, width, 1)
        self.receive_mix = MaterialMix(width, rank, (IQ_SAMPLES, spec["receivers"]), mode)
        self.head = nn.Conv2d(width, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def template(self):
        return self.base.template

    @property
    def rf_scale(self):
        return self.base.rf_scale

    def encode_detail(self, medium):
        h, w = self.native_shape
        if medium.ndim != 4 or tuple(medium.shape[1:]) != (2, h, w):
            raise ValueError("Expected native c/rho maps in physical units")
        speed, density = medium.float().unbind(1)
        log_fields = torch.stack(((speed / C_REF).log(), (density / RHO_REF).log()), 1)
        smooth = F.avg_pool2d(F.pad(log_fields, (4, 4, 0, 0), mode="replicate"), (1, 9), stride=1)
        smooth = F.avg_pool2d(F.pad(smooth, (0, 0, 4, 4), mode="replicate"), (9, 1), stride=1)
        # Fixed scales retain relative scattering amplitudes across records.
        detail = (log_fields - smooth) / log_fields.new_tensor([.005, .008])[None, :, None, None]
        smooth = smooth / smooth.new_tensor([.05, .1])[None, :, None, None]
        pad = (0, -w % 4, 0, -h % 4)
        packed = F.pixel_unshuffle(F.pad(detail, pad, mode="replicate"), 4)
        background = F.avg_pool2d(F.pad(smooth, pad, mode="replicate"), 4)
        delay = torch.cumsum((speed.reciprocal() - 1 / C_REF) * self.base.post_probe, dim=1)
        delay = F.avg_pool2d(F.pad(delay[:, None] * (2e6 * self.base.dx), pad, mode="replicate"), 4)
        coordinates = self.base.coords.expand(medium.shape[0], -1, -1, -1)
        return self.stem(torch.cat((packed, background, delay, coordinates), 1))

    def slowness_depth(self, medium, indices, time, x):
        """Invert the differentiable column travel-time coordinate.

        This supplies a material-dependent query coordinate, not a full travel-time
        solver. Oblique refraction and multiple paths remain for the network.
        """
        speed = medium[:, 0].float()
        h, w = self.native_shape
        inverse = speed.reciprocal()
        increments = .5 * (inverse[:, 1:] + inverse[:, :-1]) * (self.base.dx * 1e6)
        cumulative = F.pad(torch.cumsum(increments, 1), (0, 0, 1, 0))
        probe_index = round(-self.spec["z0_m"] / self.base.dx)
        cumulative = cumulative - cumulative[:, probe_index:probe_index+1]
        column = (x - self.spec["x0_m"]) / self.base.dx
        left = column.floor().long().clamp(0, w - 2)
        fraction = (column - left).clamp(0, 1)
        travel = cumulative[:, :, left] * (1-fraction) + cumulative[:, :, left+1] * fraction
        travel = travel.permute(0, 2, 1).repeat_interleave(indices.shape[1], 0).contiguous()
        index = indices.flatten()
        target = (time[None, :, None] - self.base.tref[index, None, None]
                  - x[None, None, :] * self.base.sin_theta[index, None, None] / C_REF)
        target = target / (1 + self.base.cos_theta[index, None, None])
        target = (target.transpose(-1, -2) * 1e6).contiguous()
        upper = torch.searchsorted(travel, target).clamp(1, h-1)
        lower_time = travel.gather(-1, upper-1)
        upper_time = travel.gather(-1, upper)
        blend = (target - lower_time) / (upper_time - lower_time).clamp_min(1e-8)
        depth = self.spec["z0_m"] + (upper - 1 + blend) * self.base.dx
        return depth.transpose(-1, -2)

    def sample_native(self, features, indices, medium):
        h, w = self.native_shape
        native = F.pixel_shuffle(features, 4)[..., :h, :w]
        receivers = self.spec["receivers"]
        aperture = (receivers - 1) * self.spec["pitch_m"]
        x = torch.linspace(-aperture / 2, aperture / 2, receivers, device=features.device)
        time = torch.arange(IQ_SAMPLES, device=features.device) * (RF_SAMPLES / (RF_FS * IQ_SAMPLES))
        index = indices.flatten()
        sin_theta = self.base.sin_theta[index, None, None]
        cos_theta = self.base.cos_theta[index, None, None]
        tref = self.base.tref[index, None, None]
        z = (C_REF * (time[None, :, None] - tref) - x[None, None, :] * sin_theta) / (1 + cos_theta)
        if self.spec.get("scatter_query", "nominal") == "slowness":
            z = self.slowness_depth(medium, indices, time, x)
        gz = 2 * (z - self.spec["z0_m"]) / (self.base.dx * (h - 1)) - 1
        gx = 2 * (x - self.spec["x0_m"]) / (self.base.dx * (w - 1)) - 1
        grid = torch.stack((gx.expand(len(index), IQ_SAMPLES, receivers), gz), -1)
        sampled = F.grid_sample(native.float(), grid, padding_mode="zeros", align_corners=True)
        return self.native_readout(sampled.to(features.dtype))

    def predict_iq(self, medium, angle_indices=None):
        if angle_indices is None:
            indices = torch.arange(self.template.shape[0], device=medium.device)
        else:
            indices = torch.as_tensor(angle_indices, device=medium.device, dtype=torch.long)
        if indices.ndim == 1:
            indices = indices[None].expand(medium.shape[0], -1)
        if indices.ndim != 2 or indices.shape[0] != medium.shape[0]:
            raise ValueError("Use one shared angle list or one list per medium")
        baseline = self.base.predict_iq(medium, indices)
        features = self.encode_detail(medium).repeat_interleave(indices.shape[1], 0)
        condition = torch.stack((self.base.sin_theta[indices], self.base.cos_theta[indices],
                                 self.base.tref[indices] * 1e6), -1).flatten(0, 1)
        gain, bias = self.angle_condition(condition).chunk(2, 1)
        features = features * (1 + gain[..., None, None]) + bias[..., None, None]
        for block in self.material_mix:
            features = block(features)
        # Restore subpixel locations before querying: packed values are not spatial averages.
        if self.spec.get("scatter_readout", "coarse") == "native":
            features = self.sample_native(features, indices, medium)
        else:
            features = self.base.sample(features, 4, (IQ_SAMPLES, self.spec["receivers"]),
                                        indices.flatten()[:, None])
        features = self.receive_mix(features)
        analytic = self.head(features).transpose(-1, -2).float()
        analytic = analytic.reshape(*indices.shape, 2, self.spec["receivers"], IQ_SAMPLES)
        real, imag = analytic.unbind(2)
        cosine, sine = self.base.carrier_cos, self.base.carrier_sin
        correction = torch.stack((real * cosine + imag * sine, imag * cosine - real * sine), 2)
        return baseline + self.rf_scale * correction

    def forward(self, medium, angle_indices=None):
        return decode(self.predict_iq(medium, angle_indices))
