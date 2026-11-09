"""Independent, differentiable V5 material-to-channel-RF neural operator."""
import torch
from torch import nn
from torch.nn import functional as F

from .rf_representation import CENTRE_BIN, IQ_SAMPLES, RF_FS, RF_SAMPLES, decode, time_windows

C_REF = 1540.
RHO_REF = 1000.


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class DecoderStage(nn.Module):
    def __init__(self, skip_channels, previous_channels, width):
        super().__init__()
        self.fuse = nn.Conv2d(skip_channels + previous_channels, width, 3, padding=1)
        self.condition = nn.Linear(64, 2 * width)
        self.block = ResidualBlock(width)

    def forward(self, feature, angle, previous=None):
        if previous is not None:
            previous = F.interpolate(previous, feature.shape[-2:], mode="bilinear", align_corners=True)
            feature = torch.cat((feature, previous), dim=1)
        feature = self.fuse(feature)
        gain, bias = self.condition(angle).chunk(2, dim=1)
        feature = feature * (1 + gain[..., None, None]) + bias[..., None, None]
        return self.block(feature)


class ForwardOperator(nn.Module):
    """Predict signed RF [batch, angle, receiver, time] from native SI-unit c/rho."""
    def __init__(self, spec, template=None, scale=1.):
        super().__init__()
        self.spec = spec
        self.height, self.width = spec["medium_shape"][-2:]
        self.dx = float(spec["dx_m"])
        self.x_start = float(spec["x0_m"]) + 1.5 * self.dx
        self.z_start = float(spec["z0_m"]) + 1.5 * self.dx
        receivers = int(spec["receivers"])
        if receivers % 8 or IQ_SAMPLES % 8:
            raise ValueError("Decoder shape must be divisible by eight")
        self.receivers = receivers
        shape = (len(spec["angles_deg"]), 2, receivers, IQ_SAMPLES)
        self.register_buffer("template", torch.zeros(shape) if template is None else template.float())
        # A fixed positive change of output coordinates; targets retain physical units.
        # Old checkpoints use a scalar rf_scale and do not have this spec entry.
        if "output_window_scales" in spec:
            windows = time_windows(IQ_SAMPLES, 1e6 * RF_SAMPLES / RF_FS)
            scales = spec["output_window_scales"]
            if len(scales) != len(windows) or any(not 0 < value < float("inf") for value in scales):
                raise ValueError("Output scales must be finite and positive for every time window")
            scale = torch.ones(IQ_SAMPLES, dtype=torch.float32)
            for window, value in zip(windows, scales):
                scale[window] = value
        self.register_buffer("rf_scale", torch.as_tensor(scale, dtype=torch.float32))
        if spec.get("residual_frame", "baseband") == "analytic":
            # Exact inverse demodulation of the representation's integer carrier bin.
            # This is a unitary output-coordinate change, not a propagation model.
            phase = 2 * torch.pi * CENTRE_BIN * torch.arange(IQ_SAMPLES, dtype=torch.float32) / IQ_SAMPLES
            self.register_buffer("carrier_cos", phase.cos())
            self.register_buffer("carrier_sin", phase.sin())

        theta = torch.deg2rad(torch.tensor(spec["angles_deg"], dtype=torch.float32))
        tref = torch.tensor(spec["source_tref_s"], dtype=torch.float32)
        self.register_buffer("sin_theta", theta.sin())
        self.register_buffer("cos_theta", theta.cos())
        self.register_buffer("tref", tref)
        native_z = torch.arange(self.height, dtype=torch.float32) * self.dx + spec["z0_m"]
        self.register_buffer("post_probe", (native_z >= -1e-8)[None, :, None])
        x = torch.linspace(-1., 1., (self.width + 3) // 4)
        z = torch.linspace(-1., 1., (self.height + 3) // 4)
        zz, xx = torch.meshgrid(z, x, indexing="ij")
        self.register_buffer("coords", torch.stack((zz, xx))[None])
        self.angle_encoder = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64))
        widths = (24, 40, 64, 96)
        self.stem = nn.Sequential(nn.Conv2d(35, widths[0], 3, padding=1), ResidualBlock(widths[0]))
        self.encoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a, b, 3, stride=2, padding=1), ResidualBlock(b))
            for a, b in zip(widths[:-1], widths[1:])
        ])
        decoded = (64, 48, 32, 24)
        self.decoder = nn.ModuleList([
            DecoderStage(skip, previous, width)
            for skip, previous, width in zip(reversed(widths), (0, *decoded[:-1]), decoded)
        ])
        self.head = nn.Conv2d(decoded[-1], 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def encode_medium(self, medium):
        if medium.ndim != 4 or medium.shape[1:] != (2, self.height, self.width):
            raise ValueError(f"Expected native medium [B,2,{self.height},{self.width}]")
        speed, density = medium.float().unbind(1)
        normalized = torch.stack(((speed / C_REF).log() / .05,
                                  (density / RHO_REF).log() / .1), dim=1)
        pad = (0, -self.width % 4, 0, -self.height % 4)
        folded = F.pixel_unshuffle(F.pad(normalized, pad, mode="replicate"), 4)
        axial_delay_us = torch.cumsum((speed.reciprocal() - 1 / C_REF) * self.post_probe, dim=1)
        axial_delay_us = axial_delay_us[:, None] * (2e6 * self.dx)
        axial_delay_us = F.avg_pool2d(F.pad(axial_delay_us, pad, mode="replicate"), 4)
        stem = self.stem(torch.cat((folded, axial_delay_us, self.coords.expand(speed.shape[0], -1, -1, -1)), dim=1))
        levels = [stem]
        for stage in self.encoder:
            levels.append(stage(levels[-1]))
        return levels

    def sample(self, feature, spatial_scale, shape, indices):
        batch, angles = indices.shape
        height, width = shape
        device = feature.device
        aperture = (self.receivers - 1) * self.spec["pitch_m"]
        x = torch.linspace(-aperture / 2, aperture / 2, width, device=device)
        max_time = (IQ_SAMPLES - 1) * RF_SAMPLES / (RF_FS * IQ_SAMPLES)
        time = torch.linspace(0, max_time, height, device=device)
        sin_theta = self.sin_theta[indices].reshape(-1, 1, 1)
        cos_theta = self.cos_theta[indices].reshape(-1, 1, 1)
        t_ref = self.tref[indices].reshape(-1, 1, 1)
        # Nominal depth of an axial receive path for this plane-wave transmit.
        z = (C_REF * (time[None, :, None] - t_ref) - x[None, None, :] * sin_theta) / (1 + cos_theta)
        z = 2 * (z - self.z_start) / (self.dx * spatial_scale * (feature.shape[-2] - 1)) - 1
        x = 2 * (x - self.x_start) / (self.dx * spatial_scale * (feature.shape[-1] - 1)) - 1
        grid = torch.stack((x.expand(batch * angles, height, width), z), dim=-1)
        repeated = feature.repeat_interleave(angles, dim=0)
        return F.grid_sample(repeated.float(), grid, padding_mode="border",
                             align_corners=True).to(feature.dtype)

    def predict_iq(self, medium, angle_indices=None):
        if angle_indices is None:
            angle_indices = torch.arange(self.template.shape[0], device=medium.device)
        else:
            angle_indices = torch.as_tensor(angle_indices, dtype=torch.long, device=medium.device)
        if angle_indices.ndim == 1:
            angle_indices = angle_indices[None].expand(medium.shape[0], -1)
        if angle_indices.ndim != 2 or angle_indices.shape[0] != medium.shape[0]:
            raise ValueError("Use a shared angle list or one angle list per medium")
        features = self.encode_medium(medium)
        geometry = torch.stack((self.sin_theta[angle_indices], self.cos_theta[angle_indices],
                                self.tref[angle_indices] * 1e6), dim=-1)
        angle = self.angle_encoder(geometry.flatten(0, 1))
        decoded = None
        for level, stage in enumerate(self.decoder):
            scale = 2 ** (3 - level)
            sampled = self.sample(features[-1 - level], 4 * scale,
                                  (IQ_SAMPLES // scale, self.receivers // scale), angle_indices)
            decoded = stage(sampled, angle, decoded)
        residual = self.head(decoded).transpose(-1, -2)
        residual = residual.reshape(*angle_indices.shape, 2, self.receivers, IQ_SAMPLES)
        if self.spec.get("residual_frame", "baseband") == "analytic":
            real, imag = residual.float().unbind(dim=2)
            residual = torch.stack((real * self.carrier_cos + imag * self.carrier_sin,
                                    imag * self.carrier_cos - real * self.carrier_sin), dim=2)
        return self.template[angle_indices] + self.rf_scale * residual.float()

    def forward(self, medium, angle_indices=None):
        return decode(self.predict_iq(medium, angle_indices))


def make_forward_model(spec, template=None, scale=1.):
    if spec.get("architecture") == "ray_arrival_v1":
        from .ray_model import RayArrivalForwardOperator
        return RayArrivalForwardOperator(spec)
    if spec.get("architecture") == "fine_scatter_v1":
        from .scatter_model import FineScatterForwardOperator
        return FineScatterForwardOperator(spec, template, scale)
    return ForwardOperator(spec, template, scale)


def load_forward_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = make_forward_model(checkpoint["spec"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


