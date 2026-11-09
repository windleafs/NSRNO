"""Coherent native-grid scattering with input-dependent straight-ray delays.

This is a differentiable single-scattering feature operator, not a wave solver.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .forward_model import C_REF, RHO_REF
from .rf_representation import CENTRE_BIN, IQ_SAMPLES, RF_FS, RF_SAMPLES


class RayPropagation(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.dx = float(spec["dx_m"])
        self.h, self.w = spec["medium_shape"][-2:]
        self.probe = round(-spec["z0_m"] / self.dx)
        self.nz = self.h - self.probe
        self.fan_count = int(spec.get("ray_fan_count", 513))
        self.fan_limit = math.radians(89.9)
        self.oversample = int(spec.get("ray_time_oversample", 4))
        self.nt = IQ_SAMPLES * self.oversample
        self.duration = RF_SAMPLES / RF_FS
        self.fc = CENTRE_BIN / self.duration
        self.chunk = int(spec.get("ray_receiver_chunk", 8))
        for key, tensor in {
            "x": torch.arange(self.w) * self.dx + spec["x0_m"],
            "z": torch.arange(self.nz) * self.dx,
            "rx": torch.linspace(-(spec["receivers"]-1)*spec["pitch_m"]/2,
                                  (spec["receivers"]-1)*spec["pitch_m"]/2, spec["receivers"]),
            "theta": torch.deg2rad(torch.tensor(spec["angles_deg"])),
            "tref": torch.tensor(spec["source_tref_s"]),
            "fan": torch.linspace(-self.fan_limit, self.fan_limit, self.fan_count),
        }.items():
            self.register_buffer(key, tensor.float(), persistent=False)

    def _grid(self, x, z):
        gx = 2 * (x - self.spec["x0_m"]) / (self.dx * (self.w - 1)) - 1
        gz = 2 * (z - self.spec["z0_m"]) / (self.dx * (self.h - 1)) - 1
        return torch.stack(torch.broadcast_tensors(gx, gz), -1)

    def detail_sources(self, medium):
        c, rho = medium.float().unbind(1)
        fields = torch.stack(((c / C_REF).log(), (rho / RHO_REF).log()), 1)
        smooth = F.avg_pool2d(F.pad(fields, (4, 4, 0, 0), mode="replicate"), (1, 9), stride=1)
        smooth = F.avg_pool2d(F.pad(smooth, (0, 0, 4, 4), mode="replicate"), (9, 1), stride=1)
        dc, dr = (fields - smooth).unbind(1)
        # Linearized relative compressibility and relative density contrasts.
        source = torch.stack(((-2 * dc - dr) / .01, dr / .008), 1)[:, :, self.probe:]
        # The z=0 sample is on the source/receiver boundary, not a volume cell.
        active = (self.z > 0).to(source.dtype)[None, None, :, None]
        return source * active

    def _path_average(self, values):
        increments = .5 * (values[..., 1:, :] + values[..., :-1, :]) * self.dx
        integral = F.pad(torch.cumsum(increments, dim=-2), (0, 0, 1, 0))
        average = integral / self.z.clamp_min(self.dx)[..., None]
        return torch.cat((values[..., :1, :], average[..., 1:, :]), -2)

    def receive_time(self, medium, receivers, material=True):
        """[B,R,Z,X] seconds; exact geometric term plus average slowness correction."""
        rx = self.rx[receivers]
        dz = self.z[None, :, None]
        lateral = self.x[None, None, :] - rx[:, None, None]
        distance = torch.sqrt(dz.square() + lateral.square()).clamp_min(self.dx / 2)
        nominal = distance / C_REF
        if not material:
            return nominal[None].expand(medium.shape[0], -1, -1, -1)
        delta = medium[:, :1].float().reciprocal() - 1 / C_REF
        ray_x = rx[:, None, None] + self.z[None, :, None] * self.fan.tan()[None, None, :]
        grid = self._grid(ray_x, self.z[None, :, None]).reshape(1, len(rx)*self.nz, self.fan_count, 2)
        sampled = F.grid_sample(delta, grid.expand(medium.shape[0], -1, -1, -1),
                                padding_mode="border", align_corners=True)
        sampled = sampled.reshape(medium.shape[0], len(rx), self.nz, self.fan_count)
        average = self._path_average(sampled)
        phi = torch.atan2(lateral.expand(-1, self.nz, -1), dz.expand(len(rx), -1, self.w))
        query = torch.stack((phi / self.fan_limit,
                            (2*self.z/self.z[-1]-1)[None, :, None].expand_as(phi)), -1)
        queried = F.grid_sample(average.flatten(0, 1)[:, None],
                               query.repeat(medium.shape[0], 1, 1, 1),
                               padding_mode="border", align_corners=True)
        correction = queried.reshape(medium.shape[0], len(rx), self.nz, self.w) * distance
        return nominal[None] + correction

    def transmit_time(self, medium, angles, material=True):
        """[B,A,Z,X] seconds with the fixed V5 plane-wave boundary clock."""
        theta = self.theta[angles]
        z, x = self.z[None, :, None], self.x[None, None, :]
        nominal = self.tref[angles, None, None] + (
            x*theta.sin()[:, None, None] + z*theta.cos()[:, None, None]) / C_REF
        if not material:
            return nominal[None].expand(medium.shape[0], -1, -1, -1)
        delta = medium[:, :1].float().reciprocal() - 1 / C_REF
        # Extend the ray-entry coordinate so oblique paths retain all native endpoints.
        extra = math.ceil(float(self.z[-1]) * math.tan(math.radians(max(abs(v) for v in self.spec["angles_deg"]))) / self.dx) + 1
        entry = torch.arange(-extra, self.w+extra, device=medium.device) * self.dx + self.spec["x0_m"]
        path_x = entry[None, None, :] + z * theta.tan()[:, None, None]
        grid = self._grid(path_x, z).reshape(1, len(theta)*self.nz, len(entry), 2)
        sampled = F.grid_sample(delta, grid.expand(medium.shape[0], -1, -1, -1),
                                padding_mode="border", align_corners=True)
        average = self._path_average(sampled.reshape(medium.shape[0], len(theta), self.nz, len(entry)))
        entry_at_point = x - z * theta.tan()[:, None, None]
        gx = 2*(entry_at_point-entry[0])/(entry[-1]-entry[0])-1
        gz = (2*self.z/self.z[-1]-1)[None, :, None].expand_as(gx)
        query = torch.stack((gx, gz), -1)
        queried = F.grid_sample(average.flatten(0, 1)[:, None], query.repeat(medium.shape[0], 1, 1, 1),
                               padding_mode="border", align_corners=True)
        correction = queried.reshape(medium.shape[0], len(theta), self.nz, self.w)
        return nominal[None] + correction * z[None] / theta.cos()[None, :, None, None]

    def _deposit(self, source, time, geometry):
        """Carrier-phase preserving deposition, followed by band-limited resampling."""
        b, r, h, w = time.shape
        bins = time.reshape(b, r, -1) * (self.nt / self.duration)
        lower = bins.floor().long()
        fraction = bins - lower
        phase = (-2*math.pi*self.fc) * time.reshape(b, r, -1)
        amplitude = source[:, None].flatten(-2) * geometry[None].flatten(-2)
        # Fixed geometric normalization; neural input scales are fitted on train only.
        amplitude = amplitude / 32.
        real = amplitude * phase.cos()[:, :, None]
        imag = amplitude * phase.sin()[:, :, None]
        outputs = []
        for value in (real, imag):
            output = value.new_zeros(b, r, 2, self.nt)
            for offset, weight in ((0, 1-fraction), (1, fraction)):
                destination = lower + offset
                valid = (destination >= 0) & (destination < self.nt)
                output.scatter_add_(-1, destination.clamp(0, self.nt-1)[:, :, None].expand_as(value),
                                    value * (weight * valid)[:, :, None])
            outputs.append(output)
        events = torch.complex(outputs[0], outputs[1])
        spectrum = torch.fft.fft(events, dim=-1)
        half = IQ_SAMPLES // 2
        spectrum = torch.cat((spectrum[..., :half], spectrum[..., -half:]), -1)
        result = torch.fft.ifft(spectrum, dim=-1)
        return torch.stack((result.real, result.imag), 3).permute(0, 2, 3, 1, 4)

    def forward(self, medium, angles=None, receivers=None, material=True):
        if medium.ndim != 4 or tuple(medium.shape[1:]) != (2, self.h, self.w):
            raise ValueError("Expected native c/rho [B,2,H,W] in physical units")
        device = medium.device
        angles = torch.arange(len(self.theta), device=device) if angles is None else torch.as_tensor(angles, device=device)
        receivers = torch.arange(len(self.rx), device=device) if receivers is None else torch.as_tensor(receivers, device=device)
        if angles.ndim != 1 or receivers.ndim != 1:
            raise ValueError("Propagation accepts shared angle and receiver lists")
        source = self.detail_sources(medium)
        tx = self.transmit_time(medium, angles, material)
        per_receiver = []
        z, x = self.z[None, :, None], self.x[None, None, :]
        aperture = (self.spec["receivers"]-1) * self.spec["pitch_m"]
        for start in range(0, len(receivers), self.chunk):
            selected = receivers[start:start+self.chunk]
            rx = self.rx[selected, None, None]
            distance = torch.sqrt(z.square() + (rx-x).square()).clamp_min(self.dx/2)
            spreading = torch.sqrt(.02 / distance)
            receive = self.receive_time(medium, selected, material)
            per_angle = []
            for j, ai in enumerate(angles):
                theta = self.theta[ai]
                entry = x-z*theta.tan()
                apod = torch.where(entry.abs() <= aperture/2,
                                   torch.cos(math.pi*entry/aperture).square(), 0.)
                dipole = ((rx-x)*theta.sin() - z*theta.cos()) / distance
                weight = spreading * apod
                geometry = torch.stack((weight, weight*dipole), 1)
                per_angle.append(self._deposit(source, receive + tx[:, j:j+1], geometry))
            per_receiver.append(torch.stack(per_angle, 1))
        return torch.cat(per_receiver, dim=-2)
