"""Material-conditioned finite-secant RF inverse UNO; no patient solver imports."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .raw_rf_inverse_uno import SourceRFEncoder, IntegralAttention, fourier_positions
from .uno_rf import UShape
from .inverse_uno import HEIGHT, WIDTH


class SecantSourceEncoder(SourceRFEncoder):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(101, 24, 1)

    def forward(self, signal, geometry):
        b = signal.shape[0]
        packed = F.pixel_unshuffle(signal, 4)
        h, w = packed.shape[-2:]
        rx = torch.linspace(-1, 1, 192, device=signal.device).reshape(1, 1, 192, 1).expand(b, 1, 192, 544)
        rx = F.avg_pool2d(rx, 4)
        geom = geometry[:, :, None, None].expand(-1, -1, h, w)
        encoded = self.uno(self.stem(torch.cat((packed, rx, geom), 1))).flatten(2).transpose(1, 2)
        valid = F.avg_pool2d(signal[:, 4:5], 4).flatten(2).transpose(1, 2)
        time = F.avg_pool2d(signal[:, 5:6], 4).flatten(2).transpose(1, 2) / valid.clamp_min(1 / 16)
        positions = torch.cat((rx.flatten(2).transpose(1, 2), time,
                               geometry[:, :3, None].transpose(1, 2).expand(-1, h * w, -1)), 2)
        query_positions = torch.cat((self.query_coordinates[None].expand(b, -1, -1),
                                     geometry[:, :3, None].transpose(1, 2).expand(-1, 512, -1)), 2)
        queries = self.queries + self.position(fourier_positions(query_positions))
        keys = self.key(encoded) + self.position(fourier_positions(positions))
        distance = query_positions[:, :, :2, None] - positions[:, :, :2].transpose(1, 2)[:, None]
        bias = -.5 * ((distance[:, :, 0] / .16) ** 2 + (distance[:, :, 1] / .055) ** 2)
        bias = bias.masked_fill(valid[..., 0][:, None, :] == 0, float('-inf'))[:, None]
        tokens = self.readout(queries, keys, self.value(encoded), bias)
        return tokens + self.refine(tokens), self.position(fourier_positions(query_positions))


class ConditionalSecantUNO(nn.Module):
    def __init__(self, query_chunk=1024):
        super().__init__()
        self.query_chunk = query_chunk
        self.encoder = SecantSourceEncoder()
        angles = torch.linspace(-8, 8, 11)
        tref = 4 / 3e6 + .0191 * torch.sin(angles * math.pi / 180).abs() / 1540
        phase = 2 * math.pi * 3e6 * tref
        self.register_buffer('geometry', torch.stack((angles / 8, tref / 10e-6, torch.sin(phase), torch.cos(phase)), 1))
        z, x = torch.meshgrid(torch.linspace(0, 1, HEIGHT), torch.linspace(-1, 1, WIDTH), indexing='ij')
        self.register_buffer('coordinates', torch.stack((x, z))[None])
        self.register_buffer('query_features', fourier_positions(torch.stack((x, z), -1).reshape(-1, 2)))
        self.register_buffer('scales', torch.tensor([80., 100., .2])[None, :, None, None])
        self.register_buffer('centres', torch.tensor([1540., 1000., .5])[None, :, None, None])
        self.register_buffer('gel', torch.arange(HEIGHT)[None, None, :, None] < 5)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GELU())
        cz, cx = torch.meshgrid(torch.linspace(-3 / 43, 45 / 43, 8),
                               torch.linspace(-28 / 19.1, 28 / 19.1, 16), indexing='ij')
        self.register_buffer('context_positions', fourier_positions(torch.stack((cx, cz), -1).reshape(-1, 2)))
        self.context_position = nn.Linear(34, 64)
        self.context_type = nn.Parameter(torch.randn(1, 1, 64) * .02)
        self.query = nn.Sequential(nn.Linear(38, 64), nn.GELU(), nn.Linear(64, 64))
        self.readout = IntegralAttention()
        self.to_image = nn.Linear(64, 24)
        self.stem = nn.Conv2d(29, 24, 1)
        self.decoder = UShape([24, 32, 48, 32, 24],
                             [(16, 16), (12, 12), (8, 8), (12, 12), (16, 16)], True)
        self.shape_head = nn.Sequential(nn.Conv2d(24, 24, 1), nn.GELU(), nn.Conv2d(24, 3, 1))
        self.gain_head = nn.Sequential(nn.Linear(135, 64), nn.GELU(), nn.Linear(64, 3))
        nn.init.normal_(self.shape_head[-1].weight, std=.001)
        nn.init.zeros_(self.shape_head[-1].bias)
        nn.init.normal_(self.gain_head[-1].weight, std=.001)
        nn.init.constant_(self.gain_head[-1].bias, -2.)
        self.config = {
            'architecture': 'conditional_finite_secant_RF_UNO_v1',
            'RF_sources': 11, 'receivers': 192, 'complex_RF_samples_padded': 544,
            'RF_inputs': ['current_RF/global_scale', 'unit_RMS_RF_difference', 'valid', 'time'],
            'material_context': 'known current full-domain 241x281 and ROI216x192',
            'output': 'independent c/rho/tissue-alpha correction',
            'correction': 'S*r*exp(log_gain)*unit_RMS_shape',
            'residual_RMS_is_retained': True, 'zero_RF_difference_exact_zero_update': True,
            'forced_smooth_c': False, 'shared_material_edges': False,
            'query_chunk': query_chunk, 'source_tokens': 512, 'context_tokens': 128,
            'proposer_is_not_completed_native_joint_reconstruction': True,
        }

    def forward(self, signal, baseline_roi, context, residual_rms):
        if signal.ndim != 5 or signal.shape[1:] != (11, 6, 192, 544):
            raise ValueError('Complete 11-shot conditional RF input required')
        b = signal.shape[0]
        if baseline_roi.shape != (b, 3, HEIGHT, WIDTH) or context.shape != (b, 3, 241, 281):
            raise ValueError('Current ROI and full propagation context required')
        r = residual_rms.reshape(b, 1).float()
        log_r = r.clamp_min(1e-12).log()
        base = (baseline_roi - self.centres) / self.scales
        ctx = (context - self.centres) / self.scales
        values, positions = [], []
        for shot in range(11):
            geometry = self.geometry[shot][None].expand(b, -1)
            if torch.is_grad_enabled():
                value, position = checkpoint(self.encoder, signal[:, shot], geometry, use_reentrant=False)
            else:
                value, position = self.encoder(signal[:, shot], geometry)
            values.append(value); positions.append(position)
        rf_values = torch.cat(values, 1)
        ctx_values = F.adaptive_avg_pool2d(self.context_encoder(ctx), (8, 16)).flatten(2).transpose(1, 2)
        keys = torch.cat((rf_values + torch.cat(positions, 1),
                          ctx_values + self.context_position(self.context_positions)[None] + self.context_type), 1)
        all_values = torch.cat((rf_values, ctx_values), 1)
        base_flat = base.flatten(2).transpose(1, 2)

        def decode_chunk(position, state, k, v, logr):
            inputs = torch.cat((position[None].expand(b, -1, -1), state,
                                logr[:, None].expand(-1, position.shape[0], -1)), -1)
            return self.to_image(self.readout(self.query(inputs), k, v))

        image = []
        for start in range(0, HEIGHT * WIDTH, self.query_chunk):
            q = self.query_features[start:start + self.query_chunk]
            state = base_flat[:, start:start + self.query_chunk]
            if torch.is_grad_enabled():
                image.append(checkpoint(decode_chunk, q, state, keys, all_values, log_r, use_reentrant=False))
            else:
                image.append(decode_chunk(q, state, keys, all_values, log_r))
        image = torch.cat(image, 1).transpose(1, 2).reshape(b, 24, HEIGHT, WIDTH)
        shape_raw = self.shape_head(self.decoder(self.stem(torch.cat(
            (image, self.coordinates.expand(b, -1, -1, -1), base), 1))))
        shape_raw = shape_raw.masked_fill(self.gel, 0.)
        shape_rms = shape_raw[:, :, 5:].square().mean((-2, -1), keepdim=True).add(1e-12).sqrt()
        shape = shape_raw / shape_rms
        gain_input = torch.cat((rf_values.mean(1), ctx_values.mean(1), base[:, :, 5:].mean((-2, -1)),
                                base[:, :, 5:].std((-2, -1), unbiased=False), log_r), 1)
        log_gain = 14 * torch.tanh(self.gain_head(gain_input) / 14)
        amplitude = r * log_gain.exp()
        delta = amplitude[:, :, None, None] * shape
        prediction = baseline_roi + self.scales * delta
        prediction = torch.where(self.gel, baseline_roi, prediction)
        return {'prediction': prediction, 'delta_scaled': delta, 'shape': shape,
                'log_gain': log_gain, 'amplitude_scaled': amplitude, 'residual_RMS': r}


def secant_loss(result, baseline, target):
    scale = target.new_tensor([80., 100., .2])[None, :, None, None]
    exact = ((target - baseline) / scale)[:, :, 5:]
    amplitude = exact.square().mean((-2, -1)).sqrt()
    floor = target.new_tensor([5 / 80, 5 / 100, .02 / .2])[None]
    relative = ((result['delta_scaled'][:, :, 5:] - exact).square().mean((-2, -1)) /
                amplitude.square().clamp_min(floor.square()))
    active = amplitude > 1e-5
    exact_shape = exact / amplitude.clamp_min(1e-12)[:, :, None, None]
    shape_error = (result['shape'][:, :, 5:] - exact_shape).square().mean((-2, -1))
    shape_loss = (shape_error * active).sum() / active.sum().clamp_min(1)
    wanted_gain = (amplitude.clamp_min(1e-12) / result['residual_RMS'].clamp_min(1e-12)).log()
    gain_error = (result['log_gain'] - wanted_gain).square()
    gain_loss = (gain_error * active).sum() / active.sum().clamp_min(1)
    loss = relative.mean() + .5 * shape_loss + .1 * gain_loss
    return loss, {
        'relative_c_rho_alpha': relative.detach().mean(0).cpu().tolist(),
        'shape_loss': float(shape_loss.detach()), 'log_gain_loss': float(gain_loss.detach()),
        'target_amplitude_scaled': amplitude.detach().cpu().tolist(),
        'predicted_amplitude_scaled': result['amplitude_scaled'].detach().cpu().tolist(),
    }
