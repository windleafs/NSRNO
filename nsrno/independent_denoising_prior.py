"""Independent frozen material denoisers; no acoustic or patient data readers.

The residual energy below includes the denoiser Jacobian under autograd.
It is an explicit chosen energy, not a claim of a normalized prior density.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ConvPair(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.SiLU(), nn.Conv2d(outgoing, outgoing, 3, padding=1), nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class MaterialDenoiser(nn.Module):
    """One material only; scales 0/native,1/4x mean,2/full FOV."""
    def __init__(self):
        super().__init__()
        self.enc0 = ConvPair(5, 12)
        self.enc1 = ConvPair(12, 24)
        self.enc2 = ConvPair(24, 48)
        self.bottom = ConvPair(48, 72)
        self.dec2 = ConvPair(72+48, 48)
        self.dec1 = ConvPair(48+24, 24)
        self.dec0 = ConvPair(24+12, 12)
        self.output = nn.Conv2d(12, 1, 1)

    def forward(self, noisy, sigma, scale):
        b, _, h, w = noisy.shape
        sigma = torch.as_tensor(sigma, dtype=noisy.dtype, device=noisy.device).reshape(-1, 1, 1, 1)
        if len(sigma) == 1:
            sigma = sigma.expand(b, -1, -1, -1)
        if len(sigma) != b or bool(torch.any(sigma <= 0)):
            raise ValueError('One positive sigma per patch required')
        scale = torch.as_tensor(scale, device=noisy.device, dtype=torch.long).reshape(-1)
        if len(scale) == 1:
            scale = scale.expand(b)
        if len(scale) != b or bool(torch.any((scale < 0)|(scale > 2))):
            raise ValueError('One of three declared scales per patch required')
        sd2 = .5**2
        cin = torch.rsqrt(sigma.square()+sd2)
        cskip = sd2/(sigma.square()+sd2)
        cout = .5*sigma*cin
        label = F.one_hot(scale, 3).to(noisy.dtype)[:, :, None, None].expand(-1, -1, h, w)
        x = torch.cat((cin*noisy, (sigma.log()/4).expand(-1, -1, h, w), label), 1)
        e0 = self.enc0(x)
        e1 = self.enc1(F.avg_pool2d(e0, 2))
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        x = self.bottom(F.avg_pool2d(e2, 2))
        x = self.dec2(torch.cat((F.interpolate(x, size=e2.shape[-2:], mode='bilinear', align_corners=False), e2), 1))
        x = self.dec1(torch.cat((F.interpolate(x, size=e1.shape[-2:], mode='bilinear', align_corners=False), e1), 1))
        x = self.dec0(torch.cat((F.interpolate(x, size=e0.shape[-2:], mode='bilinear', align_corners=False), e0), 1))
        return cskip*noisy+cout*self.output(x)


class IndependentMaterialDenoisers(nn.Module):
    def __init__(self):
        super().__init__()
        self.fields = nn.ModuleList([MaterialDenoiser() for _ in range(3)])
        self.config = {'version': 'independent_material_denoising_v1',
            'channels': [12, 24, 48, 72], 'patch_size': 128,
            'normalization_layers': False, 'shared_field_parameters': False,
            'input_channels': 'one material,log_sigma,three_scale_one_hot',
            'features': ['(c-1540)/80', '(rho-1000)/100', 'log(alpha_tissue/.5)/.7'],
            'sigma_data': .5, 'training_sigma_range': [.03, 1.5]}


def denoising_energy(model, value, sigma, scale):
    """Explicit squared fixed-point residual, retaining all chain derivatives."""
    denoised = model(value, sigma, scale)
    sigma = torch.as_tensor(sigma, dtype=value.dtype, device=value.device).reshape(-1, 1, 1, 1)
    return (.5*(value-denoised).square()/sigma.square()).mean()


def training_loss(model, clean, sigma, scale, noise):
    sigma = sigma.reshape(-1, 1, 1, 1)
    output = model(clean+sigma*noise, sigma, scale)
    weight = (sigma.square()+.5**2)/(.5*sigma).square()
    return (weight*(output-clean).square()).mean()
