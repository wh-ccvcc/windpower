"""Fixed Mexican hat continuous wavelet transform (CWT) along the time axis."""

import torch
from torch import nn
from torch.nn import functional as F


class WindCWT(nn.Module):
    """Fixed Mexican hat responses by scale at each observed time step."""

    def __init__(self, scales):
        super().__init__()
        radius = 4 * max(scales)
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
        scale = torch.tensor(scales, dtype=torch.float32)[:, None]
        scaled = offsets / scale
        support = scaled.abs() <= 4
        kernels = (1 - scaled.square()) * torch.exp(-scaled.square() / 2) * support
        kernels = (kernels - kernels.sum(dim=-1, keepdim=True) / support.sum(dim=-1, keepdim=True)) * support
        self.register_buffer("kernels", F.normalize(kernels, dim=-1).unsqueeze(1))
        self.radius = radius
        self.register_buffer("scale_radii", torch.tensor(scales) * 4, persistent=False)

    def forward(self, signal, *, full_support_only=False):
        padded = F.pad(signal.unsqueeze(1), (self.radius, self.radius), mode="replicate")
        coefficients = F.conv1d(padded, self.kernels)
        if full_support_only:
            time = torch.arange(signal.shape[-1], device=signal.device)
            distance_to_edge = torch.minimum(time, time.flip(0))
            coefficients = coefficients.masked_fill(distance_to_edge < self.scale_radii[:, None], 0)
        return coefficients
