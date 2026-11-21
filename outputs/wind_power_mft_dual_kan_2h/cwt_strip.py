"""AAAI 2026 StripModule adapted for narrow [B, C, S=4, T=96] wavelet maps."""

from torch import nn


class StripConv2d(nn.Module):
    """Decoupled time-strip and scale-strip depthwise attention."""

    def __init__(self, dim, time_kernel=11, scale_kernel=3):
        super().__init__()
        self.local = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.time_strip = nn.Conv2d(
            dim, dim, kernel_size=(1, time_kernel), padding=(0, time_kernel // 2), groups=dim
        )
        self.scale_strip = nn.Conv2d(
            dim, dim, kernel_size=(scale_kernel, 1), padding=(scale_kernel // 2, 0), groups=dim
        )
        self.mix = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        attn = self.local(x)
        attn = self.time_strip(attn)
        attn = self.scale_strip(attn)
        attn = self.gate(self.norm(self.mix(attn)))
        return x * attn


class StripBlock(nn.Module):
    """Residual StripModule block preserving scale and time resolution."""

    def __init__(self, dim, time_kernel=11, scale_kernel=3, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.strip = StripConv2d(dim, time_kernel=time_kernel, scale_kernel=scale_kernel)
        self.proj_out = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return x + self.proj_out(self.strip(self.proj_in(x)))


class CWTStripEncoder(nn.Module):
    """Lightweight multi-block StripModule encoder replacing VGG-16 for CWT maps."""

    def __init__(self, channels, hidden=32, depth=2, *, passthrough=True, dropout=0.1):
        super().__init__()
        self.passthrough = passthrough
        self.stem = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*(
            StripBlock(hidden, time_kernel=11, scale_kernel=3, dropout=dropout)
            for _ in range(depth)
        ))
        self.project = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, wavelet_map):
        encoded = self.project(self.blocks(self.stem(wavelet_map)))
        return wavelet_map + encoded if self.passthrough else encoded
