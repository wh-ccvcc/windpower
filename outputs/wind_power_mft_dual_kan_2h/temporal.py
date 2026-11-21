"""MFT encoder blocks: MHCB local features, TCN/FAN feature module, and Pre-LN residual TAN."""

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from mhcb import MHCBAdapter


class TemporalResidualBlock(nn.Module):
    def __init__(self, width, dilation, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel_size=3, dilation=dilation)
        self.norm = nn.GroupNorm(num_groups=min(8, width), num_channels=width)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.left_padding = 2 * dilation

    def forward(self, x):
        h = self.conv(F.pad(x, (self.left_padding, 0)))
        return x + self.drop(F.gelu(self.norm(h)))


class TemporalConvFeatures(nn.Module):
    """Extract local temporal features while retaining one vector per time step."""

    def __init__(self, width, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential(
            TemporalResidualBlock(width, dilation=1, dropout=dropout),
            TemporalResidualBlock(width, dilation=2, dropout=dropout),
        )

    def forward(self, tokens):
        return self.blocks(tokens.transpose(1, 2)).transpose(1, 2)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_size, output_size),
            nn.Linear(output_size, 2 * output_size),
            nn.GLU(dim=-1),
        )
        self.skip = (nn.Identity() if input_size == output_size
                     else nn.Linear(input_size, output_size))
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        return self.norm(self.skip(x) + self.transform(x))


class ChannelSoftmaxAttention(nn.Module):
    """Mix neighboring channel scores and normalize the channel weights."""

    def __init__(self):
        super().__init__()
        self.kernel = nn.Parameter(torch.tensor([[[0.0, 1.0, 0.0]]]))

    def forward(self, scores):
        mixed = F.conv1d(scores.unsqueeze(1), self.kernel, padding=1).squeeze(1)
        return mixed.softmax(dim=-1) * (scores.shape[-1] / 2)


class DualPoolFeatureAttention(nn.Module):
    """Apply independent mean and max channel gates over observed time."""

    def __init__(self, width, reduction=4, *, weighting=nn.Sigmoid):
        super().__init__()
        bottleneck = max(1, width // reduction)

        def make_gate():
            return nn.Sequential(
                nn.Linear(width, bottleneck, bias=False),
                nn.ReLU(),
                nn.Linear(bottleneck, width, bias=False),
                weighting(),
            )

        self.mean_gate = make_gate()
        self.max_gate = make_gate()

    def forward(self, x):
        mean_weight = self.mean_gate(x.mean(dim=1)).unsqueeze(1)
        max_weight = self.max_gate(x.max(dim=1).values).unsqueeze(1)
        return x * mean_weight + x * max_weight + x


class TemporalAttention(nn.Module):
    """Pre-LN multi-head self-attention + FFN with residual connections."""

    def __init__(self, width, num_heads=4, dropout=0.1):
        super().__init__()
        heads = num_heads if width % num_heads == 0 else 1
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(2 * width, width),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        return x + self.ffn(self.norm2(x))


class VANFANTANEncoder(nn.Module):
    """Encode [batch, history, variables] into [batch, history, width]."""

    def __init__(self, history, variables, width=48, *, fan="tcn", dropout=0.1):
        super().__init__()
        if min(history, variables, width) < 1:
            raise ValueError("history, variables and width must be positive")
        feature_modules = {
            "dual-attn": lambda w: DualPoolFeatureAttention(w, weighting=ChannelSoftmaxAttention),
            "tcn": lambda w: TemporalConvFeatures(w, dropout=dropout),
        }
        if fan not in feature_modules:
            raise ValueError(f"Unknown feature module: {fan}")
        self.input_shape = (history, variables)
        mhcb_channels = max(16, width // 2)
        self.van = nn.Sequential(
            nn.Linear(variables, width),
            nn.LayerNorm(width),
            MHCBAdapter(width, channels=mhcb_channels),
        )
        self.fan = feature_modules[fan](width)
        self.position = nn.Parameter(torch.empty(history, width))
        nn.init.normal_(self.position, std=0.02)
        self.tan = TemporalAttention(width, num_heads=4, dropout=dropout)

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != self.input_shape:
            raise ValueError(f"Expected [batch, {self.input_shape[0]}, {self.input_shape[1]}]")
        tokens = self.fan(self.van(history))
        return self.tan(tokens + self.position)
