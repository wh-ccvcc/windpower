"""Dual-branch MFT + CWT-Strip forecaster with AAAI 2026 CAFR cross-channel fusion."""

import torch
from torch import nn

from cwt_attention import WindCWT
from cwt_strip import CWTStripEncoder
from temporal import TemporalResidualBlock, VANFANTANEncoder


class CAFR1D(nn.Module):
    """AAAI 2026 Cross-Feature Refinement Attention adapted for aligned [B, T, D] branches."""

    def __init__(self, width, dropout=0.1):
        super().__init__()
        self.norm_l = nn.LayerNorm(width)
        self.norm_r = nn.LayerNorm(width)
        self.fc_l = nn.Linear(width, width)
        self.fc_r = nn.Linear(width, width)
        self.scale = width ** -0.5
        self.refine = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, left, right):
        ln_l, ln_r = self.norm_l(left), self.norm_r(right)
        v_l = self.fc_l(ln_l.mean(dim=1)).unsqueeze(-1)
        v_r = self.fc_r(ln_r.mean(dim=1)).unsqueeze(-1)
        affinity = (v_l @ v_r.transpose(1, 2) * self.scale).softmax(dim=-1)
        w_l = torch.sigmoid(affinity.transpose(-1, -2) @ v_l).transpose(1, 2)
        w_r = torch.sigmoid(affinity @ v_r).transpose(1, 2)
        fused = left * w_l + right * w_r
        return fused + self.refine(fused)


class GatedFusion1D(nn.Module):
    """Simple per-step softmax gated fusion baseline for ablation against CAFR1D."""

    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(2 * width, 2)
        self.proj = nn.Linear(2 * width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, left, right):
        both = torch.cat((left, right), dim=-1)
        w_l, w_r = self.gate(both).softmax(dim=-1).split(1, dim=-1)
        return self.norm(self.proj(torch.cat((w_l * left, w_r * right), dim=-1)))


class RevIN(nn.Module):
    """Reversible Instance Normalization (ICLR 2022) for direct non-stationary forecasting."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, 1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        normed = (x - mean) / std * self.weight + self.bias
        return normed, mean, std

    def denormalize_target(self, y, mean, std, target_index=0):
        w = self.weight[:, 0, target_index:target_index + 1]
        b = self.bias[:, 0, target_index:target_index + 1]
        return ((y - b) / (w + self.eps)) * std[:, 0, target_index:target_index + 1] + mean[:, 0, target_index:target_index + 1]


class DualMFTForecaster(nn.Module):
    """Fuse Pre-LN residual MFT temporal features and CWT-Strip wavelet features."""

    def __init__(
        self,
        history,
        horizon,
        variables,
        width=32,
        *,
        fan="tcn",
        cwt_scales=(1, 2, 4, 8),
        cwt_encoder="strip",
        cwt_full_support=False,
        temporal_rnn="none",
        fusion="cafr",
        output_mode="anchor",
        cwt_wind_indices=(1, 2, 3, 4),
        native_power_anchor=True,
        wind_selection=True,
        input_layout="power_cwt",
        dropout=0.2,
    ):
        super().__init__()
        if input_layout not in ("wind_cwt", "power_cwt"):
            raise ValueError(f"Unknown input layout: {input_layout}")
        if cwt_encoder not in ("strip", "original"):
            raise ValueError(f"Unknown CWT encoder: {cwt_encoder}")
        if temporal_rnn not in ("none", "lstm", "bilstm", "bigru"):
            raise ValueError(f"Unknown temporal RNN: {temporal_rnn}")
        if fusion not in ("cafr", "gated"):
            raise ValueError(f"Unknown fusion mode: {fusion}")
        if output_mode not in ("anchor", "revin"):
            raise ValueError(f"Unknown output mode: {output_mode}")

        self.power_cwt = input_layout == "power_cwt"
        self.output_mode = output_mode
        self.native_power_anchor = native_power_anchor and output_mode == "anchor"
        self.revin = RevIN(variables) if output_mode == "revin" else None
        self.cwt_full_support = cwt_full_support
        self.variables = variables
        self.cwt_indices = (0,) if self.power_cwt else tuple(cwt_wind_indices)
        self.wind_indices = tuple(range(1, variables)) if self.power_cwt else self.cwt_indices
        self.scale_count = len(cwt_scales) * len(self.cwt_indices)

        # Multi-height dynamic wind gate
        self.wind_selector = None
        if wind_selection:
            self.wind_selector = nn.Sequential(
                nn.Conv1d(variables, width, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(width, len(self.wind_indices), kernel_size=1),
            )
            nn.init.zeros_(self.wind_selector[-1].weight)
            nn.init.zeros_(self.wind_selector[-1].bias)

        # Left branch: MFT (MHCB + TCN/FAN + Pre-LN Residual TAN) + optional single RNN
        temporal_variables = variables if self.power_cwt else 1
        self.temporal_encoder = VANFANTANEncoder(history, temporal_variables, width, fan=fan, dropout=dropout)
        self.temporal_rnn = None
        self.rnn_proj = None
        if temporal_rnn == "lstm":
            self.temporal_rnn = nn.LSTM(width, width, batch_first=True)
        elif temporal_rnn == "bilstm":
            self.temporal_rnn = nn.LSTM(width, width // 2, batch_first=True, bidirectional=True)
            self.rnn_proj = nn.LayerNorm(width)
        elif temporal_rnn == "bigru":
            self.temporal_rnn = nn.GRU(width, width // 2, batch_first=True, bidirectional=True)
            self.rnn_proj = nn.LayerNorm(width)

        # Right branch: CWT + AAAI 2026 StripModule (replacing VGG-16) + lightweight causal TCN
        self.wavelet = WindCWT(cwt_scales)
        self.cwt_encoder = (
            CWTStripEncoder(len(self.cwt_indices), hidden=max(16, width // 2), depth=2,
                            passthrough=True, dropout=dropout)
            if cwt_encoder == "strip" else None
        )
        wavelet_channels = self.scale_count if self.power_cwt else variables + self.scale_count
        self.wavelet_proj = nn.Sequential(
            nn.Conv1d(wavelet_channels, width, kernel_size=1),
            nn.GELU(),
            TemporalResidualBlock(width, dilation=1, dropout=dropout),
            TemporalResidualBlock(width, dilation=2, dropout=dropout),
        )

        # Cross-feature fusion (no redundant second decoder LSTM)
        self.fusion = CAFR1D(width, dropout=dropout) if fusion == "cafr" else GatedFusion1D(width)
        self.pool_score = nn.Linear(width, 1, bias=False)
        self.readout = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(width, horizon),
        )
        if output_mode == "anchor":
            nn.init.zeros_(self.readout[-1].weight)
            nn.init.zeros_(self.readout[-1].bias)

    def forward(self, history):
        if self.native_power_anchor:
            if history.shape[-1] != self.variables + 1:
                raise ValueError("Expected the latest native power in the extra input channel")
            anchor = history[:, -1, self.variables:]
            history = history[..., :self.variables]
        else:
            history = history[..., :self.variables]
            anchor = history[:, -1, :1]

        revin_stats = None
        if self.revin is not None:
            history, mean, std = self.revin.normalize(history)
            revin_stats = (mean, std)

        observed = history.transpose(1, 2)
        raw = observed
        wind_weights = None
        if self.wind_selector is not None:
            wind_weights = self.wind_selector(observed).softmax(dim=1) * len(self.wind_indices)
            raw = observed.clone()
            raw[:, self.wind_indices, :] = observed[:, self.wind_indices, :] * wind_weights

        # Left temporal branch
        temporal_input = raw.transpose(1, 2) if self.power_cwt else history[..., :1]
        left_steps = self.temporal_encoder(temporal_input)
        if self.temporal_rnn is not None:
            rnn_out, _ = self.temporal_rnn(left_steps)
            if self.rnn_proj is not None:
                rnn_out = self.rnn_proj(rnn_out)
            left_steps = left_steps + rnn_out

        # Right wavelet branch
        signal = observed[:, self.cwt_indices, :]
        bands = self.wavelet(signal.flatten(0, 1), full_support_only=self.cwt_full_support)
        wavelet_map = bands.unflatten(0, signal.shape[:2])
        if wind_weights is not None and not self.power_cwt:
            wavelet_map = wavelet_map * wind_weights.unsqueeze(2)
        if self.cwt_encoder is not None:
            wavelet_map = self.cwt_encoder(wavelet_map)
        bands = wavelet_map.reshape(raw.shape[0], self.scale_count, raw.shape[-1])
        wavelet_input = bands if self.power_cwt else torch.cat((raw, bands), dim=1)
        right_steps = self.wavelet_proj(wavelet_input).transpose(1, 2)

        # Cross-feature fusion & readout
        fused = self.fusion(left_steps, right_steps)
        pool_weights = self.pool_score(fused).softmax(dim=1)
        pooled = (fused * pool_weights).sum(dim=1)
        out = self.readout(torch.cat((fused[:, -1], pooled), dim=-1))
        if self.revin is not None:
            mean, std = revin_stats
            return self.revin.denormalize_target(out, mean, std, target_index=0)
        return anchor + out
