# -*- coding: utf-8 -*-
"""标准 Transformer 编码器（时间维 attention）。"""
from __future__ import annotations

import torch
from torch import nn

from ..config import TrainConfig
from .base import BaseForecaster, register
from .common import PositionalEncoding


@register("transformer")
class Transformer(BaseForecaster):
    def __init__(self, input_channels: int, horizon: int, cfg: TrainConfig, *, readout="mean"):
        super().__init__(input_channels, horizon)
        if readout not in ("mean", "last"):
            raise ValueError(f"Unknown Transformer readout: {readout}")
        self.readout = readout
        self.input_proj = nn.Linear(input_channels, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff,
                                           cfg.dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos(x.shape[1])
        h = self.norm(self.encoder(h))
        pooled = h[:, -1] if self.readout == "last" else h.mean(dim=1)
        return self.head(pooled)
