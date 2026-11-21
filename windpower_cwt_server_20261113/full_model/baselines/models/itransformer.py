# -*- coding: utf-8 -*-
"""iTransformer（倒置 Transformer，变量维 attention）简化实现。

核心思想与标准 Transformer 相反：每个变量是一条 token（该变量的整段历史
作为 token 特征），attention 作用在变量之间。功率为目标变量（通道 0）。
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import TrainConfig
from .base import BaseForecaster, register
from .common import PositionalEncoding


@register("itransformer")
class ITransformer(BaseForecaster):
    def __init__(self, input_channels: int, horizon: int, cfg: TrainConfig):
        super().__init__(input_channels, horizon)
        # 变量时间序列 -> token：共享权重，作用于每个变量（历史长度由数据决定）
        self.var_embed = nn.Linear(cfg.history, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, max_len=256)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff,
                                           cfg.dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, C] -> [B, C, L] -> 每变量 token [B, C, d_model]
        tokens = self.var_embed(x.transpose(1, 2))
        tokens = tokens + self.pos(x.shape[2])
        tokens = self.norm(self.encoder(tokens))
        return self.head(tokens[:, 0, :])  # 通道 0 = power_mw
