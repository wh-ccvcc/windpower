# -*- coding: utf-8 -*-
"""共享组件：位置编码、趋势/季节分解（Autoformer 用）。"""
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """标准 sin/cos 绝对位置编码 [1, L, d]（或 [1, C, d] 用于变量维度）。"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length]


class SeriesDecomp(nn.Module):
    """移动平均趋势分解（Autoformer 用），返回 (seasonal, trend)。"""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                padding=(kernel_size - 1) // 2, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, D]
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend
