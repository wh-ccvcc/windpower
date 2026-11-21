# -*- coding: utf-8 -*-
"""RNN 家族：LSTM / 双头LSTM / GRU / CNN+LSTM。"""
from __future__ import annotations

import torch
from torch import nn

from ..config import TrainConfig
from .base import BaseForecaster, register


def _last_state(output: torch.Tensor) -> torch.Tensor:
    return output[:, -1, :]


@register("lstm")
class LSTM(BaseForecaster):
    """单 LSTM 编码器 + 线性头。"""

    def __init__(self, input_channels: int, horizon: int, cfg: TrainConfig):
        super().__init__(input_channels, horizon)
        self.lstm = nn.LSTM(input_channels, cfg.rnn_hidden, cfg.rnn_layers,
                            batch_first=True, dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0)
        self.head = nn.Linear(cfg.rnn_hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(_last_state(out))


@register("dual_lstm")
class DualLSTM(BaseForecaster):
    """双头 LSTM：两个不同容量的 LSTM 分支并行编码，拼接后融合输出。

    简化实现：头1 用 rnn_hidden，头2 用 rnn_hidden//2（下限 16），
    各自取最后隐态，concat 后经线性层预测 horizon。
    """

    def __init__(self, input_channels: int, horizon: int, cfg: TrainConfig):
        super().__init__(input_channels, horizon)
        h1 = cfg.rnn_hidden
        h2 = max(16, cfg.rnn_hidden // 2)
        self.lstm1 = nn.LSTM(input_channels, h1, cfg.rnn_layers, batch_first=True,
                             dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0)
        self.lstm2 = nn.LSTM(input_channels, h2, cfg.rnn_layers, batch_first=True,
                             dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0)
        self.head = nn.Linear(h1 + h2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(x)
        fused = torch.cat([_last_state(out1), _last_state(out2)], dim=-1)
        return self.head(fused)


@register("gru")
class GRU(BaseForecaster):
    """单 GRU 编码器 + 线性头。"""

    def __init__(self, input_channels: int, horizon: int, cfg: TrainConfig):
        super().__init__(input_channels, horizon)
        self.gru = nn.GRU(input_channels, cfg.rnn_hidden, cfg.rnn_layers,
                          batch_first=True, dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0)
        self.head = nn.Linear(cfg.rnn_hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(_last_state(out))


@register("cnn_lstm")
class CNNLSTM(BaseForecaster):
    """1D-CNN 局部特征提取 + LSTM 时序建模 + 线性头。"""

    def __init__(self, input_channels: int, horizon: int, cfg: TrainConfig):
        super().__init__(input_channels, horizon)
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, cfg.rnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(cfg.rnn_hidden, cfg.rnn_hidden, cfg.rnn_layers,
                            batch_first=True, dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0)
        self.head = nn.Linear(cfg.rnn_hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, L, hidden]
        out, _ = self.lstm(h)
        return self.head(_last_state(out))
