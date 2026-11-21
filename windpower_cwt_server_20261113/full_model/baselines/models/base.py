# -*- coding: utf-8 -*-
"""统一模型接口 + 模型注册表（放在 models 包内，避免与 registry 循环导入）。

约定输入/输出：
    forward(x) : x [B, L, C] -> 预测 [B, H]（单目标功率，标准化空间）
统一签名是预留 tblock 混合接口的基础：TemporalBlock 为 [B,L,C]->[B,H,1]，
基线输出 [B,H] 可经 tblock_adapter 包装成同形状后替换/融合。
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ..config import TrainConfig

_REGISTRY: dict[str, Callable[..., "BaseForecaster"]] = {}


def register(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"模型名重复: {name}")
        _REGISTRY[name] = cls
        return cls
    return deco


def get_model(name: str, input_channels: int, horizon: int,
              cfg: TrainConfig) -> "BaseForecaster":
    if name == "arima":
        from .arima import ARIMAForecaster
        return ARIMAForecaster(order=cfg.arima_order)  # 统计模型，非 nn.Module
    if name not in _REGISTRY:
        raise KeyError(f"未知模型: {name}，可用: {sorted(_REGISTRY)}")
    return _REGISTRY[name](input_channels, horizon, cfg)


def available_models() -> list[str]:
    return sorted(_REGISTRY)


class BaseForecaster(nn.Module):
    def __init__(self, input_channels: int, horizon: int):
        super().__init__()
        self.input_channels = input_channels
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
