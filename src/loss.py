"""自定义损失函数定义。"""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianNLLLoss(nn.Module):
    """异方差高斯负对数似然损失（预测 mu 与 log_var）。

    公式：
        loss = 0.5 * exp(-log_var) * (target - mu)^2 + 0.5 * log_var

    数值稳定性：
    - 对 log_var 进行 clamp，避免 exp 溢出或极端梯度。
    - 分母相关计算加 eps 防止数值边界问题。
    """

    def __init__(
        self,
        min_log_var: float = -10.0,
        max_log_var: float = 10.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.eps = eps

    def forward(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """计算 batch 平均损失。"""
        stable_log_var = torch.clamp(log_var, min=self.min_log_var, max=self.max_log_var)
        inv_var = torch.exp(-stable_log_var).clamp_min(self.eps)
        sq_error = (target - mu) ** 2
        loss = 0.5 * inv_var * sq_error + 0.5 * stable_log_var
        return loss.mean()
