"""模型定义：支持 HH-Swin (双通道非负软锚点)、Swin+Linear、ResNet1D 三种方案。"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.transpose(1, 2)  # [B, Np, D]

class WindowAttention1D(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        pad_len = (self.window_size - (seq_len % self.window_size)) % self.window_size
        if pad_len > 0:
            x = torch.cat([x, x.new_zeros(bsz, pad_len, dim)], dim=1)

        padded_len = x.shape[1]
        num_windows = padded_len // self.window_size

        x = x.reshape(bsz * num_windows, self.window_size, dim)
        attn_out, _ = self.attn(x, x, x)
        x = attn_out.reshape(bsz, padded_len, dim)

        if pad_len > 0:
            x = x[:, :seq_len, :]
        return x

class SwinTransformerBlock1D(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int = 0) -> None:
        super().__init__()
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention1D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)
        x = self.attn(x)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class SwinBackbone1D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, window_size: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding1D(in_channels, embed_dim, patch_size)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock1D(
                dim=embed_dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
            ) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

class PhysicsInformedHHHead(nn.Module):
    """双通道非负物理启发头"""
    def __init__(self, embed_dim: int, init_pka: float = 7.0) -> None:
        super().__init__()
        self.global_q = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.normal_(self.global_q, std=0.02)

        self.pka_prime = nn.Parameter(torch.tensor([float(init_pka)], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

        self.fc_var = nn.Linear(1, 1)
        nn.init.zeros_(self.fc_var.weight)
        nn.init.constant_(self.fc_var.bias, -2.0)

    def forward(self, k_feat: torch.Tensor, m_pos: torch.Tensor, m_neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = k_feat.shape

        score = torch.matmul(self.global_q.expand(bsz, -1, -1), k_feat.transpose(1, 2)) / math.sqrt(dim)
        score = score.squeeze(1)
        w_map = F.softplus(score)

        eps = 1e-3
        i_pos = torch.sum(w_map * m_pos.unsqueeze(0), dim=1, keepdim=True) + eps
        i_neg = torch.sum(w_map * m_neg.unsqueeze(0), dim=1, keepdim=True) + eps

        mu = self.pka_prime + self.alpha * torch.log10(i_pos / i_neg)
        
        w_map_mean = w_map.mean(dim=1, keepdim=True)
        log_var = self.fc_var(w_map_mean)
        
        return mu, log_var, w_map

class SwinLinearHeadModel(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, window_size: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.backbone = SwinBackbone1D(in_channels, embed_dim, patch_size, window_size, num_heads, num_layers)
        self.fc_mu = nn.Linear(embed_dim, 1)
        self.fc_var = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        pooled = feat.mean(dim=1)
        mu = self.fc_mu(pooled)
        log_var = self.fc_var(pooled)
        w_dummy = feat.new_zeros(feat.shape[0], feat.shape[1])
        return mu, log_var, w_dummy



class pHPredictionModel(nn.Module):
    def __init__(
        self,
        model_type: str = "hh_swin",
        in_channels: int = 1,
        embed_dim: int = 64,
        patch_size: int = 4,
        window_size: int = 8,
        num_heads: int = 4,
        num_layers: int = 3,
        hh_init_pka: float = 7.0,
        seq_len_after_patch: int = 350,
        learnable_v_prior: bool = True,
        prior_residual_scale: float = 5.0, # 5倍扩音器
        resnet_base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.model_type = model_type

        if model_type == "hh_swin":
            self.backbone = SwinBackbone1D(
                in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size,
                window_size=window_size, num_heads=num_heads, num_layers=num_layers,
            )
            self.hh_head = PhysicsInformedHHHead(embed_dim, init_pka=hh_init_pka)
            
            self.learnable_v_prior = learnable_v_prior
            self.prior_residual_scale = prior_residual_scale
            
            self.res_pos = nn.Parameter(torch.randn(seq_len_after_patch) * 0.01, requires_grad=learnable_v_prior)
            self.res_neg = nn.Parameter(torch.randn(seq_len_after_patch) * 0.01, requires_grad=learnable_v_prior)

        elif model_type == "swin_linear":
            self.swin_linear_model = SwinLinearHeadModel(
                in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size,
                window_size=window_size, num_heads=num_heads, num_layers=num_layers,
            )



    @staticmethod
    def align_prior_to_token_length(v_prior: torch.Tensor, target_len: int) -> torch.Tensor:
        if v_prior.dim() != 1:
            raise ValueError("v_prior 必须为一维张量")
        if v_prior.shape[0] == target_len:
            return v_prior
        vp = v_prior.view(1, 1, -1).float()
        vp = F.interpolate(vp, size=target_len, mode="nearest")
        return vp.view(-1)

    def forward(self, x: torch.Tensor, v_prior: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model_type == "hh_swin":
            feat = self.backbone(x)
            token_len = feat.shape[1]

            if v_prior is None:
                aligned_prior = torch.zeros(token_len, device=feat.device, dtype=feat.dtype)
            else:
                aligned_prior = self.align_prior_to_token_length(v_prior=v_prior, target_len=token_len)
                aligned_prior = aligned_prior.to(device=feat.device, dtype=feat.dtype)

            # 【核心架构跃迁：Soft Anchoring 软锚定】
            # 将外部传入的绝对硬先验 (1.0, -1.0) 整体缩放为 0.5，给 AI 留下上下探索的充足空间
            soft_anchor_scale = 0.3
            scaled_prior = aligned_prior * soft_anchor_scale
            
            # 使用 ReLU 分解为双通道非负基底 [0, 0.5]
            base_pos = F.relu(scaled_prior)
            base_neg = F.relu(-scaled_prior)

            if self.learnable_v_prior:
                r_pos = self.align_prior_to_token_length(self.res_pos, token_len).to(feat.device)
                r_neg = self.align_prior_to_token_length(self.res_neg, token_len).to(feat.device)
                
                # 双路独立残差累加
                learned_pos = base_pos + self.prior_residual_scale * torch.tanh(r_pos)
                learned_neg = base_neg + self.prior_residual_scale * torch.tanh(r_neg)
                
                # 铁闸依然设在 1.0，这意味着 AI 最高可以把 0.5 的峰放大到 1.0
                m_pos = torch.clamp(learned_pos, min=0.0, max=1.0)
                m_neg = torch.clamp(learned_neg, min=0.0, max=1.0)
            else:
                m_pos, m_neg = base_pos, base_neg

            return self.hh_head(feat, m_pos, m_neg)

        if self.model_type == "swin_linear":
            return self.swin_linear_model(x)

        return self.resnet_model(x)

def build_model(model_cfg: dict) -> pHPredictionModel:
    model_type = model_cfg.get("model_type", "hh_swin")
    in_channels = int(model_cfg.get("in_channels", 1))
    embed_dim = int(model_cfg.get("embed_dim", 64))
    patch_size = int(model_cfg.get("patch_size", 4))
    window_size = int(model_cfg.get("window_size", 8))
    num_heads = int(model_cfg.get("num_heads", 4))
    num_layers = int(model_cfg.get("num_layers", 3))
    
    hh_init_pka = float(model_cfg.get("hh_init_pka", 7.0))
    seq_len_after_patch = int(model_cfg.get("seq_len_after_patch", 350))
    learnable_v_prior = bool(model_cfg.get("learnable_v_prior", True))
    prior_residual_scale = float(model_cfg.get("prior_residual_scale", 5.0)) 
    
    resnet_base_channels = int(model_cfg.get("resnet_base_channels", 64))

    model = pHPredictionModel(
        model_type=model_type,
        in_channels=in_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        window_size=window_size,
        num_heads=num_heads,
        num_layers=num_layers,
        hh_init_pka=hh_init_pka,
        seq_len_after_patch=seq_len_after_patch,
        learnable_v_prior=learnable_v_prior,
        prior_residual_scale=prior_residual_scale,
        resnet_base_channels=resnet_base_channels,
    )
    return model