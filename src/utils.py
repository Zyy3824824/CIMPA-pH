"""通用工具函数。"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml



def set_seed(seed: int) -> None:
    """设置随机种子，保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    return cfg


def ensure_dir(path: str) -> None:
    """确保目录存在。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    """保存 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_regression_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> Dict[str, float]:
    """计算常见回归指标（MAE、RMSE、R2）。"""
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()

    return {"mae": mae, "rmse": rmse, "r2": r2}


def compute_ensemble_uncertainty(
    mus: torch.Tensor,
    log_vars: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算集成均值、总方差与总标准差。

    Args:
        mus: [M, B, 1]
        log_vars: [M, B, 1]

    Returns:
        mu_ens: [B, 1]
        var_total: [B, 1]
        std_total: [B, 1]
    """
    mu_ens = mus.mean(dim=0)
    aleatoric_var = torch.exp(log_vars).mean(dim=0)
    epistemic_var = mus.var(dim=0, unbiased=False)
    var_total = aleatoric_var + epistemic_var
    std_total = torch.sqrt(var_total + 1e-12)
    return mu_ens, var_total, std_total


def get_device() -> torch.device:
    """获取训练设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_init_wandb(config: Dict[str, Any], run_name: str) -> Any:
    """按需初始化 wandb。"""
    use_wandb = bool(config.get("train", {}).get("use_wandb", False))
    if not use_wandb:
        return None

    try:
        import wandb  # type: ignore

        wandb.init(project=config["project"]["name"], name=run_name, config=config)
        return wandb
    except Exception:
        return None


def resolve_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """解析并创建核心输出目录。"""
    ckpt_dir = config["project"].get("checkpoint_dir", "checkpoints")
    log_dir = config["project"].get("log_dir", "logs")
    output_dir = config["project"].get("output_dir", "outputs")

    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)
    ensure_dir(output_dir)

    return {"ckpt_dir": ckpt_dir, "log_dir": log_dir, "output_dir": output_dir}


def generate_chemical_prior(
    wavenumbers: np.ndarray,
    window: float = 10.0,
    pos_peaks: List[float] | None = None,
    neg_peaks: List[float] | None = None,
) -> torch.Tensor:
    """根据波数轴生成化学先验向量 v_prior（取值为 +1/-1/0）。"""
    if pos_peaks is None:
        pos_peaks = [589.0, 681.0, 834.0]
    if neg_peaks is None:
        neg_peaks = [1252.0, 1437.0]

    v_prior = np.zeros_like(wavenumbers, dtype=np.float32)

    for peak in pos_peaks:
        mask = (wavenumbers >= peak - window) & (wavenumbers <= peak + window)
        v_prior[mask] = 1.0

    for peak in neg_peaks:
        mask = (wavenumbers >= peak - window) & (wavenumbers <= peak + window)
        v_prior[mask] = -1.0

    return torch.from_numpy(v_prior)


def build_v_prior_from_config(config: Dict[str, Any]) -> torch.Tensor:
    """从配置构造光谱轴并生成化学先验向量。"""
    data_cfg = config.get("data", {})
    prior_cfg = config.get("prior", {})

    signal_length = int(data_cfg.get("signal_length", 1401))
    x_min = float(prior_cfg.get("x_min", 200.0))
    x_max = float(prior_cfg.get("x_max", 1600.0))
    window = float(prior_cfg.get("window", 10.0))

    pos_peaks = [float(p) for p in prior_cfg.get("pos_peaks", [589.0, 681.0, 834.0])]
    neg_peaks = [float(p) for p in prior_cfg.get("neg_peaks", [1252.0, 1437.0])]

    wavenumbers = np.linspace(x_min, x_max, signal_length, dtype=np.float32)
    return generate_chemical_prior(
        wavenumbers=wavenumbers,
        window=window,
        pos_peaks=pos_peaks,
        neg_peaks=neg_peaks,
    )
