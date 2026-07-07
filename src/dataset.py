"""数据集与 DataLoader 定义。

支持两种模式：
1) dummy：用于流程联调
2) real：读取真实数据集（X_pre.dat + y.npy + split.json + meta.json）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class RealSignalDataset(Dataset):
    """真实 1D 信号数据集。

    读取：
    - X_pre.dat: float32, shape=[N, L]
    - y.npy: shape=[N]
    - split.json: train/val/test 索引
    - meta.json: N, L 等信息

    输出：
    - x: [C, L]，默认 C=1
    - y: [1]
    """

    def __init__(
        self,
        x_memmap: np.memmap,
        y_array: np.ndarray,
        indices: np.ndarray,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("当前真实数据读取仅支持 in_channels=1。")

        self.x_memmap = x_memmap
        self.y_array = y_array
        self.indices = indices.astype(np.int64)
        self.in_channels = in_channels

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])

        # x: [L] -> [1, L]
        # memmap 切片通常是只读视图，这里 copy 一份避免不可写告警
        x_np = np.asarray(self.x_memmap[real_idx], dtype=np.float32).copy()
        x = torch.from_numpy(x_np).unsqueeze(0)

        # y: 标量 -> [1]
        y_value = np.float32(self.y_array[real_idx])
        y = torch.tensor([y_value], dtype=torch.float32)
        return x, y


@dataclass
class DataBundle:
    """统一返回训练/验证/测试 DataLoader。"""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

def _build_real_dataloaders(config: Dict[str, Any]) -> DataBundle:
    data_cfg = config["data"]

    dataset_root = Path(data_cfg.get("dataset_root", "dataset"))
    meta_path = dataset_root / "meta.json"
    split_path = dataset_root / "split.json"
    x_path = dataset_root / "X_pre.dat"
    y_path = dataset_root / "y.npy"

    if not (meta_path.exists() and split_path.exists() and x_path.exists() and y_path.exists()):
        raise FileNotFoundError(
            "真实数据文件不完整，请确认 dataset_root 下存在 meta.json/split.json/X_pre.dat/y.npy"
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_samples = int(meta["N"])
    signal_length = int(meta["L"])

    x_memmap = np.memmap(
        x_path,
        dtype=np.float32,
        mode="r",
        shape=(n_samples, signal_length),
    )
    y_array = np.load(y_path).astype(np.float32)

    if y_array.shape[0] != n_samples:
        raise ValueError(f"y 长度 {y_array.shape[0]} 与 meta.N={n_samples} 不一致。")

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    in_channels = int(data_cfg.get("in_channels", 1))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    train_ds = RealSignalDataset(x_memmap, y_array, train_idx, in_channels=in_channels)
    val_ds = RealSignalDataset(x_memmap, y_array, val_idx, in_channels=in_channels)
    test_ds = RealSignalDataset(x_memmap, y_array, test_idx, in_channels=in_channels)

    return DataBundle(
        train_loader=DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        val_loader=DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        test_loader=DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    )


def build_dataloaders(config: Dict[str, Any], seed: int = 42) -> DataBundle:
    """根据配置创建 DataLoader。"""
    dataset_type = config["data"].get("dataset_type", "dummy").lower()

    if dataset_type == "real":
        return _build_real_dataloaders(config=config)

    raise ValueError("dataset_type 仅支持 'dummy' 或 'real'。")
