# CIMPA-pH

Chemically Informed AI Decoding of Full-Spectrum SERS Fingerprints for pH-Guided Intraoperative Melanoma Margin Delineation

## 1D Swin Transformer + Physics-Informed HH Head + Deep Ensemble

本项目用于 1D 拉曼信号回归（pH 预测）。

## 关键特性

- 主干：1D Swin Transformer（3层，W-MSA/SW-MSA交替）
- 解码头：Physics-Informed HH Head（Henderson-Hasselbalch 约束）
- 输出：
  - 预测均值 \(\mu\)
  - 偶然不确定性对数方差 \(\log\sigma^2\)
  - 动态强度图 \(W\)（可解释性）
- 不确定性：Deep Ensemble（默认 M=5）

\[
\text{std}_{total} = \sqrt{\mathbb{E}[\exp(\log\sigma^2)] + \mathrm{Var}(\mu)}
\]

## 目录结构

- `configs/default_config.yaml`：配置（含 `prior` 化学先验峰）
- `src/dataset.py`：数据加载
- `src/model.py`：Swin 主干 + HH Head
- `src/loss.py`：异方差 Gaussian NLL
- `src/utils.py`：工具函数（含化学先验向量生成）
- `train.py`：单模型训练（记录 `pka_prime`、`alpha`）
- `train_ensemble.sh`：顺序训练 5 个独立模型

## 环境安装

```bash
pip install -r requirements.txt
```

## 单模型训练

```bash
python train.py --config configs/default_config.yaml --model_id 1 --seed 42
```

训练日志会记录：
- `loss/mae/rmse/r2`
- `physics/pka_prime`
- `physics/alpha`

## 集成训练（5个模型）

```bash
bash train_ensemble.sh
```

## 集成推理

```bash
python inference.py --config configs/default_config.yaml --ckpt_dir checkpoints --ensemble_size 5
```

将保存：
- `outputs/mu_ens.npy`
- `outputs/std_total.npy`
- `outputs/y_true.npy`
- `outputs/w_ens.npy`（当 `--save_attention_map` 启用时）

## 化学先验配置

在 `configs/default_config.yaml` 的 `prior` 段中设置：
- `pos_peaks`: 去质子化增强峰（+1）
- `neg_peaks`: 质子化增强峰（-1）
- `window`: 邻域半宽（cm^-1）

默认已配置：
- 正相关峰：589, 681, 834
- 负相关峰：1252, 1437
