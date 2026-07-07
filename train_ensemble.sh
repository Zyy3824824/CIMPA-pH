#!/usr/bin/env bash

# HH-Head + Deep Ensemble 训练脚本
# 顺序训练 5 个独立模型，每次仅训练 1 个模型（工程与显存更稳健）。

set -e

CONFIG_PATH="configs/default_config.yaml"
SEEDS=(42 52 62 72 82)

for i in {1..5}
do
  seed=${SEEDS[$((i-1))]}
  echo "[Ensemble] Start training model_${i} with seed=${seed}"
  python train.py --config ${CONFIG_PATH} --model_id ${i} --seed ${seed}
  echo "[Ensemble] Finished model_${i}"
  echo "---------------------------------------------"
done

echo "All ensemble members are trained."
