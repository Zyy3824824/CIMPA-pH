"""单模型训练脚本。

说明：
- 本脚本一次只训练 1 个模型。
- 通过 --seed 与 --model_id 控制不同成员模型训练。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.loss import GaussianNLLLoss
from src.model import build_model
from src.utils import (
    build_v_prior_from_config,
    compute_regression_metrics,
    get_device,
    load_config,
    maybe_init_wandb,
    resolve_paths,
    save_json,
    set_seed,
)

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = value < (self.best - self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: GaussianNLLLoss,
    mse_criterion: nn.MSELoss,
    device: torch.device,
    v_prior: torch.Tensor,
    optimizer: Optional[AdamW] = None,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    desc: str = "train",
    loss_mode: str = "nll",
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    total_loss = 0.0
    all_mu = []
    all_target = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                mu, log_var, _ = model(x, v_prior=v_prior)
                
                mu = mu.view_as(y)
                log_var = log_var.view_as(y)

                if loss_mode == "mse":
                    loss = mse_criterion(mu, y)
                    loss += 0.01 * torch.mean(log_var ** 2) 
                else:
                    loss = criterion(mu, log_var, y)

            if is_train and optimizer is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * x.size(0)
        all_mu.append(mu.detach().cpu())
        all_target.append(y.detach().cpu())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = total_loss / len(loader.dataset)
    y_pred = torch.cat(all_mu, dim=0)
    y_true = torch.cat(all_target, dim=0)
    metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)
    return epoch_loss, metrics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练单个 1D Swin 不确定性模型")
    parser.add_argument("--config", type=str, default="configs\\resnet1d.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_id", type=int, default=1)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config)

    set_seed(args.seed)
    device = get_device()

    run_name = f"model_{args.model_id}_{config['model']['model_type']}_seed_{args.seed}"
    writer = SummaryWriter(log_dir=str(Path(paths["log_dir"]) / run_name))
    wandb = maybe_init_wandb(config, run_name=run_name)

    data_bundle = build_dataloaders(config=config, seed=args.seed)

    model = build_model(config["model"]).to(device)

    try:
        v_prior = build_v_prior_from_config(config).to(device=device, dtype=torch.float32)
    except Exception:
        v_prior = torch.zeros(config["model"].get("seq_len_after_patch", 350), device=device)

    train_cfg = config["train"]
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
    criterion = GaussianNLLLoss()
    mse_criterion = nn.MSELoss()
    
    # 【给双路先验残差发免死金牌】
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "res_pos" in name or "res_neg" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": float(train_cfg.get("weight_decay", 1e-4))},
            {"params": no_decay_params, "weight_decay": 0.0}
        ],
        lr=float(train_cfg.get("lr", 1e-3)),
    )
    
    scheduler = StepLR(
        optimizer,
        step_size=int(train_cfg.get("lr_step_size", 20)),
        gamma=float(train_cfg.get("lr_gamma", 0.1)),
    )

    early_stopping = EarlyStopping(
        patience=int(train_cfg.get("early_stopping_patience", 10)),
        min_delta=float(train_cfg.get("early_stopping_min_delta", 1e-4)),
    )

    best_ckpt_path = Path(paths["ckpt_dir"]) / f"model_{args.model_id}.pth"
    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_mae": [], "val_mae": [],
        "train_rmse": [], "val_rmse": [],
        "train_r2": [], "val_r2": [],
        "pka_prime": [], "alpha": [],
    }

    epochs = int(train_cfg.get("epochs", 50))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    use_amp = bool(train_cfg.get("use_amp", False))

    for epoch in range(1, epochs + 1):
        loss_mode = "mse" if epoch <= warmup_epochs else "nll"

        train_loss, train_metrics = run_epoch(
            model=model, loader=data_bundle.train_loader, criterion=criterion,
            mse_criterion=mse_criterion, device=device, v_prior=v_prior,
            optimizer=optimizer, grad_clip=grad_clip, use_amp=use_amp,
            desc=f"train {epoch}/{epochs}", loss_mode=loss_mode,
        )

        val_loss, val_metrics = run_epoch(
            model=model, loader=data_bundle.val_loader, criterion=criterion,
            mse_criterion=mse_criterion, device=device, v_prior=v_prior,
            optimizer=None, grad_clip=grad_clip, use_amp=use_amp,
            desc=f"val {epoch}/{epochs}", loss_mode=loss_mode,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])
        history["train_rmse"].append(train_metrics["rmse"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["train_r2"].append(train_metrics["r2"])
        history["val_r2"].append(val_metrics["r2"])

        pka_prime_value = float(getattr(model, "hh_head", None).pka_prime.detach().cpu().item()) if hasattr(model, "hh_head") else 0.0
        alpha_value = float(getattr(model, "hh_head", None).alpha.detach().cpu().item()) if hasattr(model, "hh_head") else 0.0
        history["pka_prime"].append(pka_prime_value)
        history["alpha"].append(alpha_value)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("mae/train", train_metrics["mae"], epoch)
        writer.add_scalar("mae/val", val_metrics["mae"], epoch)
        writer.add_scalar("rmse/train", train_metrics["rmse"], epoch)
        writer.add_scalar("rmse/val", val_metrics["rmse"], epoch)
        writer.add_scalar("r2/train", train_metrics["r2"], epoch)
        writer.add_scalar("r2/val", val_metrics["r2"], epoch)
        writer.add_scalar("physics/pka_prime", pka_prime_value, epoch)
        writer.add_scalar("physics/alpha", alpha_value, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"[Epoch {epoch:03d}][{loss_mode.upper()}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_mae={train_metrics['mae']:.4f}, val_mae={val_metrics['mae']:.4f}, "
            f"train_r2={train_metrics['r2']:.4f}, val_r2={val_metrics['r2']:.4f}, "
            f"pka_prime={pka_prime_value:.2f}, alpha={alpha_value:.2f}, lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        scheduler.step()

        improved = early_stopping.step(val_metrics['mae'])
        if improved:
            best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "seed": args.seed,
                    "model_id": args.model_id,
                    "epoch": epoch,
                    "best_val_mae": val_metrics['mae'], 
                },
                best_ckpt_path,
            )
            print(f"🔥 保存最佳权重 (基于 Val MAE={val_metrics['mae']:.4f}) 到: {best_ckpt_path}")

        if early_stopping.should_stop:
            print("触发 Early Stopping，提前结束训练。")
            break

    history_path = Path(paths["output_dir"]) / f"history_model_{args.model_id}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(history, str(history_path))
    print(f"训练历史已保存: {history_path}")

    writer.close()
    if wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main()