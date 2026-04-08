#!/usr/bin/env python
'''
用于训练单目深度估计模型的主脚本
基于 ScanNet 数据集，使用 ResNet50DepthModel 作为模型架构
通过尺度不变对数损失（SiLog loss）进行优化
并定期保存检查点。

'''
from __future__ import annotations
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# 尝试导入 tqdm 用于进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Install it with 'pip install tqdm' for progress bars.")

# ===== ADDED: 导入 matplotlib 用于绘图
import matplotlib.pyplot as plt
# ===== END ADDED

from depth_model import ResNet50DepthModel
from scannet_dataset import ScanNetDepthDataset, build_train_val_scenes



def _parse_args() -> argparse.Namespace:
    '''
        解析命令行参数
        指定 ScanNet 根目录 --scannet_root、输出目录 --output_dir 和训练集划分文件 --train_split_file（如 scannetv2_train.txt）。
        可配置训练轮数、批次大小、学习率、图像尺寸、深度范围等超参数。
        支持从检查点恢复训练 --resume 和禁用预训练主干 --no_pretrained_backbone。
    '''
    parser = argparse.ArgumentParser("ScanNet monocular depth training.")
    parser.add_argument("--scannet_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_split_file", type=str, required=True, help="scannetv2_train.txt")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=10.0)

    parser.add_argument("--max_train_samples", type=int, default=10000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 加速设置：关闭确定性，启用 benchmark
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _to_device(batch: Dict[str, object], device: torch.device) -> Tuple[torch.Tensor, ...]:
    image = batch["image"].to(device, non_blocking=True)
    depth = batch["depth"].to(device, non_blocking=True)
    valid_mask = batch["valid_mask"].to(device, non_blocking=True)
    return image, depth, valid_mask


def _silog_loss(

# 尺度不变对数损失。对每个样本的有效像素计算 log(depth) 的方差
# 并引入variance_focus 平衡方差与均值的影响。若有效像素太少（<16）则跳过该样本。

    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    variance_focus: float = 0.85,
) -> torch.Tensor:
    losses = []
    for b in range(pred.shape[0]):
        mask = valid_mask[b, 0]
        if mask.sum().item() < 16:
            continue
        log_diff = torch.log(torch.clamp(pred[b, 0][mask], min=1e-3)) - torch.log(
            torch.clamp(target[b, 0][mask], min=1e-3)
        )
        mean = log_diff.mean()
        sq = (log_diff * log_diff).mean()
        silog = torch.sqrt(torch.clamp(sq - variance_focus * mean * mean, min=1e-6))
        losses.append(silog)

    if not losses:
        return pred.sum() * 0.0

    return torch.stack(losses).mean()



def _save_checkpoint(
    
   # _save_checkpoint：保存模型、优化器状态及当前 epoch 和最佳指标。
    
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_abs_rel: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_abs_rel": best_abs_rel,
            "args": vars(args),
        },
        path,
    )

####？？？？？？？？？？？？？？？？？
def _append_log_row(log_path: Path, row: Dict[str, float]) -> None:
    is_new = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def main() -> None:

# 数据准备
# 通过 build_train_val_scenes 函数从划分文件中读取训练场景列表。
# 构建 ScanNetDepthDataset 训练集，启用数据增强（augment=True），并限制最大样本数 --max_train_samples（默认 10000）。
# 创建 DataLoader，设置多线程加载、打乱、锁页内存等。

    args = _parse_args()
    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    log_path = output_dir / "train_log.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (args.image_height, args.image_width)

    train_scenes = build_train_val_scenes(
        scannet_root=args.scannet_root,
        split_file=args.train_split_file,
    )

    print(f"Train scenes: {len(train_scenes)}")

    train_dataset = ScanNetDepthDataset(
        scannet_root=args.scannet_root,
        scenes=train_scenes,
        image_size=image_size,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        augment=False,
        max_samples=args.max_train_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=args.num_workers > 0,  # 加速：保持 worker 进程
    )
    '''
    模型与优化器
    实例化 ResNet50DepthModel，可选择是否加载 ImageNet 预训练权重。
    使用 AdamW 优化器和余弦退火学习率调度器 CosineAnnealingLR。
    '''
    model = ResNet50DepthModel(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        pretrained_backbone=not args.no_pretrained_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1)

    # 加速：混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    start_epoch = 0
    best_loss = float("inf")
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_loss  = float(ckpt.get("best_abs_rel", best_loss ))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

        '''
        训练循环
    对每个 epoch：
        遍历数据加载器，将图像、深度、有效掩码移至 GPU。
        前向传播得到预测深度，计算 SiLog 损失。
        反向传播，更新模型参数。
        每 50 步打印一次当前损失。
    epoch 结束后更新学习率，保存当前检查点（包含模型状态、优化器状态、最佳指标等）。
        '''
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        valid_steps = 0
        tic = time.time()
        
        # 使用进度条包装数据加载器
        if TQDM_AVAILABLE:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}", unit="batch", leave=False)
        else:
            pbar = train_loader
            print(f"Epoch {epoch+1:03d} starting...")

        for step, batch in enumerate(pbar):
            image, depth, valid_mask = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            # 混合精度前向传播========
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(image)
                loss = _silog_loss(pred, depth, valid_mask)

            if not torch.isfinite(loss):
                print(f"Skip non-finite loss at step {step}: {float(loss.item())}")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            valid_steps += 1
            
            # 如果是 tqdm 进度条，更新后处理信息
            if TQDM_AVAILABLE:
                pbar.set_postfix(loss=float(loss.item()))

        scheduler.step()
        mean_train_loss = epoch_loss / max(1, valid_steps)
        lr = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - tic
        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={mean_train_loss:.4f} | "
            f"time={elapsed:.1f}s"
        )
        if mean_train_loss < best_loss:
            best_loss = mean_train_loss
            _save_checkpoint(
                output_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_abs_rel=best_loss,
                args=args,
            )

        # 记录日志（CSV）
        _append_log_row(
            log_path,
            {
                "epoch": epoch + 1,
                "train_loss": mean_train_loss,
                "lr": lr,
                "seconds": elapsed,
            },
        )

        '''
        检查点保存
        每个 epoch 保存一个 .pth 文件到输出目录，命名格式为 {epoch}.pth。
        同时将命令行参数保存为 args.json。
        '''
        _save_checkpoint(
            output_dir / f"epoch_{epoch + 1:03d}.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_abs_rel =best_loss ,
            args=args,
        )
    # ===== ADDED: 训练结束后绘制损失曲线
    if log_path.exists():
        epochs = []
        losses = []
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                losses.append(float(row['train_loss']))

        plt.figure(figsize=(8,5))
        plt.plot(epochs, losses, 'b-o', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('SiLog Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        curve_path = output_dir / "loss_curve.png"
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"Loss curve saved to {curve_path}")
    # ===== END ADDED

if __name__ == "__main__":
    main()
