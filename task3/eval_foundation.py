#!/usr/bin/env python
# 从t3.4移动到task3
"""
统一评估脚本，支持 Depth Anything V2 (相对深度) 和 ZoeDepth (metric深度)
"""
from __future__ import annotations
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math
from pathlib import Path
from typing import List, Callable
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入你自己的数据集和指标
from scannet_dataset import ScanNetDepthDataset
from metrics import abs_rel_metric, solve_scale_shift

# 导入基础模型包装类
# 注意：请根据实际路径修改导入
import sys
sys.path.append('/225040103/t3.4/d_any/Depth-Anything-V2')
sys.path.append('/225040103/t3.4/d_zoe/ZoeDepth')
# 导入包装类
from depth_anything_wrapper import DepthAnythingV2Wrapper
from zoe_wrapper import ZoeDepthWrapper


def parse_args():
    parser = argparse.ArgumentParser("Evaluate foundation depth models on ScanNet")
    # 数据参数
    parser.add_argument("--scannet_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--vis_dir", type=str, default=None, help="Directory to save visualizations")

    # 模型选择
    parser.add_argument("--model", type=str, required=True, choices=['da2', 'zoe'],
                        help="Foundation model: da2 (Depth Anything V2), zoe (ZoeDepth)")

    # 模型特定参数
    parser.add_argument("--da2_ckpt", type=str, default="/225040103/t3.4/d_any/checkpoints/depth_anything_v2_vitb.pth")
    parser.add_argument("--zoe_ckpt", type=str, default="/225040103/t3.4/d_zoe/ZoeDepth/checkpoints/ZoeD_M12_N.pt")
    return parser.parse_args()

def read_scene_file(path: str) -> List[str]:
    scenes = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        scenes.append(line)
    return scenes

def build_dataloader(args):
    scenes = read_scene_file(args.split_file)
    print(f"Evaluation scenes: {len(scenes)}")
    dataset = ScanNetDepthDataset(
        scannet_root=args.scannet_root,
        scenes=scenes,
        image_size=(args.image_height, args.image_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        augment=False,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return loader

@torch.no_grad()
def evaluate_model(dataloader, inference_fn, model_type, device, vis_dir=None):
    """
    inference_fn: callable, input (B,3,H,W) tensor on GPU, output (B,1,H,W) depth
    model_type: 'relative' or 'metric'
    """
    total_abs_rel = 0.0
    count = 0

    # 用于可视化的反归一化参数（与训练时一致）
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)

    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        image = batch["image"].to(device, non_blocking=True)
        depth_gt = batch["depth"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        # 模型推理
        pred = inference_fn(image)
        if isinstance(pred, dict):
            # 如果返回的是字典，假设键名为 'metric_depth' 或 'depth'
            pred = pred.get('metric_depth', pred.get('depth', pred))  # (B,1,H,W)

        # 对齐（如果需要）
        if model_type == 'relative':
            pred = solve_scale_shift(pred, depth_gt, valid_mask)

        # 计算 AbsRel
        abs_rel = abs_rel_metric(pred, depth_gt, valid_mask)
        if math.isnan(abs_rel):
            continue
        total_abs_rel += abs_rel
        count += 1

        if i % 20 == 0:
            print(f"[{model_type}] step {i}/{len(dataloader)} abs_rel={abs_rel:.4f}")

            # 可视化
            if vis_dir:
                # 反归一化图像
                img = image[0] * std + mean
                rgb = img.cpu().numpy().transpose(1,2,0).clip(0,1)
                pred_depth = pred[0,0].cpu().numpy()
                gt_depth = depth_gt[0,0].cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(15,5))
                axes[0].imshow(rgb)
                axes[0].set_title('RGB')
                axes[1].imshow(pred_depth, cmap='viridis')
                axes[1].set_title('Predicted Depth')
                axes[2].imshow(gt_depth, cmap='viridis')
                axes[2].set_title('GT Depth')
                plt.savefig(os.path.join(vis_dir, f'sample_{i}.png'))
                plt.close()

    if count == 0:
        return float("inf")
    return total_abs_rel / count

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 根据模型选择加载对应的包装类
    if args.model == 'da2':
        wrapper = DepthAnythingV2Wrapper(model_path=args.da2_ckpt)
        inference_fn = wrapper.predict_batch
        model_type = 'relative'
        model_name = "Depth Anything V2"
    elif args.model == 'zoe':
        wrapper = ZoeDepthWrapper(model_path=args.zoe_ckpt)
        inference_fn = wrapper.predict_batch
        model_type = 'metric'
        model_name = "ZoeDepth"
    else:
        raise ValueError("Unknown model")

    dataloader = build_dataloader(args)
    abs_rel = evaluate_model(dataloader, inference_fn, model_type, device, vis_dir=args.vis_dir)

    print(f"\n=== Evaluation Results for {model_name} ===")
    print(f"abs_rel: {abs_rel:.6f}")

    if args.save_json:
        metrics = {"abs_rel": float(abs_rel), "model": args.model}
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"Saved metrics to: {save_path}")

if __name__ == "__main__":
    main()
    
    
#python eval_foundation.py \    --scannet_root /path/to/scannet \    --split_file /path/to/scannetv2_val.txt \    --model da2 \    --da2_ckpt /225040103/t3.4/d_any/checkpoints/depth_anything_v2_vitb.pth \    --batch_size 4 \    --vis_dir ./vis_da2 \    --save_json ./results_da2.json
#
#
#python eval_foundation.py \    --scannet_root /path/to/scannet \    --split_file /path/to/scannetv2_val.txt \    --model zoe \    --zoe_ckpt /225040103/t3.4/d_zoe/ZoeDepth/checkpoints/ZoeD_M12_N.pt \    --batch_size 4 \    --vis_dir ./vis_zoe \    --save_json ./results_zoe.json
#
#
#
#
