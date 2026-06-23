"""监督迁移学习入口。

使用 Ultralytics 原生 freeze 参数按“层”冻结，避免旧实现把层数与
named_parameters 的参数序号混为一谈。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "yolo_source"))

from ultralytics import YOLO


def train_stage(model: YOLO, args, epochs: int, name: str, freeze: int, lr0: float):
    return model.train(
        data=args.data,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=lr0,
        lrf=0.10,
        weight_decay=5e-4,
        warmup_epochs=2,
        cos_lr=True,
        freeze=freeze,
        patience=args.patience,
        mosaic=0.2,
        mixup=0.0,
        degrees=8.0,
        translate=0.05,
        scale=0.20,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.0,
        hsv_s=0.05,
        hsv_v=0.12,
        close_mosaic=10,
        amp=True,
    )


def main():
    parser = argparse.ArgumentParser(description="YOLO 监督迁移学习")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--freeze-layers", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--project", default="runs/transfer")
    parser.add_argument("--name", default="YOLO_transfer")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"权重不存在：{args.weights}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"数据配置不存在：{args.data}")
    if not 0 <= args.freeze_epochs <= args.epochs:
        raise ValueError("freeze-epochs 必须位于 [0, epochs] 范围内")

    model = YOLO(args.weights)
    remaining = args.epochs
    if args.freeze_epochs:
        print(f"阶段 1：冻结前 {args.freeze_layers} 层，训练 {args.freeze_epochs} 轮")
        stage1 = train_stage(
            model,
            args,
            epochs=args.freeze_epochs,
            name=f"{args.name}_stage1",
            freeze=args.freeze_layers,
            lr0=args.lr0 * 0.5,
        )
        stage1_best = Path(stage1.save_dir) / "weights" / "best.pt"
        model = YOLO(str(stage1_best))
        remaining -= args.freeze_epochs

    if remaining:
        print(f"阶段 2：解冻全部层，训练 {remaining} 轮")
        train_stage(
            model,
            args,
            epochs=remaining,
            name=args.name,
            freeze=0,
            lr0=args.lr0,
        )


if __name__ == "__main__":
    main()
