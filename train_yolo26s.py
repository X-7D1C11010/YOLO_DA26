"""YOLO26s SAR 单域监督基线。

该脚本用于建立必须保留的对照组。只有当 DA 模型稳定超过该基线时，
才能说明域对抗确实有效。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "yolo_source"))

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO26s SAR 单域监督基线")
    parser.add_argument("--data", default=str(ROOT / "dataset_sar_only.yaml"))
    parser.add_argument("--weights", default="yolo26s.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="YOLO26s_SAR_baseline")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"数据集配置不存在：{args.data}")

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project="runs/detect",
        name=args.name,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.05,
        cos_lr=True,
        patience=40,
        degrees=10,
        translate=0.08,
        scale=0.30,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.0,
        hsv_s=0.05,
        hsv_v=0.15,
        mosaic=0.5,
        close_mosaic=20,
        amp=True,
    )


if __name__ == "__main__":
    main()
