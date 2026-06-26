"""YOLO26s SAR 单域监督基线。

该脚本用于建立必须保留的对照组。只有当 DA 模型稳定超过该基线时，
才能说明域对抗确实有效。

默认参数按约 12GB 可用显存和最新独立集尺度实验设置：
imgsz=576、batch=4、workers=2、nbs=16。
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "yolo_source"))

from ultralytics import YOLO

LOGGER = logging.getLogger("train_yolo26s")


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _clear_cuda_cache(reason: str):
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    LOGGER.info("%s：已清理 CUDA 缓存。", reason)


def main():
    _configure_logging()
    parser = argparse.ArgumentParser(description="YOLO26s SAR 单域监督基线")
    parser.add_argument("--data", default=str(ROOT / "dataset_sar_only.yaml"))
    parser.add_argument("--weights", default="yolo26s.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=576)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nbs", type=int, default=16)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="YOLO26s_SAR_baseline")
    parser.add_argument("--optimizer", choices=("AdamW", "Adam", "SGD"), default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--mosaic", type=float, default=0.25)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--multi-scale", type=float, default=0.10)
    parser.set_defaults(clear_cache_each_epoch=True)
    parser.add_argument("--clear-cache-each-epoch", dest="clear_cache_each_epoch", action="store_true")
    parser.add_argument("--no-clear-cache-each-epoch", dest="clear_cache_each_epoch", action="store_false")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"数据集配置不存在：{args.data}")

    model = YOLO(args.weights)
    if args.clear_cache_each_epoch:
        model.add_callback(
            "on_fit_epoch_end",
            lambda trainer: _clear_cuda_cache(f"第 {trainer.epoch + 1} 轮训练/验证结束"),
        )
    LOGGER.info(
        "启动 SAR 单域基线训练：imgsz=%d，batch=%d，workers=%d，nbs=%d，每轮清理缓存=%s",
        args.imgsz,
        args.batch,
        args.workers,
        args.nbs,
        args.clear_cache_each_epoch,
    )
    if args.imgsz >= 768:
        LOGGER.warning("最新尺度评测显示 SAR-only 在独立集上 512/576 优于 768/896；若只是建立基线，建议优先使用默认 576。")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        nbs=args.nbs,
        cos_lr=True,
        patience=args.patience,
        save_period=args.save_period,
        degrees=10,
        translate=0.08,
        scale=0.30,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.0,
        hsv_s=0.05,
        hsv_v=0.15,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        multi_scale=args.multi_scale,
        cache=False,
        amp=True,
    )


if __name__ == "__main__":
    main()
