"""光学-SAR 域对抗训练的统一入口。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "yolo_source"))

from ultralytics import YOLO


def build_parser(default_version: str = "26", default_size: str = "s") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO 光学-SAR 域对抗训练")
    parser.add_argument("--version", choices=("11", "26"), default=default_version, help="YOLO 架构版本")
    parser.add_argument("--size", choices=("n", "s", "m", "l", "x"), default=default_size, help="模型规模")
    parser.add_argument("--data", default=str(ROOT / "dataset.yaml"), help="单类别、同一类别映射的双域数据 YAML")
    parser.add_argument("--weights", default=None, help="初始化权重；默认使用对应官方预训练权重")
    parser.add_argument("--fresh", action="store_true", help="完全随机初始化，不加载预训练权重")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default=None)
    parser.add_argument("--domain-weight", type=float, default=0.02)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _base_model(trainer):
    model = trainer.model
    return model.module if hasattr(model, "module") else model


def _register_da_callbacks(model: YOLO, alpha_max: float) -> None:
    def set_alpha(trainer):
        total = max(int(trainer.args.epochs) - 1, 1)
        progress = min(max(float(trainer.epoch) / total, 0.0), 1.0)
        alpha = alpha_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
        base = _base_model(trainer)
        base.da_alpha = alpha
        trainer.da_alpha = alpha

    def report_domain_loss(trainer):
        base = _base_model(trainer)
        criterion = getattr(base, "criterion", None)
        domain_loss = getattr(criterion, "last_domain_loss", None)
        if domain_loss is not None:
            print(
                f"[DA] epoch={trainer.epoch + 1} "
                f"alpha={getattr(trainer, 'da_alpha', 0.0):.4f} "
                f"domain_loss={float(domain_loss):.6f}"
            )

    model.add_callback("on_train_epoch_start", set_alpha)
    model.add_callback("on_train_epoch_end", report_domain_loss)


def _load_pretrained(model: YOLO, weights: str) -> None:
    """加载预训练权重，并显式处理新增 DomainClassifier 导致的 Detect 层号偏移。"""
    source = YOLO(weights)
    model.model.load(source.model)

    source_head = next(
        (layer for layer in reversed(source.model.model) if layer.__class__.__name__ == "Detect"),
        None,
    )
    target_head = next(
        (layer for layer in reversed(model.model.model) if layer.__class__.__name__ == "Detect"),
        None,
    )
    if source_head is None or target_head is None:
        raise RuntimeError("无法在预训练模型或 DA 模型中定位 Detect 检测头。")

    source_state = source_head.state_dict()
    target_state = target_head.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    target_head.load_state_dict(compatible, strict=False)
    print(f"检测头显式迁移：{len(compatible)}/{len(target_state)} 个张量")


def run_training(args: argparse.Namespace):
    data_path = Path(args.data).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置不存在：{data_path}")
    if args.domain_weight < 0:
        raise ValueError("--domain-weight 不能小于 0")

    model_yaml = ROOT / "yolo_source" / "ultralytics" / "cfg" / "models" / args.version
    model_yaml = model_yaml / f"yolo{args.version}{args.size}-da.yaml"
    model = YOLO(str(model_yaml))

    if not args.fresh:
        weights = args.weights or f"yolo{args.version}{args.size}.pt"
        print(f"加载检测预训练权重：{weights}")
        _load_pretrained(model, weights)

    has_domain_classifier = any(
        layer.__class__.__name__ == "DomainClassifier" for layer in model.model.model
    )
    if not has_domain_classifier:
        raise RuntimeError("DA 模型中未发现 DomainClassifier，已中止以避免再次进行伪 DA 训练。")

    _register_da_callbacks(model, args.alpha_max)
    run_name = args.name or f"YOLO{args.version}{args.size}_DA_fixed"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"启动真实域对抗训练：model=YOLO{args.version}{args.size}, "
        f"imgsz={args.imgsz}, domain_weight={args.domain_weight}"
    )
    return model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        exist_ok=False,
        pretrained=not args.fresh,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=5e-4,
        warmup_epochs=5,
        cos_lr=True,
        domain_weight=args.domain_weight,
        patience=args.patience,
        save_period=args.save_period,
        seed=args.seed,
        deterministic=True,
        # SAR 方向/尺度变化明显；颜色扰动保持温和，避免破坏灰度散射统计。
        degrees=10.0,
        translate=0.08,
        scale=0.30,
        shear=2.0,
        perspective=0.0002,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.01,
        hsv_s=0.20,
        hsv_v=0.20,
        # 跨域 Mosaic/MixUp 会把一张合成图变成“混合域”，单一域标签因此失真。
        mosaic=0.0,
        mixup=0.0,
        close_mosaic=0,
        multi_scale=0.15,
        rect=False,
        amp=True,
    )


def main(default_version: str = "26", default_size: str = "s"):
    args = build_parser(default_version, default_size).parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
