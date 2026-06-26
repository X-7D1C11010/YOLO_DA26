#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成域均衡的光学/SAR 域自适应训练 YAML。

当前 `dataset_aircraft_da.yaml` 中 optical:SAR 约为 1662:21765，普通随机采样会让
训练 batch 主要由 SAR 图像组成，域对抗分支难以获得稳定的双域信号。本脚本通过
重复采样 optical aircraft-only 图像，生成一个新的 train txt 和 YAML，不修改原始数据。

推荐用法：

    python make_balanced_da_dataset.py \
      --data dataset_aircraft_da.yaml \
      --target-optical-ratio 0.5 \
      --output-yaml dataset_aircraft_da_balanced.yaml

随后训练：

    python train_26.py --data dataset_aircraft_da_balanced.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
YOLO_SOURCE = ROOT / "yolo_source"
if str(YOLO_SOURCE) not in sys.path:
    sys.path.insert(0, str(YOLO_SOURCE))

from ultralytics.utils import YAML  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OPTICAL_TOKENS = ("/vis", "/optical", "/rgb", "vis_aircraft_singleclass")
SAR_TOKENS = ("/sar", "sar_", "_sar", "synthetic_aperture")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _resolve_base(data_path: Path, config: dict[str, Any]) -> Path:
    raw = config.get("path") or data_path.parent
    base = Path(str(raw)).expanduser()
    if not base.is_absolute():
        base = data_path.parent / base
    return base.absolute()


def _resolve_entry(base: Path, data_path: Path, raw_entry: str) -> Path:
    path = Path(raw_entry).expanduser()
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate.absolute()
    return (data_path.parent / path).absolute()


def _collect_images(entry: Path) -> list[Path]:
    if entry.is_file() and entry.suffix.lower() == ".txt":
        images: list[Path] = []
        for line in entry.read_text(encoding="utf-8", errors="ignore").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            image = Path(value).expanduser()
            if not image.is_absolute():
                image = entry.parent / image
            if image.suffix.lower() in IMAGE_SUFFIXES:
                images.append(image.absolute())
        return images
    if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
        return [entry.absolute()]
    if entry.is_dir():
        return sorted(path.absolute() for path in entry.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return []


def _domain_of(path: Path) -> str:
    text = path.as_posix().lower()
    if any(token in text for token in SAR_TOKENS):
        return "sar"
    if any(token in text for token in OPTICAL_TOKENS):
        return "optical"
    return "unknown"


def _calculate_repeat(optical_count: int, sar_count: int, target_optical_ratio: float, max_repeat: int) -> int:
    if optical_count <= 0 or sar_count <= 0:
        return 1
    target_optical_ratio = min(max(target_optical_ratio, 0.01), 0.95)
    repeat = math.ceil(target_optical_ratio * sar_count / (optical_count * (1.0 - target_optical_ratio)))
    return max(1, min(repeat, max_repeat))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成 optical/SAR 域均衡训练 YAML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default=str(ROOT / "dataset_aircraft_da.yaml"), help="原始 DA 数据集 YAML")
    parser.add_argument("--output-yaml", default=str(ROOT / "dataset_aircraft_da_balanced.yaml"), help="输出的新 YAML")
    parser.add_argument(
        "--train-list",
        default=str(ROOT / "runs" / "datasets" / "da_balanced" / "train_balanced.txt"),
        help="输出的均衡 train txt",
    )
    parser.add_argument("--summary", default=str(ROOT / "runs" / "datasets" / "da_balanced" / "summary.json"), help="输出统计摘要")
    parser.add_argument("--target-optical-ratio", type=float, default=0.5, help="均衡后希望 optical 在 train 中占比")
    parser.add_argument("--max-repeat", type=int, default=12, help="optical 图像最大重复次数，避免训练集过大")
    parser.add_argument("--max-sar", type=int, default=0, help="最多保留多少张 SAR 图像；0 表示保留全部")
    parser.add_argument("--seed", type=int, default=0, help="SAR 子采样随机种子")
    parser.add_argument("--shuffle", action="store_true", help="输出前打乱 train txt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_path = Path(args.data).expanduser().absolute()
    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置不存在：{data_path}")

    config = YAML.load(data_path)
    base = _resolve_base(data_path, config)
    train_entries = _as_list(config.get("train"))

    optical: list[Path] = []
    sar: list[Path] = []
    unknown: list[Path] = []
    for raw_entry in train_entries:
        entry = _resolve_entry(base, data_path, raw_entry)
        for image in _collect_images(entry):
            domain = _domain_of(image)
            if domain == "optical":
                optical.append(image)
            elif domain == "sar":
                sar.append(image)
            else:
                unknown.append(image)

    optical = list(dict.fromkeys(optical))
    sar = list(dict.fromkeys(sar))
    unknown = list(dict.fromkeys(unknown))

    if not optical:
        raise RuntimeError("未识别到 optical 图像，请检查 YAML 路径是否包含 VIS/optical/RGB 等关键词。")
    if not sar:
        raise RuntimeError("未识别到 SAR 图像，请检查 YAML 路径是否包含 SAR 等关键词。")

    rng = random.Random(args.seed)
    if args.max_sar > 0 and len(sar) > args.max_sar:
        sar = sorted(rng.sample(sar, args.max_sar))

    repeat = _calculate_repeat(len(optical), len(sar), args.target_optical_ratio, args.max_repeat)
    balanced = sar + optical * repeat + unknown
    if args.shuffle:
        rng.shuffle(balanced)

    train_list = Path(args.train_list).expanduser().absolute()
    train_list.parent.mkdir(parents=True, exist_ok=True)
    train_list.write_text("\n".join(path.as_posix() for path in balanced) + "\n", encoding="utf-8")

    output_yaml = Path(args.output_yaml).expanduser().absolute()
    output_config = dict(config)
    output_config["train"] = str(train_list)
    output_config["path"] = str(base)
    YAML.save(
        output_yaml,
        output_config,
        header="# 由 make_balanced_da_dataset.py 生成：用于 optical/SAR 域均衡 DA 训练。\n",
    )

    optical_after = len(optical) * repeat
    total_after = len(balanced)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_yaml": str(data_path),
        "output_yaml": str(output_yaml),
        "train_list": str(train_list),
        "source_counts": {
            "optical": len(optical),
            "sar": len(sar),
            "unknown": len(unknown),
        },
        "repeat_optical": repeat,
        "balanced_counts": {
            "optical": optical_after,
            "sar": len(sar),
            "unknown": len(unknown),
            "total": total_after,
            "optical_ratio": optical_after / total_after if total_after else 0.0,
        },
        "params": {
            "target_optical_ratio": args.target_optical_ratio,
            "max_repeat": args.max_repeat,
            "max_sar": args.max_sar,
            "seed": args.seed,
            "shuffle": args.shuffle,
        },
    }
    summary_path = Path(args.summary).expanduser().absolute()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n已生成域均衡训练配置：{output_yaml}")
    print(f"下一步可运行：python train_26.py --data {output_yaml} --imgsz 704 --batch 4")


if __name__ == "__main__":
    main()
