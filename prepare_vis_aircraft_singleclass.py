#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把多类别 VIS 光学数据转换为单类别 aircraft 数据。

当前 `dataset.yaml` 使用 `nc: 1`，但 VIS/all/labels 中仍然包含 0~5 多个类别，
训练日志中已经出现大量：

    Label class X exceeds dataset class count 1

这会导致 VIS 图像被 Ultralytics 判为 corrupt 并丢弃，DA 训练等价于“伪双域训练”。

本脚本会：

1. 从 VIS 多类别标签中只保留 aircraft 类；
2. 把 aircraft 的类别编号重映射为 0；
3. 通过软链接/硬链接/复制复用原图像；
4. 生成符合 YOLO 单类别格式的新目录；
5. 可选写出新的 DA 数据 YAML。

根据当前 `train_sardet.yaml`，VIS 的 aircraft 类别是 1，因此默认 `--source-class 1`。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_to_label_path(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    """根据图像相对路径定位标签文件。"""

    rel = image_path.relative_to(images_root)
    return (labels_root / rel).with_suffix(".txt")


def read_and_remap_label(label_path: Path, source_class: int) -> Tuple[List[str], List[str]]:
    """读取标签，仅保留指定类别并重映射为 0。"""

    kept: List[str] = []
    errors: List[str] = []
    if not label_path.exists():
        return kept, [f"标签缺失：{label_path}"]

    for line_no, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split()
        if len(fields) < 5:
            errors.append(f"{label_path}:{line_no} 字段数不足")
            continue
        try:
            cls = int(float(fields[0]))
            coords = [float(value) for value in fields[1:5]]
        except ValueError:
            errors.append(f"{label_path}:{line_no} 非数字字段")
            continue
        if cls != source_class:
            continue
        if not all(0.0 <= value <= 1.0 for value in coords):
            errors.append(f"{label_path}:{line_no} 坐标越界")
            continue
        if coords[2] <= 0 or coords[3] <= 0:
            errors.append(f"{label_path}:{line_no} 宽高非正")
            continue
        # 保留除类别以外的额外字段，兼容可能存在的扩展标注。
        kept.append("0 " + " ".join(fields[1:]))
    return kept, errors


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    """链接或复制文件。"""

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "hardlink":
        os.link(src, dst)
        return "hardlink"
    if mode == "symlink":
        os.symlink(src, dst)
        return "symlink"

    # auto：优先软链接，再硬链接，最后复制。
    for candidate in ("symlink", "hardlink", "copy"):
        try:
            return link_or_copy(src, dst, candidate)
        except OSError:
            continue
    raise OSError(f"无法链接或复制文件：{src} -> {dst}")


def write_dataset_yaml(path: Path, dataset_root: Path, sar_train: str, sar_val: str, sar_test: str) -> None:
    """写出新的 DA 单类别 YAML。"""

    content = f"""# 自动生成：光学 aircraft 单类 + SAR aircraft 单类 DA 训练配置。
# VIS 光学标签已由 prepare_vis_aircraft_singleclass.py 重映射为 class 0。
path: {dataset_root.parent.as_posix()}
train:
  - {dataset_root.name}/images/train
  - {sar_train}
val: {sar_val}
test: {sar_test}

nc: 1
names:
  0: aircraft
"""
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""

    parser = argparse.ArgumentParser(description="VIS 多类别标签转 aircraft 单类别标签")
    parser.add_argument("--vis-images", required=True, help="原 VIS 图像目录，例如 /.../VIS/all/images")
    parser.add_argument("--vis-labels", required=True, help="原 VIS 标签目录，例如 /.../VIS/all/labels")
    parser.add_argument("--output", required=True, help="输出目录，例如 /.../Datasets/VIS_aircraft_singleclass")
    parser.add_argument("--source-class", type=int, default=1, help="原 VIS 中 aircraft 的类别编号")
    parser.add_argument(
        "--keep-background",
        action="store_true",
        help="保留不含 aircraft 的图像作为背景负样本；默认只保留至少含一个 aircraft 的图像",
    )
    parser.add_argument("--link-mode", choices=("auto", "symlink", "hardlink", "copy"), default="auto")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出目录")
    parser.add_argument("--write-yaml", default=str(ROOT / "dataset_aircraft_da.yaml"), help="写出的新 DA YAML；传空字符串则不写")
    parser.add_argument("--sar-train", default="SAR_Aircraft_noMSAR_jpg_split/images/train")
    parser.add_argument("--sar-val", default="SAR_Aircraft_noMSAR_jpg_split/images/val")
    parser.add_argument("--sar-test", default="SAR_Aircraft_noMSAR_jpg_split/images/test")
    return parser


def main() -> None:
    """主入口。"""

    args = build_parser().parse_args()
    images_root = Path(args.vis_images).expanduser().resolve()
    labels_root = Path(args.vis_labels).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    out_images = output_root / "images" / "train"
    out_labels = output_root / "labels" / "train"

    if not images_root.exists():
        raise FileNotFoundError(f"VIS 图像目录不存在：{images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"VIS 标签目录不存在：{labels_root}")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空：{output_root}。如需覆盖请加 --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    kept_images = 0
    kept_boxes = 0
    background_images = 0
    label_errors: List[str] = []
    link_modes = {}

    for image_path in images:
        label_path = image_to_label_path(image_path, images_root, labels_root)
        remapped_lines, errors = read_and_remap_label(label_path, args.source_class)
        label_errors.extend(errors[:10])
        if not remapped_lines and not args.keep_background:
            continue

        rel = image_path.relative_to(images_root)
        dst_image = out_images / rel
        dst_label = (out_labels / rel).with_suffix(".txt")
        mode_used = link_or_copy(image_path, dst_image, args.link_mode)
        link_modes[mode_used] = link_modes.get(mode_used, 0) + 1
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")

        kept_images += 1
        kept_boxes += len(remapped_lines)
        if not remapped_lines:
            background_images += 1

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vis_images": str(images_root),
        "vis_labels": str(labels_root),
        "output": str(output_root),
        "source_class": args.source_class,
        "total_images": len(images),
        "kept_images": kept_images,
        "kept_boxes": kept_boxes,
        "background_images": background_images,
        "label_error_count": len(label_errors),
        "label_error_examples": label_errors[:100],
        "link_modes": link_modes,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.write_yaml:
        write_dataset_yaml(
            Path(args.write_yaml).expanduser().resolve(),
            output_root,
            args.sar_train,
            args.sar_val,
            args.sar_test,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write_yaml:
        print(f"已写出新 DA YAML：{Path(args.write_yaml).expanduser().resolve()}")


if __name__ == "__main__":
    main()
