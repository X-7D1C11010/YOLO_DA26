#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从独立目标域数据集中抽取少量样本，并复制成可微调的数据集。

本脚本用于当前项目的“小样本目标域适应”实验准备。它会从一个 YOLO
数据集 YAML 指定的 split 中随机抽取指定比例或数量的图像，再划分为
train/test 两个子集，并完整复制图像与可用标注到新的项目目录中。

典型用途：
    python select_target_subset.py \
        --data test.yaml \
        --ratio 0.20 \
        --train-ratio 0.70 \
        --seed 2026 \
        --output runs/target_subsets/independent_20p_seed2026 \
        --overwrite

输出目录结构：
    output/
      images/train/*.jpg
      images/test/*.jpg
      images/no_leak_test/*.jpg
      images/strict_holdout/*.jpg
      labels/train/*.txt
      labels/test/*.txt
      labels/no_leak_test/*.txt
      labels/strict_holdout/*.txt
      target_subset.yaml
      target_no_leak_test.yaml
      target_strict_holdout.yaml
      manifest.json
      train_images.txt
      test_images.txt
      no_leak_test_images.txt
      strict_holdout_images.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
YOLO_SOURCE = ROOT / "yolo_source"
if str(YOLO_SOURCE) not in sys.path:
    sys.path.insert(0, str(YOLO_SOURCE))

from ultralytics.utils import YAML  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    """一张图像及其标注、相对复制路径。"""

    image: Path
    label: Path
    relative: Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="从独立测试/目标域数据集中随机抽样并复制为微调数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default=str(ROOT / "test.yaml"), help="源数据集 YAML，例如 test.yaml")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="从 YAML 的哪个 split 抽样")

    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument("--ratio", type=float, default=0.20, help="抽取比例，范围 (0, 1]")
    sample_group.add_argument("--count", type=int, help="抽取固定图像数量")

    parser.add_argument("--train-ratio", type=float, default=0.70, help="抽中样本中划入 train 的比例")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子，保证可复现")
    parser.add_argument(
        "--output",
        default=str(ROOT / "runs" / "target_subsets" / "independent_subset"),
        help="输出数据集目录",
    )
    parser.add_argument("--overwrite", action="store_true", help="若输出目录已存在，则先删除再重建")
    parser.add_argument(
        "--label-mode",
        choices=("keep", "drop-train", "drop-all"),
        default="keep",
        help="标注复制策略：keep=全保留；drop-train=训练集不复制标注；drop-all=全部不复制标注",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="允许源图像缺少标注文件；未复制标注时会写空标签以保持目录结构",
    )
    parser.add_argument("--nc", type=int, help="覆盖 YAML 中的类别数")
    parser.add_argument("--name", default="aircraft", help="当 YAML 没有 names 时使用的类别名")
    parser.add_argument(
        "--final-test-mode",
        choices=("none", "no-leak", "strict", "both"),
        default="both",
        help=(
            "额外生成无泄漏测试集：no-leak=仅排除微调训练图像；"
            "strict=排除全部抽样图像；both=两者都生成"
        ),
    )
    return parser.parse_args()


def _as_list(value: Any) -> list[Any]:
    """把 YAML 中的单值/list split 统一为列表。"""

    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _resolve_base(config: dict[str, Any], yaml_path: Path) -> Path:
    """解析 YOLO YAML 的 path 字段。"""

    base = Path(config.get("path") or yaml_path.parent).expanduser()
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    return base


def _label_from_image(image: Path) -> Path:
    """按 YOLO 常规目录结构从 image 路径推导 label 路径。"""

    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        label_parts = parts[:index] + ["labels"] + parts[index + 1 :]
        return Path(*label_parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _iter_images_from_txt(txt_path: Path, base: Path) -> Iterable[Path]:
    """读取 YOLO txt 清单中的图像路径。"""

    for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line).expanduser()
        candidates = [path] if path.is_absolute() else [txt_path.parent / path, base / path]
        for candidate in candidates:
            if candidate.exists():
                yield candidate.resolve()
                break
        else:
            # 保留一个可诊断路径，后续会在存在性检查中报错。
            yield candidates[0].resolve()


def _collect_from_entry(entry: str, base: Path) -> list[tuple[Path, Path]]:
    """从一个 YAML split entry 中收集图像，并返回图像及其源根目录。"""

    raw = Path(str(entry)).expanduser()
    path = raw if raw.is_absolute() else (base / raw)
    path = path.resolve()

    if path.is_file() and path.suffix.lower() == ".txt":
        images = sorted(_iter_images_from_txt(path, base))
        return [(image, image.parent) for image in images]
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [(path, path.parent)]
    if path.is_dir():
        images = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        return [(image.resolve(), path) for image in images]

    raise FileNotFoundError(f"无法解析数据路径：{path}")


def collect_records(data_yaml: Path, split: str) -> tuple[list[ImageRecord], dict[str, Any]]:
    """从 YOLO YAML 中收集指定 split 的图像记录。"""

    config = YAML.load(data_yaml)
    base = _resolve_base(config, data_yaml)
    entries = _as_list(config.get(split))
    if not entries:
        raise ValueError(f"{data_yaml} 中没有 split={split} 的路径配置")

    records: list[ImageRecord] = []
    used_relatives: set[Path] = set()
    for entry in entries:
        for image, root in _collect_from_entry(str(entry), base):
            if not image.exists():
                raise FileNotFoundError(f"图像不存在：{image}")
            try:
                relative = image.relative_to(root)
            except ValueError:
                relative = Path(image.name)
            if relative in used_relatives:
                relative = Path(image.parent.name) / image.name
            suffix = 1
            original_relative = relative
            while relative in used_relatives:
                relative = original_relative.with_name(f"{original_relative.stem}_{suffix}{original_relative.suffix}")
                suffix += 1
            used_relatives.add(relative)
            records.append(ImageRecord(image=image, label=_label_from_image(image), relative=relative))

    if not records:
        raise FileNotFoundError(f"{data_yaml} 的 {split} split 中没有找到图像")
    return records, config


def choose_subset(records: list[ImageRecord], args: argparse.Namespace) -> list[ImageRecord]:
    """按比例或数量抽取样本。"""

    if args.count is not None:
        if args.count <= 0:
            raise ValueError("--count 必须为正整数")
        sample_count = min(args.count, len(records))
    else:
        if not 0 < args.ratio <= 1:
            raise ValueError("--ratio 必须位于 (0, 1] 范围内")
        sample_count = max(1, round(len(records) * args.ratio))

    rng = random.Random(args.seed)
    selected = records[:]
    rng.shuffle(selected)
    return selected[:sample_count]


def split_subset(selected: list[ImageRecord], train_ratio: float, seed: int) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """把抽中样本划分为 train/test。"""

    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio 必须位于 (0, 1) 范围内")
    rng = random.Random(seed + 17)
    shuffled = selected[:]
    rng.shuffle(shuffled)
    train_count = max(1, round(len(shuffled) * train_ratio))
    if len(shuffled) > 1:
        train_count = min(train_count, len(shuffled) - 1)
    return shuffled[:train_count], shuffled[train_count:]


def prepare_output(output: Path, overwrite: bool) -> None:
    """准备输出目录。"""

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在：{output}。如需重建请添加 --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def _should_copy_label(split: str, label_mode: str) -> bool:
    """判断当前 split 是否复制标注。"""

    if label_mode == "drop-all":
        return False
    if label_mode == "drop-train" and split == "train":
        return False
    return True


def copy_records(
    records: list[ImageRecord],
    split: str,
    output: Path,
    label_mode: str,
    allow_missing_labels: bool,
) -> list[dict[str, str]]:
    """复制图像和标注，并返回 manifest 明细。"""

    rows: list[dict[str, str]] = []
    image_root = output / "images" / split
    label_root = output / "labels" / split
    for record in records:
        dst_image = image_root / record.relative
        dst_label = (label_root / record.relative).with_suffix(".txt")
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.image, dst_image)

        copy_label = _should_copy_label(split, label_mode)
        label_status = "dropped"
        if copy_label:
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            if record.label.exists():
                shutil.copy2(record.label, dst_label)
                label_status = "copied"
            elif allow_missing_labels:
                dst_label.write_text("", encoding="utf-8")
                label_status = "empty_created"
            else:
                raise FileNotFoundError(f"图像缺少对应标注：{record.image} -> {record.label}")

        rows.append(
            {
                "split": split,
                "source_image": str(record.image),
                "source_label": str(record.label),
                "target_image": str(dst_image),
                "target_label": str(dst_label),
                "label_status": label_status,
            }
        )
    return rows


def _format_names(config: dict[str, Any], fallback_name: str) -> list[str]:
    """规范化类别名。"""

    names = config.get("names")
    if isinstance(names, list):
        return [str(item) for item in names]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return [fallback_name]


def write_yaml(output: Path, config: dict[str, Any], args: argparse.Namespace) -> Path:
    """写出可直接用于 YOLO 微调的数据集 YAML。"""

    names = _format_names(config, args.name)
    nc = args.nc if args.nc is not None else int(config.get("nc") or len(names))
    yaml_path = output / "target_subset.yaml"
    lines = [
        "# 由 select_target_subset.py 自动生成，用于目标域小样本微调。",
        f"path: {json.dumps(str(output.resolve()), ensure_ascii=False)}",
        "train: images/train",
        "val: images/test",
        "test: images/test",
        f"nc: {nc}",
        "names:",
    ]
    for index, name in enumerate(names[:nc]):
        lines.append(f"  {index}: {json.dumps(name, ensure_ascii=False)}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def write_eval_yaml(
    output: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    split_dir: str,
    filename: str,
    comment: str,
) -> Path:
    """写出只用于评估的无泄漏数据集 YAML。"""

    names = _format_names(config, args.name)
    nc = args.nc if args.nc is not None else int(config.get("nc") or len(names))
    yaml_path = output / filename
    lines = [
        f"# {comment}",
        f"path: {json.dumps(str(output.resolve()), ensure_ascii=False)}",
        "train: images/train",
        f"val: images/{split_dir}",
        f"test: images/{split_dir}",
        f"nc: {nc}",
        "names:",
    ]
    for index, name in enumerate(names[:nc]):
        lines.append(f"  {index}: {json.dumps(name, ensure_ascii=False)}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def write_lists(
    output: Path,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    extra_rows: dict[str, list[dict[str, str]]] | None = None,
) -> None:
    """写出图像清单和 CSV 明细，便于复查。"""

    (output / "train_images.txt").write_text(
        "\n".join(row["target_image"] for row in train_rows) + "\n",
        encoding="utf-8",
    )
    (output / "test_images.txt").write_text(
        "\n".join(row["target_image"] for row in test_rows) + "\n",
        encoding="utf-8",
    )
    extra_rows = extra_rows or {}
    for split_name, rows in extra_rows.items():
        (output / f"{split_name}_images.txt").write_text(
            "\n".join(row["target_image"] for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
    with (output / "selection_detail.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "source_image", "source_label", "target_image", "target_label", "label_status"],
        )
        writer.writeheader()
        all_rows = [*train_rows, *test_rows]
        for rows in extra_rows.values():
            all_rows.extend(rows)
        writer.writerows(all_rows)


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据集 YAML 不存在：{data_yaml}")

    output = Path(args.output).expanduser().resolve()
    records, config = collect_records(data_yaml, args.split)
    selected = choose_subset(records, args)
    train_records, test_records = split_subset(selected, args.train_ratio, args.seed)
    train_keys = {record.image.resolve() for record in train_records}
    selected_keys = {record.image.resolve() for record in selected}
    no_leak_records = [record for record in records if record.image.resolve() not in train_keys]
    strict_holdout_records = [record for record in records if record.image.resolve() not in selected_keys]

    prepare_output(output, args.overwrite)
    train_rows = copy_records(train_records, "train", output, args.label_mode, args.allow_missing_labels)
    test_rows = copy_records(test_records, "test", output, args.label_mode, args.allow_missing_labels)
    yaml_path = write_yaml(output, config, args)

    extra_rows: dict[str, list[dict[str, str]]] = {}
    no_leak_yaml: Path | None = None
    strict_yaml: Path | None = None
    if args.final_test_mode in {"no-leak", "both"}:
        no_leak_rows = copy_records(no_leak_records, "no_leak_test", output, "keep", args.allow_missing_labels)
        extra_rows["no_leak_test"] = no_leak_rows
        no_leak_yaml = write_eval_yaml(
            output,
            config,
            args,
            "no_leak_test",
            "target_no_leak_test.yaml",
            "无泄漏测试集：排除了微调训练图像，可用于训练后测试。",
        )
    if args.final_test_mode in {"strict", "both"}:
        if strict_holdout_records:
            strict_rows = copy_records(strict_holdout_records, "strict_holdout", output, "keep", args.allow_missing_labels)
            extra_rows["strict_holdout"] = strict_rows
            strict_yaml = write_eval_yaml(
                output,
                config,
                args,
                "strict_holdout",
                "target_strict_holdout.yaml",
                "严格最终测试集：排除了全部抽样图像，适合最终无偏评估。",
            )
        else:
            print("警告：严格 holdout 为空，未生成 target_strict_holdout.yaml。请降低 --ratio 或 --count。")

    write_lists(output, train_rows, test_rows, extra_rows)

    manifest = {
        "source_yaml": str(data_yaml),
        "source_split": args.split,
        "seed": args.seed,
        "total_source_images": len(records),
        "selected_images": len(selected),
        "train_images": len(train_records),
        "test_images": len(test_records),
        "ratio": args.ratio if args.count is None else None,
        "count": args.count,
        "train_ratio": args.train_ratio,
        "label_mode": args.label_mode,
        "dataset_yaml": str(yaml_path),
        "no_leak_test_images": len(no_leak_records) if args.final_test_mode in {"no-leak", "both"} else 0,
        "strict_holdout_images": len(strict_holdout_records) if args.final_test_mode in {"strict", "both"} else 0,
        "no_leak_test_yaml": str(no_leak_yaml) if no_leak_yaml else "",
        "strict_holdout_yaml": str(strict_yaml) if strict_yaml else "",
        "output": str(output),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("目标域子集构建完成：")
    print(f"  源图像总数：{len(records)}")
    print(f"  抽取图像数：{len(selected)}")
    print(f"  训练子集：{len(train_records)}")
    print(f"  测试/验证子集：{len(test_records)}")
    print(f"  数据集 YAML：{yaml_path}")
    if no_leak_yaml:
        print(f"  无泄漏测试 YAML：{no_leak_yaml}（排除训练子集，图像数={len(no_leak_records)}）")
    if strict_yaml:
        print(f"  严格最终测试 YAML：{strict_yaml}（排除全部抽样子集，图像数={len(strict_holdout_records)}）")
    print(f"  明细文件：{output / 'manifest.json'}")


if __name__ == "__main__":
    main()
