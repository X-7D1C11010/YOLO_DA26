"""YOLO 数据集结构、标签质量与划分泄漏审计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "yolo_source"))

from ultralytics.utils import YAML

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_entries(yaml_path: Path, split: str) -> tuple[dict, list[Path]]:
    config = YAML.load(yaml_path)
    base = Path(config.get("path") or yaml_path.parent)
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    value = config.get(split, [])
    values = value if isinstance(value, list) else [value]
    entries = []
    for item in values:
        if not item:
            continue
        path = Path(str(item))
        entries.append(path if path.is_absolute() else (base / path).resolve())
    return config, entries


def collect_images(entry: Path) -> list[Path]:
    if entry.is_file() and entry.suffix.lower() == ".txt":
        images = []
        for line in entry.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value:
                continue
            path = Path(value)
            images.append(path if path.is_absolute() else (entry.parent / path).resolve())
        return images
    if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
        return [entry]
    if entry.is_dir():
        return sorted(path for path in entry.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return []


def image_to_label(image: Path) -> Path:
    parts = list(image.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return (image.parent.parent / "labels" / image.name).with_suffix(".txt")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(images: list[Path], nc: int, use_hash: bool) -> dict:
    class_counts = Counter()
    domain_counts = Counter()
    box_areas: list[float] = []
    box_widths: list[float] = []
    box_heights: list[float] = []
    missing_labels = []
    empty_labels = []
    invalid_labels = []
    hashes = {}

    for image in images:
        normalized_path = str(image).replace("\\", "/").lower()
        domain = "sar" if "sar" in normalized_path else "optical" if any(
            token in normalized_path for token in ("/vis", "/optical", "/rgb")
        ) else "unknown"
        domain_counts[domain] += 1

        if use_hash and image.exists():
            hashes.setdefault(file_sha1(image), []).append(str(image))

        label = image_to_label(image)
        if not label.exists():
            missing_labels.append(str(image))
            continue
        lines = [line.strip() for line in label.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        if not lines:
            empty_labels.append(str(label))
            continue

        for line_number, line in enumerate(lines, 1):
            fields = line.split()
            if len(fields) < 5:
                invalid_labels.append(f"{label}:{line_number} 字段数={len(fields)}")
                continue
            try:
                cls_id = int(float(fields[0]))
                x, y, width, height = map(float, fields[1:5])
            except ValueError:
                invalid_labels.append(f"{label}:{line_number} 非数值标签")
                continue
            if not 0 <= cls_id < nc:
                invalid_labels.append(f"{label}:{line_number} 类别 {cls_id} 超出 [0,{nc - 1}]")
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                invalid_labels.append(f"{label}:{line_number} 存在 NaN/Inf")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                invalid_labels.append(f"{label}:{line_number} 坐标越界")
            class_counts[cls_id] += 1
            box_widths.append(width)
            box_heights.append(height)
            box_areas.append(width * height)

    duplicate_hash_groups = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "images": len(images),
        "existing_images": sum(path.exists() for path in images),
        "domains": dict(domain_counts),
        "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "missing_label_count": len(missing_labels),
        "empty_label_count": len(empty_labels),
        "invalid_label_count": len(invalid_labels),
        "missing_label_examples": missing_labels[:20],
        "invalid_label_examples": invalid_labels[:20],
        "box_area": {
            "p10": percentile(box_areas, 0.10),
            "p50": percentile(box_areas, 0.50),
            "p90": percentile(box_areas, 0.90),
            "mean": statistics.fmean(box_areas) if box_areas else None,
        },
        "box_width_p50": percentile(box_widths, 0.50),
        "box_height_p50": percentile(box_heights, 0.50),
        "duplicate_hash_groups": duplicate_hash_groups[:20],
    }


def main():
    parser = argparse.ArgumentParser(description="审计 YOLO 数据集")
    parser.add_argument("yamls", nargs="+", help="一个或多个数据集 YAML")
    parser.add_argument("--hash", action="store_true", help="计算图像 SHA1，检查重复文件（较慢）")
    parser.add_argument("--max-images", type=int, default=0, help="每个 split 最多审计多少张，0 表示全部")
    parser.add_argument("--output", default=str(ROOT / "dataset_audit.json"))
    args = parser.parse_args()

    report = {"datasets": {}, "warnings": []}
    split_stems: dict[str, set[str]] = {}

    for yaml_value in args.yamls:
        yaml_path = Path(yaml_value).expanduser().resolve()
        if not yaml_path.exists():
            report["warnings"].append(f"配置不存在：{yaml_path}")
            continue
        config = YAML.load(yaml_path)
        nc = int(config["nc"])
        dataset_report = {"nc": nc, "names": config.get("names"), "splits": {}}
        for split in ("train", "val", "test"):
            _, entries = resolve_entries(yaml_path, split)
            images = []
            for entry in entries:
                images.extend(collect_images(entry))
            images = list(dict.fromkeys(path.resolve() for path in images))
            if args.max_images > 0:
                images = images[: args.max_images]
            dataset_report["splits"][split] = audit_split(images, nc, args.hash)
            split_stems[f"{yaml_path.name}:{split}"] = {path.stem for path in images}

        names = config.get("names", {})
        class_zero = names[0] if isinstance(names, list) and names else names.get(0, names.get("0", ""))
        train_stats = dataset_report["splits"]["train"]
        if train_stats["domains"].get("optical", 0) and str(class_zero).lower() not in {"aircraft", "plane", "飞机"}:
            report["warnings"].append(
                f"{yaml_path.name} 含光学数据，但类别 0 名称为 {class_zero!r}；请检查光学 aircraft 是否已重映射。"
            )
        if train_stats["missing_label_count"]:
            report["warnings"].append(f"{yaml_path.name} 训练集缺失 {train_stats['missing_label_count']} 个标签文件。")
        if train_stats["invalid_label_count"]:
            report["warnings"].append(f"{yaml_path.name} 训练集存在 {train_stats['invalid_label_count']} 条非法标签。")

        report["datasets"][str(yaml_path)] = dataset_report

    keys = list(split_stems)
    overlaps = {}
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            common = split_stems[left] & split_stems[right]
            if common:
                overlaps[f"{left} <-> {right}"] = sorted(common)[:100]
    report["stem_overlaps"] = overlaps
    if overlaps:
        report["warnings"].append("检测到不同配置/划分间存在同名图像，请进一步排查数据泄漏。")

    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n审计报告已保存：{output.resolve()}")


if __name__ == "__main__":
    main()
