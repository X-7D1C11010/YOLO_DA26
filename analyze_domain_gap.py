#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练 SAR 数据与独立 SAR 测试数据的域差异分析脚本。

该脚本用于回答一个关键问题：

    “独立测试集为什么从同域 val 的 mAP50≈0.89 掉到独立域的 mAP50≈0.49？”

它会比较两组数据在以下方面的差异：

1. 图像数量、分辨率、宽高比；
2. 灰度均值、方差、对比度、熵、亮斑比例、暗区比例；
3. 梯度/边缘强度、拉普拉斯方差等清晰度代理指标；
4. YOLO 标签框的数量、归一化尺度、像素尺度、宽高比；
5. 两组分布之间的 KS 统计量和分位数差异；
6. 可选图像哈希重复检查。

输出：
    runs/analysis/domain_gap_xxx/
      - domain_gap_report.json        机器可读完整报告
      - domain_gap_summary.md         中文摘要与建议
      - source_image_metrics.csv      训练域图像级统计
      - target_image_metrics.csv      独立测试域图像级统计
      - source_box_metrics.csv        训练域标签框统计
      - target_box_metrics.csv        独立测试域标签框统计

建议先运行：

    python analyze_domain_gap.py \
      --source-data dataset_sar_only.yaml --source-split train \
      --target-data test.yaml --target-split val \
      --max-images 1200 --hash
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - 服务器环境通常有 numpy
    raise SystemExit("缺少 numpy，请在训练环境中安装或使用 Ultralytics 环境运行。") from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - 服务器环境通常有 pillow
    raise SystemExit("缺少 Pillow，请安装 pillow 后再运行。") from exc

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EPS = 1e-12


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML，优先使用 PyYAML；若不可用则回退到 Ultralytics YAML。"""

    if yaml is not None:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            data = yaml.safe_load(file)
        return data or {}

    sys.path.insert(0, str(ROOT / "yolo_source"))
    from ultralytics.utils import YAML  # noqa: WPS433

    return YAML.load(path)


def as_list(value: Any) -> List[str]:
    """把 YAML split 字段统一成列表。"""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def resolve_root(data_file: Path, data: Dict[str, Any]) -> Path:
    """解析数据 YAML 的 path 字段。"""

    raw_root = data.get("path")
    if not raw_root:
        return data_file.parent
    root = Path(str(raw_root)).expanduser()
    if not root.is_absolute():
        root = data_file.parent / root
    return root


def resolve_entry(root: Path, data_file: Path, raw_entry: str) -> Path:
    """解析 split 中的单个路径。"""

    entry = Path(raw_entry).expanduser()
    if entry.is_absolute():
        return entry
    candidate = root / entry
    if candidate.exists():
        return candidate
    fallback = data_file.parent / entry
    return fallback if fallback.exists() else candidate


def iter_images_from_txt(txt_path: Path) -> Iterable[Path]:
    """从 txt 图像清单中读取图像路径。"""

    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = txt_path.parent / path
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def collect_images(data_file: Path, split: str) -> List[Path]:
    """收集某个 split 的全部图像路径。"""

    data = load_yaml(data_file)
    root = resolve_root(data_file, data)
    images: List[Path] = []

    for raw_entry in as_list(data.get(split)):
        entry = resolve_entry(root, data_file, raw_entry)
        if entry.is_file() and entry.suffix.lower() == ".txt":
            images.extend(iter_images_from_txt(entry))
        elif entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
            images.append(entry)
        elif entry.is_dir():
            images.extend(
                path
                for path in entry.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )

    return sorted(dict.fromkeys(path.resolve() for path in images))


def image_to_label_path(image_path: Path) -> Path:
    """按 YOLO 目录约定把 images 路径转换为 labels 路径。"""

    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_label_file(label_path: Path, nc: int) -> Tuple[List[Dict[str, float]], List[str]]:
    """读取并校验 YOLO 标签。"""

    boxes: List[Dict[str, float]] = []
    errors: List[str] = []

    if not label_path.exists():
        return boxes, [f"标签缺失：{label_path}"]

    for line_no, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            errors.append(f"{label_path}:{line_no} 字段数不足")
            continue
        try:
            cls = int(float(fields[0]))
            x, y, w, h = [float(value) for value in fields[1:5]]
        except ValueError:
            errors.append(f"{label_path}:{line_no} 非数字字段")
            continue
        if cls < 0 or cls >= nc:
            errors.append(f"{label_path}:{line_no} 类别越界 cls={cls}")
            continue
        if not all(0.0 <= value <= 1.0 for value in (x, y, w, h)):
            errors.append(f"{label_path}:{line_no} 坐标越界")
            continue
        if w <= 0 or h <= 0:
            errors.append(f"{label_path}:{line_no} 宽高非正")
            continue
        boxes.append({"cls": cls, "x": x, "y": y, "w": w, "h": h})

    return boxes, errors


def percentile(values: Sequence[float], qs: Sequence[float]) -> Dict[str, Optional[float]]:
    """计算分位数。"""

    clean = np.asarray([value for value in values if value is not None and math.isfinite(float(value))], dtype=float)
    if clean.size == 0:
        return {f"p{int(q)}": None for q in qs}
    return {f"p{int(q)}": float(np.percentile(clean, q)) for q in qs}


def numeric_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """数值列表摘要。"""

    clean = np.asarray([value for value in values if value is not None and math.isfinite(float(value))], dtype=float)
    if clean.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, **percentile([], [1, 5, 10, 25, 50, 75, 90, 95, 99])}
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        **percentile(clean.tolist(), [1, 5, 10, 25, 50, 75, 90, 95, 99]),
    }


def entropy_from_uint8(gray: np.ndarray) -> float:
    """计算 8bit 灰度熵。"""

    hist = np.bincount(gray.reshape(-1), minlength=256).astype(float)
    prob = hist / max(hist.sum(), EPS)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def image_hash(path: Path, block_size: int = 1024 * 1024) -> str:
    """计算文件 SHA1。"""

    hasher = hashlib.sha1()
    with path.open("rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def image_metrics(path: Path, compute_hash: bool, max_side: int) -> Dict[str, Any]:
    """计算单张图像的统计特征。"""

    with Image.open(path) as image:
        width, height = image.size
        gray_image = image.convert("L")
        if max(width, height) > max_side:
            gray_image.thumbnail((max_side, max_side))
        gray = np.asarray(gray_image, dtype=np.float32)

    p1, p5, p50, p95, p99 = np.percentile(gray, [1, 5, 50, 95, 99])
    mean = float(gray.mean())
    std = float(gray.std())
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    lap = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    bright_threshold = p95
    dark_threshold = p5
    edge_threshold = float(grad.mean() + grad.std())

    record: Dict[str, Any] = {
        "image": str(path),
        "width": width,
        "height": height,
        "aspect": width / max(height, 1),
        "pixels": width * height,
        "gray_mean": mean,
        "gray_std": std,
        "gray_cv": std / max(mean, EPS),
        "gray_p1": float(p1),
        "gray_p5": float(p5),
        "gray_p50": float(p50),
        "gray_p95": float(p95),
        "gray_p99": float(p99),
        "contrast_p95_p5": float(p95 - p5),
        "entropy": entropy_from_uint8(gray.astype(np.uint8)),
        "bright_ratio_p95": float((gray >= bright_threshold).mean()),
        "dark_ratio_p5": float((gray <= dark_threshold).mean()),
        "gradient_mean": float(grad.mean()),
        "gradient_p95": float(np.percentile(grad, 95)),
        "edge_density": float((grad > edge_threshold).mean()),
        "laplacian_var": float(lap.var()),
    }
    if compute_hash:
        record["sha1"] = image_hash(path)
    return record


def collect_dataset_metrics(
    name: str,
    data_file: Path,
    split: str,
    max_images: int,
    max_side: int,
    seed: int,
    compute_hash: bool,
) -> Dict[str, Any]:
    """收集一个数据集 split 的图像与标签统计。"""

    data = load_yaml(data_file)
    nc = int(data.get("nc", 1))
    images = collect_images(data_file, split)
    rng = random.Random(seed)
    sample_images = images[:]
    if max_images > 0 and len(sample_images) > max_images:
        sample_images = rng.sample(sample_images, max_images)
        sample_images.sort()

    image_records: List[Dict[str, Any]] = []
    image_errors: List[str] = []
    label_errors: List[str] = []
    box_records: List[Dict[str, Any]] = []
    boxes_per_image: List[int] = []
    class_counts: Counter[int] = Counter()

    print(f"[{name}] 图像总数={len(images)}，图像统计抽样={len(sample_images)}")

    for image_path in sample_images:
        try:
            image_records.append(image_metrics(image_path, compute_hash, max_side))
        except Exception as exc:  # noqa: BLE001
            image_errors.append(f"{image_path}: {exc}")

    size_lookup = {record["image"]: (record["width"], record["height"]) for record in image_records}
    for image_path in images:
        label_path = image_to_label_path(image_path)
        boxes, errors = read_label_file(label_path, nc)
        label_errors.extend(errors[:5])
        boxes_per_image.append(len(boxes))
        width_height = size_lookup.get(str(image_path))
        if width_height is None:
            try:
                with Image.open(image_path) as image:
                    width_height = image.size
            except Exception:
                width_height = (0, 0)
        width, height = width_height
        for box in boxes:
            cls = int(box["cls"])
            class_counts[cls] += 1
            box_records.append(
                {
                    "image": str(image_path),
                    "cls": cls,
                    "x": box["x"],
                    "y": box["y"],
                    "w_norm": box["w"],
                    "h_norm": box["h"],
                    "area_norm": box["w"] * box["h"],
                    "aspect": box["w"] / max(box["h"], EPS),
                    "w_px": box["w"] * width,
                    "h_px": box["h"] * height,
                    "area_px": box["w"] * width * box["h"] * height,
                }
            )

    duplicate_hash_groups: List[List[str]] = []
    if compute_hash and image_records:
        by_hash: Dict[str, List[str]] = {}
        for record in image_records:
            by_hash.setdefault(record.get("sha1", ""), []).append(record["image"])
        duplicate_hash_groups = [paths for digest, paths in by_hash.items() if digest and len(paths) > 1]

    return {
        "name": name,
        "data": str(data_file),
        "split": split,
        "image_count": len(images),
        "sampled_image_count": len(sample_images),
        "image_records": image_records,
        "box_records": box_records,
        "image_errors": image_errors[:100],
        "label_errors": label_errors[:100],
        "label_error_count": len(label_errors),
        "class_counts": dict(class_counts),
        "duplicate_hash_groups": duplicate_hash_groups[:50],
        "summaries": {
            "image": summarize_records(image_records),
            "box": summarize_records(box_records),
            "boxes_per_image": numeric_summary(boxes_per_image),
        },
    }


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """对记录中的数值字段做摘要。"""

    if not records:
        return {}
    numeric_keys = sorted(
        key
        for key in records[0].keys()
        if isinstance(records[0].get(key), (int, float)) and key not in {"cls"}
    )
    return {key: numeric_summary([float(record[key]) for record in records if key in record]) for key in numeric_keys}


def ks_statistic(a_values: Sequence[float], b_values: Sequence[float]) -> Optional[float]:
    """计算两组一维数值的双样本 KS 统计量。"""

    a = np.sort(np.asarray([v for v in a_values if math.isfinite(float(v))], dtype=float))
    b = np.sort(np.asarray([v for v in b_values if math.isfinite(float(v))], dtype=float))
    if a.size == 0 or b.size == 0:
        return None
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def compare_records(source: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    """比较源域与目标域的关键分布。"""

    comparisons: Dict[str, Any] = {}
    for group in ("image_records", "box_records"):
        source_records = source[group]
        target_records = target[group]
        keys = sorted(
            set().union(*(record.keys() for record in source_records[:20] + target_records[:20]))
        )
        group_cmp: Dict[str, Any] = {}
        for key in keys:
            source_values = [record[key] for record in source_records if isinstance(record.get(key), (int, float))]
            target_values = [record[key] for record in target_records if isinstance(record.get(key), (int, float))]
            if not source_values or not target_values:
                continue
            src_summary = numeric_summary(source_values)
            tgt_summary = numeric_summary(target_values)
            src_med = src_summary.get("p50")
            tgt_med = tgt_summary.get("p50")
            group_cmp[key] = {
                "source": src_summary,
                "target": tgt_summary,
                "median_ratio_target_over_source": None
                if not src_med
                else float(tgt_med / max(src_med, EPS)),
                "ks": ks_statistic(source_values, target_values),
            }
        comparisons[group] = group_cmp
    return comparisons


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """写 CSV。"""

    if not records:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(record.keys() for record in records)))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def fmt(value: Any, digits: int = 4) -> str:
    """格式化数字。"""

    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def make_recommendations(report: Dict[str, Any]) -> List[str]:
    """根据统计结果生成中文建议。"""

    cmp = report["comparisons"]
    recommendations: List[str] = []

    def ratio(group: str, key: str) -> Optional[float]:
        item = cmp.get(group, {}).get(key, {})
        return item.get("median_ratio_target_over_source")

    width_ratio = ratio("image_records", "width")
    height_ratio = ratio("image_records", "height")
    area_ratio = ratio("box_records", "area_norm")
    box_px_ratio = ratio("box_records", "area_px")
    contrast_ratio = ratio("image_records", "contrast_p95_p5")
    entropy_ratio = ratio("image_records", "entropy")

    if width_ratio and height_ratio and (abs(width_ratio - 1) > 0.2 or abs(height_ratio - 1) > 0.2):
        recommendations.append(
            f"独立域分辨率与训练域明显不同：宽度中位比={width_ratio:.2f}，高度中位比={height_ratio:.2f}。"
            "建议做尺度归一化实验，或使用切片训练/切片推理。"
        )
    if area_ratio and (area_ratio > 1.5 or area_ratio < 0.67):
        recommendations.append(
            f"目标归一化面积中位数差异明显，目标域/训练域={area_ratio:.2f}。"
            "建议按目标尺寸分桶评估 AP，并调整多尺度训练范围。"
        )
    if box_px_ratio and (box_px_ratio > 1.8 or box_px_ratio < 0.55):
        recommendations.append(
            f"目标像素面积中位数差异明显，目标域/训练域={box_px_ratio:.2f}。"
            "这会改变检测头看到的目标尺度，建议尝试 copy-paste/随机缩放或切片策略。"
        )
    if contrast_ratio and (contrast_ratio > 1.25 or contrast_ratio < 0.8):
        recommendations.append(
            f"目标域灰度对比度与训练域差异较大，目标域/训练域={contrast_ratio:.2f}。"
            "建议加入 SAR 风格增强：随机 gamma、CLAHE、轻度噪声/模糊，而不是强 HSV。"
        )
    if entropy_ratio and (entropy_ratio > 1.15 or entropy_ratio < 0.87):
        recommendations.append(
            f"目标域灰度熵差异较大，目标域/训练域={entropy_ratio:.2f}。"
            "这通常意味着散斑、背景复杂度或成像处理链不同，建议做无监督/少样本域适应。"
        )
    if report["target"]["label_error_count"] > 0:
        recommendations.append("独立测试域存在标签格式异常，请先修复，否则 mAP 会被低估。")
    if not recommendations:
        recommendations.append("未发现单一压倒性数据统计差异，建议重点查看预测错误分析中的 FP/FN 模式。")
    return recommendations


def write_markdown(output: Path, report: Dict[str, Any]) -> None:
    """写中文 Markdown 摘要。"""

    cmp = report["comparisons"]
    lines = [
        "# SAR 训练域与独立测试域差异分析报告",
        "",
        f"- 源域：`{report['source']['data']}` split=`{report['source']['split']}`，图像数={report['source']['image_count']}",
        f"- 目标域：`{report['target']['data']}` split=`{report['target']['split']}`，图像数={report['target']['image_count']}",
        "",
        "## 关键分布对比",
        "",
        "| 指标 | 源域中位数 | 目标域中位数 | 目标/源 | KS |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, keys in {
        "image_records": ["width", "height", "gray_mean", "gray_std", "contrast_p95_p5", "entropy", "gradient_p95", "laplacian_var"],
        "box_records": ["area_norm", "w_norm", "h_norm", "area_px", "w_px", "h_px", "aspect"],
    }.items():
        for key in keys:
            item = cmp.get(group, {}).get(key)
            if not item:
                continue
            lines.append(
                f"| {group}.{key} | {fmt(item['source'].get('p50'))} | {fmt(item['target'].get('p50'))} | "
                f"{fmt(item.get('median_ratio_target_over_source'))} | {fmt(item.get('ks'))} |"
            )

    lines.extend(["", "## 自动建议", ""])
    for rec in report["recommendations"]:
        lines.append(f"- {rec}")

    lines.extend(
        [
            "",
            "## 标签与图像异常",
            "",
            f"- 源域标签异常数：{report['source']['label_error_count']}",
            f"- 目标域标签异常数：{report['target']['label_error_count']}",
            f"- 源域图像读取异常数：{len(report['source']['image_errors'])}",
            f"- 目标域图像读取异常数：{len(report['target']['image_errors'])}",
            "",
            "完整机器可读结果见 `domain_gap_report.json`。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""

    parser = argparse.ArgumentParser(description="训练 SAR 与独立 SAR 测试域差异分析")
    parser.add_argument("--source-data", default=str(ROOT / "dataset_sar_only.yaml"), help="训练 SAR 数据 YAML")
    parser.add_argument("--source-split", default="train", help="源域 split")
    parser.add_argument("--target-data", default=str(ROOT / "test.yaml"), help="独立测试域 YAML")
    parser.add_argument("--target-split", default="val", help="目标域 split")
    parser.add_argument("--output", default=None, help="输出目录；默认 runs/analysis/domain_gap_时间戳")
    parser.add_argument("--max-images", type=int, default=1200, help="每个域最多抽样多少张图像做灰度统计；<=0 表示全部")
    parser.add_argument("--max-side", type=int, default=512, help="图像统计时最长边缩放到该值以加速")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hash", action="store_true", help="计算抽样图像 SHA1，用于重复图检查，速度较慢")
    return parser


def main() -> None:
    """主入口。"""

    args = build_parser().parse_args()
    output = Path(args.output) if args.output else ROOT / "runs" / "analysis" / f"domain_gap_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)

    source = collect_dataset_metrics(
        "source",
        Path(args.source_data).expanduser().resolve(),
        args.source_split,
        args.max_images,
        args.max_side,
        args.seed,
        args.hash,
    )
    target = collect_dataset_metrics(
        "target",
        Path(args.target_data).expanduser().resolve(),
        args.target_split,
        args.max_images,
        args.max_side,
        args.seed,
        args.hash,
    )

    report: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {key: value for key, value in source.items() if key not in {"image_records", "box_records"}},
        "target": {key: value for key, value in target.items() if key not in {"image_records", "box_records"}},
        "comparisons": compare_records(source, target),
    }
    report["recommendations"] = make_recommendations(report)

    write_csv(output / "source_image_metrics.csv", source["image_records"])
    write_csv(output / "target_image_metrics.csv", target["image_records"])
    write_csv(output / "source_box_metrics.csv", source["box_records"])
    write_csv(output / "target_box_metrics.csv", target["box_records"])
    (output / "domain_gap_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output / "domain_gap_summary.md", report)

    print(f"分析完成：{output}")
    print(f"中文摘要：{output / 'domain_gap_summary.md'}")
    print(f"完整 JSON：{output / 'domain_gap_report.json'}")


if __name__ == "__main__":
    main()
