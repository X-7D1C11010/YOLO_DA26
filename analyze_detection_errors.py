#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立测试集逐图检测错误分析脚本。

该脚本用于分析某个 best.pt 在独立测试集上的具体失败模式：

1. 每张图 TP / FP / FN 数量；
2. 重复预测框数量；
3. 每个 GT 的最佳 IoU 分布；
4. 不同目标尺度上的召回率；
5. 最严重的漏检图、误检图、重复框图；
6. 可选保存带标注叠加的 hard case 图片。

输出：
    runs/analysis/error_xxx/
      - detection_error_report.json
      - detection_error_summary.md
      - per_image_errors.csv
      - gt_match_details.csv
      - fp_details.csv
      - overlays/*.jpg

推荐运行：

    python analyze_detection_errors.py \
      --weights runs/detect/YOLO26s_DA_DANN-4/weights/best.pt \
      --data test.yaml --split val --imgsz 640 --conf 0.001 \
      --match-iou 0.5 --save-overlays 40
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 numpy，请在训练环境中运行。") from exc

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 Pillow，请安装 pillow 后再运行。") from exc

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parent
YOLO_SOURCE = ROOT / "yolo_source"
if str(YOLO_SOURCE) not in sys.path:
    sys.path.insert(0, str(YOLO_SOURCE))

from ultralytics import YOLO  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EPS = 1e-12


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML。"""

    if yaml is not None:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            return yaml.safe_load(file) or {}
    from ultralytics.utils import YAML  # noqa: WPS433

    return YAML.load(path)


def as_list(value: Any) -> List[str]:
    """把 YAML split 字段转成列表。"""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def resolve_root(data_file: Path, data: Dict[str, Any]) -> Path:
    """解析 path 字段。"""

    raw = data.get("path")
    if not raw:
        return data_file.parent
    root = Path(str(raw)).expanduser()
    if not root.is_absolute():
        root = data_file.parent / root
    return root


def resolve_entry(root: Path, data_file: Path, raw_entry: str) -> Path:
    """解析 split 路径。"""

    entry = Path(raw_entry).expanduser()
    if entry.is_absolute():
        return entry
    candidate = root / entry
    if candidate.exists():
        return candidate
    fallback = data_file.parent / entry
    return fallback if fallback.exists() else candidate


def iter_images_from_txt(txt_path: Path) -> Iterable[Path]:
    """从 txt 图像清单读取图像路径。"""

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
    """收集指定 split 的图像。"""

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
    """图像路径转标签路径。"""

    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def xywhn_to_xyxy(box: Sequence[float], width: int, height: int) -> List[float]:
    """归一化 xywh 转像素 xyxy。"""

    x, y, w, h = box
    cx = x * width
    cy = y * height
    bw = w * width
    bh = h * height
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def read_gt(image_path: Path, width: int, height: int, nc: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """读取 GT 标签。"""

    label_path = image_to_label_path(image_path)
    if not label_path.exists():
        return [], [f"标签缺失：{label_path}"]

    gts: List[Dict[str, Any]] = []
    errors: List[str] = []
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
        if not all(0 <= value <= 1 for value in (x, y, w, h)) or w <= 0 or h <= 0:
            errors.append(f"{label_path}:{line_no} 坐标异常")
            continue
        xyxy = xywhn_to_xyxy([x, y, w, h], width, height)
        area_norm = w * h
        gts.append(
            {
                "cls": cls,
                "xyxy": xyxy,
                "x": x,
                "y": y,
                "w_norm": w,
                "h_norm": h,
                "area_norm": area_norm,
                "w_px": w * width,
                "h_px": h * height,
                "area_px": w * width * h * height,
                "size_bin": size_bin(area_norm),
            }
        )
    return gts, errors


def size_bin(area_norm: float) -> str:
    """按归一化面积划分目标尺度。"""

    if area_norm < 0.001:
        return "tiny(<0.001)"
    if area_norm < 0.005:
        return "small(0.001-0.005)"
    if area_norm < 0.02:
        return "medium(0.005-0.02)"
    return "large(>=0.02)"


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """计算两个 xyxy 框 IoU。"""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    return inter / max(area_a + area_b - inter, EPS)


def match_predictions(
    gts: List[Dict[str, Any]],
    preds: List[Dict[str, Any]],
    match_iou: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """贪心匹配预测框与 GT。"""

    gt_matched = [False] * len(gts)
    pred_records: List[Dict[str, Any]] = []
    gt_records: List[Dict[str, Any]] = []
    duplicate_fp = 0

    for pred_index, pred in enumerate(sorted(preds, key=lambda item: item["conf"], reverse=True)):
        ious = [box_iou(pred["xyxy"], gt["xyxy"]) if int(pred["cls"]) == int(gt["cls"]) else 0.0 for gt in gts]
        best_gt = int(np.argmax(ious)) if ious else -1
        best_iou = float(ious[best_gt]) if best_gt >= 0 else 0.0
        matched = best_gt >= 0 and best_iou >= match_iou and not gt_matched[best_gt]
        duplicate = best_gt >= 0 and best_iou >= match_iou and gt_matched[best_gt]
        if matched:
            gt_matched[best_gt] = True
        if duplicate:
            duplicate_fp += 1
        pred_records.append(
            {
                **pred,
                "pred_index": pred_index,
                "best_gt": best_gt,
                "best_iou": best_iou,
                "matched": matched,
                "duplicate": duplicate,
            }
        )

    for gt_index, gt in enumerate(gts):
        best_iou = 0.0
        best_conf = None
        for pred in preds:
            iou = box_iou(gt["xyxy"], pred["xyxy"]) if int(pred["cls"]) == int(gt["cls"]) else 0.0
            if iou > best_iou:
                best_iou = iou
                best_conf = pred["conf"]
        gt_records.append(
            {
                **gt,
                "gt_index": gt_index,
                "matched": gt_matched[gt_index],
                "best_iou": best_iou,
                "best_conf": best_conf,
            }
        )

    tp = [record for record in pred_records if record["matched"]]
    fp = [record for record in pred_records if not record["matched"]]
    fn = [record for record in gt_records if not record["matched"]]
    return tp, fp, fn, duplicate_fp


def result_to_predictions(result: Any) -> List[Dict[str, Any]]:
    """Ultralytics result 转预测记录。"""

    preds: List[Dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return preds
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
    for box, score, cls_id in zip(xyxy, conf, cls):
        preds.append({"xyxy": [float(v) for v in box], "conf": float(score), "cls": int(cls_id)})
    return preds


def safe_font() -> Optional[Any]:
    """获取绘图字体。"""

    try:
        return ImageFont.truetype("arial.ttf", 14)
    except Exception:
        return None


def draw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], color: str, text: str = "") -> None:
    """绘制框和文本。"""

    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    if text:
        draw.text((x1, max(y1 - 16, 0)), text, fill=color, font=safe_font())


def save_overlay(
    image_path: Path,
    output_path: Path,
    gt_records: Sequence[Dict[str, Any]],
    pred_records: Sequence[Dict[str, Any]],
) -> None:
    """保存 GT/预测叠加图。"""

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for gt in gt_records:
        if gt["matched"]:
            draw_box(draw, gt["xyxy"], "lime", "GT✓")
        else:
            draw_box(draw, gt["xyxy"], "yellow", "FN")

    for pred in pred_records:
        if pred["matched"]:
            draw_box(draw, pred["xyxy"], "cyan", f"TP {pred['conf']:.2f}")
        elif pred["duplicate"]:
            draw_box(draw, pred["xyxy"], "orange", f"DUP {pred['conf']:.2f}")
        else:
            draw_box(draw, pred["xyxy"], "red", f"FP {pred['conf']:.2f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """写 CSV。"""

    if not records:
        path.write_text("", encoding="utf-8")
        return
    clean_records: List[Dict[str, Any]] = []
    for record in records:
        clean = {}
        for key, value in record.items():
            if key == "xyxy" and isinstance(value, (list, tuple)):
                clean[key] = " ".join(f"{float(v):.3f}" for v in value)
            else:
                clean[key] = value
        clean_records.append(clean)
    keys = sorted(set().union(*(record.keys() for record in clean_records)))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for record in clean_records:
            writer.writerow(record)


def numeric_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """数值摘要。"""

    clean = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "p10": float(np.percentile(clean, 10)),
        "p50": float(np.percentile(clean, 50)),
        "p90": float(np.percentile(clean, 90)),
    }


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    """执行预测与错误分析。"""

    data_file = Path(args.data).expanduser().resolve()
    data = load_yaml(data_file)
    nc = int(data.get("nc", 1))
    images = collect_images(data_file, args.split)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"未找到待分析图像：data={data_file}, split={args.split}")

    output = Path(args.output) if args.output else ROOT / "runs" / "analysis" / f"errors_{Path(args.weights).stem}_imgsz{args.imgsz}_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    per_image: List[Dict[str, Any]] = []
    gt_details: List[Dict[str, Any]] = []
    fp_details: List[Dict[str, Any]] = []
    label_errors: List[str] = []
    overlay_candidates: List[Tuple[float, Path, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]] = []

    print(f"开始预测与错误分析：图像数={len(images)}, imgsz={args.imgsz}, conf={args.conf}, match_iou={args.match_iou}")
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        batch=args.batch,
        device=args.device,
        stream=True,
        verbose=False,
    )

    for result in results:
        image_path = Path(result.path).resolve()
        height, width = result.orig_shape
        gts, errors = read_gt(image_path, width, height, nc)
        label_errors.extend(errors[:20])
        preds = result_to_predictions(result)
        full_tp, full_fp, full_fn, duplicate_fp = match_predictions(gts, preds, args.match_iou)
        pred_records_all = full_tp + full_fp
        # 根据匹配结果重建所有 GT 状态。
        matched_gt_indices = {record["best_gt"] for record in pred_records_all if record["matched"]}
        gt_records_all = []
        for index, gt in enumerate(gts):
            best_iou = max([box_iou(gt["xyxy"], pred["xyxy"]) for pred in preds], default=0.0)
            best_conf = None
            if preds:
                best_pred = max(preds, key=lambda pred: box_iou(gt["xyxy"], pred["xyxy"]))
                best_conf = best_pred["conf"]
            gt_records_all.append({**gt, "gt_index": index, "matched": index in matched_gt_indices, "best_iou": best_iou, "best_conf": best_conf})

        row = {
            "image": str(image_path),
            "width": width,
            "height": height,
            "gt": len(gts),
            "pred": len(preds),
            "tp": len(full_tp),
            "fp": len(full_fp),
            "fn": len(full_fn),
            "duplicate_fp": duplicate_fp,
            "recall_image": len(full_tp) / max(len(gts), 1),
            "precision_image": len(full_tp) / max(len(preds), 1),
            "mean_gt_best_iou": float(np.mean([record["best_iou"] for record in gt_records_all])) if gt_records_all else None,
            "max_conf": max([pred["conf"] for pred in preds], default=None),
        }
        per_image.append(row)

        for gt in gt_records_all:
            gt_details.append(
                {
                    "image": str(image_path),
                    "matched": gt["matched"],
                    "best_iou": gt["best_iou"],
                    "best_conf": gt["best_conf"],
                    "size_bin": gt["size_bin"],
                    "area_norm": gt["area_norm"],
                    "w_norm": gt["w_norm"],
                    "h_norm": gt["h_norm"],
                    "area_px": gt["area_px"],
                    "w_px": gt["w_px"],
                    "h_px": gt["h_px"],
                }
            )
        for pred in full_fp:
            fp_details.append(
                {
                    "image": str(image_path),
                    "conf": pred["conf"],
                    "best_iou": pred["best_iou"],
                    "duplicate": pred["duplicate"],
                    "x1": pred["xyxy"][0],
                    "y1": pred["xyxy"][1],
                    "x2": pred["xyxy"][2],
                    "y2": pred["xyxy"][3],
                }
            )

        severity = row["fn"] * 2.0 + row["fp"] + row["duplicate_fp"] * 0.5
        overlay_candidates.append((severity, image_path, gt_records_all, pred_records_all, row))

    total_tp = sum(row["tp"] for row in per_image)
    total_fp = sum(row["fp"] for row in per_image)
    total_fn = sum(row["fn"] for row in per_image)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPS)

    by_size: Dict[str, Dict[str, Any]] = {}
    for size_name in sorted({record["size_bin"] for record in gt_details}):
        subset = [record for record in gt_details if record["size_bin"] == size_name]
        matched = sum(1 for record in subset if record["matched"])
        by_size[size_name] = {
            "gt": len(subset),
            "matched": matched,
            "recall": matched / max(len(subset), 1),
            "best_iou": numeric_summary([record["best_iou"] for record in subset]),
        }

    top_fn = sorted(per_image, key=lambda row: (row["fn"], row["fp"]), reverse=True)[:30]
    top_fp = sorted(per_image, key=lambda row: (row["fp"], row["fn"]), reverse=True)[:30]
    top_duplicate = sorted(per_image, key=lambda row: row["duplicate_fp"], reverse=True)[:30]

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": str(Path(args.weights).expanduser().resolve()),
        "data": str(data_file),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "nms_iou": args.iou,
        "match_iou": args.match_iou,
        "image_count": len(per_image),
        "gt_count": sum(row["gt"] for row in per_image),
        "pred_count": sum(row["pred"] for row in per_image),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "duplicate_fp": sum(row["duplicate_fp"] for row in per_image),
        "precision_at_conf": precision,
        "recall_at_conf": recall,
        "f1_at_conf": f1,
        "gt_best_iou": numeric_summary([record["best_iou"] for record in gt_details]),
        "fp_best_iou": numeric_summary([record["best_iou"] for record in fp_details]),
        "by_size": by_size,
        "label_error_count": len(label_errors),
        "label_error_examples": label_errors[:100],
        "top_fn_images": top_fn,
        "top_fp_images": top_fp,
        "top_duplicate_images": top_duplicate,
    }

    write_csv(output / "per_image_errors.csv", per_image)
    write_csv(output / "gt_match_details.csv", gt_details)
    write_csv(output / "fp_details.csv", fp_details)
    (output / "detection_error_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(output / "detection_error_summary.md", report)

    if args.save_overlays > 0:
        overlay_dir = output / "overlays"
        for rank, (_, image_path, gt_records, pred_records, row) in enumerate(
            sorted(overlay_candidates, key=lambda item: item[0], reverse=True)[: args.save_overlays],
            start=1,
        ):
            out_name = f"{rank:03d}_fn{row['fn']}_fp{row['fp']}_dup{row['duplicate_fp']}_{image_path.name}"
            save_overlay(image_path, overlay_dir / out_name, gt_records, pred_records)

    print(f"错误分析完成：{output}")
    print(f"中文摘要：{output / 'detection_error_summary.md'}")
    print(f"完整 JSON：{output / 'detection_error_report.json'}")
    return report


def write_summary(path: Path, report: Dict[str, Any]) -> None:
    """写中文摘要。"""

    lines = [
        "# 独立测试集检测错误分析报告",
        "",
        f"- 权重：`{report['weights']}`",
        f"- 数据：`{report['data']}` split=`{report['split']}`",
        f"- imgsz={report['imgsz']}，conf={report['conf']}，NMS IoU={report['nms_iou']}，匹配 IoU={report['match_iou']}",
        "",
        "## 总体结果",
        "",
        f"- 图像数：{report['image_count']}",
        f"- GT 数：{report['gt_count']}",
        f"- 预测数：{report['pred_count']}",
        f"- TP/FP/FN：{report['tp']} / {report['fp']} / {report['fn']}",
        f"- 重复 FP：{report['duplicate_fp']}",
        f"- 当前 conf 下 Precision：{report['precision_at_conf']:.4f}",
        f"- 当前 conf 下 Recall：{report['recall_at_conf']:.4f}",
        f"- 当前 conf 下 F1：{report['f1_at_conf']:.4f}",
        f"- GT 最佳 IoU 中位数：{report['gt_best_iou']['p50']}",
        "",
        "## 按目标尺度召回",
        "",
        "| 尺度 | GT | 命中 | Recall | 最佳 IoU 中位数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for size_name, item in report["by_size"].items():
        lines.append(
            f"| {size_name} | {item['gt']} | {item['matched']} | {item['recall']:.4f} | {item['best_iou']['p50']} |"
        )

    lines.extend(
        [
            "",
            "## 自动判断",
            "",
        ]
    )
    if report["fn"] > report["fp"] * 1.5:
        lines.append("- 漏检明显多于误检，优先提升召回：降低 conf、增强目标域微调、切片推理或提高目标域覆盖。")
    elif report["fp"] > report["fn"] * 1.5:
        lines.append("- 误检明显多于漏检，优先提升背景抑制：加入 hard negative、提高伪标签阈值、增强 SAR 背景多样性。")
    else:
        lines.append("- 漏检和误检都比较明显，需要同时处理目标域适应与背景误检。")
    if report["duplicate_fp"] > report["tp"] * 0.2:
        lines.append("- 重复预测较多，建议测试更高 NMS IoU/更低 max_det，或检查密集目标标注与 NMS 参数。")
    if report["gt_best_iou"]["p50"] is not None and report["gt_best_iou"]["p50"] < 0.75:
        lines.append("- GT 最佳 IoU 中位数低于 0.75，AP75 低主要来自定位不准或框尺度不匹配。")
    if report["label_error_count"] > 0:
        lines.append("- 标签存在异常，请先修复标签，否则错误分析会偏差。")

    lines.extend(
        [
            "",
            "## 最严重漏检图 Top 10",
            "",
            "| image | GT | TP | FP | FN | DUP |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["top_fn_images"][:10]:
        lines.append(f"| `{row['image']}` | {row['gt']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['duplicate_fp']} |")

    lines.extend(["", "完整机器可读结果见 `detection_error_report.json`。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""

    parser = argparse.ArgumentParser(description="独立测试集逐图检测错误分析")
    parser.add_argument("--weights", required=True, help="待分析 best.pt")
    parser.add_argument("--data", default=str(ROOT / "test.yaml"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7, help="预测阶段 NMS IoU")
    parser.add_argument("--match-iou", type=float, default=0.5, help="错误分析中 TP 匹配 IoU")
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="只分析前 N 张图；0 表示全部")
    parser.add_argument("--save-overlays", type=int, default=40, help="保存多少张严重错误可视化图；0 表示不保存")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    """主入口。"""

    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
