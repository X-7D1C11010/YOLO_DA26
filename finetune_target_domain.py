#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""目标域小样本综合微调脚本。

本脚本面向当前“光学/SAR 对抗域自适应目标检测”项目，统一支持两类微调：

1. 有标注目标域微调：
   使用少量目标域人工标注数据直接微调模型。

2. 无标注目标域微调：
   使用教师模型在目标域图像上生成高置信伪标签，再用伪标签进行自训练。

脚本支持一次传入多个权重文件，逐个训练、评估并保存 summary，方便比较
epoch20/epoch30/best.pt 等不同 checkpoint 在独立目标域上的真实表现。
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# 降低长时间训练时的显存碎片风险。该环境变量需在 torch 初始化前设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
YOLO_SOURCE = ROOT / "yolo_source"
if str(YOLO_SOURCE) not in sys.path:
    sys.path.insert(0, str(YOLO_SOURCE))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOGGER = logging.getLogger("target_finetune")


@dataclass(frozen=True)
class TrainResult:
    """一次微调的核心结果。"""

    mode: str
    weight: Path
    run_dir: Path
    best: Path
    last: Path
    before_best_map50: float | None
    after_best_map50: float | None


def parse_sizes(value: str) -> list[int]:
    """解析逗号分隔的评估尺度。"""

    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("尺度列表必须为逗号分隔的正整数，例如 512,576,640")
    return sizes


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="目标域有标注/无标注综合微调脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("labeled", "unlabeled", "both"), default="labeled", help="微调模式")
    parser.add_argument("--weights", nargs="+", required=True, help="一个或多个初始权重、目录或 glob")
    parser.add_argument(
        "--data",
        help="目标域有标注数据 YAML；labeled 模式必需，unlabeled 模式中用于读取 train 图像与 val 标注",
    )
    parser.add_argument(
        "--target-images",
        help="无标注目标域图像目录或 txt；若不提供，则从 --data 的 train split 读取",
    )
    parser.add_argument(
        "--anchor-data",
        help="可选的有标注锚定数据 YAML；无标注自训练时混入其 train split，降低伪标签漂移风险",
    )
    parser.add_argument(
        "--eval-data",
        help="训练前后额外评估使用的数据 YAML；默认使用 --data",
    )
    parser.add_argument("--project", default=str(ROOT / "runs" / "target_finetune"), help="输出根目录")
    parser.add_argument("--name", default="target_ft", help="实验名前缀")
    parser.add_argument("--exist-ok", action="store_true", help="允许复用已有输出目录")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=40, help="微调轮数")
    parser.add_argument("--imgsz", type=int, default=768, help="训练尺度")
    parser.add_argument("--batch", type=int, default=8, help="训练 batch")
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--device", default="0", help="CUDA 设备，例如 0")
    parser.add_argument("--optimizer", default="AdamW", choices=("AdamW", "SGD", "auto"), help="优化器")
    parser.add_argument("--lr0", type=float, default=2e-4, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.05, help="最终学习率比例")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="权重衰减")
    parser.add_argument("--warmup-epochs", type=float, default=2.0, help="warmup 轮数")
    parser.add_argument("--freeze", type=int, default=5, help="冻结前 N 层；0 表示不冻结")
    parser.add_argument("--patience", type=int, default=12, help="早停 patience")
    parser.add_argument("--save-period", type=int, default=5, help="每隔多少轮保存 epoch*.pt；-1 表示不额外保存")
    parser.add_argument("--no-amp", action="store_true", help="关闭 AMP；若出现 NaN 可尝试开启该参数")

    # 增强参数：目标域小样本微调默认采用保守增强。
    parser.add_argument("--mosaic", type=float, default=0.0, help="mosaic 概率")
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup 概率")
    parser.add_argument("--degrees", type=float, default=3.0, help="旋转增强角度")
    parser.add_argument("--translate", type=float, default=0.04, help="平移增强比例")
    parser.add_argument("--scale", type=float, default=0.12, help="缩放增强比例")
    parser.add_argument("--flipud", type=float, default=0.5, help="上下翻转概率")
    parser.add_argument("--fliplr", type=float, default=0.5, help="左右翻转概率")
    parser.add_argument("--hsv-v", type=float, default=0.08, help="亮度扰动")
    parser.add_argument("--close-mosaic", type=int, default=0, help="关闭 mosaic 的轮数")

    # 伪标签参数
    parser.add_argument("--pseudo-conf", type=float, default=0.45, help="伪标签置信度阈值")
    parser.add_argument("--pseudo-iou", type=float, default=0.60, help="伪标签 NMS IoU")
    parser.add_argument("--min-box-area", type=float, default=1e-5, help="伪标签框最小归一化面积")
    parser.add_argument("--max-box-area", type=float, default=0.60, help="伪标签框最大归一化面积")
    parser.add_argument("--max-det", type=int, default=300, help="每张图最大检测框数")
    parser.add_argument("--min-pseudo-images", type=int, default=5, help="伪标签图像数量安全下限")
    parser.add_argument("--include-empty-pseudo", action="store_true", help="保留无伪标签图像为空标签背景图")
    parser.add_argument("--pseudo-link-mode", choices=("copy", "hardlink", "symlink"), default="copy", help="伪标签图像落盘方式")
    parser.add_argument("--overwrite-pseudo", action="store_true", help="若伪标签目录已存在则重建")

    # 评估与监控
    parser.add_argument("--eval-imgsz", type=parse_sizes, default=parse_sizes("512,576,640,704,768,1024"), help="训练前后评估尺度")
    parser.add_argument("--eval-batch", type=int, default=4, help="评估 batch")
    parser.add_argument("--eval-conf", type=float, default=0.001, help="评估置信度")
    parser.add_argument("--eval-iou", type=float, default=0.7, help="评估 NMS IoU")
    parser.add_argument("--half", action="store_true", help="评估/伪标签预测使用 FP16")
    parser.add_argument("--skip-before-eval", action="store_true", help="跳过微调前评估以节省时间")
    parser.add_argument("--no-eval-plots", action="store_true", help="关闭评估 plots")
    parser.add_argument("--clear-cache-each-epoch", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--no-clear-cache-each-epoch", dest="clear_cache_each_epoch", action="store_false", help="关闭每轮 CUDA 缓存清理")
    parser.add_argument("--stop-on-nan", action="store_true", help="检测到非有限 loss 时立即停止当前训练")
    return parser.parse_args()


def configure_logging(project: Path) -> None:
    """配置中文日志，同时写入文件和控制台。"""

    project.mkdir(parents=True, exist_ok=True)
    log_path = project / "finetune_target_domain.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    LOGGER.info("日志文件：%s", log_path)


def expand_weights(values: list[str]) -> list[Path]:
    """展开权重路径、目录和 glob。"""

    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.glob("*.pt")))
        elif any(char in value for char in "*?[]"):
            paths.extend(Path(item).expanduser() for item in sorted(glob.glob(value)))
        else:
            paths.append(path)

    unique = list(dict.fromkeys(path.resolve() for path in paths))
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError("以下权重不存在：\n" + "\n".join(missing))
    if not unique:
        raise FileNotFoundError("没有找到任何权重文件")
    return unique


def _as_list(value: Any) -> list[Any]:
    """把 YAML 单值/list 统一为列表。"""

    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _resolve_base(config: dict[str, Any], yaml_path: Path) -> Path:
    """解析 YOLO YAML 的 path。"""

    base = Path(config.get("path") or yaml_path.parent).expanduser()
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    return base


def resolve_split_entries(data_yaml: Path, split: str) -> tuple[dict[str, Any], list[str]]:
    """读取 YAML 中 split 对应的绝对路径列表。"""

    config = YAML.load(data_yaml)
    base = _resolve_base(config, data_yaml)
    entries: list[str] = []
    for item in _as_list(config.get(split)):
        path = Path(str(item)).expanduser()
        entries.append(str(path if path.is_absolute() else (base / path).resolve()))
    return config, entries


def format_names(config: dict[str, Any]) -> tuple[int, list[str]]:
    """从 YAML 中读取 nc 与 names。"""

    names = config.get("names", {0: "aircraft"})
    if isinstance(names, list):
        name_list = [str(item) for item in names]
    elif isinstance(names, dict):
        name_list = [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    else:
        name_list = ["aircraft"]
    nc = int(config.get("nc") or len(name_list))
    return nc, name_list[:nc]


def collect_images(path_like: str | Path) -> list[Path]:
    """从目录、图像文件或 txt 清单中收集图像路径。"""

    path = Path(path_like).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".txt":
        images = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            image = Path(line).expanduser()
            if not image.is_absolute():
                image = (path.parent / image).resolve()
            images.append(image)
    elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        images = [path]
    elif path.is_dir():
        images = sorted(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(f"无法读取目标域图像路径：{path}")

    missing = [str(image) for image in images if not image.exists()]
    if missing:
        raise FileNotFoundError("以下图像不存在：\n" + "\n".join(missing[:20]))
    if not images:
        raise FileNotFoundError(f"未找到图像：{path}")
    return images


def infer_target_images_from_data(data_yaml: Path) -> list[Path]:
    """从数据 YAML 的 train split 推断无标注目标图像。"""

    _, train_entries = resolve_split_entries(data_yaml, "train")
    images: list[Path] = []
    for entry in train_entries:
        images.extend(collect_images(entry))
    return sorted(dict.fromkeys(images))


def safe_run_name(prefix: str, weight: Path, mode: str) -> str:
    """构造稳定的运行名。"""

    stem = weight.stem.replace(" ", "_")
    return f"{prefix}_{stem}_{mode}"


def clear_cuda_cache(note: str) -> None:
    """主动清理 Python 与 CUDA 缓存。"""

    gc.collect()
    if torch.cuda.is_available():
        before_reserved = torch.cuda.memory_reserved() / (1024**3)
        before_allocated = torch.cuda.memory_allocated() / (1024**3)
        torch.cuda.empty_cache()
        after_reserved = torch.cuda.memory_reserved() / (1024**3)
        after_allocated = torch.cuda.memory_allocated() / (1024**3)
        LOGGER.info(
            "%s：已清理 CUDA 缓存，reserved %.2f -> %.2f GB，allocated %.2f -> %.2f GB",
            note,
            before_reserved,
            after_reserved,
            before_allocated,
            after_allocated,
        )


def register_monitor_callbacks(model: YOLO, args: argparse.Namespace) -> None:
    """注册训练监控回调：记录指标、检测 NaN、每轮清理缓存。"""

    def on_train_batch_end(trainer) -> None:  # noqa: ANN001
        loss = getattr(trainer, "loss", None)
        if loss is not None and hasattr(loss, "isfinite") and not bool(loss.isfinite()):
            message = f"检测到非有限 loss：epoch={getattr(trainer, 'epoch', '?')}，loss={loss}"
            LOGGER.error(message)
            if args.stop_on_nan:
                raise FloatingPointError(message)

    def on_fit_epoch_end(trainer) -> None:  # noqa: ANN001
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        metrics = getattr(trainer, "metrics", {}) or {}
        fitness = getattr(trainer, "fitness", None)
        metric_text = ", ".join(
            f"{key}={float(value):.4f}"
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        )
        LOGGER.info("第 %s 轮结束：fitness=%s，%s", epoch, fitness, metric_text or "暂无验证指标")
        if args.clear_cache_each_epoch:
            clear_cuda_cache(f"第 {epoch} 轮")

    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def val_once(
    weight: Path,
    data_yaml: Path,
    imgsz: int,
    args: argparse.Namespace,
    project: Path,
    name: str,
) -> dict[str, Any]:
    """在一个尺度上评估一次模型。"""

    model = YOLO(str(weight))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=args.eval_batch,
        device=args.device,
        conf=args.eval_conf,
        iou=args.eval_iou,
        max_det=args.max_det,
        half=args.half,
        rect=True,
        plots=not args.no_eval_plots,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": str(weight),
        "data": str(data_yaml),
        "imgsz": imgsz,
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "map50_95": float(metrics.box.map),
        "precision": float(getattr(metrics.box, "mp", 0.0)),
        "recall": float(getattr(metrics.box, "mr", 0.0)),
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }


def evaluate_grid(
    weight: Path,
    data_yaml: Path | None,
    args: argparse.Namespace,
    run_dir: Path,
    stage: str,
) -> tuple[list[dict[str, Any]], float | None]:
    """多尺度评估，并返回记录与最佳 mAP50。"""

    if data_yaml is None:
        return [], None
    if not data_yaml.exists():
        raise FileNotFoundError(f"评估 YAML 不存在：{data_yaml}")

    records: list[dict[str, Any]] = []
    eval_project = run_dir / "eval"
    for imgsz in args.eval_imgsz:
        LOGGER.info("评估 %s：%s @ imgsz=%s", stage, weight.name, imgsz)
        record = val_once(weight, data_yaml, imgsz, args, eval_project, f"{stage}_imgsz{imgsz}")
        records.append(record)
        LOGGER.info(
            "%s @ %s：mAP50=%.4f，mAP50-95=%.4f，P=%.4f，R=%.4f",
            stage,
            imgsz,
            record["map50"],
            record["map50_95"],
            record["precision"],
            record["recall"],
        )
    best_map50 = max((record["map50"] for record in records), default=None)
    return records, best_map50


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """追加写 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def train_kwargs(args: argparse.Namespace, data_yaml: Path, project: Path, name: str) -> dict[str, Any]:
    """集中管理 Ultralytics train 参数。"""

    return {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(project),
        "name": name,
        "exist_ok": args.exist_ok,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": True,
        "freeze": args.freeze,
        "patience": args.patience,
        "save_period": args.save_period,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "hsv_h": 0.0,
        "hsv_s": 0.04,
        "hsv_v": args.hsv_v,
        "close_mosaic": args.close_mosaic,
        "amp": not args.no_amp,
        "val": True,
        "plots": True,
    }


def train_labeled(weight: Path, args: argparse.Namespace, project: Path) -> TrainResult:
    """执行有标注目标域微调。"""

    if not args.data:
        raise ValueError("labeled 模式必须提供 --data")
    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"有标注数据 YAML 不存在：{data_yaml}")
    eval_yaml = Path(args.eval_data).expanduser().resolve() if args.eval_data else data_yaml
    run_name = safe_run_name(args.name, weight, "labeled")
    run_dir = project / run_name

    before_records: list[dict[str, Any]] = []
    before_best = None
    if not args.skip_before_eval:
        before_records, before_best = evaluate_grid(weight, eval_yaml, args, run_dir, "before")
        write_jsonl(project / "evaluation_records.jsonl", before_records)

    LOGGER.info("开始有标注微调：weight=%s，data=%s", weight, data_yaml)
    model = YOLO(str(weight))
    register_monitor_callbacks(model, args)
    results = model.train(**train_kwargs(args, data_yaml, project, run_name))
    best = Path(results.save_dir) / "weights" / "best.pt"
    last = Path(results.save_dir) / "weights" / "last.pt"

    after_records, after_best = evaluate_grid(best, eval_yaml, args, Path(results.save_dir), "after")
    write_jsonl(project / "evaluation_records.jsonl", after_records)
    return TrainResult("labeled", weight, Path(results.save_dir), best, last, before_best, after_best)


def copy_or_link_image(source: Path, destination: Path, mode: str) -> str:
    """复制、硬链接或软链接伪标签图像。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        os.symlink(source.resolve(), destination)
    else:
        raise ValueError(f"未知图像落盘方式：{mode}")
    return mode


def unique_relative_paths(images: list[Path]) -> list[tuple[Path, Path]]:
    """为伪标签图像生成不冲突的相对路径。"""

    used: set[Path] = set()
    output: list[tuple[Path, Path]] = []
    for image in images:
        relative = Path(image.name)
        if relative in used:
            relative = Path(image.parent.name) / image.name
        suffix = 1
        original = relative
        while relative in used:
            relative = original.with_name(f"{original.stem}_{suffix}{original.suffix}")
            suffix += 1
        used.add(relative)
        output.append((image, relative))
    return output


def finite_box(box: list[float]) -> bool:
    """检查 YOLO xywhn 框是否合法。"""

    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        return False
    x, y, w, h = [float(value) for value in box]
    return 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1


def generate_pseudo_dataset(
    teacher_weight: Path,
    target_images: list[Path],
    output: Path,
    nc: int,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    """生成伪标签数据集，返回伪标签图像目录和统计信息。"""

    if output.exists():
        if not args.overwrite_pseudo:
            raise FileExistsError(f"伪标签目录已存在：{output}。如需重建请添加 --overwrite-pseudo")
        shutil.rmtree(output)
    image_root = output / "images" / "train"
    label_root = output / "labels" / "train"
    label_root.mkdir(parents=True, exist_ok=True)

    teacher = YOLO(str(teacher_weight))
    stats: dict[str, Any] = {
        "teacher": str(teacher_weight),
        "source_images": len(target_images),
        "accepted_images": 0,
        "empty_images": 0,
        "accepted_boxes": 0,
        "rejected_boxes": 0,
        "pseudo_conf": args.pseudo_conf,
        "pseudo_iou": args.pseudo_iou,
    }

    image_pairs = unique_relative_paths(target_images)
    predictions = teacher.predict(
        source=[str(image) for image, _ in image_pairs],
        stream=True,
        imgsz=args.imgsz,
        conf=args.pseudo_conf,
        iou=args.pseudo_iou,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
        verbose=False,
    )

    for (source_image, relative), result in zip(image_pairs, predictions):
        accepted: list[tuple[int, float, float, float, float]] = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xywhn = boxes.xywhn.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            for cls_id, confidence, box in zip(classes, confidences, xywhn):
                class_id = int(cls_id)
                if class_id < 0 or class_id >= nc or float(confidence) < args.pseudo_conf or not finite_box(box):
                    stats["rejected_boxes"] += 1
                    continue
                x, y, w, h = [float(value) for value in box]
                area = w * h
                if area < args.min_box_area or area > args.max_box_area:
                    stats["rejected_boxes"] += 1
                    continue
                accepted.append((class_id, x, y, w, h))

        if not accepted and not args.include_empty_pseudo:
            stats["empty_images"] += 1
            continue

        dst_image = image_root / relative
        dst_label = (label_root / relative).with_suffix(".txt")
        copy_or_link_image(source_image, dst_image, args.pseudo_link_mode)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text(
            "".join(f"{cls_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for cls_id, x, y, w, h in accepted),
            encoding="utf-8",
        )
        stats["accepted_images"] += 1
        stats["accepted_boxes"] += len(accepted)

    if stats["accepted_images"] < args.min_pseudo_images:
        raise RuntimeError(
            f"伪标签图像仅 {stats['accepted_images']} 张，低于安全下限 {args.min_pseudo_images}。"
            "建议检查教师模型、降低 --pseudo-conf，或改用有标注微调。"
        )

    (output / "pseudo_manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info(
        "伪标签完成：%s/%s 张图像，%s 个框",
        stats["accepted_images"],
        stats["source_images"],
        stats["accepted_boxes"],
    )
    return image_root, stats


def write_dataset_yaml(
    output_yaml: Path,
    train_entries: list[str],
    val_entries: list[str],
    nc: int,
    names: list[str],
) -> None:
    """写出 YOLO 数据集 YAML。"""

    lines = ["# 由 finetune_target_domain.py 自动生成。", "path: /", "train:"]
    lines.extend(f"  - {json.dumps(entry, ensure_ascii=False)}" for entry in train_entries)
    lines.append("val:")
    lines.extend(f"  - {json.dumps(entry, ensure_ascii=False)}" for entry in val_entries)
    lines.append("test:")
    lines.extend(f"  - {json.dumps(entry, ensure_ascii=False)}" for entry in val_entries)
    lines.append(f"nc: {nc}")
    lines.append("names:")
    for index, name in enumerate(names[:nc]):
        lines.append(f"  {index}: {json.dumps(name, ensure_ascii=False)}")
    output_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_unlabeled(weight: Path, args: argparse.Namespace, project: Path) -> TrainResult:
    """执行无标注目标域伪标签微调。"""

    if not args.data and not args.target_images:
        raise ValueError("unlabeled 模式需要 --target-images，或提供含 train split 的 --data")
    if not args.data:
        raise ValueError("unlabeled 模式仍需要 --data 提供 val/test 标注，用于训练后验证与选型")

    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"验证数据 YAML 不存在：{data_yaml}")
    config, val_entries = resolve_split_entries(data_yaml, "val")
    if not val_entries:
        raise ValueError("unlabeled 模式的 --data 必须包含带标注的 val split，用于监控微调效果")
    nc, names = format_names(config)

    target_images = collect_images(args.target_images) if args.target_images else infer_target_images_from_data(data_yaml)
    eval_yaml = Path(args.eval_data).expanduser().resolve() if args.eval_data else data_yaml
    run_name = safe_run_name(args.name, weight, "unlabeled")
    run_dir = project / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    before_records: list[dict[str, Any]] = []
    before_best = None
    if not args.skip_before_eval:
        before_records, before_best = evaluate_grid(weight, eval_yaml, args, run_dir, "before")
        write_jsonl(project / "evaluation_records.jsonl", before_records)

    pseudo_root = run_dir / "pseudo_dataset"
    pseudo_images, pseudo_stats = generate_pseudo_dataset(weight, target_images, pseudo_root, nc, args)

    train_entries = [str(pseudo_images.resolve())]
    if args.anchor_data:
        anchor_yaml = Path(args.anchor_data).expanduser().resolve()
        if not anchor_yaml.exists():
            raise FileNotFoundError(f"锚定数据 YAML 不存在：{anchor_yaml}")
        _, anchor_train = resolve_split_entries(anchor_yaml, "train")
        train_entries.extend(anchor_train)
        LOGGER.info("无标注微调混入锚定数据：%s", anchor_yaml)

    generated_yaml = run_dir / "pseudo_train.yaml"
    write_dataset_yaml(generated_yaml, train_entries, val_entries, nc, names)
    (run_dir / "pseudo_stats.json").write_text(json.dumps(pseudo_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("开始无标注伪标签微调：weight=%s，pseudo_yaml=%s", weight, generated_yaml)
    model = YOLO(str(weight))
    register_monitor_callbacks(model, args)
    results = model.train(**train_kwargs(args, generated_yaml, project, run_name))
    best = Path(results.save_dir) / "weights" / "best.pt"
    last = Path(results.save_dir) / "weights" / "last.pt"

    after_records, after_best = evaluate_grid(best, eval_yaml, args, Path(results.save_dir), "after")
    write_jsonl(project / "evaluation_records.jsonl", after_records)
    return TrainResult("unlabeled", weight, Path(results.save_dir), best, last, before_best, after_best)


def append_summary(project: Path, result: TrainResult, args: argparse.Namespace) -> None:
    """追加写入微调 summary。"""

    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": result.mode,
        "initial_weight": str(result.weight),
        "run_dir": str(result.run_dir),
        "best": str(result.best),
        "last": str(result.last),
        "before_best_map50": result.before_best_map50,
        "after_best_map50": result.after_best_map50,
        "delta_map50": (
            None
            if result.before_best_map50 is None or result.after_best_map50 is None
            else result.after_best_map50 - result.before_best_map50
        ),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": args.lr0,
        "freeze": args.freeze,
    }
    with (project / "finetune_summary.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("微调完成：mode=%s，best=%s，after_best_map50=%s", result.mode, result.best, result.after_best_map50)


def validate_args(args: argparse.Namespace) -> None:
    """检查关键参数组合。"""

    if args.epochs <= 0:
        raise ValueError("--epochs 必须为正整数")
    if args.batch <= 0:
        raise ValueError("--batch 必须为正整数")
    if args.lr0 <= 0:
        raise ValueError("--lr0 必须为正数")
    if args.mode in {"labeled", "both"} and not args.data:
        raise ValueError("labeled/both 模式必须提供 --data")
    if not 0 <= args.pseudo_conf <= 1:
        raise ValueError("--pseudo-conf 必须位于 [0, 1]")
    if not 0 < args.min_box_area < args.max_box_area <= 1:
        raise ValueError("伪标签面积阈值需满足 0 < min < max <= 1")


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    validate_args(args)
    project = Path(args.project).expanduser().resolve()
    configure_logging(project)
    weights = expand_weights(args.weights)

    LOGGER.info("待微调权重数：%s", len(weights))
    LOGGER.info("微调模式：%s；训练尺度=%s；batch=%s；lr0=%s", args.mode, args.imgsz, args.batch, args.lr0)

    for weight in weights:
        LOGGER.info("处理权重：%s", weight)
        if args.mode in {"labeled", "both"}:
            result = train_labeled(weight, args, project)
            append_summary(project, result, args)
            clear_cuda_cache("有标注微调结束")
        if args.mode in {"unlabeled", "both"}:
            result = train_unlabeled(weight, args, project)
            append_summary(project, result, args)
            clear_cuda_cache("无标注微调结束")

    LOGGER.info("全部任务完成。汇总文件：%s", project / "finetune_summary.jsonl")
    LOGGER.info("评估明细：%s", project / "evaluation_records.jsonl")


if __name__ == "__main__":
    main()
