#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO26 光学图像与 SAR 图像域对抗训练脚本。

本脚本是项目中 YOLO26-DANN 训练流程的主入口，负责：

1. 在训练启动前检查数据集 YAML、图像目录和标签文件，避免训练进入 dataloader 后才报错；
2. 构建带 DomainClassifier 的 YOLO26 域对抗模型；
3. 加载检测预训练权重，并显式迁移检测头中形状兼容的权重；
4. 配置检测损失 + 域对抗损失、优化器、学习率、数据增强和保存策略；
5. 注册中文训练日志回调，持续输出域对抗系数与域分类损失；
6. 按 24GB 显存场景提供 1024 尺度训练默认参数，并在每轮结束后主动清理 CUDA 缓存。

说明：
    逐 batch 的前向、反向传播、损失汇总、EMA、验证和 checkpoint 保存由本项目
    yolo_source 中的 Ultralytics Trainer 执行。这样可以复用已有 YOLO 训练能力，
    同时由本脚本集中管理域自适应训练所需的额外配置与安全检查。
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 在导入 torch 前设置，降低长时间训练中的 CUDA 显存碎片风险。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


ROOT = Path(__file__).resolve().parent
YOLO_SOURCE = ROOT / "yolo_source"
if str(YOLO_SOURCE) not in sys.path:
    sys.path.insert(0, str(YOLO_SOURCE))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import IMG_FORMATS  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402


LOGGER = logging.getLogger("train_26")
IMAGE_SUFFIXES = {f".{suffix.lower()}" for suffix in IMG_FORMATS}
SOURCE_DOMAIN_TOKENS = ("/vis", "/optical", "/rgb", "/source")
TARGET_DOMAIN_TOKENS = ("/sar", "sar_", "_sar", "synthetic_aperture", "/target")


class DataConfigError(RuntimeError):
    """数据集配置错误。"""


@dataclass
class EntryReport:
    """单个数据路径的检查结果。"""

    raw: str
    path: Path
    exists: bool
    image_count: int = 0
    examples: List[Path] = field(default_factory=list)
    domain: str = "未知域"


@dataclass
class SplitReport:
    """train/val/test 一个数据划分的检查结果。"""

    name: str
    entries: List[EntryReport]
    image_count: int
    label_checked: int = 0
    label_missing: int = 0
    label_empty: int = 0
    label_invalid: int = 0

    @property
    def missing_entries(self) -> List[EntryReport]:
        return [entry for entry in self.entries if not entry.exists]


def _configure_logging(verbose: bool = False) -> None:
    """配置中文日志格式。"""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _positive_int(value: str) -> int:
    """argparse 用的正整数校验。"""

    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("该参数必须是正整数")
    return number


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(
        description="YOLO26 光学-SAR 域对抗训练入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 数据配置
    parser.add_argument("--data", default=str(ROOT / "dataset_aircraft_da.yaml"), help="训练数据集 YAML 配置文件")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("YOLO_DA_DATA_ROOT"),
        help="覆盖 YAML 中的 path 字段；也可通过环境变量 YOLO_DA_DATA_ROOT 设置",
    )
    parser.add_argument(
        "--no-auto-data-root",
        action="store_true",
        help="关闭数据根目录自动探测；默认会尝试 ROOT/Datasets、上级 Datasets 等常见位置",
    )
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        help="允许训练集中存在缺失或空标签；默认只告警，不阻断训练",
    )
    parser.add_argument(
        "--label-check-samples",
        type=int,
        default=300,
        help="每个划分最多抽查多少张图像的标签文件",
    )
    parser.add_argument(
        "--precheck-only",
        action="store_true",
        help="只执行数据集与模型配置预检查，不启动训练",
    )

    # 模型配置
    parser.add_argument("--size", choices=("n", "s", "m", "l", "x"), default="s", help="YOLO26 模型规模")
    parser.add_argument(
        "--weights",
        default=None,
        help="检测预训练权重路径；默认使用 yolo26{size}.pt，若本地不存在则由 Ultralytics 尝试下载",
    )
    parser.add_argument("--fresh", action="store_true", help="完全随机初始化，不加载检测预训练权重")

    # 训练流程
    parser.add_argument("--epochs", type=_positive_int, default=120, help="训练轮数")
    parser.add_argument("--imgsz", type=_positive_int, default=1024, help="输入图像尺寸；24GB 显存主实验建议使用 1024")
    parser.add_argument("--batch", type=_positive_int, default=24, help="batch size；24GB 显存 YOLO26s@1024 可先尝试 24，OOM 时降到 20 或 16")
    parser.add_argument("--device", default="0", help="训练设备，例如 0、0,1 或 cpu")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数；24GB 单卡建议 4，CPU/内存紧张时降到 2")
    parser.add_argument("--project", default=str(ROOT / "runs" / "detect"), help="训练输出根目录")
    parser.add_argument("--name", default=None, help="本次训练名称；默认自动生成")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖同名训练目录")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--patience", type=int, default=30, help="早停耐心轮数")
    parser.add_argument("--save-period", type=int, default=10, help="每隔多少轮额外保存一次 checkpoint")
    parser.add_argument("--verbose", action="store_true", help="输出更详细的调试日志")
    parser.add_argument("--max-vram-gb", type=float, default=24.0, help="用于显存风险提示的可用显存估计值")
    parser.set_defaults(clear_cache_each_epoch=True)
    parser.add_argument(
        "--clear-cache-each-epoch",
        dest="clear_cache_each_epoch",
        action="store_true",
        help="每轮训练/验证结束后主动执行 gc.collect 与 torch.cuda.empty_cache",
    )
    parser.add_argument(
        "--no-clear-cache-each-epoch",
        dest="clear_cache_each_epoch",
        action="store_false",
        help="关闭每轮 CUDA 缓存清理",
    )

    # 优化器与损失
    parser.add_argument("--optimizer", choices=("AdamW", "Adam", "SGD"), default="AdamW", help="优化器")
    parser.add_argument("--lr0", type=float, default=0.0007, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.05, help="最终学习率比例")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="权重衰减")
    parser.add_argument("--warmup-epochs", type=float, default=5.0, help="学习率 warmup 轮数")
    parser.add_argument("--nbs", type=_positive_int, default=64, help="名义 batch size；24GB 大 batch 训练建议 64")
    parser.add_argument("--domain-weight", type=float, default=0.002, help="域对抗损失权重")
    parser.add_argument("--alpha-max", type=float, default=1.0, help="梯度反转层最大对抗系数")

    # 数据增强。默认关闭 Mosaic/MixUp，因为跨域拼接会破坏单张图像的域标签。
    parser.add_argument("--degrees", type=float, default=10.0, help="随机旋转角度")
    parser.add_argument("--translate", type=float, default=0.08, help="随机平移比例")
    parser.add_argument("--scale-aug", type=float, default=0.30, help="随机缩放比例")
    parser.add_argument("--shear", type=float, default=0.0, help="随机错切角度")
    parser.add_argument("--perspective", type=float, default=0.0, help="随机透视变换强度")
    parser.add_argument("--flipud", type=float, default=0.5, help="上下翻转概率")
    parser.add_argument("--fliplr", type=float, default=0.5, help="左右翻转概率")
    parser.add_argument("--hsv-h", type=float, default=0.0, help="色调扰动")
    parser.add_argument("--hsv-s", type=float, default=0.05, help="饱和度扰动")
    parser.add_argument("--hsv-v", type=float, default=0.20, help="亮度扰动")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Mosaic 概率；域对抗训练建议保持 0")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp 概率；域对抗训练建议保持 0")
    parser.add_argument("--multi-scale", type=float, default=0.10, help="多尺度训练幅度")
    parser.add_argument("--no-amp", action="store_true", help="关闭自动混合精度 AMP")

    return parser


def _as_list(value: Any) -> List[str]:
    """把 YAML 中的 split 字段统一成字符串列表。"""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _normalise_path_text(path: Path) -> str:
    """把路径转换成便于域类型判断的统一形式。"""

    return path.as_posix().lower()


def _guess_domain(path: Path) -> str:
    """根据路径名称推断图像属于光学源域、SAR 目标域还是未知域。"""

    text = _normalise_path_text(path)
    if any(token in text for token in TARGET_DOMAIN_TOKENS):
        return "SAR目标域"
    if any(token in text for token in SOURCE_DOMAIN_TOKENS):
        return "光学源域"
    return "未知域"


def _resolve_yaml_root(data_file: Path, raw_root: Optional[str]) -> Path:
    """解析 YAML 中的 path 字段。"""

    if raw_root:
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = data_file.parent / root
        return root
    return data_file.parent


def _resolve_split_entry(
    root: Path,
    data_file: Path,
    raw_entry: str,
    allow_yaml_relative_fallback: bool = True,
) -> Path:
    """解析 train/val/test 中的单个路径。"""

    entry = Path(raw_entry).expanduser()
    if entry.is_absolute():
        return entry
    # Ultralytics 的数据 YAML 语义是：相对路径优先相对于 path 字段。
    candidate = root / entry
    if candidate.exists():
        return candidate
    # 兼容少量历史 YAML：split 直接相对于 YAML 文件所在目录。
    yaml_relative = data_file.parent / entry
    if allow_yaml_relative_fallback and yaml_relative.exists():
        return yaml_relative
    return candidate


def _iter_images_from_txt(txt_path: Path) -> Iterable[Path]:
    """从图片列表 txt 中读取图像路径。"""

    try:
        lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []

    images: List[Path] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        image_path = Path(value).expanduser()
        if not image_path.is_absolute():
            image_path = txt_path.parent / image_path
        if image_path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(image_path)
    return images


def _scan_images(path: Path, sample_limit: int) -> Tuple[int, List[Path]]:
    """统计图像数量，并保留少量样例用于标签抽查。"""

    count = 0
    examples: List[Path] = []

    if not path.exists():
        return count, examples

    if path.is_file() and path.suffix.lower() == ".txt":
        iterator = _iter_images_from_txt(path)
    elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        iterator = [path]
    elif path.is_dir():
        iterator = (
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        iterator = []

    for image_path in iterator:
        count += 1
        if len(examples) < sample_limit:
            examples.append(image_path)

    return count, examples


def _image_to_label_path(image_path: Path) -> Path:
    """按 YOLO 目录约定把 images 路径转换为 labels 路径。"""

    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _check_label_file(label_path: Path, nc: int) -> Tuple[bool, bool, bool]:
    """检查单个 YOLO 标签文件。

    Returns:
        (missing, empty, invalid): 是否缺失、是否为空、是否格式异常。
    """

    if not label_path.exists():
        return True, False, False

    try:
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    except OSError:
        return False, False, True

    valid_lines = [line for line in lines if line and not line.startswith("#")]
    if not valid_lines:
        return False, True, False

    for line in valid_lines:
        fields = line.split()
        if len(fields) < 5:
            return False, False, True
        try:
            cls = int(float(fields[0]))
            values = [float(value) for value in fields[1:5]]
        except ValueError:
            return False, False, True
        if cls < 0 or cls >= nc:
            return False, False, True
        if not all(0.0 <= value <= 1.0 for value in values):
            return False, False, True

    return False, False, False


def _inspect_split(
    name: str,
    raw_entries: Sequence[str],
    root: Path,
    data_file: Path,
    nc: int,
    label_check_samples: int,
) -> SplitReport:
    """检查一个数据划分中的路径、图像数量和标签质量。"""

    entry_reports: List[EntryReport] = []
    examples_for_label_check: List[Path] = []
    total_images = 0

    for raw_entry in raw_entries:
        resolved = _resolve_split_entry(root, data_file, raw_entry)
        count, examples = _scan_images(resolved, label_check_samples)
        entry_reports.append(
            EntryReport(
                raw=raw_entry,
                path=resolved,
                exists=resolved.exists(),
                image_count=count,
                examples=examples[:5],
                domain=_guess_domain(resolved),
            )
        )
        total_images += count
        remaining = max(label_check_samples - len(examples_for_label_check), 0)
        if remaining:
            examples_for_label_check.extend(examples[:remaining])

    label_checked = 0
    label_missing = 0
    label_empty = 0
    label_invalid = 0
    for image_path in examples_for_label_check[:label_check_samples]:
        label_checked += 1
        missing, empty, invalid = _check_label_file(_image_to_label_path(image_path), nc)
        label_missing += int(missing)
        label_empty += int(empty)
        label_invalid += int(invalid)

    return SplitReport(
        name=name,
        entries=entry_reports,
        image_count=total_images,
        label_checked=label_checked,
        label_missing=label_missing,
        label_empty=label_empty,
        label_invalid=label_invalid,
    )


def _root_has_required_splits(root: Path, data: Dict[str, Any], data_file: Path) -> bool:
    """判断一个候选根目录是否包含训练所需路径。"""

    for split in ("train", "val"):
        entries = _as_list(data.get(split))
        if not entries:
            return False
        for raw_entry in entries:
            if not _resolve_split_entry(
                root,
                data_file,
                raw_entry,
                allow_yaml_relative_fallback=False,
            ).exists():
                return False
    return True


def _candidate_roots(data_file: Path, original_root: Path) -> List[Path]:
    """生成常见数据根目录候选项。"""

    candidates = [
        original_root,
        ROOT / "Datasets",
        ROOT.parent / "Datasets",
        ROOT.parent.parent / "Datasets",
        Path.cwd() / "Datasets",
    ]
    env_root = os.environ.get("YOLO_DA_DATA_ROOT")
    if env_root:
        candidates.insert(0, Path(env_root).expanduser())

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _select_data_root(args: argparse.Namespace, data: Dict[str, Any], data_file: Path) -> Path:
    """选择最终用于训练的数据根目录。"""

    yaml_root = _resolve_yaml_root(data_file, data.get("path"))

    if args.data_root:
        root = Path(args.data_root).expanduser()
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        LOGGER.info("使用命令行或环境变量指定的数据根目录：%s", root)
        return root

    if _root_has_required_splits(yaml_root, data, data_file):
        return yaml_root

    if not args.no_auto_data_root:
        for candidate in _candidate_roots(data_file, yaml_root):
            if candidate == yaml_root:
                continue
            if _root_has_required_splits(candidate, data, data_file):
                LOGGER.warning("YAML 中的数据根目录不可用，已自动切换到：%s", candidate)
                return candidate

    return yaml_root


def _class_zero_name(data: Dict[str, Any]) -> str:
    """读取类别 0 的名称。"""

    names = data.get("names", {})
    if isinstance(names, dict):
        return str(names.get(0, names.get("0", "")))
    if isinstance(names, list) and names:
        return str(names[0])
    return ""


def _format_split_report(report: SplitReport) -> str:
    """把数据划分检查结果格式化为中文日志。"""

    lines = [f"{report.name}: 图像数={report.image_count}, 路径数={len(report.entries)}"]
    for entry in report.entries:
        state = "存在" if entry.exists else "缺失"
        lines.append(
            f"  - [{state}] {entry.raw} -> {entry.path} | 图像={entry.image_count} | 域={entry.domain}"
        )
    if report.label_checked:
        lines.append(
            "  标签抽查："
            f"抽查={report.label_checked}, 缺失={report.label_missing}, "
            f"空标签={report.label_empty}, 异常={report.label_invalid}"
        )
    return "\n".join(lines)


def _build_missing_data_message(data_file: Path, root: Path, reports: Sequence[SplitReport]) -> str:
    """生成数据路径错误的中文诊断信息。"""

    missing_paths: List[str] = []
    empty_splits: List[str] = []
    for report in reports:
        for entry in report.missing_entries:
            missing_paths.append(str(entry.path))
        if report.name in {"train", "val"} and report.image_count == 0:
            empty_splits.append(report.name)

    lines = [
        "数据集检查失败，训练未启动。",
        f"当前数据 YAML：{data_file}",
        f"当前数据根目录：{root}",
    ]
    if missing_paths:
        lines.append("缺失路径：")
        lines.extend(f"  - {path}" for path in missing_paths)
    if empty_splits:
        lines.append("以下关键划分没有找到任何图像：" + "、".join(empty_splits))

    lines.extend(
        [
            "建议修复方式：",
            "  1. 确认训练机上真实数据目录是否已经挂载；",
            "  2. 若真实数据根目录不是 YAML 中的 path，请使用：",
            "     python train_26.py --data-root /你的真实/Datasets",
            "  3. 或直接修改 dataset.yaml 的 path 字段；",
            "  4. 若 VIS 与 SAR 的类别编号不一致，请先把 aircraft 统一映射为类别 0。",
        ]
    )
    return "\n".join(lines)


def prepare_dataset(args: argparse.Namespace) -> Tuple[Path, Dict[str, Any], Dict[str, SplitReport]]:
    """读取、检查并生成最终传给 Ultralytics 的数据 YAML。"""

    data_file = Path(args.data).expanduser()
    if not data_file.exists():
        raise DataConfigError(f"数据集 YAML 不存在：{data_file}")
    data_file = data_file.resolve()

    data = YAML.load(data_file)
    if not isinstance(data, dict):
        raise DataConfigError(f"数据集 YAML 内容异常，未解析为字典：{data_file}")

    nc = int(data.get("nc", 0) or 0)
    if nc <= 0:
        raise DataConfigError("数据集 YAML 中 nc 必须是正整数。")

    class0 = _class_zero_name(data)
    if nc == 1 and class0 and class0.lower() not in {"aircraft", "airplane", "plane", "飞机", "飞行器"}:
        LOGGER.warning("当前单类别名称为 %s，请确认类别 0 是否确实代表 aircraft。", class0)

    root = _select_data_root(args, data, data_file)
    split_entries = {
        "train": _as_list(data.get("train")),
        "val": _as_list(data.get("val")),
        "test": _as_list(data.get("test")),
    }

    if not split_entries["train"]:
        raise DataConfigError("数据集 YAML 缺少 train 字段。")
    if not split_entries["val"]:
        raise DataConfigError("数据集 YAML 缺少 val 字段；训练过程需要验证集监控 mAP。")

    reports = {
        split: _inspect_split(
            split,
            entries,
            root,
            data_file,
            nc,
            max(args.label_check_samples, 0),
        )
        for split, entries in split_entries.items()
        if entries
    }

    for report in reports.values():
        LOGGER.info("\n%s", _format_split_report(report))

    critical_reports = [reports[name] for name in ("train", "val") if name in reports]
    has_missing = any(report.missing_entries for report in critical_reports)
    has_empty = any(report.image_count == 0 for report in critical_reports)
    if has_missing or has_empty:
        raise DataConfigError(_build_missing_data_message(data_file, root, critical_reports))

    train_domains = {entry.domain for entry in reports["train"].entries}
    if args.domain_weight > 0 and not {"光学源域", "SAR目标域"}.issubset(train_domains):
        LOGGER.warning(
            "训练集路径未同时识别到“光学源域”和“SAR目标域”。"
            "请确认路径中包含 VIS/Optical/RGB 与 SAR 等域标识，否则域标签可能全部落到同一类。"
        )

    label_problem_reports = [
        report
        for report in critical_reports
        if report.label_missing or report.label_invalid or (report.label_empty and not args.allow_empty_labels)
    ]
    if label_problem_reports:
        for report in label_problem_reports:
            LOGGER.warning(
                "%s 划分存在标签风险：缺失=%d，空标签=%d，异常=%d。"
                "若这是负样本请保留空 txt；若不是，请先修复标签。",
                report.name,
                report.label_missing,
                report.label_empty,
                report.label_invalid,
            )

    resolved_data = dict(data)
    resolved_data["path"] = str(root)

    resolved_dir = ROOT / "runs" / "_resolved_data"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml = resolved_dir / f"{data_file.stem}_resolved_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
    YAML.save(
        resolved_yaml,
        resolved_data,
        header="# 由 train_26.py 自动生成：用于记录本次训练实际采用的数据根目录。\n",
    )
    LOGGER.info("已生成本次训练使用的数据配置：%s", resolved_yaml)
    return resolved_yaml, resolved_data, reports


def _model_yaml_for_scale(size: str) -> Path:
    """返回带尺度信息的 YOLO26-DA YAML 路径。

    项目中真实文件名是 yolo26-da.yaml。Ultralytics 会把 yolo26s-da.yaml 这类
    “虚拟文件名”统一映射回 yolo26-da.yaml，同时从文件名中解析 scale=s。
    因此这里故意返回虚拟路径，以确保 n/s/m/l/x 尺度生效。
    """

    generic_yaml = YOLO_SOURCE / "ultralytics" / "cfg" / "models" / "26" / "yolo26-da.yaml"
    if not generic_yaml.exists():
        raise FileNotFoundError(f"模型结构文件不存在：{generic_yaml}")
    return generic_yaml.with_name(f"yolo26{size}-da.yaml")


def _base_model(trainer: Any) -> Any:
    """兼容单卡/多卡训练，获取真实模型对象。"""

    model = trainer.model
    return model.module if hasattr(model, "module") else model


def _register_da_callbacks(model: YOLO, alpha_max: float) -> None:
    """注册域对抗训练回调，动态调整 GRL 系数并输出中文日志。"""

    def set_alpha(trainer: Any) -> None:
        total = max(int(trainer.args.epochs) - 1, 1)
        progress = min(max(float(trainer.epoch) / total, 0.0), 1.0)
        alpha = alpha_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
        base = _base_model(trainer)
        base.da_alpha = alpha
        trainer.da_alpha = alpha

    def report_domain_loss(trainer: Any) -> None:
        base = _base_model(trainer)
        criterion = getattr(base, "criterion", None)
        domain_loss = getattr(criterion, "last_domain_loss", None)
        if domain_loss is None:
            LOGGER.info("第 %d 轮结束：暂未记录域分类损失。", trainer.epoch + 1)
            return
        LOGGER.info(
            "第 %d 轮结束：GRL系数=%.4f，域分类损失=%.6f",
            trainer.epoch + 1,
            float(getattr(trainer, "da_alpha", 0.0)),
            float(domain_loss),
        )

    model.add_callback("on_train_epoch_start", set_alpha)
    model.add_callback("on_train_epoch_end", report_domain_loss)


def _gb(value: int) -> float:
    """字节转 GB。"""

    return float(value) / 1024**3


def _cuda_memory_status() -> str:
    """返回当前 CUDA 显存状态的中文字符串。"""

    if not torch.cuda.is_available():
        return "CUDA 不可用"

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return (
            f"空闲/总计={_gb(free_bytes):.2f}/{_gb(total_bytes):.2f}GB，"
            f"已分配={_gb(allocated):.2f}GB，已保留={_gb(reserved):.2f}GB"
        )
    except Exception as exc:  # pragma: no cover - 仅用于日志兜底
        return f"显存状态读取失败：{exc}"


def _clear_cuda_cache(reason: str) -> None:
    """主动清理 Python 垃圾对象和 CUDA 缓存，缓解显存碎片。"""

    if not torch.cuda.is_available():
        return

    before = _cuda_memory_status()
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        # ipc_collect 在部分 CUDA/PyTorch 环境中不可用；empty_cache 已经完成主要工作。
        pass
    after = _cuda_memory_status()
    LOGGER.info("%s：已清理 CUDA 缓存。清理前：%s；清理后：%s", reason, before, after)


def _register_memory_callbacks(model: YOLO, enabled: bool) -> None:
    """注册每轮结束后的显存缓存清理回调。"""

    if not enabled:
        LOGGER.info("已关闭每轮 CUDA 缓存清理。")
        return

    def clear_after_epoch(trainer: Any) -> None:
        _clear_cuda_cache(f"第 {trainer.epoch + 1} 轮训练/验证结束")

    model.add_callback("on_fit_epoch_end", clear_after_epoch)
    LOGGER.info("已启用每轮结束后的 CUDA 缓存清理。")


def _log_memory_plan(args: argparse.Namespace) -> None:
    """根据 12GB 显存约束输出训练参数建议与风险提示。"""

    LOGGER.info(
        "训练显存配置：size=%s，imgsz=%d，batch=%d，nbs=%d，workers=%d，AMP=%s，"
        "每轮清理缓存=%s",
        args.size,
        args.imgsz,
        args.batch,
        args.nbs,
        args.workers,
        not args.no_amp,
        args.clear_cache_each_epoch,
    )

    if torch.cuda.is_available():
        LOGGER.info("当前 CUDA 显存状态：%s", _cuda_memory_status())

    if args.max_vram_gb <= 12.5:
        if args.size in {"m", "l", "x"}:
            LOGGER.warning("12GB 显存不建议直接训练 YOLO26%s；优先使用 size=s，必要时退到 size=n。", args.size)
        if args.imgsz >= 1024 and args.batch > 2:
            LOGGER.warning("12GB 显存下 imgsz>=1024 且 batch>2 很容易 OOM；当前独立集也不支持盲目增大到 1024/1280，建议优先 imgsz=640 或 704。")
        elif args.imgsz >= 768 and args.batch > 4:
            LOGGER.warning("12GB 显存下 imgsz>=768 且 batch>4 有 OOM 风险；建议 batch<=4。")
        elif args.imgsz <= 640:
            LOGGER.info("当前尺度较省显存；最新独立测试显示 640/704 比 1024/1280 更稳。")
    elif args.max_vram_gb >= 20:
        if args.imgsz == 1024 and args.batch < 16:
            LOGGER.warning("24GB 显存下 imgsz=1024,batch=%d 可能偏保守；若显存长期低于 18GB，可尝试 batch=20 或 24。", args.batch)
        if args.imgsz == 1024 and args.batch >= 24:
            LOGGER.info("当前为 24GB 高显存配置，目标显存占用约 20~22GB；若 OOM，优先降 batch 到 20 或 16。")
        if args.imgsz > 1024:
            LOGGER.warning("imgsz>1024 会明显增加显存和误检风险，建议仅作为消融实验。")

    if args.batch <= 2 and args.domain_weight > 0:
        LOGGER.warning(
            "当前 batch 较小，单个 batch 内可能偶尔只含一个域，域对抗损失会变得更 noisy；"
            "这是低显存折中，建议保持 nbs=16 并观察 domain_loss。"
        )


def _find_detect_head(torch_model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """在 YOLO 模型中查找 Detect 检测头。"""

    layers = getattr(torch_model, "model", [])
    return next((layer for layer in reversed(layers) if layer.__class__.__name__ == "Detect"), None)


def _load_pretrained(model: YOLO, weights: str) -> None:
    """加载检测预训练权重，并处理新增域判别器导致的层号偏移。"""

    LOGGER.info("加载检测预训练权重：%s", weights)
    source = YOLO(weights)

    # 先让 Ultralytics 执行常规的形状兼容权重加载。
    model.model.load(source.model)

    # 由于 DA 结构在 Detect 前插入了 DomainClassifier，检测头层号发生偏移。
    # 这里再显式迁移一次 Detect 中形状完全一致的张量，避免检测头没有正确初始化。
    source_head = _find_detect_head(source.model)
    target_head = _find_detect_head(model.model)
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
    LOGGER.info("检测头显式迁移完成：%d/%d 个张量形状兼容。", len(compatible), len(target_state))


def build_model(args: argparse.Namespace) -> YOLO:
    """构建 YOLO26-DA 模型并完成权重初始化。"""

    model_yaml = _model_yaml_for_scale(args.size)
    LOGGER.info("构建 YOLO26%s-DA 模型：%s", args.size, model_yaml)
    model = YOLO(str(model_yaml))

    has_domain_classifier = any(
        layer.__class__.__name__ == "DomainClassifier" for layer in model.model.model
    )
    if not has_domain_classifier:
        raise RuntimeError("模型结构中未发现 DomainClassifier，已中止以避免伪域对抗训练。")

    if args.fresh:
        LOGGER.warning("当前启用随机初始化。除非做消融实验，否则检测精度通常会明显下降。")
    else:
        weights = args.weights or f"yolo26{args.size}.pt"
        _load_pretrained(model, weights)

    model.model.da_alpha = 0.0
    _register_da_callbacks(model, args.alpha_max)
    _register_memory_callbacks(model, args.clear_cache_each_epoch)
    return model


def _write_run_manifest(args: argparse.Namespace, data_yaml: Path, reports: Dict[str, SplitReport]) -> Path:
    """写入本次训练的启动配置，便于复现实验。"""

    manifest_dir = ROOT / "runs" / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"train26_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "脚本": str(Path(__file__).resolve()),
        "数据配置": str(data_yaml),
        "启动时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "参数": vars(args),
        "数据检查": {
            name: {
                "图像数": report.image_count,
                "标签抽查数": report.label_checked,
                "缺失标签": report.label_missing,
                "空标签": report.label_empty,
                "异常标签": report.label_invalid,
                "路径": [
                    {
                        "原始": entry.raw,
                        "解析后": str(entry.path),
                        "存在": entry.exists,
                        "图像数": entry.image_count,
                        "域": entry.domain,
                    }
                    for entry in report.entries
                ],
            }
            for name, report in reports.items()
        },
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("已写入训练启动清单：%s", manifest_path)
    return manifest_path


def run_training(args: argparse.Namespace) -> Any:
    """执行完整训练流程。"""

    if args.domain_weight < 0:
        raise ValueError("--domain-weight 不能小于 0。")
    if args.alpha_max < 0:
        raise ValueError("--alpha-max 不能小于 0。")

    _log_memory_plan(args)

    data_yaml, _, reports = prepare_dataset(args)
    _write_run_manifest(args, data_yaml, reports)

    if args.precheck_only:
        LOGGER.info("预检查完成，按照 --precheck-only 要求不启动训练。")
        return None

    model = build_model(args)
    run_name = args.name or f"YOLO26{args.size}_DA_DANN"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        LOGGER.info("CUDA 可用，已清理显存缓存。")
    else:
        LOGGER.warning("当前未检测到 CUDA，将使用 CPU 或 Ultralytics 指定设备；训练速度可能很慢。")

    LOGGER.info(
        "启动训练：模型=YOLO26%s-DA，epochs=%d，imgsz=%d，batch=%d，optimizer=%s，"
        "lr0=%.6f，domain_weight=%.4f，AMP=%s",
        args.size,
        args.epochs,
        args.imgsz,
        args.batch,
        args.optimizer,
        args.lr0,
        args.domain_weight,
        not args.no_amp,
    )
    LOGGER.info("说明：检测损失由 YOLO26 DetectLoss 计算，域对抗损失由 yolo_source 中的自定义 loss 叠加。")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        exist_ok=args.exist_ok,
        pretrained=not args.fresh,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        nbs=args.nbs,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=True,
        domain_weight=args.domain_weight,
        patience=args.patience,
        save=True,
        save_period=args.save_period,
        seed=args.seed,
        deterministic=True,
        cache=False,
        plots=True,
        # 数据预处理与增强：SAR 方向和尺度变化明显，但颜色扰动保持温和。
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale_aug,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        # 跨域 Mosaic/MixUp 会让单张合成图同时含多个域，导致域标签失真。
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=0,
        multi_scale=args.multi_scale,
        rect=False,
        amp=not args.no_amp,
    )

    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        LOGGER.info("训练结束，模型权重保存目录：%s", Path(save_dir) / "weights")
        LOGGER.info("通常重点查看：best.pt、last.pt、results.csv、args.yaml。")
    else:
        LOGGER.info("训练结束。")
    return results


def main() -> None:
    """命令行入口。"""

    args = build_parser().parse_args()
    _configure_logging(args.verbose)
    try:
        run_training(args)
    except DataConfigError as exc:
        LOGGER.error("%s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        LOGGER.warning("收到中断信号，训练已停止。")
        sys.exit(130)
    except Exception as exc:
        LOGGER.exception("训练启动或执行过程中出现异常：%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
