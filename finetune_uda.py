"""安全的无监督域自适应：高置信伪标签 + 有标签锚定训练。

旧实现只最小化域分类损失，会在没有检测约束时破坏主干特征并导致预测全零。
本实现不直接修改 checkpoint 内部对象，而是使用 Ultralytics 标准训练与保存流程。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "yolo_source"))

from ultralytics import YOLO
from ultralytics.utils import YAML

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(root: Path) -> list[Path]:
    images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"目标域目录中未找到图像：{root}")
    return images


def link_image(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return "exists"

    modes = [mode] if mode != "auto" else ["symlink", "hardlink", "copy"]
    for current in modes:
        try:
            if current == "symlink":
                os.symlink(source.resolve(), destination)
            elif current == "hardlink":
                os.link(source, destination)
            else:
                shutil.copy2(source, destination)
            return current
        except OSError:
            continue
    raise OSError(f"无法链接或复制图像：{source}")


def resolve_dataset_entries(data_yaml: Path, key: str) -> tuple[dict, list[str]]:
    config = YAML.load(data_yaml)
    base = Path(config.get("path") or data_yaml.parent)
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()

    value = config.get(key, [])
    values = value if isinstance(value, list) else [value]
    resolved = []
    for item in values:
        if not item:
            continue
        path = Path(str(item))
        resolved.append(str(path if path.is_absolute() else (base / path).resolve()))
    return config, resolved


def write_training_yaml(
    output_path: Path,
    base_config: dict,
    anchor_train: list[str],
    pseudo_images: Path,
    validation: list[str],
) -> None:
    train_entries = [*anchor_train, str(pseudo_images.resolve())]
    lines = ["path: /", "train:"]
    lines.extend(f"  - {json.dumps(path, ensure_ascii=False)}" for path in train_entries)
    lines.append("val:")
    lines.extend(f"  - {json.dumps(path, ensure_ascii=False)}" for path in validation)
    lines.append(f"nc: {int(base_config['nc'])}")
    names = base_config.get("names", {0: "aircraft"})
    lines.append("names:")
    if isinstance(names, list):
        for index, name in enumerate(names):
            lines.append(f"  {index}: {json.dumps(str(name), ensure_ascii=False)}")
    else:
        for index, name in sorted(names.items(), key=lambda item: int(item[0])):
            lines.append(f"  {int(index)}: {json.dumps(str(name), ensure_ascii=False)}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_pseudo_dataset(args, teacher: YOLO) -> tuple[Path, dict]:
    target_root = Path(args.target_images).expanduser().resolve()
    images = collect_images(target_root)
    dataset_root = Path(args.output).expanduser().resolve() / "pseudo_dataset"
    if dataset_root.exists():
        if not args.overwrite_pseudo:
            raise FileExistsError(
                f"伪标签目录已存在：{dataset_root}。请更换 --output，"
                "或确认后使用 --overwrite-pseudo 重新生成。"
            )
        shutil.rmtree(dataset_root)
    image_root = dataset_root / "images" / "train"
    label_root = dataset_root / "labels" / "train"
    label_root.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_images": len(images),
        "accepted_images": 0,
        "empty_images": 0,
        "rejected_boxes": 0,
        "accepted_boxes": 0,
        "link_modes": {},
    }

    predictions = teacher.predict(
        source=[str(path) for path in images],
        stream=True,
        imgsz=args.imgsz,
        conf=args.pseudo_conf,
        iou=args.pseudo_iou,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
        verbose=False,
    )
    for result in predictions:
        source = Path(result.path).resolve()
        relative = source.relative_to(target_root)
        boxes = result.boxes
        accepted = []
        if boxes is not None and len(boxes):
            xywhn = boxes.xywhn.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            for cls_id, confidence, box in zip(classes, confidences, xywhn):
                x, y, width, height = box
                area = width * height
                class_id = int(cls_id)
                if (
                    class_id < 0
                    or class_id >= args.nc
                    or confidence < args.pseudo_conf
                    or area < args.min_box_area
                    or area > args.max_box_area
                ):
                    stats["rejected_boxes"] += 1
                    continue
                accepted.append((class_id, x, y, width, height, confidence))

        if not accepted and not args.include_empty:
            stats["empty_images"] += 1
            continue

        destination_image = image_root / relative
        used_mode = link_image(source, destination_image, args.link_mode)
        stats["link_modes"][used_mode] = stats["link_modes"].get(used_mode, 0) + 1
        label_path = (label_root / relative).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "".join(f"{cls_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for cls_id, x, y, w, h, _ in accepted),
            encoding="utf-8",
        )
        stats["accepted_images"] += 1
        stats["accepted_boxes"] += len(accepted)

    if stats["accepted_images"] < args.min_pseudo_images:
        raise RuntimeError(
            f"仅生成 {stats['accepted_images']} 张伪标注图像，低于安全下限 {args.min_pseudo_images}。"
            "请先提升教师模型、检查目标域图像，或谨慎降低 --pseudo-conf。"
        )

    manifest = dataset_root / "pseudo_manifest.json"
    manifest.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"伪标签完成：{stats['accepted_images']}/{stats['total_images']} 张图像，"
        f"{stats['accepted_boxes']} 个框"
    )
    return image_root, stats


def main():
    parser = argparse.ArgumentParser(description="安全 UDA：伪标签自训练")
    parser.add_argument("--weights", required=True, help="教师/初始学生权重")
    parser.add_argument("--target-images", required=True, help="无标签目标域图像根目录")
    parser.add_argument("--base-data", default=str(ROOT / "dataset_sar_only.yaml"), help="有标签锚定集与验证集")
    parser.add_argument("--output", default=str(ROOT / "runs" / "uda_self_train"))
    parser.add_argument("--pseudo-conf", type=float, default=0.70)
    parser.add_argument("--pseudo-iou", type=float, default=0.60)
    parser.add_argument("--min-box-area", type=float, default=1e-5)
    parser.add_argument("--max-box-area", type=float, default=0.50)
    parser.add_argument("--min-pseudo-images", type=int, default=100)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--overwrite-pseudo", action="store_true")
    parser.add_argument("--no-anchor", action="store_true", help="不混入有标签训练集（不推荐）")
    parser.add_argument("--link-mode", choices=("auto", "symlink", "hardlink", "copy"), default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=2e-4)
    parser.add_argument("--freeze", type=int, default=3)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    weights = Path(args.weights).expanduser()
    base_data = Path(args.base_data).expanduser()
    if not weights.exists():
        raise FileNotFoundError(f"权重不存在：{weights}")
    if not base_data.exists():
        raise FileNotFoundError(f"锚定数据配置不存在：{base_data}")
    args.nc = int(YAML.load(base_data)["nc"])

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher = YOLO(str(weights))

    print("先在有标签验证集上记录微调前基线……")
    before = teacher.val(data=str(base_data), imgsz=args.imgsz, batch=args.batch, device=args.device, plots=False)
    before_map50 = float(before.box.map50)

    pseudo_images, pseudo_stats = generate_pseudo_dataset(args, teacher)
    base_config, anchor_train = resolve_dataset_entries(base_data.resolve(), "train")
    _, validation = resolve_dataset_entries(base_data.resolve(), "val")
    if args.no_anchor:
        anchor_train = []
    if not validation:
        raise ValueError("base-data 必须提供有标签 val 集，用于阻止微调退化。")

    generated_yaml = output / "uda_train.yaml"
    write_training_yaml(generated_yaml, base_config, anchor_train, pseudo_images, validation)

    student = YOLO(str(weights))
    results = student.train(
        data=str(generated_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(output),
        name="student",
        exist_ok=False,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.10,
        weight_decay=5e-4,
        warmup_epochs=2,
        cos_lr=True,
        freeze=args.freeze,
        patience=10,
        mosaic=0.0,
        mixup=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.15,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.0,
        hsv_s=0.05,
        hsv_v=0.10,
        close_mosaic=0,
        amp=True,
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_path))
    after = best_model.val(data=str(base_data), imgsz=args.imgsz, batch=args.batch, device=args.device, plots=True)
    after_map50 = float(after.box.map50)
    summary = {
        "weights": str(weights.resolve()),
        "best": str(best_path.resolve()),
        "baseline_map50": before_map50,
        "finetuned_map50": after_map50,
        "pseudo": pseudo_stats,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"锚定验证集 mAP50：{before_map50:.4f} -> {after_map50:.4f}")
    if after_map50 < before_map50 - 0.02:
        print("警告：微调后锚定集下降超过 2 个百分点，不建议将该权重用于独立测试。")
    print(f"候选权重：{best_path}")


if __name__ == "__main__":
    main()
