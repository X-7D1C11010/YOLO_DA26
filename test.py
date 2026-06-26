"""统一、可追溯的模型评测脚本。"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "yolo_source"))

from ultralytics import YOLO


def _parse_sizes(value: str) -> list[int]:
    sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("imgsz 必须是逗号分隔的正整数")
    return sizes


def _expand_weights(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.glob("*.pt")))
        elif any(char in value for char in "*?[]"):
            paths.extend(Path(item) for item in sorted(glob.glob(value)))
        else:
            paths.append(path)
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError("以下权重不存在：\n" + "\n".join(missing))
    return unique


def main():
    parser = argparse.ArgumentParser(description="YOLO 模型独立测试集评测")
    parser.add_argument("weights", nargs="+", help="权重文件、目录或 glob")
    parser.add_argument("--data", default=str(ROOT / "test.yaml"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=_parse_sizes, default=_parse_sizes("640,1024,1280"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "evaluation_results_26s.jsonl"))
    args = parser.parse_args()

    data_path = Path(args.data).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"测试集配置不存在：{data_path}")

    weights = _expand_weights(args.weights)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best = None
    with output_path.open("a", encoding="utf-8") as log_file:
        for weight in weights:
            model = YOLO(str(weight))
            for imgsz in args.imgsz:
                print(f"\n评测：{weight.name}，imgsz={imgsz}")
                metrics = model.val(
                    data=str(data_path),
                    split=args.split,
                    imgsz=imgsz,
                    batch=args.batch,
                    device=args.device,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    half=args.half,
                    rect=True,
                    plots=True,
                    project="runs/eval",
                    name=f"{weight.stem}_imgsz{imgsz}",
                    exist_ok=True,
                )
                record = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "weights": str(weight),
                    "data": str(data_path.resolve()),
                    "split": args.split,
                    "imgsz": imgsz,
                    "conf": args.conf,
                    "iou": args.iou,
                    "map50": float(metrics.box.map50),
                    "map75": float(metrics.box.map75),
                    "map50_95": float(metrics.box.map),
                    "precision": float(getattr(metrics.box, "mp", 0.0)),
                    "recall": float(getattr(metrics.box, "mr", 0.0)),
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                if best is None or record["map50"] > best["map50"]:
                    best = record
                print(
                    f"mAP50={record['map50']:.4f}, mAP75={record['map75']:.4f}, "
                    f"mAP50-95={record['map50_95']:.4f}"
                )

    if best:
        print(
            f"\n最佳组合：{Path(best['weights']).name} @ imgsz={best['imgsz']}，"
            f"mAP50={best['map50']:.4f}"
        )
        print(f"完整结果已追加到：{output_path.resolve()}")


if __name__ == "__main__":
    main()
