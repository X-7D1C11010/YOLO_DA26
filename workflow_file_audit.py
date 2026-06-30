#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""盘点当前 YOLO 域自适应项目流程中使用和生成的文件。

该脚本用于解决项目目录逐渐混乱后“不知道哪些文件在当前方法中仍然有用”的问题。
它只执行只读扫描，最终在 runs/workflow_audit/ 下保存：
1. JSON 报告：便于程序继续解析；
2. Markdown 报告：便于人工查看和归档。

典型用法：
    python workflow_file_audit.py --root . --output-dir runs/workflow_audit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_SUFFIXES = {".txt"}

SCRIPT_ROLES: dict[str, str] = {
    "train_26.py": "主训练脚本：YOLO26 + 对抗域自适应/当前重点训练入口",
    "train_yolo26s.py": "对比训练脚本：YOLO26s 配置训练入口",
    "da_training.py": "域自适应训练公共逻辑",
    "test.py": "多权重、多尺度模型评测入口",
    "select_target_subset.py": "从独立测试/目标域数据中抽样并生成微调子集、无泄漏测试集",
    "finetune_target_domain.py": "目标域小样本综合微调脚本：支持有标注和无标注模式",
    "audit_dataset.py": "数据集结构、标注合法性审计脚本",
    "analyze_domain_gap.py": "源域/目标域图像差异分析脚本",
    "analyze_detection_errors.py": "检测错误类型与可视化分析脚本",
    "make_balanced_da_dataset.py": "构建/平衡域自适应训练数据配置",
    "prepare_vis_aircraft_singleclass.py": "VIS 单类 aircraft 标注映射预处理",
    "smoke_test_da.py": "训练流程烟雾测试脚本",
}

FLOW_STEPS = [
    {
        "stage": "1. 数据与配置审计",
        "inputs": ["test.yaml", "dataset_aircraft_da.yaml", "dataset_aircraft_da_balanced.yaml", "dataset_sar_only.yaml"],
        "scripts": ["audit_dataset.py", "analyze_domain_gap.py"],
        "outputs": ["dataset_audit*.json", "runs/analysis/"],
    },
    {
        "stage": "2. 对抗域自适应训练",
        "inputs": ["dataset_aircraft_da_balanced.yaml", "yolo_source/", "train_26.py", "train_yolo26s.py"],
        "scripts": ["train_26.py", "train_yolo26s.py", "da_training.py"],
        "outputs": ["runs/detect/*/weights/best.pt", "runs/detect/*/weights/epoch*.pt", "train_26.log", "train_yolo26s.log"],
    },
    {
        "stage": "3. 独立测试集多尺度评估",
        "inputs": ["test.yaml", "runs/detect/*/weights/*.pt"],
        "scripts": ["test.py", "analyze_detection_errors.py"],
        "outputs": ["evaluation*.jsonl", "runs/detect/runs/eval_*/", "runs/analysis/"],
    },
    {
        "stage": "4. 目标域小样本划分",
        "inputs": ["test.yaml"],
        "scripts": ["select_target_subset.py"],
        "outputs": ["runs/target_subsets/*/target_subset.yaml", "runs/target_subsets/*/target_no_leak_test.yaml", "runs/target_subsets/*/target_strict_holdout.yaml"],
    },
    {
        "stage": "5. 目标域微调",
        "inputs": ["runs/target_subsets/*/target_subset.yaml", "runs/detect/*/weights/*.pt"],
        "scripts": ["finetune_target_domain.py"],
        "outputs": ["runs/target_finetune/labeled/*/weights/best.pt", "runs/target_finetune/unlabeled/*/weights/best.pt", "runs/target_finetune/*records.jsonl"],
    },
    {
        "stage": "6. 微调后无泄漏/严格 holdout 测试",
        "inputs": ["runs/target_finetune/*/*/weights/best.pt", "runs/target_subsets/*/target_no_leak_test.yaml", "runs/target_subsets/*/target_strict_holdout.yaml"],
        "scripts": ["test.py", "finetune_target_domain.py --final-test-data"],
        "outputs": ["evaluation_target*.jsonl", "runs/target_finetune/final_test_records.jsonl"],
    },
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="盘点当前项目流程文件、模型输出、日志和评测结果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default=".", help="项目根目录")
    parser.add_argument("--output-dir", default=None, help="报告输出目录；默认写入 <root>/runs/workflow_audit")
    parser.add_argument("--max-jsonl-lines", type=int, default=200000, help="单个 JSONL 最多解析行数，防止异常大文件拖慢扫描")
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    """返回相对项目根目录的路径，便于报告阅读。"""

    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def file_info(path: Path, root: Path) -> dict[str, Any]:
    """读取文件或目录的基础信息。"""

    exists = path.exists()
    info: dict[str, Any] = {"path": rel(path, root), "exists": exists}
    if exists:
        stat = path.stat()
        info.update(
            {
                "type": "dir" if path.is_dir() else "file",
                "size_bytes": stat.st_size if path.is_file() else None,
                "modified_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            }
        )
    return info


def count_files(directory: Path, suffixes: set[str] | None = None) -> int:
    """统计目录下指定后缀文件数量。"""

    if not directory.exists() or not directory.is_dir():
        return 0
    total = 0
    for path in directory.rglob("*"):
        if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes):
            total += 1
    return total


def read_text(path: Path, max_chars: int = 200000) -> str:
    """安全读取文本文件。"""

    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError:
        return ""


def scan_scripts(root: Path) -> list[dict[str, Any]]:
    """扫描当前流程相关脚本。"""

    records: list[dict[str, Any]] = []
    for name, role in SCRIPT_ROLES.items():
        path = root / name
        record = file_info(path, root)
        record["role"] = role
        records.append(record)
    return records


def scan_dataset_yamls(root: Path) -> list[dict[str, Any]]:
    """扫描项目根目录下的数据集 YAML 配置。"""

    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.yaml")):
        text = read_text(path, max_chars=20000)
        record = file_info(path, root)
        for key in ("path", "train", "val", "test", "nc", "names"):
            record[f"has_{key}"] = f"{key}:" in text
        records.append(record)
    return records


def scan_logs(root: Path) -> list[dict[str, Any]]:
    """扫描训练、微调和测试日志。"""

    candidates = list(root.glob("*.log")) + list(root.glob("*Log*.txt"))
    runs = root / "runs"
    if runs.exists():
        candidates.extend(runs.rglob("*.log"))
    return [file_info(path, root) for path in sorted(set(candidates))]


def best_metric_from_record(record: dict[str, Any]) -> float | None:
    """从常见评测记录字段中读取 mAP50。"""

    for key in ("map50", "box_map50", "mAP50", "metrics/mAP50(B)", "metrics/mAP50"):
        value = record.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def scan_jsonl(path: Path, root: Path, max_lines: int) -> dict[str, Any]:
    """解析 JSONL 评测文件并提取最佳 mAP50 记录。"""

    record = file_info(path, root)
    total = 0
    parsed = 0
    best_map50: float | None = None
    best_record: dict[str, Any] | None = None
    weights: set[str] = set()
    data_files: set[str] = set()

    try:
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                if total >= max_lines:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                total += 1
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                parsed += 1
                if item.get("weights"):
                    weights.add(str(item["weights"]))
                if item.get("data"):
                    data_files.add(str(item["data"]))
                metric = best_metric_from_record(item)
                if metric is not None and (best_map50 is None or metric > best_map50):
                    best_map50 = metric
                    best_record = item
    except OSError as exc:
        record["error"] = str(exc)

    record.update(
        {
            "line_count_scanned": total,
            "json_records": parsed,
            "best_map50": best_map50,
            "best_record": best_record,
            "weights_count": len(weights),
            "data_count": len(data_files),
        }
    )
    return record


def scan_evaluations(root: Path, max_lines: int) -> list[dict[str, Any]]:
    """扫描根目录和 runs 目录下的评测 JSONL。"""

    candidates = list(root.glob("evaluation*.jsonl"))
    runs = root / "runs"
    if runs.exists():
        candidates.extend(runs.rglob("*records.jsonl"))
        candidates.extend(runs.rglob("evaluation*.jsonl"))
    return [scan_jsonl(path, root, max_lines) for path in sorted(set(candidates))]


def scan_target_subsets(root: Path) -> list[dict[str, Any]]:
    """扫描目标域小样本划分结果。"""

    subset_root = root / "runs" / "target_subsets"
    records: list[dict[str, Any]] = []
    if not subset_root.exists():
        return records

    for subset_dir in sorted(path for path in subset_root.iterdir() if path.is_dir()):
        record = file_info(subset_dir, root)
        yaml_files = sorted(subset_dir.glob("*.yaml"))
        record["yaml_files"] = [rel(path, root) for path in yaml_files]
        record["manifest_exists"] = (subset_dir / "manifest.json").exists()
        record["splits"] = {}
        for split in ("train", "test", "no_leak_test", "strict_holdout", "val"):
            image_dir = subset_dir / "images" / split
            label_dir = subset_dir / "labels" / split
            record["splits"][split] = {
                "images": count_files(image_dir, IMAGE_SUFFIXES),
                "labels": count_files(label_dir, LABEL_SUFFIXES),
            }
        records.append(record)
    return records


def infer_mode_from_run(run_dir: Path) -> str:
    """根据路径和目录名推断微调模式。"""

    parts = {part.lower() for part in run_dir.parts}
    name = run_dir.name.lower()
    if "labeled" in parts or name.endswith("_labeled") or "_labeled" in name:
        return "labeled"
    if "unlabeled" in parts or name.endswith("_unlabeled") or "_unlabeled" in name:
        return "unlabeled"
    return "unknown"


def scan_weight_runs(root: Path, base: Path, kind: str) -> list[dict[str, Any]]:
    """扫描包含 weights/ 的训练或微调 run。"""

    records: list[dict[str, Any]] = []
    if not base.exists():
        return records
    for weights_dir in sorted(path for path in base.rglob("weights") if path.is_dir()):
        run_dir = weights_dir.parent
        record = file_info(run_dir, root)
        record.update(
            {
                "kind": kind,
                "mode": infer_mode_from_run(run_dir),
                "best": rel(weights_dir / "best.pt", root) if (weights_dir / "best.pt").exists() else None,
                "last": rel(weights_dir / "last.pt", root) if (weights_dir / "last.pt").exists() else None,
                "epoch_checkpoints": len(list(weights_dir.glob("epoch*.pt"))),
                "args_yaml": rel(run_dir / "args.yaml", root) if (run_dir / "args.yaml").exists() else None,
                "results_csv": rel(run_dir / "results.csv", root) if (run_dir / "results.csv").exists() else None,
                "pseudo_dataset": rel(run_dir / "pseudo_dataset", root) if (run_dir / "pseudo_dataset").exists() else None,
                "final_test_dir": rel(run_dir / "final_test", root) if (run_dir / "final_test").exists() else None,
            }
        )
        records.append(record)
    return records


def scan_analysis_outputs(root: Path) -> dict[str, Any]:
    """扫描分析脚本输出目录。"""

    analysis = root / "runs" / "analysis"
    record = file_info(analysis, root)
    if analysis.exists() and analysis.is_dir():
        record["file_count"] = count_files(analysis)
        record["image_count"] = count_files(analysis, IMAGE_SUFFIXES)
        record["json_count"] = count_files(analysis, {".json", ".jsonl"})
    return record


def build_report(root: Path, max_jsonl_lines: int) -> dict[str, Any]:
    """构建完整 JSON 报告。"""

    detect_root = root / "runs" / "detect"
    finetune_root = root / "runs" / "target_finetune"

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root.resolve()),
        "flow_steps": FLOW_STEPS,
        "scripts": scan_scripts(root),
        "dataset_yamls": scan_dataset_yamls(root),
        "target_subsets": scan_target_subsets(root),
        "detect_runs": scan_weight_runs(root, detect_root, "detect_train"),
        "finetune_runs": scan_weight_runs(root, finetune_root, "target_finetune"),
        "evaluations": scan_evaluations(root, max_jsonl_lines),
        "logs": scan_logs(root),
        "analysis_outputs": scan_analysis_outputs(root),
        "important_single_files": [
            file_info(root / "dataset_audit.json", root),
            file_info(root / "dataset_audit_fixed.json", root),
            file_info(root / "target_domain_finetune_usage.md", root),
            file_info(root / "目标域数据划分与微调使用说明.md", root),
            file_info(root / "当前结果诊断与下一步方案.md", root),
        ],
    }


def yes_no(value: bool) -> str:
    """Markdown 表格中使用的中文布尔值。"""

    return "是" if value else "否"


def render_markdown(report: dict[str, Any]) -> str:
    """把 JSON 报告渲染为便于阅读的 Markdown。"""

    lines: list[str] = []
    lines.append("# YOLO 光学/SAR 域自适应项目流程文件盘点")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- 项目根目录：`{report['root']}`")
    lines.append("")

    lines.append("## 1. 当前方法流程地图")
    lines.append("")
    lines.append("| 阶段 | 主要输入 | 使用脚本 | 主要输出 |")
    lines.append("|---|---|---|---|")
    for step in report["flow_steps"]:
        lines.append(
            "| {stage} | {inputs} | {scripts} | {outputs} |".format(
                stage=step["stage"],
                inputs="<br>".join(f"`{item}`" for item in step["inputs"]),
                scripts="<br>".join(f"`{item}`" for item in step["scripts"]),
                outputs="<br>".join(f"`{item}`" for item in step["outputs"]),
            )
        )
    lines.append("")

    lines.append("## 2. 关键脚本状态")
    lines.append("")
    lines.append("| 脚本 | 是否存在 | 作用 | 修改时间 |")
    lines.append("|---|---:|---|---|")
    for item in report["scripts"]:
        lines.append(f"| `{item['path']}` | {yes_no(item['exists'])} | {item['role']} | {item.get('modified_time', '-')} |")
    lines.append("")

    lines.append("## 3. 数据集配置文件")
    lines.append("")
    lines.append("| YAML | train | val | test | nc | names | 修改时间 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for item in report["dataset_yamls"]:
        lines.append(
            f"| `{item['path']}` | {yes_no(item.get('has_train', False))} | {yes_no(item.get('has_val', False))} | "
            f"{yes_no(item.get('has_test', False))} | {yes_no(item.get('has_nc', False))} | "
            f"{yes_no(item.get('has_names', False))} | {item.get('modified_time', '-')} |"
        )
    lines.append("")

    lines.append("## 4. 目标域抽样/划分结果")
    lines.append("")
    if report["target_subsets"]:
        lines.append("| 目录 | train 图像/标签 | test 图像/标签 | no_leak 图像/标签 | strict_holdout 图像/标签 | YAML |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for item in report["target_subsets"]:
            splits = item["splits"]
            yaml_text = "<br>".join(f"`{path}`" for path in item["yaml_files"])
            lines.append(
                f"| `{item['path']}` | {splits['train']['images']}/{splits['train']['labels']} | "
                f"{splits['test']['images']}/{splits['test']['labels']} | "
                f"{splits['no_leak_test']['images']}/{splits['no_leak_test']['labels']} | "
                f"{splits['strict_holdout']['images']}/{splits['strict_holdout']['labels']} | {yaml_text} |"
            )
    else:
        lines.append("未发现 `runs/target_subsets/`。")
    lines.append("")

    lines.append("## 5. 模型训练与微调输出")
    lines.append("")
    all_runs = report["detect_runs"] + report["finetune_runs"]
    if all_runs:
        lines.append("| 类型 | 模式 | run 目录 | best.pt | last.pt | epoch*.pt | 伪标签/最终测试目录 |")
        lines.append("|---|---|---|---|---|---:|---|")
        for item in all_runs:
            extra = item.get("pseudo_dataset") or item.get("final_test_dir") or "-"
            lines.append(
                f"| {item['kind']} | {item['mode']} | `{item['path']}` | "
                f"`{item['best']}` | `{item['last']}` | {item['epoch_checkpoints']} | `{extra}` |"
            )
    else:
        lines.append("未发现包含 `weights/` 的训练或微调输出目录。")
    lines.append("")

    lines.append("## 6. 评测 JSONL 摘要")
    lines.append("")
    if report["evaluations"]:
        lines.append("| 文件 | 记录数 | 最佳 mAP50 | 最佳尺度 | 权重数 | 数据集数 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for item in report["evaluations"]:
            best = item.get("best_record") or {}
            best_map50 = item.get("best_map50")
            best_text = "-" if best_map50 is None else f"{best_map50:.4f}"
            lines.append(
                f"| `{item['path']}` | {item.get('json_records', 0)} | {best_text} | "
                f"{best.get('imgsz', '-')} | {item.get('weights_count', 0)} | {item.get('data_count', 0)} |"
            )
    else:
        lines.append("未发现评测 JSONL 文件。")
    lines.append("")

    lines.append("## 7. 日志与分析输出")
    lines.append("")
    lines.append(f"- 分析输出目录：`{report['analysis_outputs']['path']}`，存在：{yes_no(report['analysis_outputs']['exists'])}")
    lines.append("")
    if report["logs"]:
        lines.append("| 日志 | 大小/字节 | 修改时间 |")
        lines.append("|---|---:|---|")
        for item in report["logs"]:
            lines.append(f"| `{item['path']}` | {item.get('size_bytes', '-')} | {item.get('modified_time', '-')} |")
    else:
        lines.append("未发现日志文件。")
    lines.append("")

    lines.append("## 8. 建议阅读顺序")
    lines.append("")
    lines.append("1. 先看 `dataset_audit_fixed.json` 与目标域划分目录，确认数据没有泄漏。")
    lines.append("2. 再看 `evaluation*.jsonl` 的最佳尺度和 mAP50，判断当前 checkpoint 的真实独立测试表现。")
    lines.append("3. 微调实验优先比较 `runs/target_finetune/final_test_records.jsonl`，不要只看小样本内部 val。")
    lines.append("4. 如需复现实验，优先使用本报告第 1 节的流程地图和第 5 节列出的 best.pt/epoch*.pt。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"项目根目录不存在：{root}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "runs" / "workflow_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(root, args.max_jsonl_lines)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"workflow_files_{stamp}.json"
    md_path = output_dir / f"workflow_files_{stamp}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"流程文件盘点完成：{json_path}")
    print(f"Markdown 说明文档：{md_path}")


if __name__ == "__main__":
    main()
