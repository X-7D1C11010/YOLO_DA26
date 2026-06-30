# 目标域数据划分与微调使用说明

## 1. 代码状态确认

当前 `test.py` 已经能够完成多权重、多尺度独立测试，并能把结果追加保存为 JSONL 文件；当前已有的 `evaluation_DA_bal_dw002_img1024_b24_3_independent*.jsonl` 文件可视为现阶段最新一轮独立测试结果。

现有 `finetune_da.py`、`finetune_uda.py` 和 `train_transfer.py` 不能完整覆盖“目标域抽样、无泄漏测试集生成、批量配置多个模型、有标注/无标注统一微调”的新需求，因此新增：

- `select_target_subset.py`：从独立目标域数据集中随机抽样，划分微调训练/验证子集，并额外生成无泄漏测试集。
- `finetune_target_domain.py`：统一执行有标注目标域微调、无标注伪标签自训练，并支持多个 checkpoint 批量比较。

## 2. 数据划分原则

如果从原独立测试集中抽取 10% 或 20% 样本用于微调训练，则原始 `test.yaml` 不应再作为最终完整测试集使用，因为它包含已经进入微调训练的图像，会导致训练/测试重叠，mAP 被高估。

本项目现在采用三类数据配置：

- `target_subset.yaml`：用于微调训练和训练过程中的小验证；其中 `train` 为抽样训练子集，`val/test` 为抽样内部验证子集。
- `target_no_leak_test.yaml`：排除微调训练图像后的无泄漏测试集；可用于训练后普通测试。
- `target_strict_holdout.yaml`：排除全部抽样图像后的严格最终测试集；如果抽样内部验证子集参与了模型选择，最终报告应优先使用该文件。

## 3. 生成 10% 和 20% 目标域小样本数据

生成 10% 数据：

```bash
python select_target_subset.py \
  --data test.yaml \
  --split val \
  --ratio 0.10 \
  --train-ratio 0.70 \
  --seed 2026 \
  --output runs/target_subsets/independent_10p_seed2026 \
  --overwrite
```

生成 20% 数据：

```bash
python select_target_subset.py \
  --data test.yaml \
  --split val \
  --ratio 0.20 \
  --train-ratio 0.70 \
  --seed 2026 \
  --output runs/target_subsets/independent_20p_seed2026 \
  --overwrite
```

输出目录结构示例：

```text
runs/target_subsets/independent_20p_seed2026/
  images/train/
  images/test/
  images/no_leak_test/
  images/strict_holdout/
  labels/train/
  labels/test/
  labels/no_leak_test/
  labels/strict_holdout/
  target_subset.yaml
  target_no_leak_test.yaml
  target_strict_holdout.yaml
  manifest.json
  train_images.txt
  test_images.txt
  no_leak_test_images.txt
  strict_holdout_images.txt
  selection_detail.csv
```

## 4. 有标注目标域微调

建议优先使用当前独立测试表现最好的 `epoch30.pt`，而不是同源验证集最优的 `best.pt`。

10% 目标域有标注微调：

```bash
python finetune_target_domain.py \
  --mode labeled \
  --weights /ssd_data/lixiang_data/YOLO_DA/runs/detect/DA_bal_dw002_img1024_b24-3/weights/epoch30.pt \
  --data runs/target_subsets/independent_10p_seed2026/target_subset.yaml \
  --project runs/target_finetune \
  --name target10p_labeled \
  --epochs 40 \
  --imgsz 768 \
  --batch 8 \
  --lr0 0.0002 \
  --freeze 5 \
  --patience 12 \
  --save-period 5
```

20% 目标域有标注微调：

```bash
python finetune_target_domain.py \
  --mode labeled \
  --weights /ssd_data/lixiang_data/YOLO_DA/runs/detect/DA_bal_dw002_img1024_b24-3/weights/epoch30.pt \
  --data runs/target_subsets/independent_20p_seed2026/target_subset.yaml \
  --project runs/target_finetune \
  --name target20p_labeled \
  --epochs 40 \
  --imgsz 768 \
  --batch 8 \
  --lr0 0.0002 \
  --freeze 5 \
  --patience 12 \
  --save-period 5
```

## 5. 无标注目标域伪标签微调

无标注模式会使用传入权重作为教师模型，对目标域训练图像生成伪标签，再用伪标签训练学生模型。`--data` 仍需提供，因为脚本要使用其中的 `val` split 监控微调效果。

```bash
python finetune_target_domain.py \
  --mode unlabeled \
  --weights /ssd_data/lixiang_data/YOLO_DA/runs/detect/DA_bal_dw002_img1024_b24-3/weights/epoch30.pt \
  --data runs/target_subsets/independent_20p_seed2026/target_subset.yaml \
  --target-images runs/target_subsets/independent_20p_seed2026/images/train \
  --project runs/target_finetune \
  --name target20p_unlabeled \
  --epochs 30 \
  --imgsz 768 \
  --batch 8 \
  --lr0 0.0001 \
  --freeze 8 \
  --pseudo-conf 0.45 \
  --min-pseudo-images 5 \
  --overwrite-pseudo
```

如果伪标签数量过少，可逐步降低 `--pseudo-conf`，例如 0.45 → 0.35；但每次降低阈值后都应抽查伪标签质量，防止把背景误检作为正样本继续训练。

## 6. 微调后无泄漏测试

训练完成后，不要使用原始 `test.yaml` 做最终测试。应使用当前划分目录中自动生成的测试 YAML。

普通无泄漏测试：

```bash
python test.py runs/target_finetune/labeled/target20p_labeled_epoch30_labeled/weights/best.pt \
  --data runs/target_subsets/independent_20p_seed2026/target_no_leak_test.yaml \
  --imgsz 512,576,640,704,768,1024 \
  --output evaluation_target20p_no_leak.jsonl \
  --exist-ok
```

严格最终测试：

```bash
python test.py runs/target_finetune/labeled/target20p_labeled_epoch30_labeled/weights/best.pt \
  --data runs/target_subsets/independent_20p_seed2026/target_strict_holdout.yaml \
  --imgsz 512,576,640,704,768,1024 \
  --output evaluation_target20p_strict_holdout.jsonl \
  --exist-ok
```

## 7. 建议超参数

| 参数 | 建议值 | 说明 |
|---|---:|---|
| 初始权重 | `epoch30.pt` | 当前独立测试最佳 checkpoint |
| imgsz | 768 | 折中 576 测试优势与 1024 训练尺度 |
| batch | 8 | 小样本微调更重稳定性 |
| lr0 | 1e-4 ~ 2e-4 | 防止破坏已有检测能力 |
| freeze | 5~8 | 先保留底层特征 |
| epochs | 30~50 | 小样本不宜过长 |
| patience | 10~15 | 防止过拟合 |
| mosaic/mixup | 0 | 目标域小样本微调先关闭强增强 |
| pseudo_conf | 0.35~0.55 | 无标注伪标签阈值需结合可视化调整 |

## 8. 服务器端验证

当前本地 Windows 环境中的 `python.exe` 无法访问，因此建议在服务器项目目录中先执行：

```bash
python -m py_compile select_target_subset.py finetune_target_domain.py
python select_target_subset.py --help
python finetune_target_domain.py --help
```

确认无语法错误后，再执行正式抽样、微调和无泄漏测试。

## 9. 当前流程文件盘点

如果当前目录中文件较多，建议先运行流程盘点脚本，明确当前方法使用了哪些输入文件、训练脚本、模型权重、日志和评测结果：

```bash
python workflow_file_audit.py \
  --root . \
  --output-dir runs/workflow_audit
```

脚本会生成两类文件：

- `runs/workflow_audit/workflow_files_*.json`：完整结构化报告，适合后续程序读取。
- `runs/workflow_audit/workflow_files_*.md`：中文说明文档，适合人工查看当前流程地图。

## 10. 有标注/无标注微调与自动最终测试

新版 `finetune_target_domain.py` 默认会按微调模式分目录保存结果：

```text
runs/target_finetune/
  labeled/
    <实验名>_<初始权重名>_labeled/
      weights/best.pt
      weights/last.pt
      eval/
      final_test/
  unlabeled/
    <实验名>_<初始权重名>_unlabeled/
      pseudo_dataset/
      pseudo_train.yaml
      weights/best.pt
      weights/last.pt
      eval/
      final_test/
  finetune_summary.jsonl
  evaluation_records.jsonl
  final_test_records.jsonl
```

若希望一次同时运行有标注和无标注微调，并在微调后直接测试严格 holdout，可使用：

```bash
python finetune_target_domain.py \
  --mode both \
  --weights /ssd_data/lixiang_data/YOLO_DA/runs/detect/DA_bal_dw002_img1024_b24-3/weights/epoch30.pt \
  --data runs/target_subsets/independent_20p_seed2026/target_subset.yaml \
  --target-images runs/target_subsets/independent_20p_seed2026/images/train \
  --project runs/target_finetune \
  --name target20p \
  --epochs 40 \
  --imgsz 768 \
  --batch 8 \
  --lr0 0.0002 \
  --freeze 5 \
  --patience 12 \
  --save-period 5 \
  --pseudo-conf 0.45 \
  --overwrite-pseudo \
  --final-test-data runs/target_subsets/independent_20p_seed2026/target_strict_holdout.yaml \
  --final-test-imgsz 512,576,640,704,768,1024
```

说明：

- `--mode labeled`：只进行有标注目标域小样本微调。
- `--mode unlabeled`：只进行无标注伪标签自训练，`--target-images` 中的标签不会被读取。
- `--mode both`：先跑有标注，再跑无标注，二者分别保存到 `labeled/` 和 `unlabeled/`。
- `--final-test-data` 可以同时传入 `target_no_leak_test.yaml` 和 `target_strict_holdout.yaml`，脚本会把结果写入 `final_test_records.jsonl`。
- 若需要保持旧版扁平输出目录，可额外加入 `--flat-output`。
