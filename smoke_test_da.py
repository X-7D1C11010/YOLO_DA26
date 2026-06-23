"""DA 计算图快速自检，不需要数据集或预训练权重。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "yolo_source"))

from ultralytics import YOLO


def main():
    config = ROOT / "yolo_source" / "ultralytics" / "cfg" / "models" / "26" / "yolo26n-da.yaml"
    wrapper = YOLO(str(config))
    model = wrapper.model
    classifiers = [layer for layer in model.model if layer.__class__.__name__ == "DomainClassifier"]
    assert len(classifiers) == 1, f"期望 1 个 DomainClassifier，实际为 {len(classifiers)}"

    model.train()
    model.da_alpha = 0.5
    with torch.no_grad():
        output = model.predict(torch.zeros(2, 3, 256, 256))
    assert isinstance(output, dict), f"训练态输出类型异常：{type(output)}"
    assert "domain_preds" in output, "训练态输出缺少 domain_preds"
    assert tuple(output["domain_preds"].shape) == (2, 1), output["domain_preds"].shape

    model.eval()
    with torch.no_grad():
        output = model.predict(torch.zeros(1, 3, 256, 256))
    if isinstance(output, dict):
        assert "domain_preds" not in output, "推理态不应暴露域判别输出"

    print("DA smoke test 通过：域判别器已接入训练图，推理分支保持兼容。")


if __name__ == "__main__":
    main()
