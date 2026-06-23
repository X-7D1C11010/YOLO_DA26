import torch
from ultralytics import YOLO

if __name__ == "__main__":
    print("🚀 正在加载魔改版 DANN-YOLOv11...")

    # 【注意看这里！路径的开头变成了 yolo_source】
    model = YOLO("yolo_source/ultralytics/cfg/models/11/yolo11-da.yaml")

    # 加载预训练权重
    model = YOLO("yolo11n.pt")

    # 挂载数据集
    DATASET_YAML = "/ssd_data/lixiang_data/YOLO_DA/dataset.yaml"

    print("🔥 启动跨模态域对抗训练 (Domain Adversarial Training)...")
    results = model.train(
        data=DATASET_YAML,
        epochs=200,
        imgsz=1024,         # 改为1024，匹配测试图像原始尺寸
        batch=8,            # 1024²显存是640²的4倍，从32减到8
        rect=True,          # 矩形训练: 按长边等比缩放，保持宽高比
        device="0",
        name="YOLO11_DA",
        close_mosaic=10
    )