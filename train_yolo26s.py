import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "yolo_source"))
from ultralytics import YOLO


def main():
    DATASET_YAML = "/ssd_data/lixiang_data/YOLO_DA/dataset_sar_only.yaml"
    MODEL_SIZE = "s"
    EPOCHS = 100
    IMGSZ = 1024
    BATCH = 16
    DEVICE = "0"
    WORKERS = 8
    SAVE_PERIOD = 10

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU缓存已清理 (已用: {torch.cuda.memory_allocated()/1024**3:.1f}G / 总量: {total_gpu:.1f}G)")

    model_id = f"yolo26{MODEL_SIZE}-da"
    config_path = f"yolo_source/ultralytics/cfg/models/26/{model_id}.yaml"

    print(f"加载模型配置: {config_path}")
    model = YOLO(config_path)

    model.add_callback("on_train_epoch_end", _clear_gpu_cache)
    model.add_callback("on_fit_epoch_end", _clear_gpu_cache)

    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        save_period=SAVE_PERIOD,
        project="runs/detect",
        name=f"YOLO26{MODEL_SIZE}_SAR",
        exist_ok=True,
    )

    print(f"\n最佳模型 mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")


def _clear_gpu_cache(trainer):
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
