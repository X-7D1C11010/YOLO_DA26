from ultralytics import YOLO
import time
import os

if __name__ == '__main__':
    print("=" * 70)
    print("YOLO_DA 改进版测试: TTA @ imgsz=512")
    print("=" * 70)

    weight_path = "/ssd_data/lixiang_data/YOLO_DA/runs/detect/YOLO11_DA/weights/best.pt"

    for wp, label in [
        ("/ssd_data/lixiang_data/YOLO_DA/runs/detect/YOLO11_DA/weights/best.pt", "YOLO11_DA"),
        ("/ssd_data/lixiang_data/YOLO_DA/runs/detect/YOLO26n_DA-2/weights/best.pt", "YOLO26_DA"),
    ]:
        if not os.path.exists(wp):
            print(f"  {label}: 权重不存在 {wp}")
            continue

        model = YOLO(wp)
        print(f"\n  {'='*60}")
        print(f"  {label}")
        print(f"  {'='*60}")

        for imgsz_v in [640, 512, 384]:
            for aug in [False, True]:
                try:
                    metrics = model.val(
                        data="test.yaml",
                        split="val",
                        imgsz=imgsz_v,
                        batch=32,
                        device="0",
                        augment=aug,
                        conf=0.001,
                        name=f"test_{label}_imgsz{imgsz_v}_aug{aug}"
                    )
                    tag = "TTA" if aug else "NO_TTA"
                    print(f"  [{tag}] imgsz={imgsz_v}: AP50={metrics.box.map50:.4f}, "
                          f"mAP={metrics.box.map:.4f}, P={metrics.box.mp:.4f}, R={metrics.box.mr:.4f}")
                except Exception as e:
                    print(f"  [{tag}] imgsz={imgsz_v}: ERROR {e}")
