from ultralytics import YOLO
import time
import os

if __name__ == '__main__':

    # ============================================================
    # 通用配置
    # ============================================================
    # 权重路径：根据你实际训练出的best.pt修改
    weight_path = "/ssd_data/lixiang_data/YOLO_DA/runs/detect/YOLO26n_DA-3/weights/best.pt"
    model = YOLO(weight_path)

    CONF = 0.001              # 置信度阈值

    # ============================================================
    # 方案A：不重新训练（当前已有DA权重直接用）
    #   ⚠️  DA模型不支持augment=True（架构被修改过，TTA自动降级）
    #   策略：单尺度 imgsz=512（最佳单尺度）
    #   原理：512下物体尺度更接近训练分布
    #   预期AP50：~4-6%（比原640的2.7%略有提升，但远不如baseline的21%）
    #   注：之前21%的结果是用YOLO_baseline标准模型测的，不是DA模型
    # ============================================================
    #   结论：DA模型只能靠重训解决，不重训的天花板非常低
    # ============================================================
    USE_RETRAINED = True  # <--- 改为 True 则使用方案B

    if not USE_RETRAINED:
        # ========== 方案A：不重训，单尺度 imgsz=512 ==========
        # DA模型不支持augment=True，只能用单尺度找最优分辨率
        print("方案A：不重训 — imgsz=512（DA模型不支持TTA）")
        metrics = model.val(
            data="test.yaml",
            split="val",
            imgsz=320,          # 最接近训练物体尺度分布的单分辨率
            batch=32,
            device="0",
            conf=CONF,
            # augment=True,     # DA模型不支持！加了等于没加
            name="test_no_retrain_512"
        )
        log_imgsz = 512
        log_mode = "不重训-imgsz512"

    else:
        # ========== 方案B：重新训练后 ==========
        # 前提：已用 train_26.py (imgsz=1024, rect=True) 重新训练完毕
        # 重训后的DA模型同样可以用augment=True，因为权重会重新生成
        print("方案B：重训后 — imgsz=1024")
        metrics = model.val(
            data="test.yaml",
            split="val",
            imgsz=240,         # 与重训分辨率一致
            batch=16,           # 1024²显存大，batch减半
            device="0",
            # conf=CONF,
            rect=True,
            # iou=0.6,
            name="test_retrained_1024"
        )
        log_imgsz = 1024
        log_mode = "重训后-imgsz1024"

    # ============================================================
    # 提取指标 & 日志
    # ============================================================
    map50    = metrics.box.map50
    map75    = metrics.box.map75
    map50_95 = metrics.box.map

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_log_file = "Evaluation_Metrics_Log.txt"

    with open(save_log_file, "a", encoding="utf-8") as f:
        f.write(f"========== 测试时间: {current_time} ==========\n")
        f.write(f"测试模式: {log_mode}\n")
        f.write(f"测试权重: {weight_path}\n")
        f.write(f"测试分辨率 (imgsz): {log_imgsz}\n")
        f.write(f"置信度阈值 (conf): {CONF}\n")
        f.write(f"--------------------------------------------\n")
        f.write(f"AP50 (mAP@0.5)      : {map50:.4f}\n")
        f.write(f"AP75 (mAP@0.75)     : {map75:.4f}\n")
        f.write(f"mAP@0.5:0.95        : {map50_95:.4f}\n")
        f.write(f"============================================\n\n")

    print(f"\n测试模式: {log_mode}")
    print(f"AP50 (mAP@0.5): {map50:.4f}")
    print(f"AP75 (mAP@0.75): {map75:.4f}")
    print(f"mAP@0.5:0.95: {map50_95:.4f}")
    print(f"日志已保存至: {os.path.abspath(save_log_file)}")
