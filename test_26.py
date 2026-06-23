from ultralytics import YOLO
import time
import os

if __name__ == '__main__':
    print("🚀 正在加载跨模态域自适应模型 (DANN-YOLOv11)...")

    # 替换为你实际训练出来的最佳权重路径！
    # 比如: runs/detect/YOLO11_DA_Exp1/weights/best.pt
    weight_path = "/ssd_data/lixiang_data/YOLO_DA/runs/detect/finetuned_uda_yolo/epoch10_finetuned.pt"
    model = YOLO(weight_path)

    CONF = 0
    IMGSZ = 240
    print("📊 开始在 FAIR-CASR 测试集上打靶验证...")
    # 注意：在测试阶段，模型会自动忽略我们的 DomainClassifier，只走正常的检测分支
    metrics = model.val(
        data="test.yaml",
        split="val",  # 读取 yaml 里的 val 路径 (即我们设置的 test 文件夹)
        imgsz=IMGSZ,  # 保持和训练时一样的分辨率
        batch=32,
        device="0",
        # conf=CONF,  # 置信度阈值 (如果想看更高召回率，可以调成 0.001)
        # iou=0.6,  # NMS 阈值
        # augment=True,
        name="uda_test/epoch10"  # 结果保存的文件夹名称
    )

    # ==========================================
    # 提取核心评价指标
    # ==========================================
    map50 = metrics.box.map50  # 即 mAP@0.5 或 AP50
    map75 = metrics.box.map75  # 即 mAP@0.75 或 AP75 (更为严苛的指标)
    map50_95 = metrics.box.map  # 即 mAP@0.5:0.95 (COCO标准的绝对核心指标)

    # ==========================================
    # 将结果持久化保存到本地 txt 文件
    # ==========================================
    # 获取当前时间作为时间戳
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_log_file = "Test_Log.txt"

    # 使用 'a' 模式追加写入，这样你多次测试的记录都会保留在一个文件里
    with open(save_log_file, "a", encoding="utf-8") as f:
        f.write(f"========== 测试时间: {current_time} ==========\n")
        f.write(f"测试权重: {weight_path}\n")
        f.write(f"测试分辨率 (imgsz): {IMGSZ}\n")
        f.write(f"--------------------------------------------\n")
        f.write(f"AP50 (mAP@0.5)      : {map50:.4f}\n")
        f.write(f"AP75 (mAP@0.75)     : {map75:.4f}\n")
        f.write(f"mAP@0.5:0.95        : {map50_95:.4f}\n")
        f.write(f"conf                : {CONF}")
        f.write(f"model path          : {weight_path}")
        f.write(f"============================================\n\n")

    print("\n✅ 评估结束！")
    print(f"AP50 (mAP@0.5): {map50:.4f}")
    print(f"AP75 (mAP@0.75): {map75:.4f}")
    print(f"mAP@0.5:0.95: {map50_95:.4f}")
    print(f"📁 详细指标日志已永久追加保存至: {os.path.abspath(save_log_file)}")