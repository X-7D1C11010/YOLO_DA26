"""
YOLO26 域对抗训练脚本 (Domain Adversarial Training)
========================================
基于DANN (Domain Adversarial Neural Network) 实现跨模态域适应
源域: 光学图像 (Optical Images)
目标域: SAR图像 (Synthetic Aperture Radar Images)

核心原理:
1. 检测器学习域不变特征，提升在目标域的泛化能力
2. 梯度反转层(GRL)使特征提取器学习域不变表示
3. 域判别器尝试区分源域/目标域，特征提取器欺骗判别器

使用方法:
    python train_yolo26_da.py
"""

import os
import sys
import math
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import multiprocessing

# 确保单进程模式（RANK=-1）
os.environ.pop('RANK', None)
os.environ.pop('WORLD_SIZE', None)
os.environ.pop('LOCAL_RANK', None)

# 避免多进程共享内存问题
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# 添加yolo_source路径
sys.path.insert(0, str(Path(__file__).parent / "yolo_source"))

from ultralytics import YOLO


class DomainAdversarialTrainer:
    """
    域对抗训练器类
    封装了YOLO26域适应训练的完整流程
    """
    
    def __init__(
        self,
        model_size: str = 'n',
        data_yaml: str = None,
        pretrained: bool = True,
        device: str = '0',
        project_name: str = "YOLO26_DA",
        pretrained_weights: str = None,  # 最小重训指定预训练权重路径
        fresh_start: bool = False         # 全新训练True=不加载任何旧检查点，从零开始
    ):
        """
        初始化域对抗训练器
        
        Args:
            model_size: 模型尺寸 ('n', 's', 'm', 'l', 'x')
            data_yaml: 数据集配置文件路径
            pretrained: 是否使用预训练权重
            device: GPU设备ID
            project_name: 项目名称
            pretrained_weights: 最小重训预训练权重完整路径(.pt)，如为None则自动查找
            fresh_start: 全新训练True=跳过所有旧检查点，用yaml配置从零创建
        """
        self.model_size = model_size
        self.data_yaml = data_yaml
        self.device = device
        self.project_name = project_name
        self.model = None
        self.pretrained_weights = pretrained_weights
        self.fresh_start = fresh_start
        
        # 域对抗训练超参数
        self.da_config = {
            'domain_weight': 0.1,       # 域对抗损失权重
            'alpha_schedule': 'linear', # alpha调度策略 ('linear', 'exp', 'constant')
            'alpha_max': 1.0,           # 最大alpha值
        }
        
        # 加载模型
        self._load_model(pretrained)
    
    def _load_model(self, pretrained: bool):
        """加载YOLO26模型"""
        model_id = f"yolo26{self.model_size}-da"
        print(f"🚀 正在加载 {model_id.upper()} 模型...")
        
        config_path = f"yolo_source/ultralytics/cfg/models/26/{model_id}.yaml"
        
        # 【全新训练】跳过所有旧检查点，直接用yaml配置从零创建
        if self.fresh_start:
            print(f"   🆕 全新训练模式: 跳过旧检查点，从零初始化{model_id.upper()}模型")
            self.model = YOLO(config_path)
            print("✅ 模型加载完成!")
            return
        
        load_path = None
        
        # 优先使用指定的预训练权重路径（兼容DA架构）
        if pretrained and self.pretrained_weights and os.path.exists(self.pretrained_weights):
            load_path = self.pretrained_weights
            print(f"   📦 加载DA预训练权重: {load_path}")
        elif pretrained:
            # 自动查找最近训练的DA检查点
            import glob
            da_dirs = sorted(glob.glob("/ssd_data/lixiang_data/YOLO_DA/runs/detect/YOLO26n_DA-2/weights/best.pt"))
            if da_dirs:
                load_path = da_dirs[-1]  # 使用最近一次训练的DA模型
                print(f"   📦 自动加载DA预训练权重: {load_path}")
            else:
                # 回退：从头创建DA模型（无预训练）
                print(f"   ⚠️ 未找到DA预训练权重，将从头开始训练")
                load_path = config_path
        
        if load_path:
            self.model = YOLO(load_path)
        else:
            self.model = YOLO(config_path)
        
        print("✅ 模型加载完成!")
    
    def compute_alpha(self, epoch: int, total_epochs: int) -> float:
        """
        计算动态alpha值（梯度反转强度）
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
            
        Returns:
            alpha值
        """
        schedule = self.da_config['alpha_schedule']
        alpha_max = self.da_config['alpha_max']
        
        if schedule == 'linear':
            # 线性增长
            p = epoch / total_epochs
            alpha = alpha_max * (2 / (1 + math.exp(-10 * p)) - 1)
        elif schedule == 'exp':
            # 指数增长
            p = epoch / total_epochs
            alpha = alpha_max * (1 - math.exp(-5 * p))
        else:
            # 常数
            alpha = alpha_max
        
        return alpha
    
    # ============================================================
    # 最小重训 在train()方法中添加freeze参数支持
    # 调用时传入 freeze=N 即可冻结前N层（通常是backbone部分）
    # ============================================================
    def train(
        self,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        rect: bool = False,
        multi_scale: float = 0.0,  # 0.5=在512~1536间变化
        lr0: float = 0.01,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: int = 3,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        box: float = 7.5,
        cls: float = 0.5,
        dfl: float = 1.5,
        domain_weight: float = 0.1,
        close_mosaic: int = 10,
        save_period: int = -1,
        patience: int = 50,
        workers: int = 2,
        freeze: int = 0,
        **kwargs
    ):
        """
        执行域对抗训练
        
        Args:
            epochs: 训练轮数
            imgsz: 输入图像尺寸
            batch: 批次大小
            rect: 矩形训练模式，按长边等比缩放保持宽高比
            multi_scale: 每个batch随机缩放的比例，0.5=512~1536
            lr0: 初始学习率
            lrf: 最终学习率系数
            momentum: 动量
            weight_decay: 权重衰减
            warmup_epochs: 预热轮数
            warmup_momentum: 预热动量
            warmup_bias_lr: 预热偏置学习率
            box: 边框损失权重
            cls: 分类损失权重
            dfl: DFL损失权重
            domain_weight: 域对抗损失权重
            close_mosaic: 关闭mosaic增强的轮数
            save_period: 保存周期
            patience: 早停耐心值
            workers: 数据加载线程数
            freeze: 【最小重训】冻结前N层，建议10(冻结backbone)，0=不冻结
            **kwargs: 其他参数
        """
        if self.data_yaml is None:
            raise ValueError("❌ 数据集配置文件路径未设置!")
        
        # 更新域对抗配置
        self.da_config['domain_weight'] = domain_weight
        
        # 如果启用冻结，打印冻结信息
        if freeze > 0:
            print(f"🔒 最小重训模式: 冻结前 {freeze} 层 (backbone)")
        
        print("\n" + "="*60)
        print("🔥 启动 YOLO26 域对抗训练 (Domain Adversarial Training)")
        print("="*60)
        print(f"📊 训练配置:")
        print(f"   - 模型: YOLO26{self.model_size}-DA")
        print(f"   - 数据集: {self.data_yaml}")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch Size: {batch}")
        print(f"   - Image Size: {imgsz}")
        print(f"   - Rect Mode: {rect}")
        print(f"   - Multi Scale: {multi_scale}")  # 多尺度范围
        print(f"   - Learning Rate: {lr0} -> {lr0 * lrf}")
        print(f"   - Domain Weight: {domain_weight}")
        print(f"   - Freeze Layers: {freeze}")  # 最小重训显示冻结层数
        print(f"   - Device: {self.device}")
        print("="*60 + "\n")
        
        # 开始训练
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            rect=rect,
            multi_scale=multi_scale,  # 多尺度参数
            device=self.device,
            name=self.project_name,
            
            # 学习率相关
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            
            # 损失权重
            box=box,
            cls=cls,
            dfl=dfl,
            
            # 域对抗损失权重（自定义参数）
            domain_weight=domain_weight,
            
            # 数据增强
            close_mosaic=close_mosaic,
            
            # 最小重训冻结层数
            freeze=freeze,
            
            # 其他
            save_period=save_period,
            patience=patience,
            workers=workers,
            
            **kwargs
        )
        
        print("\n" + "="*60)
        print("✅ 训练完成!")
        print("="*60)
        
        return results
    
    def validate(self, split: str = 'val'):
        """
        验证模型
        
        Args:
            split: 验证集划分 ('val', 'test')
        """
        print(f"\n🔍 在 {split} 集上验证模型...")
        results = self.model.val(split=split)
        return results
    
    def export(self, format: str = 'onnx'):
        """
        导出模型
        
        Args:
            format: 导出格式 ('onnx', 'torchscript', 'engine', etc.)
        """
        print(f"\n📦 导出模型为 {format} 格式...")
        self.model.export(format=format)
        print("✅ 导出完成!")


def main():
    """主函数"""
    
    # ==================== 配置区域 ====================
    # 模型配置
    MODEL_SIZE = 's'  # 可选: 'n', 's', 'm', 'l', 'x'
    PRETRAINED = True
    
    # 最小重训开关：设为True启用最小重训，False为正常训练
    MINIMAL_RETRAIN = False
    
    # 最小重训预训练DA检查点路径：从之前完整训练的DA模型中加载权重
    PRETRAINED_BASE = ""  # 留空=自动查找最新的DA训练结果
    
    # 全新训练开关：True=不加载任何旧检查点，用yaml配置从零初始化
    FRESH_START = True
    
    # 训练配置
    EPOCHS = 100
    IMGSZ = 640
    BATCH = 16
    MULTI_SCALE = 0.3
    RECT = False
    DEVICE = '0'
    
    # 域对抗配置
    DOMAIN_WEIGHT = 0.03  # DA权重，0.03已足够让域判别器辅助但不主导
    
    if DOMAIN_WEIGHT > 0:
        DATASET_YAML = "/ssd_data/lixiang_data/YOLO_DA/train_sardet.yaml"      # VIS+SAR 双域
    else:
        DATASET_YAML = "/ssd_data/lixiang_data/YOLO_DA/dataset_sar_only.yaml"  # SAR
    
    # 学习率配置
    LR0 = 0.01
    LRF = 0.01
    
    # 损失权重
    BOX = 7.5
    CLS = 0.5
    DFL = 1.5
    
    # 其他配置
    CLOSE_MOSAIC = 30
    PATIENCE = 50           # DA关闭后收敛更快，无需过多耐心
    SAVE_PERIOD = 10
    WORKERS = 8
    
    # 最小重训
    if MINIMAL_RETRAIN:
        EPOCHS = 40
        LR0 = 0.0005
        DOMAIN_WEIGHT = 0.01
        CLOSE_MOSAIC = 0
        MULTI_SCALE = 0.3
        FREEZE = 3              # 只冻结前3层，允许backbone大部分适配
        print("=" * 60)
        print("🔒 最小重训模式 v2 (渐进式解冻) 已启用!")
        print(f"   - Epochs: {EPOCHS}")
        print(f"   - LR: {LR0} (极低学习率，防过拟合)")
        print(f"   - Domain Weight: {DOMAIN_WEIGHT} (DA近乎关闭)")
        print(f"   - Freeze: 前{FREEZE}层 (轻量冻结，允许适配)")
        print(f"   - Multi Scale: {MULTI_SCALE} (保持尺度鲁棒)")
        print(f"   - Close Mosaic: {CLOSE_MOSAIC} (纯真实分布)")
        print(f"   前提: 需要先用正常模式完成多尺度DA训练")
        print("=" * 60)
    else:
        FREEZE = 0
        PRETRAINED_BASE = ""
    
    # 检查数据集配置文件
    if not os.path.exists(DATASET_YAML):
        print(f"❌ 数据集配置文件不存在: {DATASET_YAML}")
        print("   请修改 DATASET_YAML 变量为正确的路径")
        return
    
    # 最小重训构建预训练权重路径
    if MINIMAL_RETRAIN and not PRETRAINED_BASE:
        import glob
        da_dirs = sorted(glob.glob("/ssd_data/lixiang_data/YOLO_DA/runs/detect/YOLO26s_DA-3/weights/epoch90.pt"))
        if da_dirs:
            PRETRAINED_BASE = da_dirs[-1]
            print(f"📦 自动找到DA预训练权重: {PRETRAINED_BASE}")
        else:
            print("❌ 未找到任何DA训练检查点！请先运行一次完整训练(正常模式)")
            return
    
    if MINIMAL_RETRAIN:
        print(f"📦 最小重训将从以下检查点加载: {PRETRAINED_BASE}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 已清理GPU缓存 (已用: {torch.cuda.memory_allocated()/1024**3:.1f}G / 总量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}G)")
    
    # 创建训练器
    trainer = DomainAdversarialTrainer(
        model_size=MODEL_SIZE,
        data_yaml=DATASET_YAML,
        pretrained=PRETRAINED,
        device=DEVICE,
        project_name=f"YOLO26{MODEL_SIZE}_DA",
        pretrained_weights=PRETRAINED_BASE if MINIMAL_RETRAIN else None,  # 最小重训传入DA检查点
        fresh_start=FRESH_START  # 全新训练跳过旧检查点
    )

    def _clear_cache(trainer):
        torch.cuda.empty_cache()

    trainer.model.add_callback("on_train_epoch_end", _clear_cache)
    trainer.model.add_callback("on_fit_epoch_end", _clear_cache)
    
    # 开始训练
    trainer.train(
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        rect=RECT,
        multi_scale=MULTI_SCALE,  # 多尺度训练，在512~1536间随机变化
        lr0=LR0,
        lrf=LRF,
        box=BOX,
        cls=CLS,
        dfl=DFL,
        domain_weight=DOMAIN_WEIGHT,
        close_mosaic=CLOSE_MOSAIC,
        patience=PATIENCE,
        save_period=SAVE_PERIOD,  # 每10轮保存一次模型
        workers=WORKERS,
        freeze=FREEZE  # 最小重训传递冻结参数
    )
    
    # 验证
    trainer.validate(split='val')

if __name__ == "__main__":
    main()
