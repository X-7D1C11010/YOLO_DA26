"""
YOLO26 域对抗迁移学习脚本 (Domain Adversarial Transfer Learning)
========================================
基于已训练模型进行迁移学习，适应新数据集

核心特点:
1. 加载预训练的域对抗模型权重
2. 使用小学习率进行微调，保护已学习的特征
3. 可选冻结骨干网络，仅训练头部
4. 继续使用域对抗训练，提升泛化能力
5. 针对少量数据优化的训练策略

使用方法:
    python train_transfer.py
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

# 添加yolo_source路径
sys.path.insert(0, str(Path(__file__).parent / "yolo_source"))

from ultralytics import YOLO


class DomainAdversarialTransferTrainer:
    """
    域对抗迁移学习训练器类
    封装了基于预训练模型的迁移学习流程
    """
    
    def __init__(
        self,
        model_size: str = 'n',
        data_yaml: str = None,
        pretrained_weights: str = None,
        device: str = '0',
        project_name: str = "YOLO26_DA_transfer"
    ):
        """
        初始化迁移学习训练器
        
        Args:
            model_size: 模型尺寸 ('n', 's', 'm', 'l', 'x')
            data_yaml: 新数据集配置文件路径
            pretrained_weights: 预训练模型权重路径
            device: GPU设备ID
            project_name: 项目名称
        """
        self.model_size = model_size
        self.data_yaml = data_yaml
        self.pretrained_weights = pretrained_weights
        self.device = device
        self.project_name = project_name
        self.model = None
        
        # 域对抗训练超参数
        self.da_config = {
            'domain_weight': 0.05,      # 迁移学习时使用较小的域对抗权重
            'alpha_schedule': 'linear', # alpha调度策略
            'alpha_max': 1.0,           # 最大alpha值
        }
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载预训练模型"""
        print(f"🚀 正在加载预训练模型: {self.pretrained_weights}")
        
        if self.pretrained_weights is None or not os.path.exists(self.pretrained_weights):
            raise ValueError(f"❌ 预训练权重文件不存在: {self.pretrained_weights}")
        
        # 加载预训练模型
        self.model = YOLO(self.pretrained_weights)
        print("✅ 预训练模型加载完成!")
    
    def freeze_backbone(self, freeze_ratio: float = 0.7):
        """
        冻结骨干网络的部分层
        
        Args:
            freeze_ratio: 冻结层的比例 (0-1)，1表示冻结全部骨干
        """
        if self.model is None:
            raise ValueError("❌ 模型未加载!")
        
        print(f"\n🔒 冻结骨干网络 {int(freeze_ratio*100)}% 的层...")
        
        # 获取模型的参数
        model = self.model.model.model
        
        # 计算要冻结的层数
        num_layers = len(model)
        freeze_until = int(num_layers * freeze_ratio)
        
        # 冻结指定层
        for i, (name, param) in enumerate(model.named_parameters()):
            if i < freeze_until:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 统计冻结和可训练参数
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   冻结参数: {frozen_params / 1e6:.2f}M")
        print(f"   可训练参数: {trainable_params / 1e6:.2f}M")
        print("✅ 冻结完成!")
    
    def unfreeze_all(self):
        """解冻所有层"""
        print("\n🔓 解冻所有层...")
        for param in self.model.model.model.parameters():
            param.requires_grad = True
        print("✅ 解冻完成!")
    
    def train(
        self,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 8,
        lr0: float = 0.001,  # 迁移学习使用较小的学习率
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: int = 2,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        box: float = 7.5,
        cls: float = 0.5,
        dfl: float = 1.5,
        domain_weight: float = 0.05,
        close_mosaic: int = 5,
        save_period: int = -1,
        patience: int = 20,
        workers: int = 2,
        freeze_backbone_epochs: int = 10,  # 冻结训练的epoch数
        freeze_ratio: float = 0.7,         # 冻结比例
        **kwargs
    ):
        """
        执行迁移学习训练
        
        Args:
            epochs: 训练轮数
            imgsz: 输入图像尺寸
            batch: 批次大小（小数据集建议使用较小的batch）
            lr0: 初始学习率（迁移学习建议使用较小的值）
            lrf: 最终学习率系数
            momentum: 动量
            weight_decay: 权重衰减
            warmup_epochs: 预热轮数
            warmup_momentum: 预热动量
            warmup_bias_lr: 预热偏置学习率
            box: 边框损失权重
            cls: 分类损失权重
            dfl: DFL损失权重
            domain_weight: 域对抗损失权重（迁移学习时建议较小）
            close_mosaic: 关闭mosaic增强的轮数
            save_period: 保存周期
            patience: 早停耐心值
            workers: 数据加载线程数
            freeze_backbone_epochs: 冻结骨干训练的epoch数
            freeze_ratio: 冻结骨干的比例
            **kwargs: 其他参数
        """
        if self.data_yaml is None:
            raise ValueError("❌ 数据集配置文件路径未设置!")
        
        if not os.path.exists(self.data_yaml):
            raise ValueError(f"❌ 数据集配置文件不存在: {self.data_yaml}")
        
        # 更新域对抗配置
        self.da_config['domain_weight'] = domain_weight
        
        print("\n" + "="*60)
        print("🔥 启动 YOLO26 域对抗迁移学习")
        print("="*60)
        print(f"📊 训练配置:")
        print(f"   - 模型: YOLO26{self.model_size}-DA (迁移学习)")
        print(f"   - 预训练权重: {self.pretrained_weights}")
        print(f"   - 新数据集: {self.data_yaml}")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch Size: {batch}")
        print(f"   - Image Size: {imgsz}")
        print(f"   - Learning Rate: {lr0} -> {lr0 * lrf}")
        print(f"   - Domain Weight: {domain_weight}")
        print(f"   - Freeze Epochs: {freeze_backbone_epochs}")
        print(f"   - Freeze Ratio: {int(freeze_ratio*100)}%")
        print(f"   - Device: {self.device}")
        print("="*60 + "\n")
        
        # 阶段1: 冻结骨干网络训练
        if freeze_backbone_epochs > 0:
            print(f"\n🎯 阶段1: 冻结骨干网络训练 ({freeze_backbone_epochs} epochs)")
            self.freeze_backbone(freeze_ratio=freeze_ratio)
            
            print(f"\n📈 开始第1阶段训练...")
            results = self.model.train(
                data=self.data_yaml,
                epochs=freeze_backbone_epochs,
                imgsz=imgsz,
                batch=batch,
                device=self.device,
                name=f"{self.project_name}_stage1",
                
                # 学习率（较小）
                lr0=lr0 * 0.1,  # 冻结阶段使用更小的学习率
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
                
                # 域对抗损失权重（较小）
                domain_weight=domain_weight * 0.5,
                
                # 数据增强（较小数据集使用较弱增强）
                close_mosaic=close_mosaic,
                augment=False,  # 小数据集关闭增强
                
                # 其他
                save_period=save_period,
                patience=patience,
                workers=workers,
                **kwargs
            )
            
            # 更新预训练权重路径为阶段1的最佳权重
            # 从训练结果中获取保存目录
            if hasattr(results, 'save_dir'):
                stage1_save_dir = Path(results.save_dir)
            else:
                # 如果无法从结果获取，则手动构建路径
                stage1_save_dir = Path("runs/detect") / f"{self.project_name}_stage1"
            
            self.pretrained_weights = str(stage1_save_dir / "weights" / "best.pt")
            print(f"\n🔄 阶段1完成，加载最佳权重: {self.pretrained_weights}")
            self.model = YOLO(self.pretrained_weights)
        
        # 阶段2: 解冻全部层，进行微调
        print(f"\n🎯 阶段2: 解冻全部层微调 ({epochs - freeze_backbone_epochs} epochs)")
        self.unfreeze_all()
        
        print(f"\n📈 开始第2阶段训练...")
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs - freeze_backbone_epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            name=self.project_name,
            
            # 学习率（迁移学习使用较小值）
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
            
            # 域对抗损失权重
            domain_weight=domain_weight,
            
            # 数据增强
            close_mosaic=close_mosaic,
            augment=True,
            
            # 其他
            save_period=save_period,
            patience=patience,
            workers=workers,
            **kwargs
        )
        
        print("\n" + "="*60)
        print("✅ 迁移学习训练完成!")
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
    # 预训练模型权重路径
    PRETRAINED_WEIGHTS = "/ssd_data/lixiang_data/YOLO_DA/runs/detect/YOLO26n_DA/weights/best.pt"
    
    # 新数据集配置
    NEW_DATA_YAML = "/ssd_data/lixiang_data/Datasets/MATD/matd.yaml"
    
    # 模型配置
    MODEL_SIZE = 'n'  # 可选: 'n', 's', 'm', 'l', 'x'
    DEVICE = '0'
    
    # 训练配置（针对少量数据优化）
    EPOCHS = 50        # 迁移学习使用较少epoch
    IMGSZ = 640
    BATCH = 8          # 小数据集使用较小batch
    
    # 学习率配置（迁移学习使用较小学习率）
    LR0 = 0.001        # 比从头训练小10倍
    LRF = 0.01
    
    # 域对抗配置（迁移学习使用较小权重）
    DOMAIN_WEIGHT = 0.05  # 比从头训练小
    
    # 冻结训练配置
    FREEZE_EPOCHS = 30    # 先冻结骨干训练10个epoch
    FREEZE_RATIO = 0.7    # 冻结70%的骨干层
    
    # 损失权重
    BOX = 7.5
    CLS = 0.5
    DFL = 1.5
    
    # 其他配置
    CLOSE_MOSAIC = 5
    PATIENCE = 20
    WORKERS = 2
    # ================================================
    
    # 检查预训练权重
    if not os.path.exists(PRETRAINED_WEIGHTS):
        print(f"❌ 预训练权重文件不存在: {PRETRAINED_WEIGHTS}")
        print("   请修改 PRETRAINED_WEIGHTS 变量为正确的路径")
        return
    
    # 检查新数据集配置文件
    if not os.path.exists(NEW_DATA_YAML):
        print(f"❌ 新数据集配置文件不存在: {NEW_DATA_YAML}")
        print("   请创建数据集配置文件或修改 NEW_DATA_YAML 变量")
        return
    
    # 创建迁移学习训练器
    trainer = DomainAdversarialTransferTrainer(
        model_size=MODEL_SIZE,
        data_yaml=NEW_DATA_YAML,
        pretrained_weights=PRETRAINED_WEIGHTS,
        device=DEVICE,
        project_name=f"YOLO26{MODEL_SIZE}_DA_transfer"
    )
    
    # 开始迁移学习训练
    trainer.train(
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        lr0=LR0,
        lrf=LRF,
        box=BOX,
        cls=CLS,
        dfl=DFL,
        domain_weight=DOMAIN_WEIGHT,
        close_mosaic=CLOSE_MOSAIC,
        patience=PATIENCE,
        workers=WORKERS,
        freeze_backbone_epochs=FREEZE_EPOCHS,
        freeze_ratio=FREEZE_RATIO
    )
    
    # 验证
    trainer.validate(split='val')
    
    # 可选：导出模型
    # trainer.export(format='onnx')


if __name__ == "__main__":
    main()
