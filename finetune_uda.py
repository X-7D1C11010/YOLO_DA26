"""
UDA 域对抗微调 (主干对齐与检测头隔离保护版)
=============================================================
"""

import os
import sys
import argparse
import glob
import math
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "yolo_source"))
from ultralytics.nn.modules.block import DomainClassifier

class DomainDataset(Dataset):
    """加载单域图像，不读标签"""
    def __init__(self, image_dir, imgsz=1024, sample_ratio=1.0):
        self.imgsz = imgsz
        self.image_paths = []
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            for p in Path(image_dir).rglob(f"*{ext}"):
                self.image_paths.append(str(p))
            for p in Path(image_dir).rglob(f"*{ext.upper()}"):
                self.image_paths.append(str(p))
        self.image_paths = sorted(set(self.image_paths))
        if sample_ratio < 1.0 and self.image_paths:
            random.seed(42)
            n = max(1, int(len(self.image_paths) * sample_ratio))
            self.image_paths = random.sample(self.image_paths, n)
        if not self.image_paths:
            raise ValueError(f"{image_dir} 中未找到图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        if r != 1:
            img = cv2.resize(img, (int(round(w * r)), int(round(h * r))),
                             interpolation=cv2.INTER_LINEAR)
        dw = self.imgsz - img.shape[1]
        dh = self.imgsz - img.shape[0]
        img = cv2.copyMakeBorder(img, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = img.transpose(2, 0, 1)[::-1]
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img).float() / 255.0


class UDAFineTuner:
    """DANN 域对抗微调器 — 主干隔离保护版"""

    def __init__(self, checkpoint_path, device="cuda:0", align_layer_idx=9):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model_name = Path(checkpoint_path).stem
        self.sequential = None
        self.model = None
        self.domain_classifier = None
        self.align_layer_idx = align_layer_idx # 默认在第9层(Backbone末端SPPF)做对抗对齐

        self._load()
        self._freeze_detector_layers() # 【核心修复】取代原有的全模型解冻
        self._build_classifier()
        self.initial_weight_sum = self._get_weight_sum()

    def _load(self):
        print(f"\n📦 加载模型: {self.model_name}")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        md = ckpt.get("model", ckpt)

        if md is not None and hasattr(md, "model") and isinstance(md.model, nn.Sequential):
            self.model = md
            self.sequential = self.model.model.float()
        else:
            from ultralytics.nn.tasks import DetectionModel
            self.model = DetectionModel(
                cfg="yolo_source/ultralytics/cfg/models/26/yolo26s-da.yaml", nc=1)
            ema = ckpt.get("ema")
            if ema is not None:
                sd = ema.state_dict() if hasattr(ema, "state_dict") else ema
                if isinstance(sd, dict) and len(sd) > 0:
                    self.model.load_state_dict(sd, strict=False)
            self.sequential = self.model.model.float()
        print(f"  模型总参数量: {sum(p.numel() for p in self.sequential.parameters())/1e6:.1f}M")

    def _freeze_detector_layers(self):
        """
        核心修复：由于不看检测损失，必须强行冻结 Neck 和 Head (层10及以后)，
        只允许前端 Backbone 接受对抗梯度进行跨域风格对齐，从物理上隔绝特征塌陷。
        """
        for i, m in enumerate(self.sequential):
            if i <= self.align_layer_idx:
                for p in m.parameters():
                    p.requires_grad = True
            else:
                for p in m.parameters():
                    p.requires_grad = False
        
        trainable_params = sum(p.numel() for p in self.sequential.parameters() if p.requires_grad)
        print(f"  🔒 保护机制：已冻结层 {self.align_layer_idx + 1} 及后续全部检测网络。")
        print(f"  🔓 当前可训练主干参数: {trainable_params/1e6:.1f}M")

    def _build_classifier(self):
        self.sequential.to(self.device).eval()
        with torch.no_grad():
            features = self._forward_features(torch.randn(1, 3, 1024, 1024, device=self.device))
            in_ch = features.shape[1]
        self.sequential.train()
        self.domain_classifier = DomainClassifier(c1=in_ch, c2=256).to(self.device).train()
        print(f"  🛠️ 域判别器架设成功: 输入通道={in_ch} → 隐层=256 → 输出=1")

    def _forward_features(self, imgs):
        """前向传播到指定的 Backbone 终止层提取域特征"""
        x = imgs
        y = []
        for i, m in enumerate(self.sequential):
            f_from = getattr(m, "f", -1)
            if f_from != -1:
                x = y[f_from] if isinstance(f_from, int) else [x if j == -1 else y[j] for j in f_from]
            x = m(x)
            if i == self.align_layer_idx: 
                return x
            y.append(x)
        return x

    def _get_weight_sum(self):
        with torch.no_grad():
            return sum(p.abs().sum().item() for p in self.sequential.parameters() if p.requires_grad)

    def _get_grad_norm(self, module):
        total_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _train_epoch(self, loader_a, loader_b, optimizer, alpha):
        self.sequential.train()
        self.domain_classifier.train()
        loss_sum = acc_a_sum = acc_b_sum = 0.0
        n = 0
        
        grad_classifier_sum = grad_backbone_mid_sum = grad_backbone_end_sum = 0.0

        pbar = tqdm(zip(loader_a, loader_b), desc=f"  Train (α={alpha:.2f})", leave=False, total=min(len(loader_a), len(loader_b)))
        for imgs_a, imgs_b in pbar:
            imgs_a, imgs_b = imgs_a.to(self.device), imgs_b.to(self.device)
            
            fa = self._forward_features(imgs_a)
            fb = self._forward_features(imgs_b)
            
            pa = self.domain_classifier(fa, alpha)
            pb = self.domain_classifier(fb, alpha)
            
            loss = 0.5 * (F.binary_cross_entropy_with_logits(pa, torch.zeros_like(pa)) +
                          F.binary_cross_entropy_with_logits(pb, torch.ones_like(pb)))
            
            optimizer.zero_grad()
            loss.backward()
            
            # 【核心探针】监控活动主干层的真实梯度范数
            grad_classifier_sum += self._get_grad_norm(self.domain_classifier)
            if len(self.sequential) > self.align_layer_idx:
                grad_backbone_end_sum += self._get_grad_norm(self.sequential[self.align_layer_idx]) # SPPF末端
            if len(self.sequential) > 4:
                grad_backbone_mid_sum += self._get_grad_norm(self.sequential[4]) # 主干中段

            # 联合裁切主干与判别器梯度
            torch.nn.utils.clip_grad_norm_(list(self.sequential.parameters()) + list(self.domain_classifier.parameters()), max_norm=5.0)
            optimizer.step()

            loss_sum += loss.item()
            with torch.no_grad():
                acc_a_sum += (pa.sigmoid() < 0.5).float().mean().item()
                acc_b_sum += (pb.sigmoid() > 0.5).float().mean().item()
            n += 1
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", a_acc=f"{acc_a_sum/n:.2f}", b_acc=f"{acc_b_sum/n:.2f}")

        stats = {
            "loss": loss_sum / n,
            "acc": 0.5 * (acc_a_sum + acc_b_sum) / n,
            "grad_cls": grad_classifier_sum / n,
            "grad_back_end": grad_backbone_end_sum / n,
            "grad_back_mid": grad_backbone_mid_sum / n
        }
        return stats

    @torch.no_grad()
    def _validate_domain_confusion(self, val_loader_a, val_loader_b, alpha=1.0):
        self.sequential.eval()
        self.domain_classifier.eval()
        loss_sum = acc_a_sum = acc_b_sum = 0.0
        n = 0
        
        for imgs_a, imgs_b in zip(val_loader_a, val_loader_b):
            imgs_a, imgs_b = imgs_a.to(self.device), imgs_b.to(self.device)
            fa = self._forward_features(imgs_a)
            fb = self._forward_features(imgs_b)
            
            pa = self.domain_classifier(fa, alpha)
            pb = self.domain_classifier(fb, alpha)
            
            loss = 0.5 * (F.binary_cross_entropy_with_logits(pa, torch.zeros_like(pa)) +
                          F.binary_cross_entropy_with_logits(pb, torch.ones_like(pb)))
            
            loss_sum += loss.item()
            acc_a_sum += (pa.sigmoid() < 0.5).float().mean().item()
            acc_b_sum += (pb.sigmoid() > 0.5).float().mean().item()
            n += 1
            if n >= 20: break
                
        val_acc = 0.5 * (acc_a_sum + acc_b_sum) / n
        error_rate = 1.0 - val_acc
        proxy_a_distance = 2.0 * abs(1.0 - 2.0 * error_rate) 

        return loss_sum / n, val_acc, proxy_a_distance

    def fine_tune(self, domain_a, domain_b, output_dir,
                  epochs=30, imgsz=1024, batch=8, lr=1e-3, workers=4, sample_ratio=1.0):
        print(f"\n{'='*60}")
        print(f"🚀 启动隔离保护版 UDA 对抗微调")
        print(f"{'='*60}")

        ds_a = DomainDataset(domain_a, imgsz, sample_ratio)
        ds_b = DomainDataset(domain_b, imgsz, sample_ratio)
        
        val_size_a = max(1, int(len(ds_a) * 0.1))
        val_size_b = max(1, int(len(ds_b) * 0.1))
        
        train_ds_a, val_ds_a = torch.utils.data.random_split(ds_a, [len(ds_a) - val_size_a, val_size_a])
        train_ds_b, val_ds_b = torch.utils.data.random_split(ds_b, [len(ds_b) - val_size_b, val_size_b])
        
        print(f"  源域(A): 训练 {len(train_ds_a)} | 验证 {len(val_ds_a)}")
        print(f"  目标域(B): 训练 {len(train_ds_b)} | 验证 {len(val_ds_b)}")

        dl_a = DataLoader(train_ds_a, batch, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        dl_b = DataLoader(train_ds_b, batch, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        val_dl_a = DataLoader(val_ds_a, batch, shuffle=False, num_workers=workers, pin_memory=True)
        val_dl_b = DataLoader(val_ds_b, batch, shuffle=False, num_workers=workers, pin_memory=True)

        params = [p for p in self.sequential.parameters() if p.requires_grad]
        params += list(self.domain_classifier.parameters())
        optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        print("\n>>> 微调前基线特征间隙评估 <<<")
        init_val_loss, init_val_acc, init_pad = self._validate_domain_confusion(val_dl_a, val_dl_b, alpha=0.0)
        print(f"   ➤ 初始独立验证集判别准确率: {init_val_acc:.4f}")
        print(f"   ➤ 初始 Proxy A-Distance (PAD): {init_pad:.4f}")

        best_loss = float("inf")
        for ep in range(1, epochs + 1):
            t = ep / epochs
            alpha = min(1.0, 2.0 / (1.0 + math.exp(-10.0 * t)) - 1.0)
            cos_lr = lr * 0.01 + 0.5 * (lr - lr * 0.01) * (1.0 + math.cos(math.pi * t))
            for pg in optim.param_groups:
                pg["lr"] = cos_lr
                
            train_stats = self._train_epoch(dl_a, dl_b, optim, alpha)
            val_loss, val_acc, val_pad = self._validate_domain_confusion(val_dl_a, val_dl_b, alpha)
            
            weight_diff = abs(self._get_weight_sum() - self.initial_weight_sum)
            
            print(f"\n[Epoch {ep:02d}/{epochs}] α={alpha:.3f} | lr={cos_lr:.2e} | 主干ΔW={weight_diff:.2f}")
            print(f"  ├─ 训练集表现: Loss={train_stats['loss']:.4f} | Domain Acc={train_stats['acc']:.3f}")
            print(f"  ├─ 验证集混淆: Loss={val_loss:.4f} | Domain Acc={val_acc:.3f} | A-Distance={val_pad:.4f}")
            print(f"  └─ 隔离层梯度: 判别器={train_stats['grad_cls']:.4f} | 主干末(L9)={train_stats['grad_back_end']:.4f} | 主干中(L4)={train_stats['grad_back_mid']:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss

        print("\n>>> 对抗微调结束最终评估 <<<")
        print(f"   ➤ 最终跨域验证 A-Distance: {val_pad:.4f} (期望此指标比初始降低)")
        return best_loss

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.sequential.to("cpu")
        self.domain_classifier.to("cpu")
        out = os.path.join(output_dir, f"{self.model_name}_finetuned.pt")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        
        ckpt["model"] = copy.deepcopy(self.model).half()
        
        # 强制将我们冻结保持的Head状态与优化后的Backbone完美同步给 EMA
        if "ema" in ckpt:
            ckpt["ema"] = copy.deepcopy(self.model).half()
        if "updates" in ckpt:
            ckpt["updates"] = None
        if "optimizer" in ckpt:
            ckpt["optimizer"] = None
            
        torch.save(ckpt, out)
        print(f"\n  💾 安全对抗微调模型及EMA已同步落盘: {out}")
        self.sequential.to(self.device)
        self.domain_classifier.to(self.device)


def main():
    parser = argparse.ArgumentParser(description="DANN UDA 隔离保护微调")
    parser.add_argument("--model_dir", type=str, default='/ssd_data/lixiang_data/YOLO_DA/runs/detect/runs/detect/YOLO26s_SAR/weights')
    parser.add_argument("--domain_A", type=str, default='/ssd_data/lixiang_data/Datasets/SAR_Aircraft_noMSAR_jpg_split/images/val')
    parser.add_argument("--domain_B", type=str, default='/ssd_data/lixiang_data/Datasets/data_sar_plane_coarsness/images')
    parser.add_argument("--output_dir", type=str, default="/ssd_data/lixiang_data/YOLO_DA/runs/detect/finetuned_uda_yolo")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    model_files = sorted(glob.glob(os.path.join(args.model_dir, "*.pt")))
    if not model_files:
        print(f"❌ 未找到权重文件")
        return

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for mp in model_files:
        try:
            tuner = UDAFineTuner(mp, device, align_layer_idx=9)
            tuner.fine_tune(args.domain_A, args.domain_B, args.output_dir,
                            args.epochs, args.imgsz, args.batch, args.lr, args.workers,
                            args.sample_ratio)
            tuner.save(args.output_dir)
        except Exception as e:
            print(f"❌ {Path(mp).name} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("🎉 任务结束!")

if __name__ == "__main__":
    main()