"""
Deep CORAL 域对齐微调 (带详尽日志监控与评估)
==========================================================
"""

import os
import sys
import argparse
import glob
import math
import copy
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "yolo_source"))

class DomainImageDataset(Dataset):
    def __init__(self, image_dir, imgsz=1024):
        self.imgsz = imgsz
        self.image_paths = []
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            for p in Path(image_dir).rglob(f"*{ext}"):
                self.image_paths.append(str(p))
            for p in Path(image_dir).rglob(f"*{ext.upper()}"):
                self.image_paths.append(str(p))
        self.image_paths = sorted(set(self.image_paths))
        if not self.image_paths:
            raise ValueError(f"{image_dir} 中未找到图像")
        print(f"  域数据集加载: {len(self.image_paths)} 张图像 from {image_dir}")

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

def coral_loss(features_a, features_b):
    n_a = features_a.size(0)
    n_b = features_b.size(0)

    # 中心化
    f_a = features_a - features_a.mean(0, keepdim=True)
    f_b = features_b - features_b.mean(0, keepdim=True)

    # 协方差
    cov_a = (f_a.T @ f_a) / (n_a - 1)
    cov_b = (f_b.T @ f_b) / (n_b - 1)

    # Frobenius 范数平方
    diff = cov_a - cov_b
    loss = (diff * diff).sum()

    return loss

class CORALFineTuner:
    def __init__(self, checkpoint_path, device="cuda:0", backbone_last_idx=4):
        self.device = torch.device(device)
        self.backbone_last_idx = backbone_last_idx
        self.checkpoint_path = checkpoint_path
        self.model_name = Path(checkpoint_path).stem
        self.sequential = None
        self.model = None
        self._init_params = {} 

        self._load()
        self._unfreeze_all()
        self._save_init_params()

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

        print(f"  参数量: {sum(p.numel() for p in self.sequential.parameters())/1e6:.1f}M, "
              f"层数: {len(self.sequential)}")

    def _unfreeze_all(self):
        for p in self.sequential.parameters():
            p.requires_grad = True

    def _save_init_params(self):
        for name, p in self.sequential.named_parameters():
            if p.requires_grad:
                self._init_params[name] = p.detach().clone()

    def _forward_features(self, imgs):
        x = imgs
        y = []
        for i, m in enumerate(self.sequential):
            f_from = getattr(m, "f", -1)
            if f_from != -1:
                x = y[f_from] if isinstance(f_from, int) else [x if j == -1 else y[j] for j in f_from]
            x = m(x)
            if i == 22:  # neck 最后一层 C3k2
                return x
            y.append(x)
        return x

    def _get_layer_weight_stats(self):
        """获取关键层的权重绝对值之和，用于监控参数更新情况"""
        stats = {}
        with torch.no_grad():
            # 监控全局
            stats["Total"] = sum(p.abs().sum().item() for p in self.sequential.parameters() if p.requires_grad)
            # 监控层 9 (Backbone SPPF)
            if len(self.sequential) > 9:
                stats["Layer9_SPPF"] = sum(p.abs().sum().item() for p in self.sequential[9].parameters())
            # 监控层 22 (Neck end)
            if len(self.sequential) > 22:
                stats["Layer22_Neck"] = sum(p.abs().sum().item() for p in self.sequential[22].parameters())
        return stats

    @torch.no_grad()
    def evaluate_domain_gap(self, loader_a, loader_b, num_batches=10):
        """独立评估两个域之间的特征分布差距"""
        self.sequential.to(self.device)
        self.sequential.eval()
        
        print("\n" + "-"*50)
        print("📊 正在执行域特征分布量化评估...")
        
        coral_losses = []
        mean_gaps = []
        fa_stats, fb_stats = [], []

        for i, (imgs_a, imgs_b) in enumerate(zip(loader_a, loader_b)):
            if i >= num_batches: break
            
            fa = self._forward_features(imgs_a.to(self.device)).mean(dim=[2, 3])
            fb = self._forward_features(imgs_b.to(self.device)).mean(dim=[2, 3])
            
            coral_l = coral_loss(fa, fb).item()
            mean_gap = F.l1_loss(fa.mean(0), fb.mean(0)).item()
            
            coral_losses.append(coral_l)
            mean_gaps.append(mean_gap)
            
            fa_stats.append((fa.mean().item(), fa.std().item()))
            fb_stats.append((fb.mean().item(), fb.std().item()))

        avg_coral = np.mean(coral_losses)
        avg_mean_gap = np.mean(mean_gaps)
        fa_mean_avg = np.mean([x[0] for x in fa_stats])
        fa_std_avg = np.mean([x[1] for x in fa_stats])
        fb_mean_avg = np.mean([x[0] for x in fb_stats])
        fb_std_avg = np.mean([x[1] for x in fb_stats])

        print(f"   ➤ 协方差距离 (CORAL Loss): {avg_coral:.4f}")
        print(f"   ➤ 特征中心距 (L1 Mean Gap): {avg_mean_gap:.4f}")
        print(f"   ➤ 域A特征统计: Mean={fa_mean_avg:.4f}, Std={fa_std_avg:.4f}")
        print(f"   ➤ 域B特征统计: Mean={fb_mean_avg:.4f}, Std={fb_std_avg:.4f}")
        print("-" * 50 + "\n")
        
        self.sequential.train()
        return avg_coral

    def _train_epoch(self, loader_a, loader_b, optimizer, coral_weight, l2_weight):
        self.sequential.to(self.device)
        self.sequential.train()
        
        loss_sum = coral_sum = l2_sum = 0.0
        fa_mean_sum, fb_mean_sum = 0.0, 0.0
        n = 0
        
        pbar = tqdm(zip(loader_a, loader_b), desc=f"  Training", leave=False, total=min(len(loader_a), len(loader_b)))
        for imgs_a, imgs_b in pbar:
            imgs_a = imgs_a.to(self.device)
            imgs_b = imgs_b.to(self.device)

            fa = self._forward_features(imgs_a).mean(dim=[2, 3])
            fb = self._forward_features(imgs_b).mean(dim=[2, 3])

            l_coral = coral_loss(fa, fb)

            l_l2 = 0.0
            if l2_weight > 0:
                for name, p in self.sequential.named_parameters():
                    if p.requires_grad and name in self._init_params:
                        l_l2 = l_l2 + ((p - self._init_params[name].to(p.device)) ** 2).sum()
                l_l2 = l_l2 / len(self._init_params)

            loss = coral_weight * l_coral + l2_weight * l_l2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sequential.parameters(), max_norm=10.0)
            optimizer.step()

            loss_sum += loss.item()
            coral_sum += l_coral.item()
            if isinstance(l_l2, torch.Tensor):
                l2_sum += l_l2.item()
                
            fa_mean_sum += fa.mean().item()
            fb_mean_sum += fb.mean().item()
            n += 1
            
            pbar.set_postfix(coral=f"{l_coral.item():.2f}", fa_u=f"{fa.mean().item():.2f}", fb_u=f"{fb.mean().item():.2f}")

        return loss_sum / n, coral_sum / n, l2_sum / n, fa_mean_sum / n, fb_mean_sum / n

    def fine_tune(self, domain_a, domain_b, output_dir,
                  epochs=30, imgsz=1024, batch=8, lr=5e-4, workers=4,
                  coral_weight=1.0, l2_weight=0.01):
        print(f"\n{'='*60}")
        print(f"🚀 启动 Deep CORAL 域对齐微调")
        print(f"{'='*60}")
        print(f"  源域 (A): {domain_a}")
        print(f"  目标域(B): {domain_b}")
        print(f"  超参数: epochs={epochs}, batch={batch}, lr={lr}")

        ds_a = DomainImageDataset(domain_a, imgsz)
        ds_b = DomainImageDataset(domain_b, imgsz)
        dl_a = DataLoader(ds_a, batch, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
        dl_b = DataLoader(ds_b, batch, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)

        # 【评估 1】微调前的域差距
        print("\n>>> 微调前基线评估 <<<")
        self.evaluate_domain_gap(dl_a, dl_b)

        params = [p for p in self.sequential.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
        
        initial_weights = self._get_layer_weight_stats()

        best_loss = float("inf")
        print("\n>>> 开始迭代训练 <<<")
        for ep in range(1, epochs + 1):
            t = ep / epochs
            cos_lr = lr * 0.01 + 0.5 * (lr - lr * 0.01) * (1.0 + math.cos(math.pi * t))
            for pg in optim.param_groups:
                pg["lr"] = cos_lr

            avg_loss, avg_coral, avg_l2, fa_u, fb_u = self._train_epoch(dl_a, dl_b, optim, coral_weight, l2_weight)

            # 计算权重位移
            current_weights = self._get_layer_weight_stats()
            diff_total = abs(current_weights["Total"] - initial_weights["Total"])
            diff_L9 = abs(current_weights.get("Layer9_SPPF", 0) - initial_weights.get("Layer9_SPPF", 0))
            diff_L22 = abs(current_weights.get("Layer22_Neck", 0) - initial_weights.get("Layer22_Neck", 0))

            print(f"  Ep {ep:02d}/{epochs} | lr: {cos_lr:.2e} | "
                  f"Loss: {avg_loss:.4f} (coral:{avg_coral:.4f}, l2:{avg_l2:.4f}) | "
                  f"Feat Mean(A/B): {fa_u:.2f}/{fb_u:.2f}")
            print(f"          ↳ Δ权重: 全局={diff_total:.2f}, L9(SPPF)={diff_L9:.2f}, L22(Neck)={diff_L22:.2f}")

            if avg_loss < best_loss:
                best_loss = avg_loss

        # 【评估 2】微调后的域差距
        print("\n>>> 微调后最终评估 <<<")
        self.evaluate_domain_gap(dl_a, dl_b)

        print(f"  ✅ 训练完成, best_loss={best_loss:.4f}")
        return best_loss

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.sequential.to("cpu")
        out = os.path.join(output_dir, f"{self.model_name}_finetuned.pt")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        
        # 覆写主模型
        ckpt["model"] = copy.deepcopy(self.model).half()
        
        # 【关键修复】：强制同步 EMA 模型权重，清除优化器残余
        if "ema" in ckpt:
            ckpt["ema"] = copy.deepcopy(self.model).half()
        if "updates" in ckpt:
            ckpt["updates"] = None
        if "optimizer" in ckpt:
            ckpt["optimizer"] = None
            
        torch.save(ckpt, out)
        print(f"\n  💾 模型及EMA已同步保存至: {out}")
        self.sequential.to(self.device)

def main():
    parser = argparse.ArgumentParser(description="Deep CORAL 域对齐微调 (带监控)")
    parser.add_argument("--model_dir", type=str, default='/ssd_data/lixiang_data/YOLO_DA/runs/detect/YOLO26s_DA-6/weights')
    parser.add_argument("--domain_A", type=str, default='/ssd_data/lixiang_data/Datasets/VIS/all/images')
    parser.add_argument("--domain_B", type=str, default='/ssd_data/lixiang_data/Datasets/SAR_Aircraft_noMSAR_jpg_split/images/val')
    parser.add_argument("--output_dir", type=str,
                        default="/ssd_data/lixiang_data/YOLO_DA/runs/detect/finetuned_coral")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--coral_weight", type=float, default=1.0)
    parser.add_argument("--l2_weight", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    model_files = sorted(glob.glob(os.path.join(args.model_dir, "*.pt")))
    if not model_files:
        print(f"❌ 未找到 .pt 文件: {args.model_dir}")
        return

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for mp in model_files:
        try:
            tuner = CORALFineTuner(mp, device)
            tuner.fine_tune(args.domain_A, args.domain_B, args.output_dir,
                            args.epochs, args.imgsz, args.batch, args.lr, args.workers,
                            args.coral_weight, args.l2_weight)
            tuner.save(args.output_dir)
        except Exception as e:
            print(f"❌ {Path(mp).name} 处理失败: {e}")
            import traceback
            traceback.print_exc()

    print("🎉 批量微调任务结束!")

if __name__ == "__main__":
    main()