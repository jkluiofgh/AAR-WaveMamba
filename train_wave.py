import argparse
import csv
import os
import platform
import copy
import time
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import importlib 
# 如果在没有图形界面的服务器上运行，请保留下一行
mpl.use('agg') 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _WeightedLoss
from conf import settings
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
from utils import get_mydataloader, get_network, get_weighted_mydataloader
import pandas as pd
import torch.nn.functional as F

# 尝试导入 thop 用于计算 FLOPs
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

# ================= 全局变量 =================
best_train_recall = 0.0
best_train_target_recall = None
best_train_pred_recall = None
best_valid_recall = 0.0
best_valid_target = None
best_valid_pred = None
final_valid_target = None
final_valid_pred = None
final_train_target = None
final_train_predict = None

Epoch_Avg_Grad_Norms = []
Train_Loss = []
Train_Accuracy = []
Valid_Loss = []
Valid_Accuracy = []
recall_s = []
class_names = ['Stationary', 'Running', 'Eating', 'Trotting', 'Walking']

# ================= Loss Functions =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

class CBLoss(_WeightedLoss):
    def __init__(self, samples_per_class, beta=0.9999, reduction='mean'):
        super().__init__(reduction=reduction)
        self.samples_per_class = samples_per_class
        self.beta = beta
        samples_per_class_tensor = torch.tensor(samples_per_class, dtype=torch.float32)
        self.effective_num = 1.0 - torch.pow(torch.tensor(beta), samples_per_class_tensor)
        self.weights = (1.0 - beta) / self.effective_num
        self.weights = self.weights / torch.sum(self.weights) * len(samples_per_class)

    def forward(self, inputs, targets):
        weights = self.weights.to(inputs.device)
        cb_loss = F.cross_entropy(inputs, targets, weight=weights, reduction=self.reduction)
        return cb_loss

class ACSL(nn.Module):
    def __init__(self, margin=0.05, scale=1.0, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.reduction = reduction

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        one_hot = F.one_hot(targets, num_classes=num_classes).float()
        probs = F.softmax(logits, dim=1)
        suppression_mask = (1 - one_hot) * (probs > self.margin).float()
        suppression = (probs * suppression_mask).sum(dim=1)
        ce_loss = F.cross_entropy(self.scale * logits, targets, reduction='none')
        loss = ce_loss + self.scale * suppression
        if self.reduction == "mean": return loss.mean()
        elif self.reduction == "sum": return loss.sum()
        return loss

class AnyLoss(nn.Module):
    def __init__(self, L=73, reduction='mean', epsilon=1e-8, metric='f1'):
        super().__init__()
        self.L = L
        self.reduction = reduction
        self.epsilon = epsilon
        self.metric = metric.lower()
        
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        probs = torch.sigmoid(self.L * (inputs - 0.5))
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        TP = targets_one_hot * probs
        FP = (1 - targets_one_hot) * probs
        FN = targets_one_hot * (1 - probs)
        if self.metric == 'recall':
            recall_per_class = torch.sum(TP, dim=0) / (torch.sum(TP + FN, dim=0) + self.epsilon)
            metric_loss = 1 - torch.mean(recall_per_class)
        elif self.metric == 'f1':
            precision_per_class = torch.sum(TP, dim=0) / (torch.sum(TP + FP, dim=0) + self.epsilon)
            recall_per_class = torch.sum(TP, dim=0) / (torch.sum(TP + FN, dim=0) + self.epsilon)
            f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + self.epsilon)
            metric_loss = 1 - torch.mean(f1_per_class)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        return metric_loss
    
def get_loss_function(loss_type='ce', loss_params=None):
    if loss_params is None: loss_params = {}
    if loss_type == 'ce': return nn.CrossEntropyLoss()
    elif loss_type == 'focal': return FocalLoss(**loss_params)
    elif loss_type == 'cb': return CBLoss(**loss_params)
    elif loss_type == 'acsl': return ACSL(**loss_params)
    elif loss_type == 'any': return AnyLoss(**loss_params)
    else: raise ValueError(f"Unsupported loss type: {loss_type}")   

def save_hyperparameters(args, model_configs, save_path):
    train_hyperparams = [
        ('network', args.net), ('gpu', args.gpu), ('data_path', args.data_path),
        ('save_path', args.save_path), ('batch_size', args.b), ('',''),
        ('learning_rate', args.lr), ('weight_decay', args.weight_d), ('epochs', args.epoch),
        ('',''), ('loss_type', args.loss_type), ('focal_alpha', args.focal_alpha),
        ('focal_gamma', args.focal_gamma), ('cb_beta', args.cb_beta),
        ('acsl_margin', args.acsl_margin), ('acsl_scale', args.acsl_scale),
        ('',''), ('seed', args.seed), ('decomp_type', args.decomp_type),
    ]
    model_hyperparams = []
    if model_configs:
        for key, value in model_configs.__dict__.items():
            model_hyperparams.append((f"model_{key}", value))
    all_lines = ["hyperparameters of train_code:"]
    for k, v in train_hyperparams:
        all_lines.append("" if k == "" else f"  {k}: {v}")
    all_lines.append("")
    all_lines.append("hyperparameters of model_code:")
    for k, v in model_hyperparams:
        all_lines.append("" if k == "" else f"  {k}: {v}")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines) + "\n")

# ================= 训练与评估核心函数 =================
def train(train_loader, network, optimizer, epoch, criterion, samples_per_cls, device):
    epoch_grad_norms = []
    start = time.time()
    network.train()
    train_acc_process = []
    train_loss_process = []
    class_target_train = []
    class_predict_train = []
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs, _, _ = network(images, return_attentions=False)
        loss = criterion(outputs, labels)

        loss.backward()
        total_norm = 0.0
        for p in network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        epoch_grad_norms.append(total_norm)

        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        _, preds = outputs.max(1)
        correct_n = preds.eq(labels).sum()
        acc_iter = correct_n.float() / labels.size(0)

        train_acc_process.append(acc_iter.item())
        train_loss_process.append(loss.item())

        class_target_train.append(labels.cpu())
        class_predict_train.append(preds.cpu())

    Train_Loss.append(np.mean(train_loss_process))
    Train_Accuracy.append(np.mean(train_acc_process))
    Epoch_Avg_Grad_Norms.append(np.mean(epoch_grad_norms))

    print(f"[Epoch {epoch}] Train Loss: {Train_Loss[-1]:.4f}, "
          f"Acc: {Train_Accuracy[-1]:.4f}, GradNorm: {Epoch_Avg_Grad_Norms[-1]:.4f}, "
          f"Time: {time.time() - start:.2f}s, lr: {optimizer.param_groups[0]['lr']:.6f}")
    
    return network, torch.cat(class_target_train), torch.cat(class_predict_train)

@torch.no_grad()
def eval_training(valid_loader, network, criterion, epoch=0, log_attention=False, save_path=None, device='cpu'):
    start = time.time()
    network.eval()
    valid_loss = 0.0
    correct = 0
    class_target, class_predict = [], []
    epoch_channel_weights = []

    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, channel_weights, band_weights = network(images, return_attentions=log_attention)
        
        if channel_weights is not None:
            epoch_channel_weights.append(channel_weights.cpu())

        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        class_target.append(labels.cpu())
        class_predict.append(preds.cpu())

    avg_loss = valid_loss / len(valid_loader)
    acc = correct / len(valid_loader.dataset)
    class_target = torch.cat(class_target).numpy()
    class_predict = torch.cat(class_predict).numpy()
    recall = recall_score(class_target, class_predict, average='macro', zero_division=0) 
    
    Valid_Loss.append(avg_loss)
    Valid_Accuracy.append(acc)
    recall_s.append(recall) 
    
    print(f" Valid Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Recall: {recall:.4f}, Time: {time.time() - start:.2f}s")

    cm = confusion_matrix(class_target, class_predict)
    print("Confusion Matrix (counts):")
    print(cm)

    # === 修改：只打印归一化后的百分比 ===
    if epoch_channel_weights:
        all_weights = torch.cat(epoch_channel_weights, dim=0)
        if all_weights.dim() == 3:
            all_weights = all_weights.squeeze(-1)
            
        avg_weights = all_weights.mean(dim=0).numpy() # [3]
        
        # 归一化权重 (X+Y+Z=1)
        norm_weights = avg_weights / (avg_weights.sum() + 1e-8)
        
        # 打印格式修改：X: 33.12%, ...
        print(f"  Avg Channel Weights -> X: {norm_weights[0]:.2%}, Y: {norm_weights[1]:.2%}, Z: {norm_weights[2]:.2%}")

    network.train()
    return acc, avg_loss, recall, class_target, class_predict

# ================= 可视化与评估函数 =================

def plot_confusion_matrix_grid(
    best_train_target, best_train_pred,
    final_train_target, final_train_pred,
    best_valid_target, best_valid_pred,
    final_valid_target, final_valid_pred,
    save_path
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    datasets = [
        (best_train_target, best_train_pred, "Train Set (Best Recall Epoch)"),
        (final_train_target, final_train_pred, "Train Set (Final Epoch)"),
        (best_valid_target, best_valid_pred, "Valid Set (Best Epoch)"),
        (final_valid_target, final_valid_pred, "Valid Set (Final Epoch)")
    ]
    
    for i, (target, pred, title) in enumerate(datasets):
        ax = axes[i//2, i%2]
        if target is None or pred is None:
            continue
        cm = confusion_matrix(target, pred)
        row_sums = cm.sum(axis=1)
        cm_precision = cm.astype('float') / (row_sums[:, np.newaxis] + 1e-9)
        
        sns.heatmap(
            cm_precision, cmap="coolwarm", linecolor="white", linewidths=1,
            xticklabels=class_names, 
            yticklabels=[f"{c} (n={n})" for c,n in zip(class_names, row_sums)],
            annot=True, fmt=".2f", vmin=0.0, vmax=1.0, ax=ax
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle("Confusion Matrices Comparison (Row-wise Precision)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()

def plot_channel_importance_heatmap(network, data_loader, save_path, device, class_names):
    """
    生成 Behavior - Sensors Channel 热力图。
    统计测试集中每个类别在 X, Y, Z 通道上的平均权重。
    """
    print("正在生成通道权重热力图 (Channel Importance Heatmap)...")
    network.eval()
    
    # 存储结构：{class_index: [ [w_x, w_y, w_z], ... ]}
    class_weights = {i: [] for i in range(len(class_names))}
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            # 获取通道权重 channel_w
            _, channel_w, _ = network(images, return_attentions=True)
            
            if channel_w is not None:
                # channel_w shape: [B, 3, 1] or [B, 3] -> 统一转为 [B, 3]
                w = channel_w.squeeze(-1).cpu().numpy()
                lbls = labels.cpu().numpy()
                
                for i in range(len(lbls)):
                    label_idx = lbls[i]
                    class_weights[label_idx].append(w[i])

    # 计算平均值
    heatmap_data = []
    for i in range(len(class_names)):
        weights_list = np.array(class_weights[i])
        if len(weights_list) > 0:
            avg_w = np.mean(weights_list, axis=0)
        else:
            avg_w = np.array([0.0, 0.0, 0.0])
        heatmap_data.append(avg_w)
    
    heatmap_data = np.array(heatmap_data) # [5, 3]

    # 绘图
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".4f", 
        cmap="Reds", 
        xticklabels=['X', 'Y', 'Z'], 
        yticklabels=class_names,
        linewidths=1, 
        linecolor='white',
        cbar_kws={'label': 'Average Attention Weight'}
    )
    plt.title("Average Channel Weights per Class (DAE Module)", fontsize=14)
    plt.xlabel("Sensor Channel", fontsize=12)
    plt.ylabel("Behavior", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Channel_Importance_Heatmap.png"), dpi=300)
    plt.close()
    print("  - 热力图已保存: Channel_Importance_Heatmap.png")

def plot_attention_heatmaps(network, data_loader, save_path, device, class_names):
    """
    修改：仅生成频带热力图 (Frequency_Activation_Heatmap.png)。
    """
    print("正在生成注意力热力图 (Frequency Only)...")
    network.eval()
    all_band_weights = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            _, channel_w, band_w = network(images, return_attentions=True)
            
            # 我们这里只关心 band_w
            if band_w is not None:
                if band_w.dim() == 3: band_w = band_w.squeeze(-1)
                all_band_weights.append(band_w.cpu().numpy())
                
            all_labels.append(labels.cpu().numpy())

    if not all_band_weights:
        print("警告: 未能获取到频带注意力权重，跳过频带热力图绘制。")
        return

    bnd_weights = np.concatenate(all_band_weights, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 1. Band Heatmap (恢复原顺序，开启数字显示)
    band_cols = [f'B{i+1}' for i in range(bnd_weights.shape[1])]
    df_bnd = pd.DataFrame(bnd_weights, columns=band_cols)
    df_bnd['label'] = labels
    avg_bnd = df_bnd.groupby('label').mean()
    avg_bnd.index = [class_names[i] for i in avg_bnd.index]

    plt.figure(figsize=(10, 6))
    sns.heatmap(avg_bnd, annot=True, cmap='Reds', fmt='.2f', linewidths=1, linecolor='white')
    plt.title('Frequency Activation Heatmap\n(Wavelet Band Weights)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Frequency_Activation_Heatmap.png'), dpi=600)
    plt.close()
    print("  - 热力图已保存: Frequency_Activation_Heatmap.png")

def visualize_dae_effect(model, test_loader, device, save_path):
    """
    只保留 DAE_Signal_Analysis.png (折线图)，体现信号增强前后的对比。
    """
    print("绘制 DAE 信号增强对比图 (Line Plot)...")
    model.eval()
    images, labels = next(iter(test_loader))
    images = images.to(device)
    
    with torch.no_grad():
        x_enhanced, weights = model.channel_router(images)
    
    # 选取第一个样本进行绘制
    idx = 0
    raw_sig = images[idx].cpu().numpy()   # [3, L]
    enh_sig = x_enhanced[idx].cpu().numpy() # [3, L]
    w = weights[idx].cpu().numpy().flatten()
    lbl = class_names[labels[idx].item()]
    
    channels = ['X', 'Y', 'Z']

    # === DAE_Signal_Analysis.png (折线图) ===
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axes1[i].plot(raw_sig[i], linestyle='--', color='gray', alpha=0.6, label='Raw Input')
        axes1[i].plot(enh_sig[i], color='red', linewidth=1.5, label='Enhanced (DAE)')
        axes1[i].set_title(f"Channel {channels[i]} (Weight: {w[i]:.4f})")
        axes1[i].legend(loc='upper right')
        axes1[i].grid(True, alpha=0.3)
    
    plt.suptitle(f"DAE Signal Analysis (Class: {lbl})")
    plt.xlabel("Time Steps")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "DAE_Signal_Analysis.png"), dpi=300)
    plt.close(fig1)
    print("  - 折线图已保存: DAE_Signal_Analysis.png")

def visualize_fusion_comparison_tsne(model_ours, test_loader, device, save_path, class_names, baseline_ckpt=None):
    """
    修改后的 t-SNE：IEEE 论文格式
    (a) Baseline (Standard Mamba)
    (b) WPD w/o Fusion Strategy
    (c) Proposed Adaptive Fusion (Ours)
    """
    print("正在准备 t-SNE 对比可视化 (Paper Format)...")
    model_ours.eval()
    
    # 1. 加载 Baseline (如果存在)
    model_baseline = None
    if baseline_ckpt and os.path.exists(baseline_ckpt):
        try:
            module = importlib.import_module("models.onlymamba")
            BaseModel = getattr(module, "MambaWithDWT", None) or getattr(module, "onlymamba", None)
            BaseConfig = getattr(module, "ModelConfigs", None)
            if BaseModel and BaseConfig:
                base_cfg = BaseConfig()
                model_baseline = BaseModel(base_cfg).to(device)
                model_baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device))
                model_baseline.eval()
                print("✅ Baseline 模型加载成功")
        except Exception as e:
            print(f"⚠️ Baseline 加载失败: {e}")
            
    # 数据容器
    data = {
        "baseline": [], 
        "wpd_raw": [],   # WPD 原始拼接 (模拟无 Fusion/Router)
        "fusion_out": [], # Feature Fusion 模块输出 (Ours)
        "labels": []
    }

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            data["labels"].append(lbls.numpy())
            
            # --- 1. Baseline 特征 ---
            if model_baseline:
                try:
                    out_base, _, _ = model_baseline(imgs, return_attentions=False)
                    data["baseline"].append(out_base.cpu().numpy())
                except:
                     pass
            
            # --- 2. Ours: WPD Raw & Fusion Output ---
            if hasattr(model_ours, 'forward_for_tsne'):
                feat_raw, feat_fused, _ = model_ours.forward_for_tsne(imgs)
                data["wpd_raw"].append(feat_raw.cpu().numpy())
                data["fusion_out"].append(feat_fused.cpu().numpy())
            else:
                pass

    # 拼接数据
    for k in data:
        if len(data[k]) > 0: data[k] = np.concatenate(data[k], axis=0)

    # 采样
    if len(data["labels"]) > 2000:
        idx = np.random.choice(len(data["labels"]), 2000, replace=False)
        for k in data:
            if len(data[k]) > 0: data[k] = data[k][idx]

    # 绘图配置 - IEEE 论文标题映射
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    
    # 定义绘图顺序和对应的底部标题
    plot_items = []
    if len(data["baseline"]) > 0:
        plot_items.append(("baseline", "(a) Baseline (Standard Mamba)"))
    if len(data["wpd_raw"]) > 0:
        plot_items.append(("wpd_raw", "(b) WPD w/o Fusion Strategy"))
    if len(data["fusion_out"]) > 0:
        plot_items.append(("fusion_out", "(c) Proposed Adaptive Fusion (Ours)"))

    if not plot_items:
        print("无数据可画 t-SNE")
        return

    # 设置画布，留出底部空间给标题
    fig, axes = plt.subplots(1, len(plot_items), figsize=(5.5 * len(plot_items), 5))
    if len(plot_items) == 1: axes = [axes] 

    for i, (key, bottom_label) in enumerate(plot_items):
        ax = axes[i]
        print(f"计算 t-SNE: {key}...")
        if len(data[key]) == 0: continue
        
        embed = tsne.fit_transform(data[key])
        
        # 绘制散点
        for li, name in enumerate(class_names):
            ids = data["labels"] == li
            ax.scatter(embed[ids, 0], embed[ids, 1], label=name, alpha=0.7, s=10)
        
        # === IEEE 格式修改 ===
        # 1. 移除顶部 set_title
        # 2. 移除坐标轴刻度，但保留边框（或者 ax.axis('off') 完全移除）
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 3. 在图下方添加 (a) (b) (c) 标签
        # transform=ax.transAxes 坐标系：(0,0)左下, (1,1)右上
        # y=-0.15 将文字放到轴下方
        ax.text(0.5, -0.12, bottom_label, transform=ax.transAxes, 
                ha='center', va='top', fontsize=14, fontweight='normal', color='black')

        # 图例只在最后一张图显示，或者第一张，视论文排版而定。这里每张都画太挤，只在最后一张画
        if i == len(plot_items) - 1:
            ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

    # 移除 suptitle
    plt.tight_layout()
    # 增加底部边距以容纳 ax.text
    plt.subplots_adjust(bottom=0.15) 
    
    save_file = os.path.join(save_path, "Module_Comparison_tSNE.png")
    plt.savefig(save_file, dpi=600, bbox_inches='tight') # bbox_inches='tight' 会自动包裹底部文字
    plt.close()
    print(f"  - t-SNE 对比图已保存 (论文格式): {save_file}")


def visualize_dwt_vs_wpd_tsne(model_wpd, test_loader, device, save_path, class_names, dwt_ckpt=None):
    """
    修改后的 DWT vs WPD t-SNE 对比图 (IEEE 论文格式)。
    (a) DWT (Partial Decomposition)
    (b) WPD (Full-Band Decomposition, Ours)
    """
    print("正在准备 DWT vs WPD t-SNE 对比 (Paper Format)...")
    model_wpd.eval()
    
    # 1. 加载 DWT 模型
    model_dwt = None
    if dwt_ckpt and os.path.exists(dwt_ckpt):
        try:
            module = importlib.import_module("models.MambaWithDWT11")
            ModelClass = getattr(module, "MambaWithDWT")
            ConfigClass = getattr(module, "ModelConfigs")
            dwt_cfg = ConfigClass(decomposition_type='dwt') 
            model_dwt = ModelClass(dwt_cfg).to(device)
            model_dwt.load_state_dict(torch.load(dwt_ckpt, map_location=device))
            model_dwt.eval()
            print(f"✅ DWT 对照模型已加载: {dwt_ckpt}")
        except Exception as e:
            print(f"❌ 加载 DWT 模型失败: {e}")
            return
    else:
        print("⚠️ 未提供有效的 --dwt_ckpt，跳过 DWT vs WPD 对比图。")
        return

    data = {"dwt": [], "wpd": [], "labels": []}

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            data["labels"].append(lbls.numpy())
            
            # --- 1. DWT 特征 ---
            if hasattr(model_dwt, 'forward_for_tsne'):
                _, feat_dwt, _ = model_dwt.forward_for_tsne(imgs)
                data["dwt"].append(feat_dwt.cpu().numpy())
            
            # --- 2. WPD 特征 (Ours) ---
            if hasattr(model_wpd, 'forward_for_tsne'):
                _, feat_wpd, _ = model_wpd.forward_for_tsne(imgs)
                data["wpd"].append(feat_wpd.cpu().numpy())

    # 拼接与采样
    for k in data:
        if len(data[k]) > 0: data[k] = np.concatenate(data[k], axis=0)
    if len(data["labels"]) > 2000:
        idx = np.random.choice(len(data["labels"]), 2000, replace=False)
        for k in data:
            if len(data[k]) > 0: data[k] = data[k][idx]

    # 绘图
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # 配置列表：(Key, Bottom Label, Axis Object)
    plot_configs = [
        ("dwt", "(a) DWT (Partial Decomposition)", axes[0]),
        ("wpd", "(b) WPD (Full-Band Decomposition, Ours)", axes[1])
    ]

    for key, bottom_label, ax in plot_configs:
        print(f"计算 t-SNE: {key}...")
        if len(data[key]) == 0: continue
        
        embed = tsne.fit_transform(data[key])
        
        for li, name in enumerate(class_names):
            ids = data["labels"] == li
            ax.scatter(embed[ids, 0], embed[ids, 1], label=name, alpha=0.7, s=15)
        
        # === IEEE 格式修改 ===
        # 1. 移除顶部标题
        # 2. 移除坐标轴数值
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 3. 添加底部子标题
        ax.text(0.5, -0.12, bottom_label, transform=ax.transAxes, 
                ha='center', va='top', fontsize=14, fontweight='normal', color='black')

        # 图例只在第二张图显示，避免重复
        if key == "wpd":
            ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

    # 移除总标题 suptitle
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) # 留出底部空间
    
    save_file = os.path.join(save_path, "DWT_vs_WPD_tSNE.png")
    plt.savefig(save_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  - DWT vs WPD 对比图已保存 (论文格式): {save_file}")


def evaluate_efficiency(model, device, out_txtsavedpath):
    print("\n========= Efficiency Evaluation =========")
    model.eval()
    
    # 构建与您真实输入尺寸一致的虚拟张量 (Batch_Size=1 用于测 FLOPs)
    # 假设您的输入是 3 通道，序列长度 50
    dummy_input = torch.randn(1, 3, 50).to(device) 
    
    # 1. 计算参数量 Params (M)
    params = sum(p.numel() for p in model.parameters())
    params_m = params / 1e6  # 强制转换为 Million (M)
    print(f"🔹 Params (M): {params_m:.4f}")

    # 2. 计算 FLOPs (G)
    flops_g = 0.0
    if THOP_AVAILABLE:
        try:
            # 【核心修复点】：为 thop 创建一个深拷贝的独立模型，防止 hook 污染原模型
            model_for_flops = copy.deepcopy(model)
            model_for_flops.eval()
            
            flops, _ = profile(model_for_flops, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9  # 强制转换为 Giga (G)
            print(f"🔹 FLOPs (G): {flops_g:.4f}")
            
            # 释放深拷贝模型占用的显存，避免影响后续的极限吞吐量测试
            del model_for_flops
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"🔹 FLOPs (G): N/A (thop error: {e})")
    else:
        print("🔹 FLOPs (G): N/A (请先 pip install thop)")
    
    # 3. 计算吞吐量 Throughput (samples/s)
    # 为了测试极限吞吐量，通常使用较大的 Batch Size (如 256)
    batch_size = 256 
    dummy_batch = torch.randn(batch_size, 3, 50).to(device)
    repetitions = 100
    
    # GPU 预热 (消除初始化带来的时间误差)
    # 【注意】：由于上面的 thop 运行在副本上，这里的原 model 是干净的，不会再报错
    for _ in range(10): 
        _ = model(dummy_batch) 
    
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        with torch.no_grad():
            for _ in range(repetitions):
                _ = model(dummy_batch)
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender) / 1000.0  # 毫秒转秒
    else:
        t0 = time.time()
        with torch.no_grad():
            for _ in range(repetitions):
                _ = model(dummy_batch)
        total_time = time.time() - t0

    throughput = (batch_size * repetitions) / total_time
    print(f"🔹 Throughput (samples/s): {throughput:.0f}")
    print("=========================================\n")

    # === 将结果追加写入到 output.txt 日志中 ===
    if out_txtsavedpath:
        with open(out_txtsavedpath, 'a', encoding='utf-8') as f:
            f.write('\n========= Efficiency Evaluation =========\n')
            f.write(f'Params (M): {params_m:.4f}\n')
            f.write(f'FLOPs (G): {flops_g:.4f}\n')
            f.write(f'Throughput (samples/s): {throughput:.0f}\n')
            f.write('=========================================\n')
            
    return params_m, flops_g, throughput


# ================= 主程序 =================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='MambaWithDWT11')
    parser.add_argument('--gpu', type=int, default=0) 
    parser.add_argument('--data_path', type=str, default='./data/004goat.pt')
    parser.add_argument('--save_path', type=str, default='./checkpoint')
    parser.add_argument('--b', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--weight_d', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=200)

    parser.add_argument('--loss_type', type=str, default='cb', choices=['ce', 'focal', 'cb', 'acsl', 'any'])
    parser.add_argument('--focal_alpha', type=float, default=0.5)
    parser.add_argument('--focal_gamma', type=float, default=6.5)
    parser.add_argument('--cb_beta', type=float, default=0.999)
    parser.add_argument('--acsl_margin', type=float, default=0.5)
    parser.add_argument('--acsl_scale', type=float, default=5)
    parser.add_argument('--any_L', type=float, default=80.0)
    parser.add_argument('--seed', type=int, default=10)
    
    # 新增参数
    parser.add_argument('--decomp_type', type=str, default='wpd', choices=['wpd', 'dwt'], help='分解模式')
    parser.add_argument('--baseline_ckpt', type=str, default=None, help='Baseline (OnlyMamba) 权重路径')
    parser.add_argument('--dwt_ckpt', type=str, default=None, help='DWT 模型权重路径，用于 t-SNE 对比')

    args = parser.parse_args()
    
    # GPU Logic
    if torch.cuda.is_available():
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # ================= 核心修改：Config Loading & Model Initialization =================
    model_configs = None
    ModelClass = None # 用于存储找到的模型类

    try:
        # 1. 尝试动态加载模块
        module = importlib.import_module(f"models.{args.net}")
        
        # 2. 尝试加载 Config
        if hasattr(module, 'ModelConfigs'):
            ModelConfigs = getattr(module, 'ModelConfigs')
            try:
                # 尝试传入 decomposition_type (针对 MambaWithDWT11)
                model_configs = ModelConfigs(decomposition_type=args.decomp_type)
            except TypeError:
                # 如果模型 (如 onlymamba) 不接受该参数，使用默认初始化
                print(f"Note: Model [{args.net}] does not accept 'decomposition_type'. Using default config.")
                model_configs = ModelConfigs()
                
        elif hasattr(module, 'MambaConfigs'):
            MambaConfigs = getattr(module, 'MambaConfigs')
            model_configs = MambaConfigs()
        else:
            print(f"Warning: models.{args.net} 中未找到 ModelConfigs")
        
        # 3. 尝试获取模型类 (用于直接实例化，跳过 get_network 的默认逻辑)
        # 尝试一些常见的类名命名习惯
        possible_class_names = ['MambaWithDWT', args.net, 'onlymamba', 'Net']
        for name in possible_class_names:
            if hasattr(module, name):
                Candidate = getattr(module, name)
                # 简单的检查：是否是类且继承自 nn.Module
                if isinstance(Candidate, type) and issubclass(Candidate, nn.Module):
                    ModelClass = Candidate
                    break

    except ImportError as e:
        print(f"ImportError: {e}")
        # 硬编码兼容性处理
        if args.net == 'MambaWithDWT11':
             from models.MambaWithDWT11 import ModelConfigs, MambaWithDWT
             try:
                 model_configs = ModelConfigs(decomposition_type=args.decomp_type)
             except TypeError:
                 model_configs = ModelConfigs()
             ModelClass = MambaWithDWT

    if model_configs is None:
        class DummyConfig: pass
        model_configs = DummyConfig()

    # 4. 实例化模型
    # 优先尝试直接使用 ModelClass(model_configs)，这样能确保 decomposition_type 传进去
    if ModelClass is not None:
        print(f"Instantiating {ModelClass.__name__} directly with config...")
        try:
            net = ModelClass(model_configs).to(device)
        except Exception as e:
            print(f"Direct instantiation failed ({e}), falling back to get_network...")
            net = get_network(args).to(device)
    else:
        # 如果找不到类，回退到 get_network (可能导致 DWT 配置丢失，但在 Baseline 上没问题)
        net = get_network(args).to(device)

    print(f"Model on device: {device}")
    print(f"Decomposition Type Request: {args.decomp_type}")
    print(f"data source:{os.path.basename(args.data_path)}")

    # Dataloaders
    sysstr = platform.system()
    num_workers = 0 if sysstr == 'Windows' else 8
    train_loader, _, number_train = get_weighted_mydataloader(
        args.data_path, data_id=0, batch_size=args.b, num_workers=num_workers, shuffle=True)
    valid_loader = get_mydataloader(
        args.data_path, data_id=1, batch_size=args.b, num_workers=num_workers, shuffle=True)
    test_loader = get_mydataloader(
        args.data_path, data_id=2, batch_size=args.b, num_workers=num_workers, shuffle=True)

    # Loss
    samples_per_cls = number_train
    loss_params = {}
    if args.loss_type == 'focal':
        loss_params = {'alpha': args.focal_alpha, 'gamma': args.focal_gamma}
    elif args.loss_type == 'cb':
        loss_params = {'samples_per_class': samples_per_cls, 'beta': args.cb_beta}
    elif args.loss_type == 'acsl':
        loss_params = {'margin': args.acsl_margin, 'scale': args.acsl_scale}
    elif args.loss_type == 'any':  
        loss_params = {'L': args.any_L}
    
    criterion = get_loss_function(args.loss_type, loss_params)

    # Optimizer
    param_optimizer = list(net.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias', 'norm.weight', 'norm.bias', 'A_log', 'D'] 
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_d},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7) 
    
    beta_weight = args.cb_beta
    num_classes = len(samples_per_cls)
    effective_num = 1.0 - np.power(beta_weight, samples_per_cls)
    weights = (1.0 - beta_weight) / (effective_num + 1e-8)
    weights = weights / np.sum(weights) * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.save_path)
    os.makedirs(checkpoint_path, exist_ok=True)
    best_weights_path = os.path.join(checkpoint_path, f"{args.net}-best.pth")

    # Training Vars
    best_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    min_delta_loss = 1e-4        
    patience_loss = 25
    wait_loss = 0      
    warmup_epochs = 3

    for epoch in range(1, args.epoch + 1):
        if epoch <= warmup_epochs:
            warmup_lr = args.lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"[Warmup] Epoch {epoch}/{warmup_epochs}, LR set to {warmup_lr:.8f}")

        net, this_train_target, this_train_predict = train(
            train_loader, net, optimizer, epoch, loss_function, samples_per_cls, device)
        
        val_acc, val_loss, val_recall, val_class_target, val_class_predict = eval_training(
            valid_loader, net, loss_function, epoch, log_attention=True, save_path=checkpoint_path, device=device
        )

        if epoch > warmup_epochs:
            scheduler.step(val_loss)

        train_recall = recall_score(this_train_target.cpu().numpy(), this_train_predict.cpu().numpy(), average='macro')
        if train_recall > best_train_recall:
            best_train_recall = train_recall
            best_train_target_recall = this_train_target.cpu().numpy()
            best_train_pred_recall = this_train_predict.cpu().numpy()

        if val_recall > best_valid_recall:
            best_valid_recall = val_recall
            best_valid_target = val_class_target
            best_valid_pred = val_class_predict
 
        final_valid_target = val_class_target
        final_valid_pred = val_class_predict

        if val_loss < best_loss - min_delta_loss:
            best_loss = val_loss
            wait_loss = 0
            torch.save(net.state_dict(), best_weights_path)
            print(f" 验证 Loss 降低至 {val_loss:.4f}，已保存模型权重。")
        else:
            wait_loss += 1
            print(f" 验证 Loss 未改善 ({wait_loss}/{patience_loss})。")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(net.state_dict(), os.path.join(checkpoint_path, f"{args.net}-best-acc.pth"))
            print(f" 验证准确率提升至 {val_acc:.4f}，已保存最佳准确率模型。")

        final_train_target = this_train_target.cpu().numpy()
        final_train_predict = this_train_predict.cpu().numpy()

        if wait_loss >= patience_loss:
            print(f"Early Stopping: 连续 {patience_loss} 个 epoch 验证 Loss 未显著提升，训练终止。")
            break
            
    # Save Results
    with open(os.path.join(checkpoint_path, 'performance.json'), 'w') as f:
        json.dump({
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'final_val_acc': Valid_Accuracy[-1] if Valid_Accuracy else 0,
            'final_train_acc': Train_Accuracy[-1] if Train_Accuracy else 0,
            'best_val_loss': best_loss,
            'final_val_loss': Valid_Loss[-1] if Valid_Loss else 0,
            'final_train_loss': Train_Loss[-1] if Train_Loss else 0
        }, f, indent=4)

    save_hyperparameters(args, model_configs, os.path.join(checkpoint_path, 'hyperparameters.yaml'))
    print(f"Training finished. Best validation Loss: {best_loss:.4f}")

    out_txtsavedpath = os.path.join(checkpoint_path, 'output.txt')
    with open(out_txtsavedpath, 'w', encoding='utf-8') as f:
        print('Setting: Seed: {}, Epoch: {}, Batch size: {}, Learning rate: {:.6f}, Weight decay: {}, gpu: {}, Data path: {}, Saved path: {}'.format(
            args.seed, len(Train_Accuracy), args.b, args.lr, args.weight_d, args.gpu, args.data_path, args.save_path), file=f)
        print('--------------------------------------------------', file=f)
        total_num_paras = sum(p.numel() for p in net.parameters())
        trainable_num_paras = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('The total number of network parameters = {}'.format(total_num_paras), file=f)
        print('The trainable number of network parameters = {}'.format(trainable_num_paras), file=f)

    # Test
    # ================= 核心修复：Test 阶段也使用正确的 ModelClass 实例化 =================
    if ModelClass is not None:
        best_net = ModelClass(model_configs).to(device)
    else:
        best_net = get_network(args).to(device)
        
    best_net.load_state_dict(torch.load(best_weights_path, map_location=device))
    best_net.eval()
    # ===================================================================================

    test_targets, test_preds = [], []
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out, _, _ = best_net(imgs, return_attentions=False)
        preds = out.argmax(dim=1)
        test_targets.extend(lbls.cpu().numpy().tolist())
        test_preds.extend(preds.cpu().numpy().tolist())

    test_acc = np.mean(np.array(test_preds) == np.array(test_targets))
    test_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0)
    test_precision = precision_score(test_targets, test_preds, average='macro', zero_division=0)
    test_recall = recall_score(test_targets, test_preds, average='macro', zero_division=0)

    with open(out_txtsavedpath, 'a', encoding='utf-8') as f:
        print('Testing network......', file=f)
        print('Test set: Accuracy: {:.6f}'.format(test_acc), file=f)
        print('f1 score = {:.6f}'.format(test_f1), file=f)
        print('precision = {:.6f}'.format(test_precision), file=f)
        print('recall = {:.6f}'.format(test_recall), file=f)
        print('Classification Report', file=f)
        print(classification_report(test_targets, test_preds, target_names=class_names, zero_division=0), file=f)

    torch.save(test_targets, os.path.join(checkpoint_path, 'test_target.pt')) 
    torch.save(test_preds, os.path.join(checkpoint_path, 'test_predict.pt')) 

    # Visualizations
    idx = list(range(1, len(Train_Accuracy) + 1))
    plt.figure(figsize=(12, 9))
    plt.title('Accuracy Curve')
    plt.plot(idx, Train_Accuracy, label='Train Acc')
    plt.plot(idx, Valid_Accuracy, label='Valid Acc')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(checkpoint_path, 'Accuracy_curve.png'))
    plt.close()

    plt.figure(figsize=(12, 9))
    plt.title('Loss Curve')
    plt.plot(idx, Train_Loss, label='Train Loss')
    plt.plot(idx, Valid_Loss, label='Valid Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(checkpoint_path, 'Loss_curve.png'))
    plt.close()
    
    plt.figure(figsize=(12, 9))
    plt.title('Recall Curve')
    plt.plot(idx, recall_s, label='Valid Recall')              
    plt.grid(True)
    plt.legend() 
    plt.savefig(os.path.join(checkpoint_path, 'Recall-score.png')) 
    plt.close()

    plot_confusion_matrix_grid(
        best_train_target_recall, best_train_pred_recall,
        final_train_target, final_train_predict,
        best_valid_target, best_valid_pred,
        final_valid_target, final_valid_pred,
        os.path.join(checkpoint_path, "Confusion_Matrix_Grid.png")
    )

    params_m, flops_g, throughput = evaluate_efficiency(best_net, device, out_txtsavedpath)    # === 修改后的可视化流程 ===
    if hasattr(best_net, 'channel_router'):
        # 1. 绘制 DAE 折线图 (保留)
        visualize_dae_effect(best_net, test_loader, device, checkpoint_path)
        
        # 2. 新增：绘制通道权重热力图 (Behavior-Sensors Heatmap)
        plot_channel_importance_heatmap(best_net, test_loader, checkpoint_path, device, class_names)
        
        # 3. 绘制频带热力图 (保留)
        plot_attention_heatmaps(best_net, test_loader, checkpoint_path, device, class_names)
    
    # 4. 特征融合模块对比 (Baseline vs WPD raw vs Fusion)
    if hasattr(best_net, 'channel_router'):
        visualize_fusion_comparison_tsne(
            best_net, test_loader, device, checkpoint_path, class_names, baseline_ckpt=args.baseline_ckpt
        )

    # 5. 分解策略对比 (DWT vs WPD)
    if args.decomp_type == 'wpd' and args.dwt_ckpt:
        visualize_dwt_vs_wpd_tsne(
            best_net, test_loader, device, checkpoint_path, class_names, dwt_ckpt=args.dwt_ckpt
        )
    
    print(f"✅ 所有结果已保存至 {checkpoint_path}")