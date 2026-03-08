# 文件名: simple_dwt_classifier.py
# 描述: 一个仅包含小波分解和MLP分类器的简化模型

import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward

# MambaConfigs 和 DWTBranch 保持不变，因为我们仍然需要它们
class MambaConfigs:
    def __init__(self):
        self.batch_size = 1
        self.channel_size = 3
        self.seq_len = 50
        self.num_classes = 5  # Standing, Running, Grazing, Trotting, Walking

        self.wavelet = "sym2"
        self.dwt_level = 3 # 保持3层分解，得到 2^3 = 8 个频带

class DWTBranch(nn.Module):
    """
    小波包分解 (只返回最后一层的叶子节点) - 这部分完全保持不变
    """
    def __init__(self, cfg):
        super().__init__()
        self.wavelet = cfg.wavelet
        self.level = cfg.dwt_level
        self.dwt = DWT1DForward(J=1, wave=self.wavelet, mode="zero")

    def forward(self, x):
        nodes_to_process = [(x, "")]
        for _ in range(self.level):
            next_nodes = []
            for sig, path in nodes_to_process:
                yl, yh = self.dwt(sig)
                yh = yh[0]
                next_nodes.append((yl, path + "a"))
                next_nodes.append((yh, path + "d"))
            nodes_to_process = next_nodes
        
        final_bands_with_paths = sorted(nodes_to_process, key=lambda item: item[1])
        final_bands = [band[0] for band in final_bands_with_paths]
        return final_bands

# ==============================================================================
# 核心修改：新的、简化的模型
# ==============================================================================
class SimpleDWTClassifier(nn.Module):
    """
    一个仅由 DWTBranch 和 Final Classifier 组成的极简模型。
    """
    def __init__(self, config: MambaConfigs):
        super().__init__()
        self.config = config
        
        # 模块1: 小波包分解
        self.dwt = DWTBranch(config)
        
        # 为了确定最终分类器的输入维度，我们需要动态计算
        # DWT分解后所有频带拼接在一起的总特征数。
        with torch.no_grad():
            # 创建一个假的输入张量来模拟真实数据
            dummy_input = torch.randn(1, config.channel_size, config.seq_len)
            # 通过DWT模块得到输出的频带列表
            dummy_bands = self.dwt(dummy_input)
            
            # 将所有频带展平并计算总特征维度
            # b.numel() 计算每个频带张量中的元素总数
            total_features = sum(b.numel() for b in dummy_bands)
            
        # 模块2: 最终分类器 (MLP)
        # 它的输入维度就是上面计算出的 total_features
        self.final_fc = nn.Linear(total_features, config.num_classes)

    def forward(self, x):
        """
        模型的前向传播
        """
        # 步骤1: 对输入信号进行小波包分解
        # 输入 x: [B, C, T] -> [Batch, 3, 50]
        # 输出 bands: 一个包含8个张量的列表, 每个张量形状类似 [B, 3, 7]
        bands = self.dwt(x)

        # 步骤2: 将所有频带展平并拼接
        # 我们使用列表推导式将每个频带张量从 [B, 3, 7] 展平为 [B, 21]
        # 注意：start_dim=1 表示保持 batch 维度不变
        flattened_bands = [b.flatten(start_dim=1) for b in bands]
        
        # 使用 torch.cat 将列表中的所有 [B, 21] 张量沿特征维度(dim=1)拼接
        # 最终得到一个大的特征向量，形状为 [B, 8 * 21] = [B, 168]
        combined_features = torch.cat(flattened_bands, dim=1)

        # 步骤3: 通过最终的全连接层得到分类结果
        # 输入: [B, 168] -> 输出: [B, 5]
        out = self.final_fc(combined_features)
        
        return out, None, None


# --- 模型测试 ---
if __name__ == '__main__':
    # 初始化配置
    cfg = MambaConfigs()
    # 实例化新模型
    model = SimpleDWTClassifier(cfg)
    print("模型初始化成功!")
    
    # 计算并打印模型总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总可训练参数: {total_params:,}") # 参数量会比之前少非常多
    
    # 创建一个模拟输入
    x = torch.randn(cfg.batch_size, cfg.channel_size, cfg.seq_len)
    
    # 前向传播测试
    out = model(x)
    
    print("\n--- 前向传播测试 ---")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")