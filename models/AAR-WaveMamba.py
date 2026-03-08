import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm import selective_scan_fn
from pytorch_wavelets import DWT1DForward

# ================= 配置类 =================
class ModelConfigs:
    def __init__(self, decomposition_type='wpd'):
        self.batch_size = 64
        self.channel_size = 3
        self.seq_len = 50
        self.num_classes = 5

        # DWT / WPD 配置
        self.wavelet = 'sym2'
        self.dwt_level = 3
        
        # 核心切换开关: 'wpd' (小波包) 或 'dwt' (离散小波)
        self.decomposition_type = decomposition_type 

        # Mamba 配置
        self.d_inner = 16
        self.dt_rank = 8
        self.d_state = 4
        self.d_conv = 3
        self.dropout_rate = 0.5
        self.router_ratio = 0.5 

# ================= (b) DAE (Channel Router) =================
class DimensionAwareEnhancement(nn.Module):
    def __init__(self, input_dim, ratio=0.5):
        super().__init__()
        hidden_dim = max(int(input_dim * ratio), 2)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )
    def forward(self, x):
        x_pool = x.mean(dim=-1) 
        weights = self.mlp(x_pool).unsqueeze(-1)
        return x + x * weights, weights

# ================= (c) Band Router =================
class FeatRoutingBlock(nn.Module):
    def __init__(self, num_bands, ratio=0.5):
        super().__init__()
        self.num_bands = num_bands 
        hidden_dim = max(int(num_bands * ratio), 4)
        self.mlp = nn.Sequential(
            nn.Linear(num_bands, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_bands), nn.Sigmoid()
        )
    def forward(self, feats_list):
        band_energies = torch.cat([f.abs().mean(dim=(1, 2)).unsqueeze(1) for f in feats_list], dim=1)
        alphas = self.mlp(band_energies)
        weighted_feats = []
        for i, feat in enumerate(feats_list):
            weighted_feats.append(feat + feat * alphas[:, i:i+1].unsqueeze(-1))
        return weighted_feats, alphas

# ================= Extractor (支持 WPD 和 DWT) =================
class WaveletExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dwt = DWT1DForward(J=1, wave=cfg.wavelet, mode='symmetric')
        self.target_level = cfg.dwt_level
        self.mode = cfg.decomposition_type # 'wpd' or 'dwt'

    def get_nodes(self, x, current_level):
        """
        递归获取节点。
        WPD: 返回全树节点 (Level 0 到 Level 3 所有节点)。
        DWT: 仅递归 Low，保留 High (金字塔结构)。
        """
        
        # --- WPD 模式: 全树分解 (Full Tree) ---
        if self.mode == 'wpd':
            nodes = [x]
            if current_level < self.target_level:
                yl, yh = self.dwt(x)
                yl, yh = yl, yh[0] 
                nodes += self.get_nodes(yl, current_level + 1)
                nodes += self.get_nodes(yh, current_level + 1)
            return nodes

        # --- DWT 模式: 金字塔分解 (Pyramid) ---
        elif self.mode == 'dwt':
            if current_level >= self.target_level:
                return [x]
            yl, yh = self.dwt(x)
            yl, yh = yl, yh[0] 
            nodes = self.get_nodes(yl, current_level + 1)
            nodes.append(yh) 
            return nodes

    def forward(self, x):
        return self.get_nodes(x, 0)

# ================= Mamba Core (不变) =================
class MambaCore(nn.Module):
    def __init__(self, cfg: ModelConfigs):
        super().__init__()
        self.cfg = cfg
        d_inner = cfg.d_inner
        self.d_state = cfg.d_state
        self.in_proj = nn.Linear(cfg.channel_size, 2 * d_inner)
        self.depth_conv = nn.Conv1d(d_inner, d_inner, kernel_size=cfg.d_conv, padding=cfg.d_conv-1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, cfg.dt_rank + 2 * cfg.d_state, bias=False)
        self.dt_proj = nn.Linear(cfg.dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_inner)
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True 
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True
        self.norm1 = nn.LayerNorm(d_inner)
        self.norm2 = nn.LayerNorm(d_inner)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, x):
        B, C, L = x.shape
        h = rearrange(x, 'b c l -> b l c')
        xz = self.in_proj(h)
        x_res, z = xz.chunk(2, dim=-1)
        x_ssm = rearrange(x_res, 'b l d -> b d l')
        x_conv = self.depth_conv(x_ssm)[..., :L]
        x_conv = F.silu(x_conv)
        flat = rearrange(x_conv, 'b d l -> (b l) d')
        dt_bc = self.x_proj(flat)
        dt_raw, B_raw, C_raw = dt_bc.split([self.cfg.dt_rank, self.cfg.d_state, self.cfg.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        dt = rearrange(dt, '(b l) d -> b d l', b=B, l=L)
        B_ssm = rearrange(B_raw, '(b l) d -> b d l', b=B, l=L)
        C_ssm = rearrange(C_raw, '(b l) d -> b d l', b=B, l=L)
        y = selective_scan_fn(x_conv, dt, -torch.exp(self.A_log), B_ssm, C_ssm, self.D, rearrange(z, 'b l d -> b d l'))
        y = rearrange(y, 'b d l -> b l d')
        y = self.norm1(y + x_res)
        y_proj = self.out_proj(y)
        y = self.norm2(y + self.dropout(y_proj))
        return y 

# ================= Main Pipeline =================
class MambaWithDWT(nn.Module):
    def __init__(self, cfg: ModelConfigs = None):
        super().__init__()
        self.cfg = cfg or ModelConfigs()
        
        self.channel_router = DimensionAwareEnhancement(self.cfg.channel_size, self.cfg.router_ratio)
        self.extractor = WaveletExtractor(self.cfg)
        
        if self.cfg.decomposition_type == 'wpd':
            self.num_bands = (2 ** (self.cfg.dwt_level + 1)) - 1
        else: 
            self.num_bands = self.cfg.dwt_level + 1
        
        print(f"Build Model with Mode: {self.cfg.decomposition_type.upper()}, Num Bands: {self.num_bands}")

        self.mamba_backbone = nn.ModuleList([MambaCore(self.cfg) for _ in range(self.num_bands)])
        self.mamba_encoder = self.mamba_backbone[0] # Alias
        self.band_router = FeatRoutingBlock(self.num_bands, self.cfg.router_ratio)
        
        self.classifier_input_dim = self.cfg.d_inner * self.num_bands
        self.classifier_head = nn.Sequential(
            nn.Linear(self.classifier_input_dim, 128), nn.ReLU(),
            nn.Dropout(self.cfg.dropout_rate), nn.Linear(128, self.cfg.num_classes)
        )

    def _align_and_fuse(self, feat_list):
        target_len = self.cfg.seq_len
        upsampled_feats = []
        for feat in feat_list:
            feat_p = feat.permute(0, 2, 1) 
            if feat_p.shape[-1] != target_len:
                feat_up = F.interpolate(feat_p, size=target_len, mode='linear', align_corners=True)
            else: feat_up = feat_p
            upsampled_feats.append(feat_up.permute(0, 2, 1))
        fused_feat = torch.cat(upsampled_feats, dim=-1)
        return fused_feat.mean(dim=1)

    def forward(self, x, return_attentions=True):
        x_hat, channel_w = self.channel_router(x)
        if not return_attentions: channel_w = None
            
        bands = self.extractor(x_hat)
        mamba_feats = [m(b) for m, b in zip(self.mamba_backbone, bands)]
        weighted_feats, band_w = self.band_router(mamba_feats)
        if not return_attentions: band_w = None
            
        fused_feat_pooled = self._align_and_fuse(weighted_feats)
        out = self.classifier_head(fused_feat_pooled)
        return out, channel_w, band_w

    def forward_for_tsne(self, x):
        """ t-SNE 专用接口：返回 Raw, Fused, Logits """
        x_hat, _ = self.channel_router(x)
        bands = self.extractor(x_hat)
        mamba_feats = [m(b) for m, b in zip(self.mamba_backbone, bands)]
        
        # 1. Raw Features (WPD Only, No Router Weighting)
        raw_feat_pooled = self._align_and_fuse(mamba_feats)

        # 2. Fused Features (WPD + Router Weighting)
        weighted_feats, _ = self.band_router(mamba_feats)
        fused_feat_pooled = self._align_and_fuse(weighted_feats)
        
        # 3. Logits
        out = self.classifier_head(fused_feat_pooled)
        
        return raw_feat_pooled, fused_feat_pooled, out