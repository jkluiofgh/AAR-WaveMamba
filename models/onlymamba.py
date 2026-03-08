import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# ================== Mamba CPU/GPU 兼容补丁 ==================
try:
    from mamba_ssm import selective_scan_fn
except ImportError:
    selective_scan_fn = None

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    u, delta = u.float(), delta.float()
    if delta_bias is not None: delta = delta + delta_bias[..., None].float()
    if delta_softplus: delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    
    deltaA = torch.exp(torch.einsum('bdl,dn->bdnl', delta, A))
    if is_variable_B: deltaB_u = torch.einsum('bdl,bnl,bdl->bdnl', delta, B, u)
    else: deltaB_u = torch.einsum('bdl,dn,bdl->bdnl', delta, B, u)
    
    x = torch.zeros((batch, dim, dstate), device=u.device)
    ys = []
    L = u.shape[-1]
    for i in range(L):
        x = deltaA[:, :, :, i] * x + deltaB_u[:, :, :, i]
        if is_variable_C: y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        else: y = torch.einsum('bdn,dn->bd', x, C)
        ys.append(y)
    
    y = torch.stack(ys, dim=2)
    if D is not None: y = y + u * D[..., None]
    if z is not None: y = y * F.silu(z)
    return y if not return_last_state else (y, x)

def selective_scan_fn_compat(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    if u.is_cuda and selective_scan_fn is not None:
        return selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
    else:
        return selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)

# ================= 配置类 =================
class ModelConfigs:
    def __init__(self):
        self.batch_size = 64
        self.channel_size = 3
        self.seq_len = 50
        self.num_classes = 5
        self.d_inner = 16
        self.dt_rank = 4
        self.d_state = 4
        self.d_conv = 3
        self.dropout_rate = 0.5

# ================= Mamba Core (修复: 返回完整序列) =================
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
        
        y = selective_scan_fn_compat(x_conv, dt, -torch.exp(self.A_log), B_ssm, C_ssm, self.D, rearrange(z, 'b l d -> b d l'))
        
        y = rearrange(y, 'b d l -> b l d')
        y = self.norm1(y + x_res)
        y_proj = self.out_proj(y)
        y = self.norm2(y + self.dropout(y_proj))
        
        # 【修改点】这里不再 mean(dim=1)，而是返回完整序列 [B, L, D]
        # 这样 Hook 就能抓到时间维度的数据，画出热力图
        return y 

# ================= 对照组模型: MambaWithDWT =================
class MambaWithDWT(nn.Module):
    def __init__(self, cfg: ModelConfigs = None):
        super().__init__()
        self.cfg = cfg or ModelConfigs()
        
        # 定义别名，方便 train_wave2.py 识别
        self.mamba_backbone = MambaCore(self.cfg)
        self.mamba_encoder = self.mamba_backbone 
        
        self.classifier_head = nn.Sequential(
            nn.Linear(self.cfg.d_inner, 128),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout_rate),
            nn.Linear(128, self.cfg.num_classes)
        )

    def forward(self, x, return_attentions=True):
        # 1. 经过 Mamba 核心，得到 [B, L, D]
        feat_seq = self.mamba_backbone(x)
        
        # 2. 【修改点】在这里进行池化，而不是在 MambaCore 内部
        feat = feat_seq.mean(dim=1)
        
        # 3. 分类
        out = self.classifier_head(feat)
        
        return out, None, None