from .common_utils_mbyolo import *

__all__ = ("VSSBlock", "SimpleStem", "VisionClueMerge", "XSSBlock")


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            forward_type="v2",
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out


class RGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSBlock(nn.Module):
    # 增加 norm_layer 参数传递
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        
        # 使用传入的 norm_layer 替代写死的 BatchNorm2d
        self.norm = norm_layer(hidden_features) 
        
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = input + self.drop(x)
        return x


class XSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            n: int = 1,
            mlp_ratio=4.0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if in_channels != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim
        # ==========SSM============================
        self.norm = norm_layer(hidden_dim)
        self.ss2d = nn.Sequential(*(SS2D(d_model=self.hidden_dim,
                                         d_state=ssm_d_state,
                                         ssm_ratio=ssm_ratio,
                                         ssm_rank_ratio=ssm_rank_ratio,
                                         dt_rank=ssm_dt_rank,
                                         act_layer=ssm_act_layer,
                                         d_conv=ssm_conv,
                                         conv_bias=ssm_conv_bias,
                                         dropout=ssm_drop_rate, ) for _ in range(n)))
        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate)

    def forward(self, input):
        input = self.in_proj(input)
        # ====================
        X1 = self.lsblock(input)
        input = input + self.drop_path(self.ss2d(self.norm(X1)))
        # ===================
        if self.mlp_branch:
            input = input + self.drop_path(self.mlp(self.norm2(input)))
        return input


class VSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        # proj
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=False)

    def forward(self, input: torch.Tensor):
        input = self.proj_conv(input)
        X1 = self.lsblock(input)
        x = input + self.drop_path(self.op(self.norm(X1)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x


class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)
        
        # 引入原生的 PixelUnshuffle，替代手动切片
        self.unshuffle = nn.PixelUnshuffle(2) 

        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        # 极其优雅且高效，C++ 部署毫无压力
        y = self.unshuffle(x) 
        return self.pw_linear(y)

import torch.nn.functional as F
class MoELayer(nn.Module):
    def __init__(self, hidden_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = 2.0 
        
        # 1. 路由逻辑优化：利用全局上下文做决策
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 全局空间池化
            nn.Flatten(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # 2. 训练期的噪声探索逻辑
        self.noise_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 1)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 计算路由概率
        logits = self.router(x) # [B, num_experts]
        
        if self.training:
            noise_logits = self.noise_linear(x)
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            logits = logits + noise
        
        # 概率分布
        prob = F.softmax(logits / self.temperature, dim=-1)
        
        # 选择 Top-K
        topk_prob, topk_idx = torch.topk(prob, self.top_k, dim=-1)
        
        # 归一化权重
        topk_prob = topk_prob / topk_prob.sum(dim=-1, keepdim=True)
        
        # 核心改进：计算 Load Balance Loss (用于辅助训练)
        # 这部分通常作为属性抛出，或者在检测头中统一收集
        # 目的是让 prob 的均值尽量接近 1/num_experts
        if self.training:
            # 重要度损失 (Importance Loss)
            importance = prob.sum(0)
            loss_aux = self.num_experts * torch.sum(F.normalize(importance, p=1, dim=0)**2) - 1
            self.aux_loss = loss_aux # 挂载在模块上
        
        # 专家融合计算 (优化后的掩码逻辑)
        final_output = 0
        for i, expert in enumerate(self.experts):
            # 找到哪些 Batch 选择了个这专家
            # mask 形状 [B, top_k] -> [B] (只要 top_k 里有 i)
            expert_mask = (topk_idx == i).any(dim=-1)
            if not expert_mask.any():
                continue
            
            # 提取权重
            # 获取每个 batch 中专家 i 对应的权重
            weight = torch.zeros(B, 1, 1, 1, device=x.device, dtype=x.dtype)
            for k in range(self.top_k):
                batch_match = (topk_idx[:, k] == i)
                weight[batch_match] = topk_prob[batch_match, k].view(-1, 1, 1, 1)
            
            # 计算专家输出并加权
            final_output += weight * expert(x)
            
        return final_output
class MambaMoEBlock(nn.Module):
    """
    针对 YOLO 框架优化的 MambaMoEBlock
    修复了通道不匹配报错，并优化了特征流拓扑
    """
    def __init__(self, c1, c2, num_experts=4, top_k=2, drop_path=0.0):
        super().__init__()
        # c1: YOLO 自动计算的输入通道 (例如 Concat 后的 1024)
        # c2: YAML 中定义的输出通道 (例如 512)
        
        # 1. 通道调整层 (解决 RuntimeError 的关键)
        self.proj = nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
        hidden_dim = c2 # 后续所有子模块均基于 c2 进行缩放
        
        # 2. 空间局部特征提取 (Local)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        
        # 3. Mamba 全局序列建模 (Global)
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.ss2d = SS2D(d_model=hidden_dim, d_state=16) 
        
        # 4. MoE 动态路由分支 (FFN Upgrade)
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        self.moe = MoELayer(hidden_dim, num_experts=num_experts, top_k=top_k)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        # 统一输入通道到目标维度
        x = self.act(self.bn(self.proj(x)))
        
        # --- 分支 1: Mamba + 局部卷积 ---
        # 采用并行残差结构，训练更稳
        res1 = x
        x1 = self.lsblock(x)
        x1 = self.ss2d(self.norm1(x1))
        x = res1 + self.drop_path(x1)
        
        # --- 分支 2: MoE 专家路由 ---
        res2 = x
        x2 = self.moe(self.norm2(x))
        x = res2 + self.drop_path(x2)
        
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F



# ==============================================================================
# 创新点二：异构专家混合 (Heterogeneous Mixture-of-Experts)
# ==============================================================================

class HeterogeneousExpert(nn.Module):
    """
    一个异构专家模块，内部包含三个具有不同感受野和功能的子专家。
    这种设计允许模型在通道层面动态选择“局部细节”、“全局上下文”或“通道聚合”特征。
    """
    def __init__(self, dim):
        super().__init__()
        
        # 异构专家 1：标准局部特征专家 (Standard Local Expert)
        # 感受野 3x3，用于捕捉常规纹理和细节
        self.expert_local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # 异构专家 2：大感受野上下文专家 (Dilated Context Expert)
        # 采用空洞卷积 (Dilation=2)，感受野增大到 5x5，不增加参数，捕捉长距离依赖
        self.expert_context = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # 异构专家 3：通道聚合专家 (Channel Aggregation Expert)
        # 感受野 1x1，用于快速的通道间信息交互和调整
        self.expert_ch_agg = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # 专家融合投影 (确保异构特征的统一表达)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x, expert_idx):
        """
        x: [B, C, H, W]
        expert_idx: 该专家模块在 MoELayer 中的索引 (0, 1, 或 2)
        """
        # 根据动态路由的索引，选择执行具体的子专家逻辑
        # 这种 Heterogeneous 设计使得 top_k=1 也能获得丰富的特征表达
        if expert_idx == 0:
            feat = self.expert_local(x)
        elif expert_idx == 1:
            feat = self.expert_context(x)
        else:
            feat = self.expert_ch_agg(x)
            
        return self.proj_out(feat)


class HeteroMoELayer(nn.Module):
    """
    异构混合专家层 (Heterogeneous MoELayer)
    采用全局路由策略和 Load Balance Loss 防止专家塌陷。
    """
    def __init__(self, hidden_dim, num_experts=3, top_k=1):
        super().__init__()
        # 强制设置 num_experts 为 3 (对应 Local, Context, Ch_Agg)
        self.num_experts = 3 
        self.top_k = top_k
        self.temperature = 2.0 
        
        # 1. 路由逻辑 (基于全局语义)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 全局空间池化
            nn.Flatten(),
            nn.Linear(hidden_dim, self.num_experts)
        )
        
        # 2. 训练期的噪声探索逻辑
        self.noise_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, self.num_experts)
        )
        
        # 3. 核心改进：异构专家库
        # 这里 HeterogeneousExpert 内部已经处理了异构逻辑
        self.experts = nn.ModuleList([
            HeterogeneousExpert(hidden_dim) for _ in range(self.num_experts)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 计算路由概率
        logits = self.router(x) # [B, num_experts]
        
        if self.training:
            noise_logits = self.noise_linear(x)
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            logits = logits + noise
        
        # 概率分布
        prob = F.softmax(logits / self.temperature, dim=-1)
        
        # 选择 Top-K
        topk_prob, topk_idx = torch.topk(prob, self.top_k, dim=-1)
        
        # 归一化权重
        topk_prob = topk_prob / topk_prob.sum(dim=-1, keepdim=True)
        
        # 计算 Load Balance Loss (用于辅助训练，挂载在模块上供Loss函数调用)
        if self.training:
            importance = prob.sum(0)
            # 计算专家负载的均方根，越小越均衡
            loss_aux = self.num_experts * torch.sum(F.normalize(importance, p=1, dim=0)**2) - 1
            self.aux_loss = loss_aux 
        
        # 异构专家融合计算
        final_output = 0
        for i, expert in enumerate(self.experts):
            expert_mask = (topk_idx == i).any(dim=-1)
            if not expert_mask.any():
                continue
            
            # 提取权重
            weight = torch.zeros(B, 1, 1, 1, device=x.device, dtype=x.dtype)
            for k in range(self.top_k):
                batch_match = (topk_idx[:, k] == i)
                weight[batch_match] = topk_prob[batch_match, k].view(-1, 1, 1, 1)
            
            # 动态调用对应的异构逻辑
            # 注意：这里我们向 forward 传递了索引 i
            final_output += weight * expert(x, expert_idx=i)
            
        return final_output

class MoEBottleneck(nn.Module):
    """
    替换传统 BottleNeck，引入异构专家路由
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, k[0], 1, padding=k[0]//2),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )
        
        # 创新点：在这里插入异构 MoE，替代单一的 3x3 卷积
        # 专家 0: 3x3 局部专家
        # 专家 1: 5x5 (通过 3x3 dilation=2) 全局上下文专家
        # 专家 2: 7x7 (通过 3x3 dilation=3) 超大感受野专家 (补偿 Mamba 的缺失)
        self.moe = HeteroMoELayer(c_, num_experts=3, top_k=1)
        
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c2, k[1], 1, padding=k[1]//2),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 1. 基础投影
        y = self.cv1(x)
        # 2. 动态路由专家处理 (核心创新)
        y = self.moe(y)
        # 3. 输出投影与残差连接
        return x + self.cv2(y) if self.add else self.cv2(y)

class MoE_C3k2(nn.Module):
    """
    基于异构 MoE 的高级特征提取模块
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, self.c, 1, 1),
            nn.BatchNorm2d(self.c),
            nn.SiLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c1, self.c, 1, 1),
            nn.BatchNorm2d(self.c),
            nn.SiLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(2 * self.c, c2, 1, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        # 堆叠 MoE 瓶颈层
        self.m = nn.ModuleList(MoEBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        for m in self.m:
            y1 = m(y1)
        return self.cv3(torch.cat((y1, y2), 1))        