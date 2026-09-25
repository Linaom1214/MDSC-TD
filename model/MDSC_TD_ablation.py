import copy
import math
from torch.nn import Conv2d
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import time
import torch.nn.functional as F
import ml_collections
from einops import rearrange
import numbers
from thop import profile # 用于计算 FLOPs
from torch.distributions.gamma import Gamma

import sys
sys.path.append('../')

# 假设这些外部依赖依然存在
from model.memory.module import MModule
from model.memory.sam2_utils import get_activation_fn, get_clones

# ===========================================================================
#  辅助模块 (Copied from MDSC_TD.py)
# ===========================================================================

class PixelShuffleHead(nn.Module):
    """
    接收 H/2, W/2 的特征图，通过 PixelShuffle 直接输出 H, W 的预测结果。
    """
    def __init__(self, in_channels, n_classes, scale=2):
        super().__init__()
        self.process = nn.Sequential(
            # 1. 特征整理
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 2. 通道扩充: 输出通道 = n_classes * scale^2
            nn.Conv2d(64, n_classes * (scale ** 2), 1, bias=False),
            # 3. 亚像素重排
            nn.PixelShuffle(scale)
        )
    def forward(self, x): return self.process(x)

def autopad(kernel_size): return (kernel_size - 1) // 2

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, autopad(kernel_size), bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, autopad(dw_size), groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.oup, :, :]

class Encoder(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2):
        super().__init__()
        self.ghost1 = GhostModule(inp, int(inp * 2), kernel_size)
        self.convdw = nn.Conv2d(int(inp * 2), int(inp * 2), kernel_size, stride, autopad(kernel_size), groups=int(inp * 2))
        self.bn = nn.BatchNorm2d(int(inp * 2))
        self.ghost2 = GhostModule(int(inp * 2), oup, kernel_size, stride=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size, stride, autopad(kernel_size), groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.ghost2(self.bn(self.convdw(self.ghost1(x)))) + self.shortcut(x)

class Decoder(nn.Module):
    def __init__(self, hidden_concat_channels, oup, kernel_size=3):
        super().__init__()
        self.ghost = GhostModule(hidden_concat_channels, oup, kernel_size)
    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        return self.ghost(torch.cat((x1, x2), dim=1))

class _ConstGate(nn.Module):
    """可学习标量门控: 输出与输入同空间尺寸的常数场 sigmoid(s)。

    用作 SHT 的正确对照组。原 --no_sht 令 dec2_enhanced = dec2, 同时移除了
    门控本身携带的常数增益; 而实测表明原 SHT 的输出恒在 [0.5, 0.5046] 区间内
    (alpha=1e-3 时 sigmoid 参数被 1e-8 钳在 [0, 18.42] 之内), 其唯一作用就是提供
    一个约 1.5 倍的常数缩放。本对照保留该常数增益且允许其被学习, 从而把
    "门控是否存在" 与 "门控内容是否携带信息" 分离: 若本组与 SHT 组无显著差异,
    则 SHT 的统计检验部分不携带信息。
    """

    def __init__(self, init: float = 0.0):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x):
        return torch.sigmoid(self.s).expand(x.shape[0], 1, x.shape[2], x.shape[3])


class _Conv1x1Gate(nn.Module):
    """1x1 卷积 + sigmoid 门控: 有可学习的逐像素空间门控, 但不含任何统计假设检验。

    用于区分 "空间自适应门控本身有用" 与 "Gamma 零假设检验有用"。
    """

    def __init__(self, c_in: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 1, kernel_size=1, bias=True)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class SHT(nn.Module):
    """
    统计假设检验特征校准头 (Statistical Hypothesis Testing head)。
    基于背景统计建模对像素激活做异常打分。支持多种背景分布假设以供消融对比。
    dist_type: 'gamma' | 'rayleigh' | 'ggd' | 'empirical'
    """
    def __init__(self, c_in: int, C: int = 8, alpha: float = 0.001, dist_type: str = 'gamma'):
        super().__init__()
        self.C = C
        self.alpha = alpha
        self.dist_type = dist_type
        self.spatial_filter = nn.Sequential(
            nn.Conv2d(c_in, C, 3, 1, 1, bias=False), nn.BatchNorm2d(C), nn.ReLU(True),
            nn.Conv2d(C, C, 3, 1, 1, bias=False), nn.BatchNorm2d(C), nn.ReLU(True)
        )
        # GGD 的形状参数 beta 作为可学习参数 (beta=2 -> 高斯, beta=1 -> 拉普拉斯)
        if dist_type == 'ggd':
            self.ggd_beta = nn.Parameter(torch.tensor(1.5))

    def _survival_gamma(self, s, x_filtered):
        lambda_rate = 1.0 / (torch.mean(x_filtered) + 1e-8)
        gamma_dist = Gamma(concentration=torch.tensor(float(self.C), device=s.device), rate=lambda_rate)
        return 1.0 - gamma_dist.cdf(s) + 1e-8

    def _survival_rayleigh(self, s, x_filtered):
        # Rayleigh: P(X>s) = exp(-s^2 / (2*sigma^2)); sigma^2 由二阶矩估计
        sigma_sq = torch.mean(x_filtered ** 2) / 2.0 + 1e-8
        return torch.exp(-(s ** 2) / (2.0 * sigma_sq)) + 1e-8

    def _survival_ggd(self, s, x_filtered):
        # 广义高斯分布尾概率的指数核近似: P(X>s) ~ exp(-(s/a)^beta)
        beta = torch.clamp(self.ggd_beta, 0.5, 4.0)
        a = torch.mean(torch.abs(x_filtered)) + 1e-8
        return torch.exp(-torch.pow(s / a, beta)) + 1e-8

    def _survival_empirical(self, s, x_filtered):
        # 经验分布: 用批内激活值的均值/标准差做标准化后的高斯尾概率近似
        mu = torch.mean(x_filtered)
        std = torch.std(x_filtered) + 1e-8
        z = (s - mu) / std
        # 标准正态尾概率 Q(z) = 0.5 * erfc(z / sqrt(2))
        return 0.5 * torch.erfc(z / math.sqrt(2.0)) + 1e-8

    def forward(self, x):
        x_filtered = self.spatial_filter(x)
        sum_activations = torch.sum(x_filtered, dim=1, keepdim=True)

        if self.dist_type == 'gamma':
            survival = self._survival_gamma(sum_activations, x_filtered)
        elif self.dist_type == 'rayleigh':
            survival = self._survival_rayleigh(sum_activations, x_filtered)
        elif self.dist_type == 'ggd':
            survival = self._survival_ggd(sum_activations, x_filtered)
        elif self.dist_type == 'empirical':
            survival = self._survival_empirical(sum_activations, x_filtered)
        else:
            raise ValueError(f"Unknown dist_type: {self.dist_type}")

        return torch.sigmoid(self.alpha * (-torch.log(survival)))

def to_4d(x, h, w): return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
def to_3d(x): return rearrange(x, 'b c h w -> b (h w) c')

class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.norm(to_3d(x)), h, w)

class Normalized_Patch_Embedding(nn.Module):
    def __init__(self, patchsize, in_channels, out_channels):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(in_channels, out_channels, kernel_size=patchsize, stride=patchsize)
    def forward(self, x): return self.patch_embeddings(x) if x is not None else None

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(True)
        self.scale_factor = scale_factor
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return self.activation(self.norm(self.conv(x)))

class STMICrossAttention(nn.Module):
    """Cross-frame memory retrieval + intra-frame context enhancement."""
    def __init__(self, config, num_layers=6):
        super(STMICrossAttention, self).__init__()
        self.attn_norm = LayerNorm3d(config.QKV_size, LayerNorm_type='WithBias')
        self.channel_attn = MModule(num_layers=num_layers, d_model=config.QKV_size)
    def forward(self, emb4, prev_state):
        normed = self.attn_norm(emb4)
        processed = normed + self.channel_attn(normed, prev_state) if prev_state is not None else normed
        return emb4 + processed, emb4 + processed

class STMIBlock(nn.Module):
    def __init__(self, config, vis):
        super(STMIBlock, self).__init__()
        self.encoder_norm4 = LayerNorm3d(config.QKV_size, LayerNorm_type='WithBias')
        self.layers = STMICrossAttention(config, num_layers=config.transformer.num_layers)
    def forward(self, emb4, prev_memory_list):
        current_emb4, output_memory_state = self.layers(emb4, prev_memory_list)
        return self.encoder_norm4(current_emb4), output_memory_state

class STMI(nn.Module):
    """Spatio-Temporal Memory Interaction module (paper Sec. III-B).

    The Sparse Prior-Guided (SPG) mask injection is nested here: the
    previous-frame mask is added to the bottleneck feature before the
    cross-frame attention, confining memory retrieval to prior-flagged
    regions.
    """
    def __init__(self, config, vis, channel_num, patchSize):
        super().__init__()
        self.embeddings_4 = Normalized_Patch_Embedding(patchSize[3], channel_num[3], config.QKV_size)
        self.stmi_block = STMIBlock(config, vis)
        self.reconstruct_4 = Reconstruct(config.QKV_size, channel_num[3], kernel_size=1, scale_factor=patchSize[3])

    def forward(self, en4, prev_memory_list, prev_mask):
        # SPG: sparse prior-guided mask injection
        if prev_mask is not None:
            en4 = en4 + prev_mask
        emb4 = self.embeddings_4(en4)
        encoded4, cur_memory_list = self.stmi_block(emb4, prev_memory_list)
        x4 = self.reconstruct_4(encoded4)
        # 全分辨率测试图(非 256 整除)时，patch 步长向下取整会使 reconstruct 的
        # scale_factor 插值与 en4 尺寸差 1~2，导致残差加法尺寸不匹配。
        # 以 en4 精确尺寸为准对齐，训练(256 整除)时为恒等、不影响结果。
        if x4.shape[-2:] != en4.shape[-2:]:
            x4 = F.interpolate(x4, size=en4.shape[-2:], mode='bilinear', align_corners=True)
        return x4 + en4, cur_memory_list

def get_MDSC_TD_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.QKV_size = 128 * 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32
    config.n_classes = 1
    return config

# ===========================================================================
#  MDSC_TD_Ablation
# ===========================================================================
class MDSC_TD_Ablation(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, vis=False,
                 mode='train', deepsuper=True,
                 # 消融实验开关与参数
                 use_stmi=True,        # 是否使用时空记忆交互模块 (STMI)
                 use_sht=True,         # 是否使用假设检验虚警抑制头 (SHT)
                 use_mask=True,        # 是否使用前序掩码引导 (SPG)
                 sht_alpha=0.001,      # SHT 灵敏度因子
                 sht_channels=8,       # SHT 通道数(Gamma分布形状参数k)
                 sht_dist='gamma',     # SHT 背景分布假设 (新增: gamma/rayleigh/ggd/empirical)
                 num_attn_layers=6,    # Memory Attention 层数 (新增)
                 downsample_rate=2,    # 瓶颈层注意力 patch 下采样步长 s (新增，默认2)
                 sht_mode='sht',       # 门控分支实现: sht / const / conv1x1 (新增)
                 base_channel=None,    # 主干宽度; None=用 config 默认(32)。容量控制消融
                 qkv_size=None,        # STMI 注意力维度; None=config 默认(512)。
                                       # 实测 STMI 占总参数 91.7%, 容量控制主要靠它
                 stmi_place='bottleneck'):  # STMI 放置: bottleneck / enc4 / dec4

        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        self.mode = mode

        # 保存消融配置
        self.use_stmi = use_stmi
        self.use_sht = use_sht
        self.use_mask = use_mask
        self.num_attn_layers = num_attn_layers
        self.downsample_rate = downsample_rate
        self.sht_mode = sht_mode
        self.stmi_place = stmi_place

        # 容量控制: 参数量约与 ch_base^2 成正比, ch_base=12 时约为默认(32)的 14%,
        # 可构造与 RFR(4.9M) 量级相当的对照, 回应 28.2M vs 基线 0.07~4.9M 的质疑。
        if base_channel is not None or qkv_size is not None:
            config = copy.deepcopy(config)
            if base_channel is not None:
                config.base_channel = int(base_channel)
            if qkv_size is not None:
                # STMI 参数约与 QKV_size^2 成正比
                config.QKV_size = int(qkv_size)
        ch_base = config.base_channel # 默认 32

        # Encoder Path
        self.layer1 = Encoder(n_channels, ch_base, stride=1)
        self.layer2 = Encoder(ch_base, ch_base * 2)
        self.layer3 = Encoder(ch_base * 2, ch_base * 4)
        self.layer4 = Encoder(ch_base * 4, ch_base * 8)
        self.layer5 = Encoder(ch_base * 8, ch_base * 8)

        # STMI at Bottleneck (根据 use_stmi 决定是否初始化)
        if self.use_stmi:
            # 动态创建配置副本以支持不同的层数与瓶颈 patch 下采样率 s
            config_dynamic = copy.deepcopy(config)
            config_dynamic.transformer.num_layers = num_attn_layers
            # downsample_rate(s) 控制瓶颈层注意力的 patch 下采样步长，
            # 决定注意力 token 数 (~1/s^2)，进而决定注意力复杂度 (~1/s^4)
            patch_sizes_dynamic = list(config.patch_sizes)
            patch_sizes_dynamic[3] = downsample_rate
            config_dynamic.patch_sizes = patch_sizes_dynamic
            # STMI 输入通道随放置位置而变: bottleneck/enc4 为 8*ch_base, dec4 为 4*ch_base。
            # dec4 空间分辨率为瓶颈的 2 倍, patch 步长同步加倍以保持 token 数可比。
            place_ch = {'bottleneck': ch_base * 8, 'enc4': ch_base * 8,
                        'dec4': ch_base * 4}[self.stmi_place]
            if self.stmi_place == 'dec4':
                patch_sizes_dynamic[3] = max(1, downsample_rate * 2)
            self.stmi = STMI(config_dynamic, vis,
                              channel_num=[ch_base, ch_base * 2, ch_base * 4, place_ch],
                              patchSize=patch_sizes_dynamic)
        else:
            self.stmi = None

        # Decoder Path
        self.up_decoder4 = Decoder(ch_base * 8 + ch_base * 8, ch_base * 4)
        self.up_decoder3 = Decoder(ch_base * 4 + ch_base * 4, ch_base * 2)
        self.up_decoder2 = Decoder(ch_base * 2 + ch_base * 2, ch_base)

        # Heads
        # Fast Head 接收融合后的特征，通过 PixelShuffle x2 放大到全尺寸
        self.fast_head = PixelShuffleHead(in_channels=ch_base, n_classes=n_classes, scale=2)

        # SHT 作用于 Decoder 2 输出 (低分辨率) (根据 use_sht 决定是否初始化)
        # sht_mode 控制门控分支的实现, 用于把"门控是否存在"与"门控内容是否携带信息"分开:
        #   'sht'     : 原统计假设检验头 (默认, 与历史行为一致)
        #   'const'   : 可学习标量, 门控退化为 dec2*(1+sigmoid(s)) —— 只保留常数增益
        #   'conv1x1' : 1x1 卷积 + sigmoid —— 有可学习的空间门控但无统计检验
        # 注: --no_sht 直接令 dec2_enhanced = dec2, 同时移除了常数增益本身,
        # 因此它不是检验 SHT 内容的正确对照 (实测把已训练模型的门控置 0 会使 Pd 塌至 80.5%)。
        if self.use_sht:
            if self.sht_mode == 'sht':
                self.sht = SHT(c_in=ch_base, C=sht_channels, alpha=sht_alpha, dist_type=sht_dist)
            elif self.sht_mode == 'const':
                self.sht = _ConstGate()
            elif self.sht_mode == 'conv1x1':
                self.sht = _Conv1x1Gate(c_in=ch_base)
            else:
                raise ValueError(f"unknown sht_mode: {self.sht_mode}")
        else:
            self.sht = None

        # Deep Supervision
        if self.deepsuper:
            self.ds_conv5 = nn.Conv2d(ch_base * 8, n_classes, kernel_size=1)
            self.ds_conv4 = nn.Conv2d(ch_base * 4, n_classes, kernel_size=1)
            self.ds_conv3 = nn.Conv2d(ch_base * 2, n_classes, kernel_size=1)
            self.ds_conv2 = nn.Conv2d(ch_base, n_classes, kernel_size=1)
            self.ds_outconv = nn.Conv2d(n_classes * 5, n_classes, kernel_size=1)

        # Mask Downscaling (固定 16x，对齐瓶颈层 en4 的空间分辨率)
        # 如果启用 stmi 且启用 mask (SPG)，才需要初始化 mask_downscaling
        if self.use_stmi and self.use_mask:
            mask_in_chans = 64
            # 原固定 256 = 8*32; ch_base 或放置位置改变时须同步, 否则残差相加通道不匹配
            embed_dim = {'bottleneck': ch_base * 8, 'enc4': ch_base * 8,
                         'dec4': ch_base * 4}[self.stmi_place]
            self.mask_downscaling = nn.Sequential(
                nn.Conv2d(1, mask_in_chans // 8, kernel_size=2, stride=2), LayerNorm3d(mask_in_chans // 8, 'WithBias'), nn.GELU(),
                nn.Conv2d(mask_in_chans // 8, mask_in_chans // 4, kernel_size=2, stride=2), LayerNorm3d(mask_in_chans // 4, 'WithBias'), nn.GELU(),
                nn.Conv2d(mask_in_chans // 4, mask_in_chans // 2, kernel_size=2, stride=2), LayerNorm3d(mask_in_chans // 2, 'WithBias'), nn.GELU(),
                nn.Conv2d(mask_in_chans // 2, mask_in_chans, kernel_size=2, stride=2), LayerNorm3d(mask_in_chans, 'WithBias'), nn.GELU(),
                nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
            )
        else:
            self.mask_downscaling = None

    def forward(self, x, prev_memory_list=None, prev_mask=None):
        # 1. Encoder
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        bottleneck = self.layer5(e4)

        # 2. STMI Module (Ablation Controlled)
        cur_memory_list = None

        if self.use_stmi:
            # Mask Prep (Ablation Controlled)
            if self.use_mask and self.mask_downscaling is not None and prev_mask is not None:
                mask = self.mask_downscaling(prev_mask)
            else:
                mask = None

            # mask_downscaling 固定 16x; dec4 分辨率为其 2 倍, 需上采样对齐
            # mask_downscaling 固定 16x, 对齐瓶颈层。enc4 与 dec4 的分辨率均为瓶颈的 2 倍,
            # 两者都需上采样, 否则残差相加尺寸不匹配 (实测 enc4 报 32 vs 16)。
            if self.stmi_place in ('dec4', 'enc4') and mask is not None:
                mask = F.interpolate(mask, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            mask = None

        # STMI 放置位置消融 (原实现固定在 bottleneck, 无任何实验依据):
        #   enc4       —— 提前一级, 作用于 layer4 输出 (不经 layer5 的进一步下采样)
        #   bottleneck —— 默认
        #   dec4       —— 推迟到解码器第一级之后
        if self.use_stmi and self.stmi_place == 'enc4':
            e4, cur_memory_list = self.stmi(e4, prev_memory_list, mask)
            stmi_bottleneck = bottleneck
        elif self.use_stmi and self.stmi_place == 'bottleneck':
            stmi_bottleneck, cur_memory_list = self.stmi(bottleneck, prev_memory_list, mask)
        else:
            stmi_bottleneck = bottleneck
            if not self.use_stmi:
                cur_memory_list = None

        # 3. Decoder
        dec4 = self.up_decoder4(stmi_bottleneck, e4)
        if self.use_stmi and self.stmi_place == 'dec4':
            dec4, cur_memory_list = self.stmi(dec4, prev_memory_list, mask)
        dec3 = self.up_decoder3(dec4, e3)
        dec2 = self.up_decoder2(dec3, e2) # (B, 32, H/2, W/2)

        # 4. Head (Ablation Controlled)
        if self.use_sht:
             # 4a. 计算 SHT 分数 (低分辨率)
            sht_score = self.sht(dec2) # (B, 1, H/2, W/2)
            # 4b. 特征融合 (广播乘法)
            dec2_enhanced = dec2 * (1 + sht_score)
        else:
            # Bypass SHT
            dec2_enhanced = dec2
            sht_score = None # 如果在其他地方使用了这个变量，需要注意

        # 4c. 通过 PixelShuffle 头直接输出全尺寸 Mask
        main_out = self.fast_head(dec2_enhanced) # (B, 1, H, W)

        if self.deepsuper:
            ds_out5 = self.ds_conv5(stmi_bottleneck)
            ds_out4 = self.ds_conv4(dec4)
            ds_out3 = self.ds_conv3(dec3)
            ds_out2 = self.ds_conv2(dec2)

            # DeepSupervision 依然需要双线性插值来对齐
            target_size = main_out.shape[2:]
            ds_out5_up = F.interpolate(ds_out5, size=target_size, mode='bilinear', align_corners=True)
            ds_out4_up = F.interpolate(ds_out4, size=target_size, mode='bilinear', align_corners=True)
            ds_out3_up = F.interpolate(ds_out3, size=target_size, mode='bilinear', align_corners=True)
            ds_out2_up = F.interpolate(ds_out2, size=target_size, mode='bilinear', align_corners=True)

            combined_ds_out = self.ds_outconv(torch.cat((ds_out2_up, ds_out3_up, ds_out4_up, ds_out5_up, main_out), dim=1))

            if self.mode == 'train':
                return (torch.sigmoid(ds_out5_up), torch.sigmoid(ds_out4_up), torch.sigmoid(ds_out3_up),
                        torch.sigmoid(ds_out2_up), torch.sigmoid(combined_ds_out), torch.sigmoid(main_out)), \
                       cur_memory_list
            else:
                return torch.sigmoid(main_out), cur_memory_list
        else:
            return torch.sigmoid(main_out), cur_memory_list

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    config_vit = get_MDSC_TD_config()

    # Example: Testing Ablation Variant (Disable STMI)
    print("\n--- Testing Variant: w/o STMI ---")
    model = MDSC_TD_Ablation(config_vit, n_channels=1, n_classes=config_vit.n_classes, mode='test',
                          deepsuper=True, vis=False,
                          use_stmi=False, use_sht=True, use_mask=True).to(device)
    model.eval()

    inputs = torch.rand(2, 1, 256, 256).to(device)
    output, _ = model(inputs, None, None)
    if isinstance(output, tuple):
        print(f"Output shape (DeepSuper): {output[-1].shape}")
    else:
        print(f"Output shape: {output.shape}")

    # Example: Testing Variant (Disable SHT)
    print("\n--- Testing Variant: w/o SHT ---")
    model_no_sht = MDSC_TD_Ablation(config_vit, n_channels=1, n_classes=config_vit.n_classes, mode='test',
                                 deepsuper=True, vis=False,
                                 use_stmi=True, use_sht=False, use_mask=True).to(device)
    model_no_sht.eval()
    output, _ = model_no_sht(inputs, None, None)
    if isinstance(output, tuple):
        print(f"Output shape (DeepSuper): {output[-1].shape}")
    else:
        print(f"Output shape: {output.shape}")
