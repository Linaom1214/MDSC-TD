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
#  辅助模块
# ===========================================================================

class Fast_MTTU_Head(nn.Module):
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

class AADHead(nn.Module):
    def __init__(self, c_in: int, C: int = 8, alpha: float = 0.001):
        super().__init__()
        self.C = C
        self.alpha = alpha
        self.spatial_filter = nn.Sequential(
            nn.Conv2d(c_in, C, 3, 1, 1, bias=False), nn.BatchNorm2d(C), nn.ReLU(True),
            nn.Conv2d(C, C, 3, 1, 1, bias=False), nn.BatchNorm2d(C), nn.ReLU(True)
        )
    def forward(self, x):
        x_filtered = self.spatial_filter(x)
        lambda_rate = 1.0 / (torch.mean(x_filtered) + 1e-8)
        sum_activations = torch.sum(x_filtered, dim=1, keepdim=True)
        gamma_dist = Gamma(concentration=torch.tensor(self.C, device=x.device), rate=lambda_rate)
        return torch.sigmoid(self.alpha * (-torch.log(1.0 - gamma_dist.cdf(sum_activations) + 1e-8)))

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

class MMAttention(nn.Module):
    def __init__(self, config):
        super(MMAttention, self).__init__()
        self.attn_norm = LayerNorm3d(config.QKV_size, LayerNorm_type='WithBias')
        self.channel_attn = MModule() 
    def forward(self, emb4, prev_state):
        normed = self.attn_norm(emb4)
        processed = normed + self.channel_attn(normed, prev_state) if prev_state is not None else normed
        return emb4 + processed, emb4 + processed

class MTTM(nn.Module):
    def __init__(self, config, vis):
        super(MTTM, self).__init__()
        self.encoder_norm4 = LayerNorm3d(config.QKV_size, LayerNorm_type='WithBias')
        self.layers = MMAttention(config)
    def forward(self, emb4, prev_memory_list):
        current_emb4, output_memory_state = self.layers(emb4, prev_memory_list)
        return self.encoder_norm4(current_emb4), output_memory_state

class MTTM_Process(nn.Module):
    def __init__(self, config, vis, channel_num, patchSize):
        super().__init__()
        self.embeddings_4 = Normalized_Patch_Embedding(patchSize[3], channel_num[3], config.QKV_size)
        self.mttm = MTTM(config, vis)
        self.reconstruct_4 = Reconstruct(config.QKV_size, channel_num[3], kernel_size=1, scale_factor=patchSize[3])

    def forward(self, en4, prev_memory_list, prev_mask):
        if prev_mask is not None:
            en4 = en4 + prev_mask
        emb4 = self.embeddings_4(en4)
        encoded4, cur_memory_list = self.mttm(emb4, prev_memory_list)
        x4 = self.reconstruct_4(encoded4)
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
#  MDSC_TD
# ===========================================================================
class MDSC_TD(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        self.mode = mode
        
        ch_base = config.base_channel # 32

        # Encoder Path
        self.layer1 = Encoder(n_channels, ch_base, stride=1) 
        self.layer2 = Encoder(ch_base, ch_base * 2)          
        self.layer3 = Encoder(ch_base * 2, ch_base * 4)      
        self.layer4 = Encoder(ch_base * 4, ch_base * 8)      
        self.layer5 = Encoder(ch_base * 8, ch_base * 8)      

        # MTTM at Bottleneck
        self.mttm_p = MTTM_Process(config, vis,
                                   channel_num=[ch_base, ch_base * 2, ch_base * 4, ch_base * 8],
                                   patchSize=config.patch_sizes)
        
        # Decoder Path
        self.up_decoder4 = Decoder(ch_base * 8 + ch_base * 8, ch_base * 4) 
        self.up_decoder3 = Decoder(ch_base * 4 + ch_base * 4, ch_base * 2) 
        self.up_decoder2 = Decoder(ch_base * 2 + ch_base * 2, ch_base)     

        # Heads
        # Fast Head 接收融合后的特征，通过 PixelShuffle x2 放大到全尺寸
        self.fast_head = Fast_MTTU_Head(in_channels=ch_base, n_classes=n_classes, scale=2)
        
        # AADHead 作用于 Decoder 2 输出 (低分辨率)
        self.cls = AADHead(c_in=ch_base, C=8, alpha=0.001)

        # Deep Supervision
        if self.deepsuper:
            self.ds_conv5 = nn.Conv2d(ch_base * 8, n_classes, kernel_size=1) 
            self.ds_conv4 = nn.Conv2d(ch_base * 4, n_classes, kernel_size=1) 
            self.ds_conv3 = nn.Conv2d(ch_base * 2, n_classes, kernel_size=1) 
            self.ds_conv2 = nn.Conv2d(ch_base, n_classes, kernel_size=1)     
            self.ds_outconv = nn.Conv2d(n_classes * 5, n_classes, kernel_size=1)

        # Mask Downscaling (16x for Bottleneck)
        mask_in_chans = 64
        embed_dim = 256
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 8, kernel_size=2, stride=2), LayerNorm3d(mask_in_chans // 8, 'WithBias'), nn.GELU(),
            nn.Conv2d(mask_in_chans // 8, mask_in_chans // 4, kernel_size=2, stride=2), LayerNorm3d(mask_in_chans // 4, 'WithBias'), nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans // 2, kernel_size=2, stride=2), LayerNorm3d(mask_in_chans // 2, 'WithBias'), nn.GELU(),
            nn.Conv2d(mask_in_chans // 2, mask_in_chans, kernel_size=2, stride=2), LayerNorm3d(mask_in_chans, 'WithBias'), nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

    def forward(self, x, prev_memory_list=None, prev_mask=None):
        # 1. Encoder
        e1 = self.layer1(x)     
        e2 = self.layer2(e1)    
        e3 = self.layer3(e2)    
        e4 = self.layer4(e3)    
        bottleneck = self.layer5(e4) 

        # 2. Mask Prep
        mask = self.mask_downscaling(prev_mask) if prev_mask is not None else None
            
        # 3. MTTM at Bottleneck
        mttm_bottleneck, cur_memory_list = self.mttm_p(bottleneck, prev_memory_list, mask)

        # 4. Decoder
        dec4 = self.up_decoder4(mttm_bottleneck, e4) 
        dec3 = self.up_decoder3(dec4, e3)
        dec2 = self.up_decoder2(dec3, e2) # (B, 32, H/2, W/2)
        
        # 5. [关键修改] Early Fusion & Fast Upsampling
        # 5a. 计算 AAD 分数 (低分辨率)
        aad_score = self.cls(dec2) # (B, 1, H/2, W/2)
        
        # 5b. 特征融合 (广播乘法)
        # 将 anomaly score 作为注意力图，增强 dec2 特征
        dec2_enhanced = dec2 * (1 + aad_score)
        
        # 5c. 通过 PixelShuffle 头直接输出全尺寸 Mask
        main_out = self.fast_head(dec2_enhanced) # (B, 1, H, W)

        if self.deepsuper:
            ds_out5 = self.ds_conv5(mttm_bottleneck) 
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
    model = MDSC_TD(config_vit, n_channels=1, n_classes=config_vit.n_classes, mode='test', deepsuper=True, vis=False)
    model = model.to(device)
    model.eval()

    inputs = torch.rand(1, 1, 512, 512).to(device) # Keeping 8 as per original request
    
    print(f"Input shape: {inputs.shape}")

    num_frames_to_test = 3
    current_memory_list = None
    print("\nTesting temporal processing:")
    for i in range(num_frames_to_test):
        with torch.no_grad():
            output, current_memory_list = model(inputs, current_memory_list)
        
        # Handle Output format (Single Tensor or Tuple)
        if isinstance(output, tuple):
             final_out = output[-1] if isinstance(output[-1], torch.Tensor) else output[0][-1]
        else:
             final_out = output
             
        mem_info = 0
        if current_memory_list is not None:
            if isinstance(current_memory_list, torch.Tensor):
                mem_info = current_memory_list.shape # 打印形状更直观
            else:
                mem_info = len(current_memory_list)

        print(f"Frame {i+1}: Output shape: {final_out.shape}, Memory Info: {mem_info}")

    try:
        # Pass dummy inputs to profile matching the forward signature
        flops, params = profile(model, inputs=(inputs, None, None), verbose=False) 
        print("-" * 50)
        print(f'FLOPs = {flops / 1000 ** 3:.2f} G (approx.)') 
        print(f'Params = {params / 1000 ** 2:.2f} M')
    except Exception as e:
        print(f"Could not compute FLOPs/Params with thop: {e}")


    # # Timing test
    # print("-" * 50)
    # print("Running timing test...")
    # num_timing_runs = 50 
    
    # # Warm-up runs
    # for _ in range(10):
    #     with torch.no_grad():
    #         _, _ = model(inputs, None)

    # start_time = time.time()
    # temp_memory = None
    # for _ in range(num_timing_runs):
    #     with torch.no_grad():
    #         _, temp_memory = model(inputs, temp_memory, None) 
    # end_time = time.time()
    
    # avg_time_per_batch = (end_time - start_time) / num_timing_runs
    # avg_time_per_frame = avg_time_per_batch / inputs.shape[0]
    # fps = 1.0 / avg_time_per_frame if avg_time_per_frame > 0 else 0

    # print(f'Average time per frame (BS={inputs.shape[0]}) = {avg_time_per_frame * 1000:.2f} ms')
    # print(f'FPS = {fps:.2f}')
    # print("-" * 50)
    def timing(model, inputs, device, num_warmup=20, num_runs=100, keep_memory=True):
        assert inputs.device.type == ("cuda" if "cuda" in device else "cpu")
        model.eval()
        def run_once(mem):
            with torch.no_grad():
                out, mem = model(inputs, mem, None)
            return out, mem
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        mem = None
        for _ in range(num_warmup):
            _, mem = run_once(mem if keep_memory else None)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        mem = None
        for _ in range(num_runs):
            _, mem = run_once(mem if keep_memory else None)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end = time.perf_counter()
        t_step = (end - start) / num_runs
        bs = inputs.shape[0]
        step_fps = 1.0 / t_step
        throughput = bs / t_step
        per_image_ms = (t_step / bs) * 1000.0
        return {
            "t_step_ms": t_step * 1000.0,
            "per_image_ms": per_image_ms,
            "step_fps": step_fps,
            "throughput_img_s": throughput,
            "bs": bs,
        }
    
    print("-" * 50)
    print("Running timing test...")
    res = timing(model, inputs, device, num_warmup=20, num_runs=100, keep_memory=True)
    print(f"Average time per step (one temporal frame, BS={res['bs']}) = {res['t_step_ms']:.2f} ms")
    print(f"Step FPS (steps/s) = {res['step_fps']:.2f}")
    print(f"Throughput (img/s) = {res['throughput_img_s']:.2f}")
    print(f"Per-image latency (ms/img) = {res['per_image_ms']:.4f}")
    print("-" * 50)