from model.memory.memory_attention import MemoryAttention, MemoryAttentionLayer
from model.memory.transformer import RoPEAttention, Attention
from model.memory.position_encoding import PositionEmbeddingRandom
from torch import nn
import torch 
from einops import rearrange

class MModule(nn.Module):
    """
    A simple module that can be used to test the MemoryAttention class.
    """
    def __init__(self, num_layers=6, d_model=512):
        # d_model 原为 6 处硬编码 512, 与 config.QKV_size 脱钩, 使容量控制失效。
        # 默认仍为 512, 既有行为不变。
        super(MModule, self).__init__()
        self.d_model = d_model
        self.att = MemoryAttention(
                d_model=d_model,
                pos_enc_at_input=True,
                layer=MemoryAttentionLayer(
                    activation = "relu",
                    cross_attention = Attention(
                            embedding_dim = d_model,
                            num_heads =  1,
                            downsample_rate = 1,
                            dropout =  0.1,
                            kv_in_dim = d_model,
                    ),
                    d_model = d_model,
                    dim_feedforward = 2048,
                    dropout = 0.1,
                    pos_enc_at_attn = True,
                    pos_enc_at_cross_attn_keys = True,
                    pos_enc_at_cross_attn_queries = True,
                    self_attention = Attention(
                            embedding_dim = d_model,
                            num_heads =  1,
                            downsample_rate = 1,
                            dropout =  0.1,
                            kv_in_dim = d_model
                    ),
                ),
                num_layers=num_layers,
                batch_first=True
            )
        
        # 输出维度为 2*num_pos_feats, 须等于 d_model, 否则位置编码相加时维度不匹配
        self.image_pos = PositionEmbeddingRandom(num_pos_feats=d_model // 2)
        self.memory_pos = PositionEmbeddingRandom(num_pos_feats=d_model // 2)

    def forward(self, curr, memory):
        # curr: torch.Tensor,  # self-attention inputs
        # memory: torch.Tensor,  # cross-attention inputs
        # curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        # memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        # num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*

        B, C, H, W = curr.shape
        image_pos = self.image_pos((H, W))
        memory_pos = self.memory_pos((H, W))
        image_pos = torch.repeat_interleave(
            image_pos.unsqueeze(0).cpu(),
            curr.shape[0],
            dim=0,
        ).to(curr.device)
        memory_pos = torch.repeat_interleave(
            memory_pos.unsqueeze(0).cpu(),
            memory.shape[0],
            dim=0,
        ).to(curr.device)
        curr = rearrange(curr, "b c h w -> b (h w) c")
        memory = rearrange(memory, "b c h w -> b (h w) c")

        image_pos = rearrange(image_pos, "b c h w -> b (h w) c")
        memory_pos = rearrange(memory_pos, "b c h w -> b (h w) c")

        output = self.att(
            curr, 
            memory, 
            curr_pos=image_pos, 
            memory_pos=memory_pos
        )
   
        output = output.permute(0, 2, 1).reshape(B, C, H, W)
        return output

if __name__ == "__main__":

    m = MModule()
    curr = torch.randn(1, 256, 8, 8)
    memory = torch.randn(1, 256, 8, 8)
    output = m(curr, memory)

    print(output.shape)  # Should be (1, 256, 8, 8)