# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from model.memory.position_encoding import apply_rotary_enc, compute_axial_cis

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention(Attention):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(64, 64),  # [w, h] for stride 16 feats at 1024 resolution
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = (
            freqs_cis.to("cuda") if torch.cuda.is_available() else freqs_cis
        )
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

  
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class RoPEAttention(Attention):
    """
    Attention with rotary position encoding.
    This version is corrected to be robust to changes in batch size and sequence length.
    """

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        rope_k_repeat=False,
        # **MODIFICATION**: Increased the pre-computation size for robustness.
        # This should be larger than any expected feature map size (e.g., 64*64=4096).
        # We use 128x128 = 16384 as a safe upper limit.
        max_feat_size: Tuple[int, int] = (128, 128),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.rope_k_repeat = rope_k_repeat

        # **MODIFICATION**: Pre-compute freqs_cis once for the maximum possible size.
        # This avoids re-computation during the forward pass.
        compute_cis_func = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = compute_cis_func(end_x=max_feat_size[0], end_y=max_feat_size[1])

        # Register freqs_cis as a buffer. This ensures it's moved to the correct
        # device (e.g., GPU) along with the model, but it's not considered a model parameter.
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)


    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # **MODIFICATION**: Dynamically slice the pre-computed freqs_cis tensor.
        # This is the core of the fix. It's efficient and robust.

        # 1. Get the actual sequence length from the input query tensor 'q'.
        # The shape of q is (batch, num_heads, seq_len, head_dim).
        actual_seq_len = q.shape[-2]

        # 2. Slice self.freqs_cis to match the actual sequence length.
        # This ensures the dimensions always match, preventing the assertion error.
        # We also ensure it's on the correct device.
        sliced_freqs_cis = self.freqs_cis[:actual_seq_len, :].to(q.device)

        # The rest of the logic remains the same, but uses the sliced tensor.
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat, "Sequence lengths of Q and K must match if rope_k_repeat is False"

        num_k_rope = k.size(-2) - num_k_exclude_rope
        
        # Apply rotary position encoding using the correctly sliced frequencies
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=sliced_freqs_cis, # Use the sliced tensor here
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # Recombine heads and project
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out