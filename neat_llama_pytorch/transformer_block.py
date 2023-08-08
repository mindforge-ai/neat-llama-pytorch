import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from .mlp import MLP
from .normalisations import RMSNorm
from .attention import Attention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        num_attention_heads: int,
        embedding_dim: int,
        norm_eps: float,
        multiple_of: int,
        ffn_dim_multiplier: int,
        max_batch_size: int,
        max_seq_len: int
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.embedding_dim = embedding_dim
        self.embedding_dim_per_attention_head = (
            self.embedding_dim // self.num_attention_heads
        )
        self.norm_eps = norm_eps
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.attention = Attention(
            num_attention_heads=self.num_attention_heads,
            embedding_dim=self.embedding_dim,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
        )
        self.feed_forward = MLP(
            dim=self.embedding_dim,
            hidden_dim=4 * self.embedding_dim,
            multiple_of=self.multiple_of,
            ffn_dim_multiplier=self.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(self.embedding_dim, norm_eps=self.norm_eps)
        self.ffn_norm = RMSNorm(self.embedding_dim, norm_eps=self.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
