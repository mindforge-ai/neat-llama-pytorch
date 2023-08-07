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
        embedding_dim: int,
        num_attention_heads: int,
        norm_eps: float,
        max_batch_size: int,
        max_seq_len: int,
        multiple_of: int,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.embedding_dim = embedding_dim
        self.head_dim = self.embedding_dim // self.num_attention_heads
        self.norm_eps = norm_eps
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.multiple_of = multiple_of

        self.attention = Attention(
            num_attention_heads=self.num_attention_heads,
            embedding_dim=self.embedding_dim,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
        )
        self.feed_forward = MLP(
            embedding_dim=self.embedding_dim,
            hidden_dim=4 * self.embedding_dim,
            multiple_of=self.multiple_of,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(self.embedding_dim, eps=self.norm_eps)
        self.ffn_norm = RMSNorm(self.embedding_dim, eps=self.norm_eps)

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
