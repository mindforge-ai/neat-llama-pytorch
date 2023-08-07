import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
import math

from .embeddings import apply_rotary_emb


class Attention(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        embedding_dim: int,
        max_batch_size: int,
        max_seq_len: int,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.embedding_dim = embedding_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.num_local_heads = self.num_attention_heads // 1
        self.head_dim = self.embedding_dim // self.num_attention_heads

        self.wq = nn.Linear(
            self.embedding_dim,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            self.embedding_dim,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            self.embedding_dim,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.embedding_dim,
            bias=False,
        )

        self.cache_k = torch.zeros(
            (self.max_batch_size, self.max_seq_len, self.num_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (self.max_batch_size, self.max_seq_len, self.num_local_heads, self.head_dim)
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.num_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, num_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, num_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)
