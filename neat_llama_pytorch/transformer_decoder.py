import torch
from torch import nn
import torch.nn.functional as F

from .transformer_block import TransformerBlock
from .normalisations import RMSNorm
from .utils import precompute_freqs_cis

from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    ColumnParallelLinear
)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_len: int,
        embedding_dim: int,
        num_layers: int,
        num_attention_heads: int,
        norm_eps: float,
        max_batch_size: int,
        max_seq_len: int,
        multiple_of: int,
        ffn_dim_multiplier: int,
    ):
        super().__init__()

        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.norm_eps = norm_eps
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier

        self.tok_embeddings = ParallelEmbedding(
            self.vocab_len, self.embedding_dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.num_layers):
            self.layers.append(
                TransformerBlock(
                    layer_id,
                    num_attention_heads=self.num_attention_heads,
                    embedding_dim=self.embedding_dim,
                    norm_eps=self.norm_eps,
                    multiple_of=self.multiple_of,
                    ffn_dim_multiplier=self.ffn_dim_multiplier,
                    max_batch_size=self.max_batch_size,
                    max_seq_len=self.max_seq_len
                )
            )

        self.norm = RMSNorm(self.embedding_dim, norm_eps=self.norm_eps)
        self.output = ColumnParallelLinear(
            self.embedding_dim, self.vocab_len, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.embedding_dim // self.num_attention_heads, self.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
