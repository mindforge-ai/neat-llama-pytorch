from dataclasses import dataclass
from typing import Optional


@dataclass
class LlamaConfig:
    embedding_dim: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    vocab_len: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-05
    max_batch_size: int = 1
    max_seq_len: int = 4096
    ffn_dim_multiplier: Optional[int] = None