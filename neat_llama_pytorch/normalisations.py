import torch
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, embedding_dim: int, norm_eps: float = 1e-6):
        super().__init__()
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
