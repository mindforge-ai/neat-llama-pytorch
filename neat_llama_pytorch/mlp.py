from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.hidden_dim = int(2 * self.hidden_dim / 3)
        self.hidden_dim = multiple_of * (
            (self.hidden_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = nn.Linear(self.embedding_dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.embedding_dim, bias=False)
        self.w3 = nn.Linear(self.embedding_dim, self.hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
