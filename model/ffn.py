
import torch
from torch import nn

from model.linear import Linear

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, dtype=dtype, device=device)
        self.w2 = Linear(d_ff, d_model, dtype=dtype, device=device)
        self.w3 = Linear(d_model, d_ff, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor):
        return self.w2(silu(self.w1(x)) * self.w3(x))
