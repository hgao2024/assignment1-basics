
import torch
from torch import nn


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = nn.Parameter(torch.zeros((d_ff, d_model), dtype=dtype, device=device))
        self.w2 = nn.Parameter(torch.zeros((d_model, d_ff), dtype=dtype, device=device))
        self.w3 = nn.Parameter(torch.zeros((d_ff, d_model), dtype=dtype, device=device))

    def forward(self, x: torch.Tensor):
        return (silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T
