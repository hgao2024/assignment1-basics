
from torch import nn
import torch
from einops import einsum


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, dtype=None, device=None):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros((d_out, d_in), dtype=dtype, device=device))
        
    def forward(self, x: torch.Tensor):
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
