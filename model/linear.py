
from torch import nn
import torch
from einops import einsum


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((d_out, d_in), dtype=dtype, device=device))
        std = (2.0 / (d_in + d_out)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
