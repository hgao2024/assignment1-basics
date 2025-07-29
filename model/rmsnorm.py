import torch

from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, in_features: torch.Tensor):
        rms = in_features.square().mean(dim=-1, keepdim=True).sqrt()
        return  in_features / (rms + self.eps) * self.weight
