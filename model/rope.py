import torch

from torch import nn


class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # Precompute the rope values
        self.register_buffer("cos_sin",  self._compute_rope())


    def _compute_rope(self):
        # [pos, 1]
        pos = torch.arange(self.max_seq_len, device=self.device).unsqueeze(-1)
        # [1, k]
        k = torch.arange(self.d_k // 2, device=self.device).unsqueeze(0)
        # calculate cos, sin
        angles = pos / (torch.pow(self.theta, k / (self.d_k // 2)))
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # [pos, d_k // 2]
        cos_sin = self.cos_sin[token_positions]
        # x1 [pos, d_k // 2]
        new_x1 = x1 * cos_sin[..., 0] - x2 * cos_sin[..., 1]
        new_x2 = x1 * cos_sin[..., 1] + x2 * cos_sin[..., 0]
        return torch.stack([new_x1, new_x2], dim=-1).flatten(start_dim=-2)