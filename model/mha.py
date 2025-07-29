import torch
from torch import nn
from einops import rearrange

from model.attention import attention
from model.linear import Linear
from model.rope import Rope


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_head = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)
    
    def forward(self, x):
        # x.shape = [batch, seq, d_model]
        seq_len, d_model = x.shape[-2], x.shape[-1]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        d_k = d_model // self.num_heads

        q = rearrange(q, "... seq (n_head d_head) -> ... n_head seq d_head", n_head = self.num_heads, d_head = d_k, seq=seq_len)
        k = rearrange(k, "... seq (n_head d_head) -> ... n_head seq d_head", n_head = self.num_heads, d_head = d_k, seq=seq_len)
        v = rearrange(v, "... seq (n_head d_head) -> ... n_head seq d_head", n_head = self.num_heads, d_head = d_k, seq=seq_len)

        mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)
        mask = torch.tril(mask)
        x = attention(q, k, v, mask)
        # x.shape: [batch, n_head seq d_head]
        x = rearrange(x, "... n_head seq d_h -> ... seq (n_head d_h)", d_h=d_k, seq=seq_len, n_head=self.num_heads)
        return self.out_proj(x)


class MultiHeadSelfAttentionWithRope(nn.Module):
    def __init__(
        self, d_model, num_heads, 
        max_seq_len: int,
        theta: float
    ):
        super().__init__()
        self.d_head = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = Rope(theta, self.d_head, max_seq_len)

    def forward(self, x, pos_ids):
        # x.shape = [batch, seq, d_model]
        seq_len, d_model = x.shape[-2], x.shape[-1]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        d_k = d_model // self.num_heads

        q = rearrange(q, "... seq (n_head d_head) -> ... n_head seq d_head", n_head = self.num_heads, d_head = d_k, seq=seq_len)
        k = rearrange(k, "... seq (n_head d_head) -> ... n_head seq d_head", n_head = self.num_heads, d_head = d_k, seq=seq_len)
        v = rearrange(v, "... seq (n_head d_head) -> ... n_head seq d_head", n_head = self.num_heads, d_head = d_k, seq=seq_len)

        q = self.rope(q, pos_ids)
        k = self.rope(k, pos_ids)

        mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)
        mask = torch.tril(mask)
        x = attention(q, k, v, mask)
        # x.shape: [batch, n_head seq d_head]
        x = rearrange(x, "... n_head seq d_h -> ... seq (n_head d_h)", d_h=d_k, seq=seq_len, n_head=self.num_heads)
        return self.output_proj(x)

    
