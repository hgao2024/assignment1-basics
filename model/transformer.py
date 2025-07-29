import torch
from torch import nn

from model.ffn import FeedForward
from model.rmsnorm import RMSNorm
from model.mha import MultiHeadSelfAttentionWithRope

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, theta, max_seq_len):
        super().__init__()
        self.attn = MultiHeadSelfAttentionWithRope(
            d_model, num_heads, max_seq_len, theta, 
        )
        self.ln1 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    
    def forward(self, x, pos_ids):
        x1 = self.ln1(x)
        x1 = self.attn(x1, pos_ids)
        x = x + x1
        x2 = self.ln2(x)
        x2 = self.ffn(x2)
        return x + x2
