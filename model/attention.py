import torch
import math


from einops import einsum

def attention(q, k, v, mask = None):
    d_k = k.shape[-1]
    score = einsum(q, k, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)

    if mask is not None:
        score = score.masked_fill(~mask, float("-inf"))
    attn_weights = torch.softmax(score, dim=-1)
    return einsum(attn_weights, v, "... q k, ... k d_v -> ... q d_v")
