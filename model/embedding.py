import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, d_model, device=None, dtype=None):
        super().__init__()
        # Create the embedding matrix as a learnable parameter
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, d_model, device=device, dtype=dtype)
        )

    def forward(self, x):
        # x: (batch, seq_len) or any shape of indices
        return self.weight[x]
