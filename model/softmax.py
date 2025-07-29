import torch

def softmax(logits, dim):
    logits = torch.transpose(logits, dim, -1)
    new_logits = logits - logits.max(dim=-1, keepdim=True).values
    exps = torch.exp(new_logits)
    scores = exps / exps.sum(dim=-1, keepdim=True)
    return scores.transpose(-1, dim)
