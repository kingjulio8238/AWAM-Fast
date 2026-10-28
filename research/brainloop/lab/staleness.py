"""T1: a staleness input for the action expert, so the motor tier can run on a code/video cache that is k control
steps old (E1 showed the released model drifts there).

The action expert's timestep embedding is wrapped: t_emb = time_embedding(sinusoid(t)) + E[k], with E zero-initialised,
so an installed but untrained module reproduces the released model exactly, for every k.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TimeEmbeddingWithStaleness(nn.Module):
    def __init__(self, base: nn.Module, hidden_dim: int, max_k: int = 32):
        super().__init__()
        self.base = base
        self.max_k = int(max_k)
        self.stale = nn.Embedding(self.max_k + 1, hidden_dim)
        nn.init.zeros_(self.stale.weight)
        self.register_buffer("k", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, x):
        t = self.base(x)
        k = self.k.clamp(0, self.max_k)
        e = self.stale(k).to(t.dtype)                     # [1 or B, hidden]
        if e.shape[0] == 1 and t.shape[0] > 1:
            e = e.expand(t.shape[0], -1)
        return t + e


def install(action_expert, max_k: int = 32) -> TimeEmbeddingWithStaleness:
    if isinstance(action_expert.time_embedding, TimeEmbeddingWithStaleness):
        return action_expert.time_embedding
    w = TimeEmbeddingWithStaleness(action_expert.time_embedding, action_expert.hidden_dim, max_k)
    p = next(action_expert.parameters())
    w.stale.to(device=p.device, dtype=p.dtype)
    w.k = w.k.to(p.device)
    action_expert.time_embedding = w
    return w


def set_k(action_expert, k) -> None:
    """k: int (one staleness for the batch) or LongTensor [B]."""
    te = action_expert.time_embedding
    if not isinstance(te, TimeEmbeddingWithStaleness):
        if (torch.is_tensor(k) and bool((k != 0).any())) or (not torch.is_tensor(k) and int(k) != 0):
            raise RuntimeError("staleness module not installed")
        return
    dev = te.stale.weight.device
    te.k = (k.to(dev).long() if torch.is_tensor(k) else torch.tensor([int(k)], device=dev, dtype=torch.long))


def state_dict(action_expert) -> dict:
    return {k: v.detach().cpu() for k, v in action_expert.state_dict().items()}
