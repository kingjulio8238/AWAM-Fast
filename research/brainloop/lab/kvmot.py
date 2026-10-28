"""MoT forward with per-layer K/V capture, and an action-only forward that reads a captured cache.

Why this is exact: in AW-0's joint mask (`AgenticWAMCosmos._build_mot_attention_mask`) code reads only code,
video reads code + video, and action reads everything. Nothing reads action, so the code and video hidden
states -- and hence their per-layer keys/values -- do not depend on the action tokens. An action-only pass
that attends to K/V captured from a joint forward at the same state reproduces that forward's action output
(up to kernel numerics). Reusing a cache captured at an EARLIER state is the staleness approximation E1 measures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from agenticwam.models.experts.wan_video_dit import flash_attention


@dataclass
class KVCache:
    """Per-layer K/V of every token that precedes the action stream, [B, S_ctx, H*Dh] each."""
    k: List[torch.Tensor] = field(default_factory=list)
    v: List[torch.Tensor] = field(default_factory=list)

    @property
    def seq_len(self) -> int:
        return int(self.k[0].shape[1]) if self.k else 0

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


def _check_action_is_leaf(mot) -> None:
    if mot.expert_order[-1] != "action":
        raise RuntimeError(f"action must be the last MoT expert, got order {mot.expert_order}")


def mot_forward(mot, tokens: Dict[str, torch.Tensor], aux, attention_mask, capture: KVCache | None = None):
    """`GenericMoT.forward`, line for line, optionally filling `capture` with the non-action K/V of each layer."""
    _check_action_is_leaf(mot)
    tokens = dict(tokens)
    for layer_idx in range(mot.num_layers):
        q_chunks, k_chunks, v_chunks, states, seq_lens = [], [], [], {}, []
        for name in mot.expert_order:
            expert = mot.experts[name]
            q, k, v, st = expert.pre_attn(expert.blocks[layer_idx], tokens[name], aux[name], layer_idx)
            q_chunks.append(q); k_chunks.append(k); v_chunks.append(v)
            seq_lens.append(q.shape[1]); states[name] = st
        q_cat, k_cat, v_cat = torch.cat(q_chunks, 1), torch.cat(k_chunks, 1), torch.cat(v_chunks, 1)
        if attention_mask.shape[-1] != q_cat.shape[1]:
            raise ValueError(f"attention_mask {tuple(attention_mask.shape)} != total tokens {q_cat.shape[1]}")
        if capture is not None:
            n_ctx = q_cat.shape[1] - seq_lens[-1]
            capture.k.append(k_cat[:, :n_ctx].detach())
            capture.v.append(v_cat[:, :n_ctx].detach())
        mixed = mot._mixed_attention(q_cat, k_cat, v_cat, attention_mask)
        start = 0
        for name, slen in zip(mot.expert_order, seq_lens):
            expert = mot.experts[name]
            tokens[name] = expert.post_attn(expert.blocks[layer_idx], states[name],
                                            mixed[:, start:start + slen, :], aux[name], layer_idx)
            start += slen
    return tokens


def action_only_forward(mot, action_tokens: torch.Tensor, action_aux, cache, force_mask: bool = False) -> torch.Tensor:
    """The action stream alone, attending to [cached code/video K/V ; its own fresh K/V]. Every action row of
    the joint mask is all-True, so no mask is needed (and SDPA may pick its fused kernels).
    `cache` is a KVCache or a (k_list, v_list) tuple (the form a compiled region takes)."""
    _check_action_is_leaf(mot)
    ks, vs = (cache.k, cache.v) if isinstance(cache, KVCache) else cache
    if len(ks) != mot.num_layers:
        raise ValueError(f"cache has {len(ks)} layers, MoT has {mot.num_layers}")
    expert = mot.experts["action"]
    x = action_tokens
    for layer_idx in range(mot.num_layers):
        block = expert.blocks[layer_idx]
        q, k, v, st = expert.pre_attn(block, x, action_aux, layer_idx)
        kc, vc = ks[layer_idx].to(k.dtype), vs[layer_idx].to(v.dtype)
        if kc.shape[0] != k.shape[0]:                       # one cache shared by a batch of action rollouts
            kc, vc = kc.expand(k.shape[0], -1, -1), vc.expand(k.shape[0], -1, -1)
        k_all = torch.cat([kc, k], 1)
        v_all = torch.cat([vc, v], 1)
        # force_mask: pass an all-True boolean mask so SDPA picks the same (masked) backend as the joint pass;
        # isolates kernel numerics from any real difference (diag_motor.py)
        m = torch.ones(q.shape[1], k_all.shape[1], dtype=torch.bool, device=q.device) if force_mask else None
        out = flash_attention(q=q, k=k_all, v=v_all, num_heads=mot.num_heads, ctx_mask=m)
        x = expert.post_attn(block, st, out, action_aux, layer_idx)
    return x


def action_rows_all_true(mask: torch.Tensor, action_len: int) -> bool:
    return bool(mask[-action_len:, :].all())
