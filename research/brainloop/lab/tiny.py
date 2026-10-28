"""A tiny AgenticWAMCosmos built from the real classes, for CPU tests of the lab code.

Same expert classes, MoT, attention mask, schedulers and joint decode as the released model; only the widths,
depths, VAE and vision tower are shrunk or faked. A test that passes here proves the lab's plumbing and its
equivalence claims (cached-K/V action pass == joint pass, lab decode == the model's own decode); it says nothing
about the trained model's behaviour.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from agenticwam.models.cosmos import (
    COSMOS_2B_CONFIG, CosmosTransformer3DModel, FlowMatchUniPCScheduler, MultiViewRoPE, SchedulerConfig,
    VideoTokenizer)
from agenticwam.models.cosmos_mot.agenticwam_model import AgenticWAMCosmos
from agenticwam.models.cosmos_mot.cosmos_video_expert import CosmosVideoExpert
from agenticwam.models.cosmos_mot.wan_experts import WanActionExpert, WanCodeExpert
from agenticwam.models.experts.code_expert import _build_mock_code_expert
from agenticwam.models.experts.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

TINY_H, TINY_W = 64, 96          # multiples of 32: 16x VAE downsample, then 2x2 patches
TEXT_IN = 40                     # stands in for the 100352-wide Cosmos-Reason1 embedding


def _byte_decode(ids, skip_special_tokens=True):
    return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


class FakeVAE(nn.Module):
    """Causal 4x-temporal / 16x-spatial 'VAE' with the WanVideoVAE38 call surface (single_encode/decode)."""

    def __init__(self, z=48):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.enc = nn.Parameter(torch.randn(z, 3, generator=g) * 0.5, requires_grad=False)
        self.dec = nn.Parameter(torch.linalg.pinv(self.enc.data), requires_grad=False)

    def single_encode(self, video):                       # [N,3,T,H,W] -> [N,48,1+(T-1)//4,H/16,W/16]
        x = F.avg_pool3d(video.float(), (1, 16, 16))
        first, rest = x[:, :, :1], x[:, :, 1:]
        if rest.shape[2] > 0:
            rest = rest.reshape(*rest.shape[:2], rest.shape[2] // 4, 4, *rest.shape[3:]).mean(3)
            x = torch.cat([first, rest], 2)
        else:
            x = first
        return torch.einsum("zc,ncthw->nzthw", self.enc.float(), x).to(video.dtype)

    def single_decode(self, lat):                         # inverse-ish, for plumbing only
        x = torch.einsum("cz,nzthw->ncthw", self.dec.float(), lat.float())
        first, rest = x[:, :, :1], x[:, :, 1:].repeat_interleave(4, dim=2)
        x = torch.cat([first, rest], 2)
        x = F.interpolate(x, scale_factor=(1, 16, 16), mode="nearest")
        return x.clamp(-1, 1).to(lat.dtype)


class FakeCodeVision(nn.Module):
    """Stands in for CodeVisionPrefix: [B,nc,3,H,W] -> [B, nc*P, D] with P = 4 tokens per view."""

    def __init__(self, out_dim, per_view=4):
        super().__init__()
        self.per_view = per_view
        self.proj = nn.Linear(3 * per_view, out_dim * per_view)
        self.out = out_dim

    def forward(self, obs):
        B, nc = obs.shape[:2]
        x = F.adaptive_avg_pool2d(obs.flatten(0, 1).float(), (2, 2)).flatten(1)          # [B*nc, 12]
        return self.proj(x.to(self.proj.weight.dtype)).view(B, nc * self.per_view, self.out)


def build_tiny_model(seed: int = 0, n_cameras: int = 3, adim: int = 14, canvas: int = 16,
                     max_code_tokens: int = 4096) -> AgenticWAMCosmos:
    torch.manual_seed(seed)
    heads, hd, layers = 2, 32, 2
    cfg = dataclasses.replace(
        COSMOS_2B_CONFIG, in_channels=49, out_channels=48, view_embed_dim=7, max_n_cameras=4,
        num_attention_heads=heads, attention_head_dim=hd, num_layers=layers, adaln_lora_dim=8,
        crossattn_proj_in_channels=TEXT_IN, text_embed_dim=32)
    dit = CosmosTransformer3DModel(cfg)
    video_expert = CosmosVideoExpert(dit, use_gradient_checkpointing=False)
    action_expert = WanActionExpert(hidden_dim=48, action_dim=adim, ffn_dim=96, text_dim=cfg.text_embed_dim,
                                    freq_dim=32, eps=1e-6, num_heads=heads, attn_head_dim=hd, num_layers=layers)
    code_expert = _build_mock_code_expert(num_layers=layers, num_heads=heads, attn_head_dim=hd,
                                          hidden_dim=heads * hd, vocab_size=259, max_code_tokens=max_code_tokens)
    code_expert.tokenizer.decode = _byte_decode
    code_expert.__class__ = WanCodeExpert
    code_expert.mask_token_id = 258
    code_expert.code_objective = "joint"
    model = AgenticWAMCosmos(
        video_expert=video_expert, action_expert=action_expert, code_expert=code_expert, vae=FakeVAE(),
        video_tokenizer=VideoTokenizer(cfg),
        video_rope=MultiViewRoPE(cfg.attention_head_dim, cfg.max_size, cfg.patch_size, rope_scale=cfg.rope_scale),
        text_encoder=None,
        video_scheduler=FlowMatchUniPCScheduler(SchedulerConfig(num_train_timesteps=1000, shift=5.0)),
        action_scheduler=WanContinuousFlowMatchScheduler(num_train_timesteps=1000),
        proprio_dim=adim, text_proj_dim=cfg.text_embed_dim, n_cameras=n_cameras, num_cond_frames=1,
        video_sigma_shift=5.0, use_code_expert=True, infer_code_tokens=canvas)
    model.code_vision = FakeCodeVision(heads * hd)
    model.code_memory_enabled = False
    model.code_memory = None
    # random init makes the AdaLN-Zero gates zero -> every expert is an identity map and every test is vacuous
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and p.abs().sum() == 0 and p.ndim >= 1:
                p.normal_(0, 0.02)
    return model.eval()


def tiny_inputs(model: AgenticWAMCosmos, seed: int = 0, prompt_len: int = 6):
    g = torch.Generator().manual_seed(seed)
    nc = model.n_cameras
    obs = torch.rand(1, 3, nc, TINY_H, TINY_W, generator=g) * 2 - 1
    proprio = torch.randn(1, model.action_expert.action_dim, generator=g)
    prompt = torch.randn(1, prompt_len, TEXT_IN, generator=g)
    return obs, proprio, prompt
