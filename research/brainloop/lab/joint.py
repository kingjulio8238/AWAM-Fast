"""The joint tri-modal decode, re-implemented with the hooks E1-E5 need.

`joint_decode` reproduces `AgenticWAMCosmos._infer_action_joint` on its DEFAULT path (no research env gates:
confidence-ordered reveal, linear reveal schedule, no CFG / remask / memory) -- the same op order and the same
noise draws, so with `generator=None` and the same global seed it returns the same action and code
(tests/test_cpu_tiny.py checks this). On top of that it can:

  * capture the code+video K/V the action stream read at the final step (`capture="final"`) or at every
    step (`capture="all"`)                                           -> E1, E5 motor tier
  * start mid-schedule from a previous prediction (`warm=`)          -> E2
  * take separate pixels for the VAE and the ViT pathway             -> E4
  * return the final video latents and per-token commit confidences  -> E3

`fast_action` is the motor tier: action-only denoising against a cache, with fresh proprio.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .kvmot import KVCache, action_only_forward, action_rows_all_true, mot_forward

# env gates that switch _infer_action_joint off its default path; the lab refuses to run under any of them
_NON_DEFAULT_ENV = (
    "AGENTICWAM_REMASK_ROUNDS", "AGENTICWAM_ORACLE_CODE_FILE", "AGENTICWAM_PREFIX_LAST", "AGENTICWAM_DECODE_TEMP",
    "AGENTICWAM_ORDER_TEMP", "AGENTICWAM_WARMUP", "AGENTICWAM_MIN_CODE_LEN", "AGENTICWAM_ABLATE_FREEZE_VIDEO",
    "AGENTICWAM_ABLATE_DROP_CODE", "AGENTICWAM_ABLATE_VIT_PREFIX", "AGENTICWAM_CFG_WC", "AGENTICWAM_CFG_WV",
    "AGENTICWAM_EOS_DEFER_FRAC", "AGENTICWAM_EOS_RUN_TRUNC", "AGENTICWAM_CODE_PREAMBLE_FILE",
    "AGENTICWAM_CODE_NO_INSTRUCTION")


def assert_default_decode_env() -> None:
    """Fail loudly if an env knob would make the model's own decode differ from this one."""
    bad = {k: os.environ[k] for k in _NON_DEFAULT_ENV if os.environ.get(k) not in (None, "", "0")}
    if os.environ.get("AGENTICWAM_CONF_THRESH") not in (None, "", "0"):
        bad["AGENTICWAM_CONF_THRESH"] = os.environ["AGENTICWAM_CONF_THRESH"]
    if os.environ.get("AGENTICWAM_REVEAL_ORDER", "confidence") != "confidence":
        bad["AGENTICWAM_REVEAL_ORDER"] = os.environ["AGENTICWAM_REVEAL_ORDER"]
    fl = os.environ.get("AGENTICWAM_FL_FRAC")
    if fl not in (None, "", "1", "1.0"):
        bad["AGENTICWAM_FL_FRAC"] = fl
    if bad:
        raise RuntimeError(f"non-default decode env set, lab decode would not match the model's: {bad}")


@dataclass
class Inputs:
    obs_video: torch.Tensor                    # [1,3,nc,H,W] in [-1,1] -> VAE pathway (video expert)
    proprio: torch.Tensor                      # [1,pdim] normalized state
    prompt_embs: torch.Tensor                  # [1,L,E] Cosmos-Reason1 embedding of the wrapped instruction
    code_instruction: str                      # the code expert's prompt text (skill block + instruction)
    obs_video_vit: Optional[torch.Tensor] = None   # E4: pixels for the ViT pathway only (None -> obs_video)
    action_horizon: int = 32
    num_video_frames: int = 9


@dataclass
class Static:
    """Everything a decode needs that depends only on the observation and prompt."""
    obs_latents: torch.Tensor
    vit_pref: Optional[torch.Tensor]
    prefix_ids: Optional[torch.Tensor]
    n_prefix: int
    pids: List[int]
    plen: int
    gen_len: int
    prompt_embs: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    Tlpv: int
    n_cond_lat: int
    cond_pos: torch.Tensor
    cond_mask5d: torch.Tensor


@dataclass
class Result:
    action: torch.Tensor                       # [Ta, adim] fp32 cpu, normalized (model) space
    canvas: torch.Tensor                       # [gen_len] final code canvas ids (cpu)
    commit_conf: torch.Tensor                  # [gen_len] confidence at commit (nan = given by warm start)
    commit_step: torch.Tensor                  # [gen_len] step index at commit (-1 = given by warm start)
    code_text: str
    latents_video: torch.Tensor                # [1,Cl,nc*Tlpv,Hl,Wl] final video latents (device, model dtype)
    latents_action: torch.Tensor               # [1,Ta,adim] final action latents (device)
    static: Static
    cache: Optional[KVCache] = None            # capture="final"
    caches: Optional[List[KVCache]] = None     # capture="all"
    steps_run: int = 0
    final_inputs: Optional[dict] = None        # capture="final": the last step's inputs -> rebuild_cache() is exact


def compute_dtype(model):
    """The frozen experts' dtype. Not the action expert's: T1 trains it in fp32 while everything else stays bf16."""
    return next(model.video_expert.parameters()).dtype


def autocast_ctx(model):
    cdt = compute_dtype(model)
    dev = model.device
    if dev.type == "cuda" and cdt in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=cdt)
    return contextlib.nullcontext()


def _randn(shape, *, device, dtype, generator):
    if generator is None:
        return torch.randn(*shape, device=device, dtype=dtype)
    return torch.randn(*shape, device=device, dtype=dtype, generator=generator)


@torch.no_grad()
def action_context(model, prompt_embs, proprio):
    cdt = compute_dtype(model)
    p = proprio
    if p is not None and p.dim() == 2:
        p = p.unsqueeze(1)
    if p is not None:
        p = p.to(device=model.device, dtype=cdt)
    return model._action_context(prompt_embs.to(device=model.device, dtype=cdt), p)


@torch.no_grad()
def prepare_static(model, inp: Inputs, gen_len: int) -> Static:
    ce = model.code_expert
    device = model.device
    cdt = compute_dtype(model)
    nc = model.n_cameras
    with autocast_ctx(model):
        obs_latents = model._encode_video_latents(inp.obs_video.to(device))
        vit_src = inp.obs_video if inp.obs_video_vit is None else inp.obs_video_vit
        vit_pref = model._vit_prefix(vit_src.to(device))
    n_prefix = int(vit_pref.shape[1]) if vit_pref is not None else 0
    prefix_ids = None
    if vit_pref is not None:
        pad = getattr(ce.tokenizer, "pad_token_id", 0) or 0
        prefix_ids = torch.full((1, n_prefix), pad, dtype=torch.long, device=device)
    n_cond_lat = obs_latents.shape[2] // nc
    Tlpv = max(model._pixel_to_latent(int(inp.num_video_frames)), n_cond_lat)
    Cl, Hl, Wl = obs_latents.shape[1], obs_latents.shape[3], obs_latents.shape[4]
    prompt = f"# Instruction: {inp.code_instruction}\n"
    try:
        pids = ce.tokenizer.encode(prompt, add_special_tokens=False)
    except TypeError:
        pids = ce.tokenizer.encode(prompt)
    plen = len(pids)
    cap = int(ce.freqs.shape[0])
    if plen + gen_len > cap:
        gen_len = cap - plen
    context, context_mask = action_context(model, inp.prompt_embs, inp.proprio)
    cond_pos = torch.zeros(1, nc * Tlpv, dtype=torch.bool, device=device)
    cond_mask5d = torch.zeros(1, 1, nc * Tlpv, Hl, Wl, device=device, dtype=cdt)
    for view in range(nc):
        d0 = view * Tlpv
        cond_pos[:, d0:d0 + n_cond_lat] = True
        cond_mask5d[:, :, d0:d0 + n_cond_lat] = 1.0
    return Static(obs_latents=obs_latents, vit_pref=vit_pref, prefix_ids=prefix_ids, n_prefix=n_prefix,
                  pids=pids, plen=plen, gen_len=int(gen_len),
                  prompt_embs=inp.prompt_embs.to(device=device, dtype=cdt), context=context,
                  context_mask=context_mask, Tlpv=Tlpv, n_cond_lat=n_cond_lat, cond_pos=cond_pos,
                  cond_mask5d=cond_mask5d)


def _clamp_cond(latents_video, st: Static, nc: int):
    for view in range(nc):
        d0, s0 = view * st.Tlpv, view * st.n_cond_lat
        latents_video[:, :, d0:d0 + st.n_cond_lat] = st.obs_latents[:, :, s0:s0 + st.n_cond_lat]
    return latents_video


def _forward(model, st: Static, latents_video, latents_action, code_ids, sigma_v, t_a, mask,
             capture: Optional[KVCache]):
    """One joint MoT step (fast_joint_infer._step_core, through kvmot.mot_forward)."""
    return _forward_t(model, latents_video, latents_action, code_ids, sigma_v, t_a, st.prompt_embs, st.context,
                      st.context_mask, st.cond_pos, st.cond_mask5d, st.vit_pref, st.prefix_ids, st.n_prefix, mask,
                      capture)


def _forward_t(model, latents_video, latents_action, code_ids, sigma_v, t_a, prompt_embs, context, context_mask,
               cond_pos, cond_mask5d, vit_pref, prefix_ids, n_prefix: int, mask, capture: Optional[KVCache]):
    """`_forward` with every per-observation tensor as an argument, so a compiled/CUDA-graphed copy can be reused
    across observations (the graph copies new inputs into its static buffers)."""
    v_sig_pf = torch.where(cond_pos, model.cond_timestep, sigma_v).to(latents_video.dtype)
    video_tokens, video_aux, shape_info = model._video_pre(latents_video, v_sig_pf, prompt_embs, cond_mask5d)
    action_pre = model.action_expert.pre_dit(action_tokens=latents_action, timestep=t_a,
                                             context=context, context_mask=context_mask)
    ce = model.code_expert
    if vit_pref is None:
        code_pre = ce.pre_dit(code_ids)
    else:
        code_pre = ce.pre_dit(torch.cat([prefix_ids, code_ids], dim=1))
        ct = code_pre["tokens"].clone()
        ct[:, :n_prefix, :] = vit_pref.to(ct.dtype)
        code_pre["tokens"] = ct
    tokens = {"video": video_tokens, "code": code_pre["tokens"], "action": action_pre["tokens"]}
    aux = {"video": video_aux,
           "code": {"t_mod": code_pre["t_mod"], "freqs": code_pre["freqs"], "context": None},
           "action": {"t_mod": action_pre["t_mod"], "freqs": action_pre["freqs"],
                      "context": {"context": action_pre["context"], "mask": action_pre["context_mask"]}}}
    if mask is None:
        mask = model._build_mot_attention_mask(video_tokens.shape[1], action_pre["tokens"].shape[1], model.device,
                                               code_seq_len=code_ids.shape[1] + n_prefix,
                                               code_prefix_len=n_prefix)
        if not action_rows_all_true(mask, action_pre["tokens"].shape[1]):
            raise RuntimeError("action rows of the joint mask are not all-True; the K/V cache would be inexact")
    out = mot_forward(model.mot, tokens, aux, mask, capture=capture)
    pred_video = model._video_post(out["video"], video_aux, shape_info)
    pred_action = model.action_expert.post_dit(out["action"], action_pre)
    code_logits = ce.post_dit(out["code"])[0][n_prefix:]
    return pred_video, pred_action, code_logits, mask


def reset_video_schedule(model, num_steps: int):
    device = model.device
    model.video_scheduler.sigmas = model._vs_sigmas0.clone().to(device)
    model.video_scheduler.timesteps = model._vs_timesteps0.clone().to(device)
    model.video_scheduler.set_timesteps(num_steps, device=device)


@torch.no_grad()
def joint_decode(model, inp: Inputs, *, num_steps: int, gen_len: int, generator=None, warm: Optional[dict] = None,
                 capture: str = "none", static: Optional[Static] = None, core=None) -> Result:
    """Default-path joint decode. `warm` = {"start": s, "video": [1,Cl,nc*Tlpv,Hl,Wl], "action": [1,Ta,adim],
    "canvas": [gen_len] ids with mask_token_id at the positions to re-decode}: the run starts at schedule index s
    from x_s = (1-sigma_s) * x_warm + sigma_s * noise (video and action), with the given canvas."""
    if capture not in ("none", "final", "all"):
        raise ValueError(capture)
    if core is not None and capture == "all":
        raise ValueError("a compiled core cannot capture K/V")
    # with a compiled core, capture="final" records the final step's inputs only (rebuild_cache() makes the cache)
    model.dit.eval()
    from . import staleness
    staleness.set_k(model.action_expert, 0)        # joint decodes are never stale (and a leftover batch k breaks B=1)
    device = model.device
    cdt = compute_dtype(model)
    adim = model.action_expert.action_dim
    nc = model.n_cameras
    ce = model.code_expert
    mid = ce.mask_token_id
    st = static if static is not None else prepare_static(model, inp, gen_len)
    gen_len = st.gen_len
    Cl, Hl, Wl = st.obs_latents.shape[1], st.obs_latents.shape[3], st.obs_latents.shape[4]
    with autocast_ctx(model):
        # noise draws in the original order (video then action), whether or not the run is warm
        noise_v = _randn((1, Cl, nc * st.Tlpv, Hl, Wl), device=device, dtype=cdt, generator=generator)
        noise_a = _randn((1, inp.action_horizon, adim), device=device, dtype=cdt, generator=generator)
        reset_video_schedule(model, num_steps)
        ts_a, deltas_a = model.action_scheduler.build_inference_schedule(num_steps, device, cdt)
        T = len(ts_a)
        start = 0
        canvas = torch.full((gen_len,), mid, dtype=torch.long, device=device)
        if warm is None:
            latents_video, latents_action = noise_v, noise_a
        else:
            start = int(warm["start"])
            if not 0 <= start < T:
                raise ValueError(f"warm start {start} outside [0,{T})")
            sv = model.video_scheduler.sigmas[start].to(device=device, dtype=cdt)
            latents_video = (1 - sv) * warm["video"].to(device=device, dtype=cdt) + sv * noise_v
            sa = (ts_a[start] / float(model.action_scheduler.num_train_timesteps)).to(cdt)
            latents_action = (1 - sa) * warm["action"].to(device=device, dtype=cdt) + sa * noise_a
            canvas = warm["canvas"].to(device=device, dtype=torch.long).clone()
            if canvas.numel() != gen_len:
                raise ValueError(f"warm canvas {canvas.numel()} != gen_len {gen_len}")
        latents_video = _clamp_cond(latents_video, st, nc)
        code_ids = torch.cat([torch.tensor([st.pids], dtype=torch.long, device=device), canvas.view(1, -1)], 1)
        plen = st.plen
        m0 = int((code_ids[0, plen:] == mid).sum())
        commit_conf = torch.full((gen_len,), float("nan"))
        commit_step = torch.full((gen_len,), -1, dtype=torch.long)
        mask = None
        cache = None
        final_inputs = None
        caches = [] if capture == "all" else None
        for step_index in range(start, T):
            t_a, d_a = ts_a[step_index], deltas_a[step_index]
            sigma_v = model.video_scheduler.sigmas[step_index].to(device=device, dtype=cdt)
            ts_v = model.video_scheduler.timesteps[step_index]
            ts_a_ = t_a.reshape(1).to(device=device, dtype=cdt)
            cap = None
            if capture == "all" or (capture == "final" and step_index == T - 1):
                cap = KVCache()
                if capture == "final":
                    final_inputs = {"latents_video": latents_video.detach().clone(),
                                    "code_ids": code_ids.detach().clone(), "sigma_v": sigma_v.detach().clone()}
            if core is not None:
                cap = None
                pred_video, pred_action, code_logits = core(latents_video, latents_action, code_ids, sigma_v, ts_a_)
            else:
                pred_video, pred_action, code_logits, mask = _forward(
                    model, st, latents_video, latents_action, code_ids, sigma_v, ts_a_, mask, cap)
            if capture == "all":
                caches.append(cap)
            elif cap is not None:
                cache = cap
            res = model.video_scheduler.step(pred_video, ts_v, latents_video, step_index)
            latents_video = res[0] if isinstance(res, (tuple, list)) else res
            latents_video = _clamp_cond(latents_video, st, nc)
            latents_action = model.action_scheduler.step(pred_action, d_a, latents_action)
            masked = (code_ids[0, plen:] == mid).nonzero(as_tuple=True)[0] + plen
            if masked.numel() > 0 and m0 > 0:
                frac = min(1.0, (step_index - start + 1) / max(1.0, float(T - start)))
                target = int(round(m0 * frac))
                n_reveal = max(0, target - (m0 - int(masked.numel())))
                if step_index == T - 1:
                    n_reveal = int(masked.numel())
                if n_reveal > 0:
                    k = min(int(n_reveal), int(masked.numel()))
                    probs = code_logits[masked].float().softmax(-1)
                    conf, tok = probs.max(-1)
                    order = torch.argsort(conf, descending=True)[:k]
                    code_ids[0, masked[order]] = tok[order].to(code_ids.dtype)
                    pos = (masked[order] - plen).cpu()
                    commit_conf[pos] = conf[order].float().cpu()
                    commit_step[pos] = step_index
    canvas = code_ids[0, plen:].detach().cpu()
    return Result(action=latents_action[0].detach().float().cpu(), canvas=canvas, commit_conf=commit_conf,
                  commit_step=commit_step, code_text=decode_canvas(ce, canvas), latents_video=latents_video,
                  latents_action=latents_action, static=st, cache=cache, caches=caches, steps_run=T - start,
                  final_inputs=final_inputs)


def decode_canvas(ce, canvas) -> str:
    mid = ce.mask_token_id
    eos = getattr(ce.tokenizer, "eos_token_id", None)
    gen = [int(t) for t in canvas.tolist() if int(t) != mid]
    if eos is not None and eos in gen:
        gen = gen[:gen.index(eos)]
    try:
        return ce.tokenizer.decode(gen, skip_special_tokens=True).strip()
    except Exception:
        return ""


@torch.no_grad()
def fast_action(model, cache, prompt_embs, proprio, *, num_steps: int, action_horizon: int = 32, generator=None,
                caches: Optional[List[KVCache]] = None, force_mask: bool = False, stale_k: int = 0,
                init: Optional[torch.Tensor] = None, start: int = 0) -> torch.Tensor:
    """init/start: warm motor -- begin at schedule index `start` from x = (1-s)*init + s*noise (init = the previous
    plan's action latents shifted to now), so the motor only corrects the old plan with fresh proprio."""
    """Motor tier: denoise the action alone against a code/video K/V cache, with FRESH proprio.
    `caches` (one per step, from capture="all" at the same num_steps) reproduces the joint decode exactly."""
    device = model.device
    cdt = compute_dtype(model)
    adim = model.action_expert.action_dim
    from . import staleness
    staleness.set_k(model.action_expert, stale_k)
    with autocast_ctx(model):
        context, context_mask = action_context(model, prompt_embs, proprio)
        latents_action = _randn((1, action_horizon, adim), device=device, dtype=cdt, generator=generator)
        ts_a, deltas_a = model.action_scheduler.build_inference_schedule(num_steps, device, cdt)
        if caches is not None and len(caches) != len(ts_a):
            raise ValueError(f"{len(caches)} per-step caches for {len(ts_a)} steps")
        if init is not None:
            sa = (ts_a[start] / float(model.action_scheduler.num_train_timesteps)).to(cdt)
            latents_action = (1 - sa) * init.to(device=device, dtype=cdt) + sa * latents_action
        for i, (t_a, d_a) in enumerate(zip(ts_a, deltas_a)):
            if i < start:
                continue
            pre = model.action_expert.pre_dit(action_tokens=latents_action, timestep=t_a.reshape(1).to(cdt),
                                              context=context, context_mask=context_mask)
            aux = {"t_mod": pre["t_mod"], "freqs": pre["freqs"],
                   "context": {"context": pre["context"], "mask": pre["context_mask"]}}
            x = action_only_forward(model.mot, pre["tokens"], aux, caches[i] if caches is not None else cache,
                                    force_mask=force_mask)
            latents_action = model.action_scheduler.step(model.action_expert.post_dit(x, pre), d_a, latents_action)
    return latents_action[0].detach().float().cpu()


@torch.no_grad()
def encode_window(model, video: torch.Tensor) -> torch.Tensor:
    """[1,3,nc*T,H,W] pixels in [-1,1] -> per-view VAE latents [1,Cl,nc*Tl,Hl,Wl] (the training encode)."""
    with autocast_ctx(model):
        return model._encode_video_latents(video.to(model.device))


@torch.no_grad()
def decode_latents(model, latents: torch.Tensor) -> torch.Tensor:
    with autocast_ctx(model):
        return model._decode_video_latents(latents)


def make_core(model, st: Static, gen_len: int, action_horizon: int = 32, compile_mode: Optional[str] = "reduce-overhead"):
    """One joint MoT step for a fixed token layout, optionally torch.compile'd (CUDA graphs). Returns bind(static) ->
    core(latents_video, latents_action, code_ids, sigma_v, t_a) -> (pred_video, pred_action, code_logits)."""
    cdt = compute_dtype(model)
    dev = model.device
    lv = torch.zeros(1, st.obs_latents.shape[1], model.n_cameras * st.Tlpv, *st.obs_latents.shape[3:], device=dev,
                     dtype=cdt)
    la = torch.zeros(1, action_horizon, model.action_expert.action_dim, device=dev, dtype=cdt)
    ids = torch.zeros(1, st.plen + gen_len, dtype=torch.long, device=dev)
    with autocast_ctx(model), torch.no_grad():
        _, _, _, mask = _forward(model, st, lv, la, ids, torch.tensor(0.5, device=dev, dtype=cdt),
                                 torch.tensor([500.0], device=dev, dtype=cdt), None, None)

    n_prefix = st.n_prefix

    def _core(latents_video, latents_action, code_ids, sigma_v, t_a, prompt_embs, context, context_mask, cond_pos,
              cond_mask5d, vit_pref, prefix_ids):
        return _forward_t(model, latents_video, latents_action, code_ids, sigma_v, t_a, prompt_embs, context,
                          context_mask, cond_pos, cond_mask5d, vit_pref, prefix_ids, n_prefix, mask, None)[:3]

    fn = _core if compile_mode is None else torch.compile(_core, mode=compile_mode, fullgraph=True)

    def bind(static: Static):
        """core for one observation: core(latents_video, latents_action, code_ids, sigma_v, t_a)."""
        if static.n_prefix != n_prefix:
            raise ValueError("observation has a different token layout than the compiled core")
        return lambda lv_, la_, ids_, sv_, ta_: fn(lv_, la_, ids_, sv_, ta_, static.prompt_embs, static.context,
                                                   static.context_mask, static.cond_pos, static.cond_mask5d,
                                                   static.vit_pref, static.prefix_ids)

    return bind


@torch.no_grad()
def rebuild_cache(model, st: Static, final_inputs: dict, action_horizon: int = 32) -> KVCache:
    """The terminal code/video K/V cache from stored final-step inputs. Action tokens never enter the cache, so a
    zero action placeholder gives the same cache the decode captured."""
    from . import staleness
    staleness.set_k(model.action_expert, 0)
    cdt = compute_dtype(model)
    dev = model.device
    la = torch.zeros(1, action_horizon, model.action_expert.action_dim, device=dev, dtype=cdt)
    cap = KVCache()
    with autocast_ctx(model):
        _forward(model, st, final_inputs["latents_video"].to(dev, cdt), la, final_inputs["code_ids"].to(dev),
                 final_inputs["sigma_v"].to(dev, cdt), torch.tensor([0.0], device=dev, dtype=cdt), None, cap)
    return cap


def reflex_action(model, st: Static, inp: Inputs, video: torch.Tensor, canvas: torch.Tensor, sigma_v: torch.Tensor,
                  *, num_steps: int, generator=None, action_horizon: int = 32) -> torch.Tensor:
    """Reflex tier: fresh vision AND fresh action against the last slow plan. `st` comes from the current observation;
    `video` is the slow plan's predicted future shifted to now (its conditioning frames are replaced by the fresh
    ones); `canvas` is the slow plan's code. One joint forward at the slow plan's final noise level rebuilds the
    code/video K/V, then the action is denoised from noise in `num_steps` action-only steps with fresh proprio."""
    dev = model.device
    fi = {"latents_video": _clamp_cond(video.clone().to(dev), st, model.n_cameras),
          "code_ids": torch.cat([torch.tensor([st.pids], dtype=torch.long), canvas.view(1, -1).long().cpu()], 1),
          "sigma_v": sigma_v}
    cache = rebuild_cache(model, st, fi, action_horizon)
    return fast_action(model, cache, inp.prompt_embs, inp.proprio, num_steps=num_steps,
                       action_horizon=action_horizon, generator=generator)
