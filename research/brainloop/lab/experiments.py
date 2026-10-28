"""E1-E4 data collection over RoboTwin demonstrations, and E5 latency.

One pass over an episode decodes the released model at every 4th frame ("anchors", two noise seeds each) and,
from each anchor, runs the cheap variants the experiments compare against those fresh decodes:

  E1 staleness   motor tier (action-only, cached code/video K/V from anchor a, fresh proprio at a+k) and plain
                 open-loop playback of anchor a's chunk, vs the fresh decode at a+k.
  E2 warm start  decode at a+k started mid-schedule from anchor a's final video/action/code, vs cold decodes.
  E3 surprise    per-anchor foresight error (predicted vs actual future latents and pixels), seed disagreement and
                 code confidence; the analysis lines them up with gripper events and with E1's errors.
  E4 occlusion   (separate pass) action/code sensitivity to dropping, blurring or occluding parts of each view,
                 per pathway (VAE -> video expert, ViT -> code expert).
  E5 latency     (separate pass) stage-level timings of the full decode, the compiled decode, and the motor tier.

Everything is written as JSON lines under out_dir; analysis.py turns them into the report.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import joint
from .demos import Episode, gripper_events
from .metrics import action_err, canvas_agreement, rel_mse, text_similarity


@dataclass
class SweepConfig:
    nis: int = 20                                      # released WAM eval setting
    stride: int = 4                                    # anchor spacing = the video frame stride
    horizon: int = 32
    ks: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 24])
    fast_steps: List[int] = field(default_factory=lambda: [1, 2, 4])
    e2_every: int = 4                                  # E2 on every 4th anchor
    warm_ks: List[int] = field(default_factory=lambda: [4, 8, 16, 24])
    warm_steps: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    warm_rho: float = 0.5                              # fraction of code positions re-masked (lowest confidence)
    warm_rho_extra: List[float] = field(default_factory=lambda: [0.0, 1.0])   # also run at warm_steps == 2
    decode_pixels: bool = True
    max_anchors: Optional[int] = None                  # cap per episode (smoke)
    seed: int = 1234


def _gen(model, *parts) -> torch.Generator:
    h = int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)
    g = torch.Generator(device=model.device)
    g.manual_seed(h)
    return g


def _sync(model):
    if model.device.type == "cuda":
        torch.cuda.synchronize()


def window_video(adapter, ep: Episode, t: int, n_frames: int = 9, stride: int = 4) -> torch.Tensor:
    """Pixels of frames t, t+4, ..., t+32 for all views, [1,3,nc*9,H,W] view-major (the training layout)."""
    frames = [adapter.obs_video(ep.obs_dict(t + stride * i)) for i in range(n_frames)]   # [1,3,nc,H,W] each
    v = torch.stack(frames, dim=3)                                                        # [1,3,nc,9,H,W]
    return v.reshape(v.shape[0], v.shape[1], v.shape[2] * v.shape[3], v.shape[4], v.shape[5])


def shift_video(latents: torch.Tensor, nc: int, tlpv: int, k: int, frames_per_latent: int = 16) -> torch.Tensor:
    """Advance a predicted future by k control steps in latent space: latent j <- interp(j, j+1) by k/16."""
    out = latents.clone()
    alpha = k / float(frames_per_latent)
    for v in range(nc):
        base = v * tlpv
        fut = latents[:, :, base + 1: base + tlpv]                     # [1,C,tlpv-1,H,W]
        n = fut.shape[2]
        new = []
        for j in range(n):
            pos = j + alpha
            lo = min(int(np.floor(pos)), n - 1)
            hi = min(lo + 1, n - 1)
            w = min(max(pos - np.floor(pos), 0.0), 1.0) if hi != lo else 0.0
            new.append((1 - w) * fut[:, :, lo] + w * fut[:, :, hi])
        out[:, :, base + 1: base + tlpv] = torch.stack(new, dim=2)
    return out


def shift_action(latents_action: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return latents_action.clone()
    a = latents_action[:, k:]
    pad = latents_action[:, -1:].expand(-1, k, -1)
    return torch.cat([a, pad], dim=1)


def remask(canvas: torch.Tensor, conf: torch.Tensor, rho: float, mask_id: int) -> torch.Tensor:
    c = canvas.clone()
    n = int(round(rho * c.numel()))
    if n <= 0:
        return c
    score = conf.clone()
    score[~torch.isfinite(score)] = float("inf")
    c[torch.argsort(score)[:n]] = mask_id
    return c


class Writer:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a")

    def __call__(self, rec: Dict):
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def _progress(out_dir: str, **kw):
    kw["ts"] = time.time()
    tmp = os.path.join(out_dir, "progress.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(kw, fh)
    os.replace(tmp, os.path.join(out_dir, "progress.json"))


def sweep_episode(adapter, ep: Episode, cfg: SweepConfig, out_dir: str, instruction: Optional[str] = None,
                  progress: Optional[Dict] = None) -> Dict:
    """Collect E1/E2/E3 records for one episode into out_dir/sweep/<task>_ep<i>.jsonl (resumable per episode)."""
    model = adapter.model
    instr = instruction or ep.instructions[0]
    gen_len = adapter.gen_len
    path = os.path.join(out_dir, "sweep", f"{ep.task}_ep{ep.index}.jsonl")
    done_flag = path + ".done"
    if os.path.exists(done_flag):
        return {"skipped": True}
    if os.path.exists(path):
        os.remove(path)                                  # partial episode: redo it whole
    w = Writer(path)
    last = ep.T - 1 - cfg.horizon                         # anchor needs t+32 for GT and the 9-frame window
    anchors = list(range(0, last + 1, cfg.stride))
    if cfg.max_anchors:
        anchors = anchors[: cfg.max_anchors]
    aset = set(anchors)
    w({"kind": "episode", "task": ep.task, "ep": ep.index, "T": ep.T, "instruction": instr,
       "anchors": anchors, "events": gripper_events(ep.qpos), "qpos": ep.qpos.round(5).tolist(),
       "cfg": cfg.__dict__})
    fresh: Dict[int, Dict] = {}
    pending_fast: List[Dict] = []
    cold_cache: Dict = {}
    t_ep = time.time()
    n_mot_steps = 0
    for ai, a in enumerate(anchors):
        t0 = time.time()
        inp = adapter.inputs(ep.obs_dict(a), instr)
        st = joint.prepare_static(model, inp, gen_len)
        rA = joint.joint_decode(model, inp, num_steps=cfg.nis, gen_len=gen_len, generator=_gen(model, cfg.seed, "A", a),
                                capture="final", static=st)
        rB = joint.joint_decode(model, inp, num_steps=cfg.nis, gen_len=gen_len, generator=_gen(model, cfg.seed, "B", a),
                                static=st)
        n_mot_steps += 2 * cfg.nis
        _sync(model)
        t_dec = (time.time() - t0) / 2
        actA, actB = adapter.denorm(rA.action), adapter.denorm(rB.action)
        # --- E3: foresight error of anchor a's imagined future vs what actually happened
        win = window_video(adapter, ep, a)
        actual = joint.encode_window(model, win)
        nc, tl = model.n_cameras, st.Tlpv
        lat_err = [[rel_mse(rA.latents_video[0, :, v * tl + j].float().cpu().numpy(),
                            actual[0, :, v * tl + j].float().cpu().numpy()) for j in range(1, tl)] for v in range(nc)]
        pix_err = None
        if cfg.decode_pixels:
            pred_pix = joint.decode_latents(model, rA.latents_video).float()
            tp = pred_pix.shape[2] // nc
            winf = win.to(pred_pix.device).float()
            pix_err = [[float(((pred_pix[0, :, v * tp + i] - winf[0, :, v * tp + i]) ** 2).mean()) for i in range(1, tp)]
                       for v in range(nc)]
        rec = {"kind": "anchor", "task": ep.task, "ep": ep.index, "a": a, "t_decode_s": round(t_dec, 3),
               "actA": actA.round(5).tolist(), "actB": actB.round(5).tolist(),
               "codeA": rA.code_text, "codeB": rB.code_text,
               "canvasA": rA.canvas.tolist(), "canvasB": rB.canvas.tolist(),
               "confA": [None if not np.isfinite(x) else round(float(x), 4) for x in rA.commit_conf.tolist()],
               "lat_err": lat_err, "pix_err": pix_err, "plen": st.plen, "gen_len": st.gen_len,
               "n_prefix": st.n_prefix, "cache_tokens": rA.cache.seq_len}
        w(rec)
        fresh[a] = {"actA": actA, "actB": actB, "canvasA": rA.canvas, "canvasB": rB.canvas, "codeA": rA.code_text,
                    "codeB": rB.code_text, "latA": rA.latents_video.detach().cpu(), "latB": rB.latents_video.detach().cpu()}
        # --- E1: motor tier from anchor a's cache, fresh proprio at a+k (compared once a+k is decoded)
        for k in cfg.ks:
            if a + k not in aset:
                continue
            obs_k = ep.obs_dict(a + k)
            prop = adapter.proprio(obs_k["joint_action"]["vector"])
            for n in cfg.fast_steps:
                af = joint.fast_action(model, rA.cache, inp.prompt_embs, prop, num_steps=n, action_horizon=cfg.horizon,
                                       generator=_gen(model, cfg.seed, "F", a, k, n))
                pending_fast.append({"a": a, "k": k, "method": f"fast{n}", "act": adapter.denorm(af)})
            pending_fast.append({"a": a, "k": k, "method": "open_loop", "act": actA[k:]})
        # --- E2: warm starts from anchor a (every e2_every-th anchor)
        if ai % cfg.e2_every == 0:
            mid = model.code_expert.mask_token_id
            for k in cfg.warm_ks:
                if a + k not in aset:
                    continue
                inp_k = adapter.inputs(ep.obs_dict(a + k), instr)
                st_k = joint.prepare_static(model, inp_k, gen_len)
                wv = shift_video(rA.latents_video, nc, tl, k)
                wa = shift_action(rA.latents_action, k)
                plans = [(n, cfg.warm_rho) for n in cfg.warm_steps]
                plans += [(2, r) for r in cfg.warm_rho_extra]
                for n, rho in plans:
                    warm = {"start": cfg.nis - n, "video": wv, "action": wa,
                            "canvas": remask(rA.canvas, rA.commit_conf, rho, mid)}
                    rw = joint.joint_decode(model, inp_k, num_steps=cfg.nis, gen_len=gen_len,
                                            generator=_gen(model, cfg.seed, "W", a, k, n, rho), warm=warm, static=st_k)
                    n_mot_steps += n
                    pending_fast.append({"a": a, "k": k, "method": f"warm{n}_rho{rho}", "act": adapter.denorm(rw.action),
                                         "canvas": rw.canvas, "code": rw.code_text, "lat": rw.latents_video.detach().cpu()})
                for n in cfg.warm_steps:                   # cold baselines with the same step budget
                    key = (a + k, n)
                    if key in cold_cache:
                        continue
                    rc = joint.joint_decode(model, inp_k, num_steps=n, gen_len=gen_len,
                                            generator=_gen(model, cfg.seed, "C", a + k, n), static=st_k)
                    n_mot_steps += n
                    cold_cache[key] = True
                    pending_fast.append({"a": a + k, "k": 0, "method": f"cold{n}", "act": adapter.denorm(rc.action),
                                         "canvas": rc.canvas, "code": rc.code_text, "lat": rc.latents_video.detach().cpu()})
        del rA, rB
        if progress is not None:
            el = time.time() - progress["t_start"]
            progress.update(anchor=ai + 1, anchors=len(anchors), elapsed_s=round(el), mot_steps=progress.get(
                "mot_steps_base", 0) + n_mot_steps, ep_task=ep.task, ep_index=ep.index, last_decode_s=round(t_dec, 3))
            _progress(out_dir, **progress)
    # --- resolve every pending comparison against the fresh decodes
    for p in pending_fast:
        tgt = p["a"] + p["k"]
        f = fresh.get(tgt)
        if f is None:
            continue
        rec = {"kind": "cmp", "task": ep.task, "ep": ep.index, "a": p["a"], "k": p["k"], "method": p["method"]}
        for h in (4, 8, 16, 24):
            rec[f"vsA_h{h}"] = action_err(p["act"], f["actA"], h)
            rec[f"vsB_h{h}"] = action_err(p["act"], f["actB"], h)
            for o in (0, 1):
                g0 = tgt + o
                rec[f"vsGT{o}_h{h}"] = action_err(p["act"], ep.qpos[g0:g0 + h], h)
        if "canvas" in p:
            rec["canvas_vsA"] = canvas_agreement(p["canvas"], f["canvasA"])
            rec["text_vsA"] = text_similarity(p["code"], f["codeA"])
            nc, tl = model.n_cameras, f["latA"].shape[2] // model.n_cameras
            fut = [v * tl + j for v in range(nc) for j in range(1, tl)]
            rec["lat_vsA"] = rel_mse(p["lat"][0, :, fut].float().numpy(), f["latA"][0, :, fut].float().numpy())
        w(rec)
    for a, f in fresh.items():                             # seed floor at every anchor
        rec = {"kind": "floor", "task": ep.task, "ep": ep.index, "a": a}
        for h in (4, 8, 16, 24):
            rec[f"AvsB_h{h}"] = action_err(f["actB"], f["actA"], h)
            for o in (0, 1):
                rec[f"AvsGT{o}_h{h}"] = action_err(f["actA"], ep.qpos[a + o:a + o + h], h)
                rec[f"holdvsGT{o}_h{h}"] = action_err(np.repeat(ep.qpos[a:a + 1], h, 0), ep.qpos[a + o:a + o + h], h)
        rec["canvas_AvsB"] = canvas_agreement(f["canvasB"], f["canvasA"])
        rec["text_AvsB"] = text_similarity(f["codeB"], f["codeA"])
        nc, tl = model.n_cameras, f["latA"].shape[2] // model.n_cameras
        fut = [v * tl + j for v in range(nc) for j in range(1, tl)]
        rec["lat_AvsB"] = rel_mse(f["latB"][0, :, fut].float().numpy(), f["latA"][0, :, fut].float().numpy())
        w(rec)
    w({"kind": "episode_done", "task": ep.task, "ep": ep.index, "wall_s": round(time.time() - t_ep, 1),
       "mot_steps": n_mot_steps, "anchors": len(anchors)})
    w.close()
    open(done_flag, "w").write("ok\n")
    return {"anchors": len(anchors), "mot_steps": n_mot_steps, "wall_s": time.time() - t_ep}


# --------------------------------------------------------------------------------------------------------------
# E4: which parts of which view the action and the code actually depend on
# --------------------------------------------------------------------------------------------------------------

def _perturb(obs: torch.Tensor, view: int, kind: str, arg) -> torch.Tensor:
    x = obs.clone()
    v = x[:, :, view]                                       # [1,3,H,W]
    H, W = v.shape[-2:]
    if kind == "drop":
        x[:, :, view] = 0.0
    elif kind == "down":
        f = int(arg)
        small = F.interpolate(v, size=(H // f, W // f), mode="area")
        x[:, :, view] = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    elif kind == "occl":
        r, c, nr, ncol = arg
        h0, h1 = r * H // nr, (r + 1) * H // nr
        w0, w1 = c * W // ncol, (c + 1) * W // ncol
        x[:, :, view, h0:h1, w0:w1] = 0.0
    else:
        raise ValueError(kind)
    return x


def e4_perturbations(nc: int = 3, grid=(2, 3)) -> List[Dict]:
    out = []
    for v in range(nc):
        for path in ("both", "vae", "vit"):
            out.append({"view": v, "kind": "drop", "arg": None, "path": path})
        for f in (2, 4):
            out.append({"view": v, "kind": "down", "arg": f, "path": "both"})
        for r in range(grid[0]):
            for c in range(grid[1]):
                out.append({"view": v, "kind": "occl", "arg": (r, c, grid[0], grid[1]), "path": "both"})
    return out


def e4_episode(adapter, ep: Episode, anchors: List[int], out_dir: str, nis: int = 20, seed: int = 1234,
               instruction: Optional[str] = None) -> None:
    model = adapter.model
    instr = instruction or ep.instructions[0]
    w = Writer(os.path.join(out_dir, "e4", f"{ep.task}_ep{ep.index}.jsonl"))
    gen_len = adapter.gen_len
    for a in anchors:
        obs = ep.obs_dict(a)
        base_inp = adapter.inputs(obs, instr)
        rA = joint.joint_decode(model, base_inp, num_steps=nis, gen_len=gen_len, generator=_gen(model, seed, "E4", a))
        rB = joint.joint_decode(model, base_inp, num_steps=nis, gen_len=gen_len, generator=_gen(model, seed, "E4B", a))
        actA, actB = adapter.denorm(rA.action), adapter.denorm(rB.action)
        w({"kind": "e4_floor", "task": ep.task, "ep": ep.index, "a": a,
           "h8": action_err(actB, actA, 8), "h24": action_err(actB, actA, 24),
           "canvas": canvas_agreement(rB.canvas, rA.canvas), "text": text_similarity(rB.code_text, rA.code_text)})
        for p in e4_perturbations(model.n_cameras):
            pert = _perturb(base_inp.obs_video, p["view"], p["kind"], p["arg"])
            inp = joint.Inputs(**{**base_inp.__dict__})
            if p["path"] in ("both", "vae"):
                inp.obs_video = pert
            if p["path"] == "vit":
                inp.obs_video_vit = pert
            elif p["path"] == "vae":
                inp.obs_video_vit = base_inp.obs_video
            r = joint.joint_decode(model, inp, num_steps=nis, gen_len=gen_len, generator=_gen(model, seed, "E4", a))
            act = adapter.denorm(r.action)
            w({"task": ep.task, "ep": ep.index, "a": a, **{k: (list(v) if isinstance(v, tuple) else v)
                                                         for k, v in p.items()}, "kind": "e4", "pert": p["kind"],
               "h8": action_err(act, actA, 8), "h24": action_err(act, actA, 24),
               "canvas": canvas_agreement(r.canvas, rA.canvas), "text": text_similarity(r.code_text, rA.code_text)})
    w.close()


# --------------------------------------------------------------------------------------------------------------
# E5: latency, by stage, on the target GPU
# --------------------------------------------------------------------------------------------------------------

class _Skip(Exception):
    pass


class CudaTimer:
    def __init__(self, model):
        self.cuda = model.device.type == "cuda"

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.cuda:
            torch.cuda.synchronize()
        self.ms = (time.perf_counter() - self.t0) * 1e3


def _stats(xs: List[float]) -> Dict[str, float]:
    xs = sorted(xs)
    return {"n": len(xs), "p50": float(np.percentile(xs, 50)), "p95": float(np.percentile(xs, 95)),
            "min": float(xs[0]), "mean": float(np.mean(xs))}


def device_report() -> Dict:
    rep = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rep.update(device=p.name, capability=f"{p.major}.{p.minor}", total_gb=round(p.total_memory / 2 ** 30, 1),
                   cuda=torch.version.cuda, sm_count=p.multi_processor_count)
    rep["is_rtx_5090"] = "5090" in rep.get("device", "")
    return rep


def e5_latency(adapter, ep: Episode, out_dir: str, *, t: int = 0, reps: int = 20, warmup: int = 3,
               nis_list=(20, 6, 4), fast_steps=(1, 2, 4), compile_modes=True) -> Dict:
    model = adapter.model
    instr = ep.instructions[0]
    rep = {"device": device_report(), "dtype": str(next(model.action_expert.parameters()).dtype), "results": {}}
    R = rep["results"]
    inp = adapter.inputs(ep.obs_dict(t), instr)
    gen_len = adapter.gen_len

    def bench(name, fn, n=reps, w_=warmup):
        for _ in range(w_):
            fn()
        ts = []
        for _ in range(n):
            with CudaTimer(model) as tm:
                fn()
            ts.append(tm.ms)
        R[name] = _stats(ts)
        return R[name]

    only = os.environ.get("BRAINLOOP_E5_ONLY", "")            # "", "motor" or "fastjoint": one compiled path, fresh process
    if os.environ.get("BRAINLOOP_FIX_FREQS") == "1":          # the one-line fix: ActionDiT RoPE table onto the GPU
        R["action_freqs_device_before_fix"] = str(model.action_expert.freqs.device)
        model.action_expert.freqs = model.action_expert.freqs.to(model.device)
    if only:
        st = joint.prepare_static(model, inp, gen_len)
        r = joint.joint_decode(model, inp, num_steps=nis_list[0], gen_len=gen_len, generator=_gen(model, 0, "E5c"),
                               capture="final", static=st)
        torch.cuda.empty_cache()
        R.update(_e5_compiled(model, adapter, inp, st, r.cache, gen_len, nis_list, fast_steps, bench, only=only))
        tag = os.environ.get("BRAINLOOP_E5_TAG", only)
        with open(os.path.join(out_dir, f"e5_latency_{tag}.json"), "w") as fh:
            json.dump(rep, fh, indent=1)
        return rep
    # stages of the observation encode
    bench("vae_encode_obs_3views", lambda: joint.encode_window(model, inp.obs_video))
    bench("vit_prefix_3views", lambda: model._vit_prefix(inp.obs_video.to(model.device)))
    st = joint.prepare_static(model, inp, gen_len)
    # one joint MoT step (no scheduler) at the decode's token counts
    cdt = next(model.action_expert.parameters()).dtype
    lv = torch.randn(1, st.obs_latents.shape[1], model.n_cameras * st.Tlpv, *st.obs_latents.shape[3:],
                     device=model.device, dtype=cdt)
    la = torch.randn(1, 32, model.action_expert.action_dim, device=model.device, dtype=cdt)
    ids = torch.cat([torch.tensor([st.pids], device=model.device),
                     torch.full((1, gen_len), model.code_expert.mask_token_id, device=model.device)], 1)
    sig = torch.tensor(0.5, device=model.device, dtype=cdt)
    ta = torch.tensor([500.0], device=model.device, dtype=cdt)
    with joint.autocast_ctx(model), torch.no_grad():
        _, _, _, mask = joint._forward(model, st, lv, la, ids, sig, ta, None, None)
    R["tokens"] = {"video": int(mask.shape[0] - ids.shape[1] - st.n_prefix - 32), "code": int(ids.shape[1] + st.n_prefix),
                   "vit_prefix": st.n_prefix, "prompt": st.plen, "canvas": gen_len, "action": 32}
    with joint.autocast_ctx(model), torch.no_grad():
        bench("joint_mot_step", lambda: joint._forward(model, st, lv, la, ids, sig, ta, mask, None), n=reps)
    # full decodes: lab engine (eager) and the model's own method (eager), per step count
    for nis in nis_list:
        bench(f"lab_decode_eager_nis{nis}_excl_encode", lambda: joint.joint_decode(
            model, inp, num_steps=nis, gen_len=gen_len, generator=_gen(model, 0, "E5"), static=st), n=max(5, reps // 2))
        bench(f"model_decode_eager_nis{nis}", lambda: model._infer_action_joint(
            inp.obs_video, inp.proprio, inp.prompt_embs, 32, num_video_frames=9, num_inference_steps=nis,
            instruction=inp.code_instruction, max_code_tokens=gen_len), n=max(5, reps // 2))
    # motor tier: action-only steps against a final-step cache, eager
    r = joint.joint_decode(model, inp, num_steps=nis_list[0], gen_len=gen_len, generator=_gen(model, 0, "E5c"),
                           capture="final", static=st)
    R["cache_tokens"] = r.cache.seq_len
    R["cache_mb"] = round(r.cache.nbytes() / 2 ** 20, 1)
    for n in fast_steps:
        bench(f"motor_eager_n{n}", lambda: joint.fast_action(model, r.cache, inp.prompt_embs, inp.proprio,
                                                             num_steps=n, generator=_gen(model, 0, "E5m")))
    if compile_modes and model.device.type == "cuda":
        R.update(_e5_compiled(model, adapter, inp, st, r.cache, gen_len, nis_list, fast_steps, bench))
    tag = "bf16" if "bfloat16" in rep["dtype"] else ("fp32" if "float32" in rep["dtype"] else "other")
    with open(os.path.join(out_dir, f"e5_latency_{tag}.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    return rep


def _e5_compiled(model, adapter, inp, st, cache, gen_len, nis_list, fast_steps, bench, only: str = "") -> Dict:
    """CUDA-graph (torch.compile reduce-overhead) versions: the repo's fast_joint decode, and the motor tier."""
    out = {}
    from agenticwam.models.experts.wan_video_dit import flash_attention  # noqa: F401  (import before compile)
    import importlib
    import sys
    from .policy_io import REPO
    sys.path.insert(0, str(REPO / "deploy" / "common"))
    try:
        if only == "motor":
            raise _Skip()
        fji = importlib.import_module("fast_joint_infer")
        t0 = time.perf_counter()
        fji.install(model)
        for nis in nis_list:
            t1 = time.perf_counter()
            model._infer_action_joint(inp.obs_video, inp.proprio, inp.prompt_embs, 32, num_video_frames=9,
                                      num_inference_steps=nis, instruction=inp.code_instruction, max_code_tokens=gen_len)
            torch.cuda.synchronize()
            out[f"fastjoint_first_call_s_nis{nis}"] = round(time.perf_counter() - t1, 2)
            bench(f"fastjoint_compiled_nis{nis}", lambda: model._infer_action_joint(
                inp.obs_video, inp.proprio, inp.prompt_embs, 32, num_video_frames=9, num_inference_steps=nis,
                instruction=inp.code_instruction, max_code_tokens=gen_len))
        out["fastjoint_install_s"] = round(time.perf_counter() - t0, 2)
        model._infer_action_joint = model._infer_action_joint_orig
    except _Skip:
        pass
    except Exception as exc:                                   # report, do not hide
        out["fastjoint_error"] = repr(exc)
    if only == "fastjoint":
        return out
    try:
        mot = model.mot
        ks, vs = list(cache.k), list(cache.v)
        # ActionDiT keeps its RoPE table as a plain CPU attribute (not a buffer), so every pre_dit copies it
        # host->device; a CUDA graph cannot contain that copy. Move it once (also recorded as a finding).
        out["action_freqs_device"] = str(model.action_expert.freqs.device)
        model.action_expert.freqs = model.action_expert.freqs.to(model.device)
        cdt = next(model.action_expert.parameters()).dtype

        from .kvmot import action_only_forward
        nl = len(ks)

        def motor_step(latents_action, t_a, d_a, context, context_mask, *kv):
            c = (kv[:nl], kv[nl:])
            pre = model.action_expert.pre_dit(action_tokens=latents_action, timestep=t_a, context=context,
                                              context_mask=context_mask)
            aux = {"t_mod": pre["t_mod"], "freqs": pre["freqs"],
                   "context": {"context": pre["context"], "mask": pre["context_mask"]}}
            x = action_only_forward(mot, pre["tokens"], aux, c)
            return latents_action + model.action_expert.post_dit(x, pre) * d_a

        cstep = torch.compile(motor_step, mode="reduce-overhead", fullgraph=True)
        with joint.autocast_ctx(model), torch.no_grad():
            context, context_mask = joint.action_context(model, inp.prompt_embs, inp.proprio)
            for n in fast_steps:
                ts_a, deltas = model.action_scheduler.build_inference_schedule(n, model.device, cdt)
                la0 = torch.randn(1, 32, model.action_expert.action_dim, device=model.device, dtype=cdt)

                def run(n=n, ts_a=ts_a, deltas=deltas):
                    la = la0
                    for i in range(n):
                        la = cstep(la, ts_a[i].reshape(1), deltas[i], context, context_mask, *ks, *vs)
                    return la

                t1 = time.perf_counter()
                run()
                torch.cuda.synchronize()
                out[f"motor_compiled_first_call_s_n{n}"] = round(time.perf_counter() - t1, 2)
                bench(f"motor_compiled_n{n}", run)
                # exactness of the compiled step vs eager (same noise, same cache)
                le = la0
                for i in range(n):
                    le = motor_step(le, ts_a[i].reshape(1), deltas[i], context, context_mask, *ks, *vs)
                out[f"motor_compiled_vs_eager_maxabs_n{n}"] = float((run() - le).abs().max())
    except Exception as exc:
        out["motor_compile_error"] = repr(exc)
    return out


# --------------------------------------------------------------------------------------------------------------
# B3: chained warm starts -- the persistent belief over many replans, not one hop from a cold decode
# --------------------------------------------------------------------------------------------------------------

def chain_episode(adapter, ep: Episode, out_dir: str, *, hop: int = 8, hops: int = 10, warm_steps=(1, 2),
                  rho: float = 0.5, nis: int = 20, start_every: int = 120, seed: int = 1234,
                  instruction: Optional[str] = None) -> Dict:
    """From a cold decode at s, replan every `hop` steps for `hops` replans, each warm-started from the previous
    replan (video/action shifted by `hop`, the lowest-confidence `rho` of the code re-masked, confidences carried
    forward). Every replan is compared with a fresh 20-step decode at the same moment, and with a cold decode of
    the same step budget."""
    model = adapter.model
    instr = instruction or ep.instructions[0]
    gen_len = adapter.gen_len
    mid = model.code_expert.mask_token_id
    path = os.path.join(out_dir, "chain", f"{ep.task}_ep{ep.index}.jsonl")
    if os.path.exists(path + ".done"):
        return {"skipped": True}
    if os.path.exists(path):
        os.remove(path)
    w = Writer(path)
    span = hop * hops
    starts = list(range(0, ep.T - 1 - 32 - span + 1, start_every))
    fresh: Dict[int, Dict] = {}

    def get_fresh(t):
        if t not in fresh:
            inp = adapter.inputs(ep.obs_dict(t), instr)
            st = joint.prepare_static(model, inp, gen_len)
            rA = joint.joint_decode(model, inp, num_steps=nis, gen_len=gen_len, generator=_gen(model, seed, "chA", t),
                                    static=st)
            rB = joint.joint_decode(model, inp, num_steps=nis, gen_len=gen_len, generator=_gen(model, seed, "chB", t),
                                    static=st)
            fresh[t] = {"inp": inp, "st": st, "rA": rA, "actA": adapter.denorm(rA.action),
                        "actB": adapter.denorm(rB.action), "codeA": rA.code_text, "canvasA": rA.canvas,
                        "latA": rA.latents_video.detach().cpu(), "latB": rB.latents_video.detach().cpu()}
        return fresh[t]

    nc = model.n_cameras
    for s in starts:
        base = get_fresh(s)
        for n in warm_steps:
            prev = base["rA"]
            conf = prev.commit_conf.clone()
            for i in range(1, hops + 1):
                t = s + hop * i
                f = get_fresh(t)
                tl = f["st"].Tlpv
                warm = {"start": nis - n, "video": shift_video(prev.latents_video, nc, tl, hop),
                        "action": shift_action(prev.latents_action, hop),
                        "canvas": remask(prev.canvas, conf, rho, mid)}
                rw = joint.joint_decode(model, f["inp"], num_steps=nis, gen_len=gen_len,
                                        generator=_gen(model, seed, "chW", s, n, i), warm=warm, static=f["st"])
                rc = joint.joint_decode(model, f["inp"], num_steps=n, gen_len=gen_len,
                                        generator=_gen(model, seed, "chC", t, n), static=f["st"])
                newc = rw.commit_conf.clone()
                keep = ~torch.isfinite(newc)
                newc[keep] = conf[keep]
                conf = newc
                fut = [v * tl + j for v in range(nc) for j in range(1, tl)]
                aw, ac = adapter.denorm(rw.action), adapter.denorm(rc.action)
                w({"kind": "chain", "task": ep.task, "ep": ep.index, "start": s, "steps": n, "hop": i, "t": t,
                   "warm_h8": action_err(aw, f["actA"], 8), "cold_h8": action_err(ac, f["actA"], 8),
                   "floor_h8": action_err(f["actB"], f["actA"], 8),
                   "warm_gt_h8": action_err(aw, ep.qpos[t + 1:t + 9], 8), "fresh_gt_h8": action_err(f["actA"], ep.qpos[t + 1:t + 9], 8),
                   "warm_text": text_similarity(rw.code_text, f["codeA"]), "cold_text": text_similarity(rc.code_text, f["codeA"]),
                   "warm_canvas": canvas_agreement(rw.canvas, f["canvasA"]),
                   "warm_lat": rel_mse(rw.latents_video[0, :, fut].float().cpu().numpy(), f["latA"][0, :, fut].float().numpy()),
                   "floor_lat": rel_mse(f["latB"][0, :, fut].float().numpy(), f["latA"][0, :, fut].float().numpy())})
                prev = rw
        for t in [t for t in fresh if t < s]:          # bounded memory: earlier starts are finished
            fresh.pop(t, None)
    w({"kind": "chain_done", "task": ep.task, "ep": ep.index, "starts": starts})
    w.close()
    open(path + ".done", "w").write("ok\n")
    return {"starts": len(starts)}


# --------------------------------------------------------------------------------------------------------------
# B4 + B5: warm-start and compiled-step latency, end to end (observation encode included)
# --------------------------------------------------------------------------------------------------------------

def e5b_latency(adapter, ep: Episode, out_dir: str, *, only: str = "core", reps: int = 20, warmup: int = 3,
                compile_mode: Optional[str] = "reduce-overhead") -> Dict:
    """only="core": eager vs CUDA-graphed lab step (cold 20/4, warm 1/2), end to end per replan.
    only="fastjoint": the repo's fast_joint decode, now under no_grad and with the RoPE fix (20/4 steps).
    One mode per process: two CUDA-graph pools plus the model do not fit a 32 GB card."""
    model = adapter.model
    instr = ep.instructions[0]
    gen_len = adapter.gen_len
    rep = {"device": device_report(), "dtype": str(next(model.action_expert.parameters()).dtype), "mode": only,
           "action_freqs_device": str(model.action_expert.freqs.device), "results": {}}
    R = rep["results"]
    obs0, obs1 = ep.obs_dict(0), ep.obs_dict(min(8, ep.T - 1))
    inp0 = adapter.inputs(obs0, instr)

    def bench(name, fn, n=reps, w_=warmup):
        for _ in range(w_):
            fn()
        ts = []
        for _ in range(n):
            with CudaTimer(model) as tm:
                fn()
            ts.append(tm.ms)
        R[name] = _stats(ts)

    cuda = model.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        prev = joint.joint_decode(model, inp0, num_steps=20, gen_len=gen_len, generator=_gen(model, 0, "B4p"),
                                  capture="final" if only == "reflex" else None)
        mid = model.code_expert.mask_token_id

        def replan(n, warm, core_bind=None, total=20):
            """one replan, end to end: encode the new observation, then decode (warm from `prev` or cold)."""
            inp = adapter.inputs(obs1, instr)
            st = joint.prepare_static(model, inp, gen_len)
            core = core_bind(st) if core_bind else None
            if warm:
                tl = st.Tlpv
                wd = {"start": total - n, "video": shift_video(prev.latents_video, model.n_cameras, tl, 8),
                      "action": shift_action(prev.latents_action, 8), "canvas": remask(prev.canvas, prev.commit_conf, 0.5, mid)}
                return joint.joint_decode(model, inp, num_steps=total, gen_len=gen_len, warm=wd, static=st, core=core,
                                          generator=_gen(model, 0, "B4w"))
            return joint.joint_decode(model, inp, num_steps=n, gen_len=gen_len, static=st, core=core,
                                      generator=_gen(model, 0, "B4c"))

        if only == "core":
            for n in (20, 4):
                bench(f"eager_cold{n}_e2e", lambda n=n: replan(n, False), n=max(5, reps // 2))
            for n in (1, 2):
                bench(f"eager_warm{n}_e2e", lambda n=n: replan(n, True))
            try:
                st0 = joint.prepare_static(model, inp0, gen_len)
                bind = joint.make_core(model, st0, gen_len, compile_mode=compile_mode)
                t1 = time.perf_counter()
                replan(2, True, bind)
                _sync(model)
                R["compiled_first_call_s"] = round(time.perf_counter() - t1, 2)
                core = bind(st0)
                cdt = next(model.action_expert.parameters()).dtype
                lv = torch.randn(1, st0.obs_latents.shape[1], model.n_cameras * st0.Tlpv, *st0.obs_latents.shape[3:],
                                 device=model.device, dtype=cdt)
                la = torch.randn(1, 32, model.action_expert.action_dim, device=model.device, dtype=cdt)
                ids = torch.cat([torch.tensor([st0.pids], device=model.device),
                                 torch.full((1, gen_len), mid, device=model.device)], 1)
                sv = torch.tensor(0.5, device=model.device, dtype=cdt)
                ta = torch.tensor([500.0], device=model.device, dtype=cdt)
                with joint.autocast_ctx(model):
                    bench("compiled_joint_step", lambda: core(lv, la, ids, sv, ta))
                for n in (20, 4):
                    bench(f"compiled_cold{n}_e2e", lambda n=n: replan(n, False, bind), n=max(5, reps // 2))
                for n in (1, 2):
                    bench(f"compiled_warm{n}_e2e", lambda n=n: replan(n, True, bind))
                a_e, a_c = adapter.denorm(replan(2, True).action), adapter.denorm(replan(2, True, bind).action)
                R["compiled_vs_eager_warm2_h8"] = action_err(a_c, a_e, 8)
            except Exception as exc:
                R["compiled_error"] = repr(exc)[:600]
        elif only == "reflex":
            # D2: the reflex tier end to end (eager), its pieces, and compiled cold-n decodes (n=8 was interpolated)
            def reflex(n):
                inp = adapter.inputs(obs1, instr)
                st = joint.prepare_static(model, inp, gen_len)
                vid = shift_video(prev.latents_video, model.n_cameras, st.Tlpv, 8)
                return joint.reflex_action(model, st, inp, vid, prev.canvas, prev.final_inputs["sigma_v"],
                                           num_steps=n, generator=_gen(model, 0, "D2r"))

            bench("eager_prepare_static", lambda: joint.prepare_static(model, adapter.inputs(obs1, instr), gen_len))
            st1 = joint.prepare_static(model, adapter.inputs(obs1, instr), gen_len)
            vid1 = shift_video(prev.latents_video, model.n_cameras, st1.Tlpv, 8)
            fi1 = {"latents_video": joint._clamp_cond(vid1.clone(), st1, model.n_cameras),
                   "code_ids": torch.cat([torch.tensor([st1.pids]), prev.canvas.view(1, -1).long().cpu()], 1),
                   "sigma_v": prev.final_inputs["sigma_v"]}
            bench("eager_rebuild_cache", lambda: joint.rebuild_cache(model, st1, fi1))
            for n in (2, 4):
                bench(f"eager_reflex{n}_e2e", lambda n=n: reflex(n))
            try:
                st0 = joint.prepare_static(model, inp0, gen_len)
                bind = joint.make_core(model, st0, gen_len, compile_mode=compile_mode)
                replan(2, False, bind)
                core = bind(st0)
                cdt = next(model.action_expert.parameters()).dtype
                lv = torch.randn(1, st0.obs_latents.shape[1], model.n_cameras * st0.Tlpv, *st0.obs_latents.shape[3:],
                                 device=model.device, dtype=cdt)
                la = torch.randn(1, 32, model.action_expert.action_dim, device=model.device, dtype=cdt)
                ids = torch.cat([torch.tensor([st0.pids], device=model.device),
                                 torch.full((1, gen_len), mid, device=model.device)], 1)
                sv = torch.tensor(0.5, device=model.device, dtype=cdt)
                ta = torch.tensor([500.0], device=model.device, dtype=cdt)
                with joint.autocast_ctx(model):
                    bench("compiled_joint_step", lambda: core(lv, la, ids, sv, ta))
                for n in (2, 3, 4, 8, 12):
                    bench(f"compiled_cold{n}_e2e", lambda n=n: replan(n, False, bind), n=max(5, reps // 2))
            except Exception as exc:
                R["compiled_error"] = repr(exc)[:600]
        elif only == "fastjoint":
            import importlib
            import sys
            from .policy_io import REPO
            sys.path.insert(0, str(REPO / "deploy" / "common"))
            try:
                fji = importlib.import_module("fast_joint_infer")
                fji.install(model)
                for n in (20, 4):
                    t1 = time.perf_counter()
                    model._infer_action_joint(inp0.obs_video, inp0.proprio, inp0.prompt_embs, 32, num_video_frames=9,
                                              num_inference_steps=n, instruction=inp0.code_instruction,
                                              max_code_tokens=gen_len)
                    torch.cuda.synchronize()
                    R[f"fastjoint_first_call_s_nis{n}"] = round(time.perf_counter() - t1, 2)
                    bench(f"fastjoint_nis{n}_e2e", lambda n=n: model._infer_action_joint(
                        inp0.obs_video, inp0.proprio, inp0.prompt_embs, 32, num_video_frames=9, num_inference_steps=n,
                        instruction=inp0.code_instruction, max_code_tokens=gen_len))
            except Exception as exc:
                R["fastjoint_error"] = repr(exc)[:600]
    R["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if cuda else 0.0
    with open(os.path.join(out_dir, f"e5b_{only}.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    return rep
