"""C3 trainers.

T1 motor tier: fine-tune the action expert (+ the zero-initialised staleness embedding) to act from a code/video cache
   that is k steps old, with fresh proprio. The cache is rebuilt exactly from D1 records (the warm-1 deliberation loop),
   and the target is the demonstration's own action chunk at a+k. Standard flow-matching loss, as in training.
   Everything except the action expert stays frozen.
T2 time-shift: a small 3D-conv net that advances the previous replan's video latents by k steps, given the new
   observation's latents. Target: the fresh 20-step decode at a+k (D2). It replaces shift_video() in the warm start.
T3 surprise: a small MLP on "foresight vs observation" features. Target: whether the plan made at a is invalid at
   a+k, i.e. open-loop error vs the fresh decode above threshold.

Split by demonstration index: T1 trains on demos < `heldout_from`; T2/T3 train on D2 episodes < `t23_eval_from`.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import joint, staleness
from .demos import open_episode
from .kvmot import action_only_forward
from .metrics import auroc

GT_OFFSET = 0          # diag_t1.py: base flow-matching loss vs the demo chunk is lower at offset 0 (0.0045) than 1 (0.0051)


def _load_dir(d) -> List[Dict]:
    return [torch.load(p, map_location="cpu", weights_only=False) for p in sorted(glob.glob(os.path.join(d, "*.pt")))]


def norm_action(adapter, qpos_chunk: np.ndarray) -> torch.Tensor:
    p = adapter.policy
    key = p.processor.shape_meta["action"][0]["key"]
    nz = p.processor.normalizer.normalizers["action"][key]
    return nz.forward(torch.as_tensor(qpos_chunk, dtype=torch.float32))


def _static_from_d1(adapter, ep, rec):
    inp = adapter.inputs(ep.obs_dict(rec["a"]), ep.instructions[0])
    return inp, joint.prepare_static(adapter.model, inp, adapter.gen_len)


def _fi(rec):
    return {"latents_video": rec["latents_video_in"], "code_ids": rec["code_ids_in"].long(),
            "sigma_v": torch.tensor(rec["sigma_v"])}


# ----------------------------------------------------------------------------------------------------------------
# T1
# ----------------------------------------------------------------------------------------------------------------

LIGHT_KEYS = ("time_embedding.", "time_projection.", ".modulation")
STALE_KEYS = ("time_embedding.stale.",)
# per-group learning rates: C1 showed that Adam at 1e-4 (light) or 1e-5 (full) on the PRETRAINED weights drifts them
# within a few hundred noisy steps (the motor got worse at every k, even k=0, and vs the demos too). New zero-init
# parameters can take a high rate; pretrained ones only a very low one.
LR_PRETRAINED_SCALE = 0.01


def train_motor(adapter, d1_dir: str, demos_root: str, out_dir: str, *, steps: int = 10000, batch: int = 8,
                lr: float = 1e-4, kmax: int = 24, heldout_from: int = 16, log_every: int = 50,
                save_every: int = 2000, seed: int = 0, max_minutes: float = 1e9, params: str = "light",
                warmup: int = 200) -> Dict:
    """params="light": only the staleness embedding, the timestep path and each block's AdaLN modulation train
    (C1: a full fine-tune at lr 1e-5 without warmup degraded the motor even at k=0 within 200 steps)."""
    torch.manual_seed(seed)
    random.seed(seed)
    model = adapter.model
    dev = model.device
    os.makedirs(out_dir, exist_ok=True)
    data = [d for d in _load_dir(d1_dir) if d["ep"] < heldout_from]
    if not data:
        raise RuntimeError(f"no D1 training episodes in {d1_dir}")
    eps = {(d["task"], d["ep"]): open_episode(demos_root, d["task"], d["ep"]) for d in data}
    pool = [(d, i) for d in data for i, r in enumerate(d["recs"]) if r["a"] + kmax + GT_OFFSET + 32 <= d["T"]]
    ae = model.action_expert
    staleness.install(ae, max_k=32)
    for p in model.parameters():
        p.requires_grad_(False)
    ae.float()
    n_train = 0
    new_p, old_p = [], []
    for name, p in ae.named_parameters():
        keys = {"full": ("",), "light": LIGHT_KEYS, "stale": STALE_KEYS}[params]
        on = any(k in name for k in keys)
        p.requires_grad_(on)
        if on:
            n_train += p.numel()
            (new_p if any(k in name for k in STALE_KEYS) else old_p).append(p)
    print(f"[T1] training {n_train / 1e6:.3f}M of {sum(p.numel() for p in ae.parameters()) / 1e6:.1f}M action-expert "
          f"params ({params}); lr new {lr:g}, pretrained {lr * LR_PRETRAINED_SCALE:g}", flush=True)
    ae.train()
    groups = [{"params": new_p, "lr": lr}]
    if old_p:
        groups.append({"params": old_p, "lr": lr * LR_PRETRAINED_SCALE})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s_: min(1.0, (s_ + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s_ / max(1, steps)))))
    sch = model.action_scheduler
    log = open(os.path.join(out_dir, "train_motor.jsonl"), "a")
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        d, i = random.choice(pool)
        rec = d["recs"][i]
        ep = eps[(d["task"], d["ep"])]
        with torch.no_grad():
            inp, st = _static_from_d1(adapter, ep, rec)
            cache = joint.rebuild_cache(model, st, _fi(rec))
        ks = torch.tensor([0] + random.choices(range(kmax + 1), k=batch - 1), dtype=torch.long)
        a = rec["a"]
        props = torch.cat([adapter.proprio(ep.qpos[a + int(k)]) for k in ks], 0)                     # [B,14]
        tgt = torch.stack([norm_action(adapter, ep.qpos[a + int(k) + GT_OFFSET: a + int(k) + GT_OFFSET + 32])
                           for k in ks], 0).to(dev)                                                    # [B,32,14]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            ctx, cmask = model._action_context(st.prompt_embs.expand(batch, -1, -1).float(),
                                               props.to(dev).float().unsqueeze(1))
            t = sch.sample_training_t(batch_size=batch, device=dev, dtype=torch.float32)
            noise = torch.randn_like(tgt)
            noisy = sch.add_noise(tgt, noise, t)
            target = sch.training_target(tgt, noise, t)
            staleness.set_k(ae, ks)
            pre = ae.pre_dit(action_tokens=noisy, timestep=t, context=ctx, context_mask=cmask)
            aux = {"t_mod": pre["t_mod"], "freqs": pre["freqs"],
                   "context": {"context": pre["context"], "mask": pre["context_mask"]}}
            x = action_only_forward(model.mot, pre["tokens"], aux, cache)
            pred = ae.post_dit(x, pre)
            w = sch.training_weight(t).to(pred.device).float().view(-1)
            loss = (((pred.float() - target.float()) ** 2).mean(dim=(1, 2)) * w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in ae.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
        if step % log_every == 0 or step == 1:
            rec_ = {"step": step, "loss": float(np.mean(hist[-log_every:])), "grad_norm": float(gn),
                    "s_per_step": round((time.time() - t0) / step, 3), "stale_norm": float(ae.time_embedding.stale.weight.norm())}
            log.write(json.dumps(rec_) + "\n"); log.flush()
            print(f"[T1] {rec_}", flush=True)
        if step % save_every == 0 or step == steps or (time.time() - t0) / 60 > max_minutes:
            torch.save({"action_expert": staleness.state_dict(ae), "step": step}, os.path.join(out_dir, "motor_t1.pt"))
            if (time.time() - t0) / 60 > max_minutes:
                break
    ae.eval()
    log.close()
    return {"steps": step, "final_loss": float(np.mean(hist[-200:])), "first_loss": float(np.mean(hist[:50])),
            "minutes": round((time.time() - t0) / 60, 1)}


def load_motor(model, path: str) -> None:
    staleness.install(model.action_expert, max_k=32)
    sd = torch.load(path, map_location="cpu")["action_expert"]
    dt = next(model.action_expert.parameters()).dtype
    model.action_expert.load_state_dict({k: v.to(dt) for k, v in sd.items()})


# ----------------------------------------------------------------------------------------------------------------
# T2
# ----------------------------------------------------------------------------------------------------------------

class ShiftNet(nn.Module):
    """Per view: [shifted future latents (2 frames) | new obs latent] -> residual on the shifted future latents."""

    def __init__(self, z: int = 48, width: int = 128):
        super().__init__()
        self.inp = nn.Conv3d(3 * z + 1, width, (1, 3, 3), padding=(0, 1, 1))
        self.body = nn.Sequential(*[nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(),
                                                  nn.Conv3d(width, width, (1, 3, 3), padding=(0, 1, 1)))
                                    for _ in range(4)])
        self.out = nn.Conv3d(width, 2 * z, (1, 3, 3), padding=(0, 1, 1))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.z = z

    def forward(self, fut_shifted, obs_lat, k):
        """fut_shifted [N,z,2,H,W], obs_lat [N,z,1,H,W], k [N] -> advanced future [N,z,2,H,W]."""
        N, z, _, H, W = fut_shifted.shape
        kk = (k.float() / 24.0).view(N, 1, 1, 1, 1).expand(N, 1, 1, H, W)
        x = torch.cat([fut_shifted.reshape(N, 2 * z, 1, H, W), obs_lat, kk], 1)
        h = self.inp(x)
        for blk in self.body:
            h = h + blk(h)
        return fut_shifted + self.out(h).reshape(N, z, 2, H, W)


def _views(lat, nc, tl):
    """[1,C,nc*tl,H,W] -> [nc,C,tl,H,W]"""
    C, H, W = lat.shape[1], lat.shape[3], lat.shape[4]
    return lat[0].reshape(C, nc, tl, H, W).permute(1, 0, 2, 3, 4)


def t23_pairs(d1_dir: str, d2_dir: str, ks=(8, 16)):
    """(D1 warm output at a, D2 fresh at a+k) for episodes present in both."""
    from .experiments import shift_video
    d1 = {(d["task"], d["ep"]): d for d in _load_dir(d1_dir)}
    out = []
    for d2 in _load_dir(d2_dir):
        key = (d2["task"], d2["ep"])
        if key not in d1:
            continue
        by_a1 = {r["a"]: r for r in d1[key]["recs"]}
        by_a2 = {r["a"]: r for r in d2["recs"]}
        for a, r1 in by_a1.items():
            for k in ks:
                if a + k in by_a2 and a in by_a2:
                    out.append({"task": key[0], "ep": key[1], "a": a, "k": k, "r1": r1, "fresh": by_a2[a + k],
                                "fresh_a": by_a2[a], "qpos": d1[key]["qpos"]})
    return out


def train_shift(pairs, out_dir: str, *, nc: int = 3, tl: int = 3, epochs: int = 30, lr: float = 3e-4,
                eval_from: int = 18, device: str = "cuda") -> Dict:
    from .experiments import shift_video
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    def tensors(ps):
        X, O, K, Y, B = [], [], [], [], []
        for p in ps:
            prev = p["r1"]["latents_video_out"].float()
            sh = shift_video(prev, nc, tl, p["k"])
            fv = _views(sh, nc, tl)[:, :, 1:]                  # [nc,C,2,H,W]
            ov = _views(p["fresh"]["obs_latents"].float(), nc, 1)
            yv = _views(p["fresh"]["latents_video"].float(), nc, tl)[:, :, 1:]
            X.append(fv); O.append(ov); Y.append(yv); K.append(torch.full((nc,), p["k"]))
            B.append(((fv - yv) ** 2).mean().item())
        return torch.cat(X), torch.cat(O), torch.cat(K), torch.cat(Y), float(np.mean(B))

    tr = [p for p in pairs if p["ep"] < eval_from]
    ev = [p for p in pairs if p["ep"] >= eval_from]
    if not tr or not ev:
        raise RuntimeError(f"T2 needs train and eval pairs (got {len(tr)}, {len(ev)})")
    Xt, Ot, Kt, Yt, base_tr = tensors(tr)
    Xe, Oe, Ke, Ye, base_ev = tensors(ev)
    scale = Yt.std().item()
    net = ShiftNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = Xt.shape[0]
    bs = 32
    hist = []
    for e in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            pred = net(Xt[j].to(dev) / scale, Ot[j].to(dev) / scale, Kt[j].to(dev))
            loss = F.mse_loss(pred, Yt[j].to(dev) / scale)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(j)
        with torch.no_grad():
            pe = torch.cat([net(Xe[i:i + 64].to(dev) / scale, Oe[i:i + 64].to(dev) / scale, Ke[i:i + 64].to(dev)).cpu()
                            for i in range(0, Xe.shape[0], 64)]) * scale
            ev_mse = float(((pe - Ye) ** 2).mean())
        hist.append({"epoch": e, "train_mse_scaled": tot / n, "eval_mse": ev_mse})
        print(f"[T2] {hist[-1]}", flush=True)
    torch.save({"net": net.state_dict(), "scale": scale}, os.path.join(out_dir, "shift_t2.pt"))
    res = {"pairs_train": len(tr), "pairs_eval": len(ev), "eval_mse_shift_video": base_ev,
           "eval_mse_t2": hist[-1]["eval_mse"], "improvement": 1 - hist[-1]["eval_mse"] / base_ev, "history": hist}
    json.dump(res, open(os.path.join(out_dir, "train_shift.json"), "w"), indent=1)
    return res


def load_shift(path: str, device):
    d = torch.load(path, map_location="cpu")
    net = ShiftNet()
    net.load_state_dict(d["net"])
    return net.to(device).eval(), d["scale"]


def apply_shift(net, scale, prev_latents, obs_latents, k, nc=3, tl=3):
    """Warm-start video init from T2: prev_latents [1,C,nc*tl,H,W], obs_latents [1,C,nc,H,W]."""
    from .experiments import shift_video
    sh = shift_video(prev_latents, nc, tl, k)
    fv = _views(sh.float(), nc, tl)[:, :, 1:]
    ov = _views(obs_latents.float(), nc, 1)
    dev = next(net.parameters()).device
    with torch.no_grad():
        out = net(fv.to(dev) / scale, ov.to(dev) / scale, torch.full((nc,), k, device=dev)) * scale   # [nc,C,2,H,W]
    full = _views(sh.float(), nc, tl).clone()
    full[:, :, 1:] = out.to(full.device)
    C, H, W = full.shape[1], full.shape[3], full.shape[4]
    return full.permute(1, 0, 2, 3, 4).reshape(1, C, nc * tl, H, W).to(prev_latents.dtype)


# ----------------------------------------------------------------------------------------------------------------
# T3
# ----------------------------------------------------------------------------------------------------------------

def surprise_features(r1, obs_lat_now, qpos_now, k, plan_qpos, nc=3, tl=3):
    """Foresight (the replan made k steps ago) vs what is observed now; plus where the arm is vs where the plan said.
    plan_qpos: that replan's action chunk, denormalized [32,14]."""
    lat = r1["latents_video_out"].float()
    j = 1 if k <= 16 else 2                                  # latent frame whose window contains "now"
    pv = _views(lat, nc, tl)[:, :, j]                       # [nc,C,H,W]
    ov = _views(obs_lat_now.float(), nc, 1)[:, :, 0]
    f_lat = (pv - ov).abs().mean(dim=(2, 3)).flatten()     # nc*C
    plan = torch.as_tensor(plan_qpos, dtype=torch.float32)
    f_arm = torch.zeros(14) if k < 1 else (plan[min(k - 1, 31)] - torch.as_tensor(qpos_now).float()).abs()
    return torch.cat([f_lat, f_arm, torch.tensor([k / 24.0])])


def train_surprise(pairs, out_dir: str, *, thr: float = 0.011, eval_from: int = 18, epochs: int = 200,
                   denorm=None) -> Dict:
    """Label = open-loop error (h8 arm, rad) of the plan from a, at a+k, vs the fresh decode at a+k > thr."""
    from .demos import ARM_DIMS
    os.makedirs(out_dir, exist_ok=True)
    X, y, split = [], [], []
    for p in pairs:
        k, a = p["k"], p["a"]
        q = p["qpos"][a + k]
        plan = denorm(p["r1"]["action"].float())
        X.append(surprise_features(p["r1"], p["fresh"]["obs_latents"], q, k, plan))
        old = plan[k:k + 8]
        new = denorm(p["fresh"]["action"].float())[:8]
        err = float(np.abs(old - new)[:, list(ARM_DIMS)].mean())
        y.append(int(err > thr))
        split.append(p["ep"] >= eval_from)
    X = torch.stack(X); y = torch.tensor(y).float(); split = torch.tensor(split)
    mu, sd = X[~split].mean(0), X[~split].std(0) + 1e-6
    Xn = (X - mu) / sd
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.SiLU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    pos = float(y[~split].mean().clamp(0.05, 0.95))
    for _ in range(epochs):
        logit = net(Xn[~split]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit, y[~split], pos_weight=torch.tensor((1 - pos) / pos))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = torch.sigmoid(net(Xn).squeeze(-1))
    res = {"n_train": int((~split).sum()), "n_eval": int(split.sum()), "pos_rate_train": float(y[~split].mean()),
           "pos_rate_eval": float(y[split].mean()), "auroc_train": auroc(s[~split].tolist(), y[~split].int().tolist()),
           "auroc_eval": auroc(s[split].tolist(), y[split].int().tolist()), "thr_rad": thr}
    torch.save({"net": net.state_dict(), "mu": mu, "sd": sd, "dim": X.shape[1]}, os.path.join(out_dir, "surprise_t3.pt"))
    json.dump(res, open(os.path.join(out_dir, "train_surprise.json"), "w"), indent=1)
    return res
