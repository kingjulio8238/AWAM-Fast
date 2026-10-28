"""C4 offline evaluation on held-out demonstrations (never used to train T1/T2/T3).

E1-T1: the motor tier (2 action steps) from a k-steps-stale warm-1 cache with fresh proprio, base vs T1 weights,
       against the fresh 20-step decode at a+k (D2) and against the demonstration. Open-loop playback as baseline.
B3-T2: warm-1 replans chained along held-out episodes, video init from shift_video() vs from T2, against D2.
Latency: the motor step with the staleness input and the T2 forward pass.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict

import numpy as np
import torch

from . import joint, staleness
from .demos import open_episode
from .experiments import CudaTimer, _gen, _stats, remask, shift_action, shift_video
from .metrics import action_err
from .train_c import GT_OFFSET, _fi, _load_dir, _static_from_d1, apply_shift, load_motor, load_shift

SEED_FLOOR_STAGE_B = 0.0055     # rad, two fresh 20-step decodes, same checkpoint (Stage B, 387 anchors)


def eval_motor(adapter, d1_dir, d2_dir, demos_root, motor_ckpt, out_dir, *, heldout_from=16, ks=(0, 8, 16, 24),
               max_anchors=200) -> Dict:
    model = adapter.model
    d1 = {(d["task"], d["ep"]): d for d in _load_dir(d1_dir) if d["ep"] >= heldout_from}
    d2 = {(d["task"], d["ep"]): {r["a"]: r for r in d["recs"]} for d in _load_dir(d2_dir)}
    items = []
    for key, d in d1.items():
        if key not in d2:
            continue
        for r in d["recs"]:
            if any(r["a"] + k in d2[key] for k in ks):
                items.append((key, r))
    items = items[:max_anchors]
    rows = []
    variants = ("base", "warm4s3", "warm2s1", "warm8s6") + (("t1",) if motor_ckpt and os.path.exists(motor_ckpt) else ())
    for variant in variants:
        if variant == "t1":
            load_motor(model, motor_ckpt)
        for key, r in items:
            ep = open_episode(demos_root, *key)
            _, st = _static_from_d1(adapter, ep, r)
            cache = joint.rebuild_cache(model, st, _fi(r))
            old = adapter.denorm(r["action"].float())
            for k in ks:
                f = d2[key].get(r["a"] + k)
                if f is None:
                    continue
                fresh = adapter.denorm(f["action"].float())
                prop = adapter.proprio(ep.qpos[r["a"] + k])
                if variant.startswith("warm"):          # zero-training warm motor: correct the shifted old plan
                    n, s0 = (int(x) for x in variant[4:].split("s"))
                    init = shift_action(r["latents_action_out"].float(), k)
                    af = adapter.denorm(joint.fast_action(model, cache, st.prompt_embs, prop, num_steps=n, start=s0,
                                                          init=init, generator=_gen(model, 7, key, r["a"], k)))
                else:
                    af = adapter.denorm(joint.fast_action(model, cache, st.prompt_embs, prop, num_steps=2,
                                                          generator=_gen(model, 7, key, r["a"], k),
                                                          stale_k=k if variant == "t1" else 0))
                gt = ep.qpos[r["a"] + k + GT_OFFSET: r["a"] + k + GT_OFFSET + 8]
                rows.append({"variant": variant, "task": key[0], "ep": key[1], "a": r["a"], "k": k,
                             "motor_vs_fresh": action_err(af, fresh, 8)["arm"],
                             "openloop_vs_fresh": action_err(old[k:], fresh, 8)["arm"],
                             "motor_vs_gt": action_err(af, gt, 8)["arm"], "fresh_vs_gt": action_err(fresh, gt, 8)["arm"]})
    summ = {}
    for v in variants:
        for k in ks:
            rr = [x for x in rows if x["variant"] == v and x["k"] == k]
            if rr:
                summ[f"{v}_k{k}"] = {m: float(np.median([x[m] for x in rr])) for m in
                                     ("motor_vs_fresh", "openloop_vs_fresh", "motor_vs_gt", "fresh_vs_gt")}
                summ[f"{v}_k{k}"]["n"] = len(rr)
    t1_8 = summ.get("t1_k8", {})
    summ["verdict_T1"] = {"pass": bool(t1_8 and t1_8["motor_vs_fresh"] <= 1.5 * SEED_FLOOR_STAGE_B
                                       and t1_8["motor_vs_fresh"] < t1_8["openloop_vs_fresh"]),
                          "seed_floor": SEED_FLOOR_STAGE_B}
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(out_dir, "eval_motor.json"), "w"), indent=1)
    return summ


def eval_chain(adapter, d1_dir, d2_dir, demos_root, shift_ckpt, out_dir, *, eval_from=18, stride=8, hops=10) -> Dict:
    """Chained warm-1 replans on held-out episodes; video init from shift_video vs T2."""
    model = adapter.model
    net, scale = load_shift(shift_ckpt, model.device) if shift_ckpt else (None, None)
    d2 = {(d["task"], d["ep"]): {r["a"]: r for r in d["recs"]} for d in _load_dir(d2_dir) if d["ep"] >= eval_from}
    gl = adapter.gen_len
    mid = model.code_expert.mask_token_id
    rows = []
    for key, by_a in d2.items():
        ep = open_episode(demos_root, *key)
        anchors = sorted(by_a)
        for init in ("shift", "t2") if net is not None else ("shift",):
            prev = None
            for hop, a in enumerate(anchors[: hops + 1]):
                inp = adapter.inputs(ep.obs_dict(a), ep.instructions[0])
                st = joint.prepare_static(model, inp, gl)
                if prev is None:
                    prev = joint.joint_decode(model, inp, num_steps=20, gen_len=gl, static=st,
                                              generator=_gen(model, 9, key, "cold"))
                    continue
                vid = (apply_shift(net, scale, prev.latents_video, st.obs_latents, stride) if init == "t2"
                       else shift_video(prev.latents_video, model.n_cameras, st.Tlpv, stride))
                warm = {"start": 19, "video": vid, "action": shift_action(prev.latents_action, stride),
                        "canvas": remask(prev.canvas, prev.commit_conf, 0.5, mid)}
                r = joint.joint_decode(model, inp, num_steps=20, gen_len=gl, static=st, warm=warm,
                                       generator=_gen(model, 9, key, a))
                keep = ~torch.isfinite(r.commit_conf)
                r.commit_conf[keep] = prev.commit_conf[keep]
                fresh = adapter.denorm(by_a[a]["action"].float())
                rows.append({"init": init, "task": key[0], "ep": key[1], "hop": hop, "a": a,
                             "warm1_vs_fresh": action_err(adapter.denorm(r.action), fresh, 8)["arm"]})
                prev = r
    summ = {}
    for init in ("shift", "t2"):
        rr = [x for x in rows if x["init"] == init]
        if rr:
            summ[init] = {"median": float(np.median([x["warm1_vs_fresh"] for x in rr])),
                          "last_hop": float(np.median([x["warm1_vs_fresh"] for x in rr if x["hop"] == max(y["hop"] for y in rr)])),
                          "n": len(rr)}
            per_task = {}
            for t in sorted({x["task"] for x in rr}):
                per_task[t] = float(np.median([x["warm1_vs_fresh"] for x in rr if x["task"] == t]))
            summ[init]["per_task"] = per_task
    if "t2" in summ:
        pt = summ["t2"]["per_task"]
        summ["verdict_T2"] = {"pass": sum(v <= 1.5 * SEED_FLOOR_STAGE_B for v in pt.values()) >= max(1, round(5 / 6 * len(pt))),
                              "tasks_within": f"{sum(v <= 1.5 * SEED_FLOOR_STAGE_B for v in pt.values())}/{len(pt)}"}
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(out_dir, "eval_chain.json"), "w"), indent=1)
    return summ


def latency_c(adapter, ep, motor_ckpt, shift_ckpt, out_dir, reps=20) -> Dict:
    model = adapter.model
    R = {}
    motor_ckpt = motor_ckpt if motor_ckpt and os.path.exists(motor_ckpt) else None
    shift_ckpt = shift_ckpt if shift_ckpt and os.path.exists(shift_ckpt) else None
    if motor_ckpt:
        load_motor(model, motor_ckpt)
    inp = adapter.inputs(ep.obs_dict(0), ep.instructions[0])
    r = joint.joint_decode(model, inp, num_steps=20, gen_len=adapter.gen_len, capture="final")

    def bench(name, fn):
        for _ in range(3):
            fn()
        ts = []
        for _ in range(reps):
            with CudaTimer(model) as tm:
                fn()
            ts.append(tm.ms)
        R[name] = _stats(ts)

    bench("motor_eager_n2_stale8", lambda: joint.fast_action(model, r.cache, inp.prompt_embs, inp.proprio, num_steps=2,
                                                             stale_k=8 if motor_ckpt else 0))
    if shift_ckpt:
        net, scale = load_shift(shift_ckpt, model.device)
        bench("t2_shift_forward", lambda: apply_shift(net, scale, r.latents_video, r.static.obs_latents, 8))
    json.dump(R, open(os.path.join(out_dir, "latency_c.json"), "w"), indent=1)
    return R
