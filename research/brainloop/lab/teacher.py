"""C2 teacher data: what the deployed deliberation tier would see and produce along each demonstration.

  D1 (T1, the motor tier): along each episode, a cold 20-step decode at t=0, then warm 1-step replans every `stride`
     frames, each warm-started from the previous replan. That is the deliberation loop of config (b)/(c). Each record
     keeps the final step's inputs (exact cache rebuild via joint.rebuild_cache), the canvas and the action chunk.
  D2 (T2 time-shift / T3 surprise): fresh cold 20-step decodes at the same anchors on a subset of episodes, with the
     final video/action latents and the observation's own latents (the "what actually happened" side of surprise).

One .pt per episode under out_dir/{d1,d2}/<task>_ep<i>.pt; existing files are skipped (resumable).
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import torch

from . import joint
from .demos import Episode
from .experiments import _gen, remask, shift_action, shift_video


def _cpu(x):
    return x.detach().to("cpu") if torch.is_tensor(x) else x


def _bf16(x):
    return x.detach().to("cpu", torch.bfloat16)


def episode_d1(adapter, ep: Episode, out_dir: str, *, stride: int = 8, bind=None, seed: int = 4321) -> Optional[Dict]:
    path = os.path.join(out_dir, "d1", f"{ep.task}_ep{ep.index}.pt")
    if os.path.exists(path):
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = adapter.model
    instr = ep.instructions[0]
    gl = adapter.gen_len
    mid = model.code_expert.mask_token_id
    recs: List[Dict] = []
    prev = None
    t0 = time.time()
    for a in range(0, ep.T - 1 - 32 + 1, stride):
        inp = adapter.inputs(ep.obs_dict(a), instr)
        st = joint.prepare_static(model, inp, gl)
        core = bind(st) if bind else None
        if prev is None:
            r = joint.joint_decode(model, inp, num_steps=20, gen_len=gl, static=st, core=core, capture="final",
                                   generator=_gen(model, seed, "d1c", ep.task, ep.index))
        else:
            warm = {"start": 19, "video": shift_video(prev.latents_video, model.n_cameras, st.Tlpv, stride),
                    "action": shift_action(prev.latents_action, stride),
                    "canvas": remask(prev.canvas, prev.commit_conf, 0.5, mid)}
            r = joint.joint_decode(model, inp, num_steps=20, gen_len=gl, static=st, core=core, capture="final",
                                   warm=warm, generator=_gen(model, seed, "d1w", ep.task, ep.index, a))
            # carry confidences of tokens the warm replan kept
            keep = ~torch.isfinite(r.commit_conf)
            r.commit_conf[keep] = prev.commit_conf[keep]
        fi = r.final_inputs
        recs.append({"a": a, "latents_video_in": _bf16(fi["latents_video"]), "code_ids_in": _cpu(fi["code_ids"]).int(),
                     "sigma_v": float(fi["sigma_v"]), "canvas": _cpu(r.canvas).int(), "action": _cpu(r.action).half(),
                     "latents_video_out": _bf16(r.latents_video), "latents_action_out": _bf16(r.latents_action),
                     "commit_conf": _cpu(r.commit_conf).half(),
                     "plen": r.static.plen, "n_prefix": r.static.n_prefix, "code": r.code_text})
        prev = r
    torch.save({"task": ep.task, "ep": ep.index, "instruction": instr, "T": ep.T, "stride": stride, "recs": recs,
                "qpos": torch.from_numpy(ep.qpos)}, path)
    return {"anchors": len(recs), "s": round(time.time() - t0, 1)}


def episode_d2(adapter, ep: Episode, out_dir: str, *, stride: int = 8, bind=None, seed: int = 4321) -> Optional[Dict]:
    path = os.path.join(out_dir, "d2", f"{ep.task}_ep{ep.index}.pt")
    if os.path.exists(path):
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = adapter.model
    instr = ep.instructions[0]
    gl = adapter.gen_len
    recs: List[Dict] = []
    t0 = time.time()
    for a in range(0, ep.T - 1 - 32 + 1, stride):
        inp = adapter.inputs(ep.obs_dict(a), instr)
        st = joint.prepare_static(model, inp, gl)
        core = bind(st) if bind else None
        r = joint.joint_decode(model, inp, num_steps=20, gen_len=gl, static=st, core=core,
                               generator=_gen(model, seed, "d2", ep.task, ep.index, a))
        recs.append({"a": a, "latents_video": _bf16(r.latents_video), "latents_action": _bf16(r.latents_action),
                     "obs_latents": _bf16(st.obs_latents), "canvas": _cpu(r.canvas).int(),
                     "action": _cpu(r.action).half(), "code": r.code_text})
    torch.save({"task": ep.task, "ep": ep.index, "instruction": instr, "T": ep.T, "stride": stride, "recs": recs,
                "qpos": torch.from_numpy(ep.qpos)}, path)
    return {"anchors": len(recs), "s": round(time.time() - t0, 1)}
