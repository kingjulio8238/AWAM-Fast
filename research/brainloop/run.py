#!/usr/bin/env python3
"""Brainloop lab entry point (run from the repo root with PYTHONPATH=src:.).

    python -u research/brainloop/run.py fetch   --assets /workspace/assets            # weights + release + demos
    python -u research/brainloop/run.py proof   --assets /workspace/assets --out /workspace/out
    python -u research/brainloop/run.py sweep   --assets ... --out ...                # E1 + E2 + E3 data
    python -u research/brainloop/run.py e4      --assets ... --out ...
    python -u research/brainloop/run.py e5      --assets ... --out ...
    python -u research/brainloop/run.py analyze --out ...
    python -u research/brainloop/run.py all     --assets ... --out ...                # proof -> sweep -> e4 -> e5 -> analyze
    python -u research/brainloop/run.py tiny    --assets <dir with release/ and robotwin/<task>> --out ...   # CPU plumbing

Every stage writes under --out and prints a terminal marker line `BRAINLOOP <stage> DONE` (or FAILED).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

TASKS = ["click_alarmclock", "beat_block_hammer", "stack_blocks_three", "handover_block", "open_microwave",
         "hanging_mug"]

WEIGHTS = {   # local dir name -> (repo id, allow_patterns or None for the whole repo)
    "Wan2.2-TI2V-5B": ("Wan-AI/Wan2.2-TI2V-5B", ["Wan2.2_VAE.pth"]),
    "Cosmos-Reason1-7B": ("nvidia/Cosmos-Reason1-7B", None),
    "SDAR-1.7B-Chat": ("JetLM/SDAR-1.7B-Chat", None),
    "Qwen3-VL-2B-Instruct": ("Qwen/Qwen3-VL-2B-Instruct", None),
}
RELEASE = ("miniFranka/agentic-wam-posttrain-robotwin-suit", ["wam.pt", "wam.yaml", "dataset_stats.json",
                                                              "task_skill_prompt.json", "manifest.json"])


def paths_for(assets: str) -> dict:
    w = Path(assets) / "weights"
    return {"wan_vae": str(w / "Wan2.2-TI2V-5B" / "Wan2.2_VAE.pth"), "text_encoder": str(w / "Cosmos-Reason1-7B"),
            "code_model": str(w / "SDAR-1.7B-Chat"), "code_vision": str(w / "Qwen3-VL-2B-Instruct")}


def marker(stage, ok=True, **kw):
    print(f"BRAINLOOP {stage} {'DONE' if ok else 'FAILED'} {json.dumps(kw, default=str)}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
def cmd_fetch(a):
    from huggingface_hub import hf_hub_download, snapshot_download
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")   # xet / hf_transfer stalled on two Vast hosts
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    t0 = time.time()
    root = Path(a.assets)
    for name, (repo, pats) in WEIGHTS.items():
        snapshot_download(repo, local_dir=str(root / "weights" / name), allow_patterns=pats, max_workers=8)
        print(f"[fetch] {repo} -> {name} ({time.time() - t0:.0f}s)", flush=True)
    rel = root / "release"
    for f in RELEASE[1]:
        hf_hub_download(RELEASE[0], f, local_dir=str(rel))
    print(f"[fetch] release ({time.time() - t0:.0f}s)", flush=True)
    import zipfile
    for t in a.tasks:
        dst = root / "robotwin" / t
        if (dst / "aloha-agilex_clean_50" / "data").exists():
            continue
        z = hf_hub_download("TianxingChen/RoboTwin2.0", f"dataset/{t}/aloha-agilex_clean_50.zip", repo_type="dataset",
                            local_dir=str(root / "robotwin_zips"))
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dst)
        print(f"[fetch] demos {t} ({time.time() - t0:.0f}s)", flush=True)
    _verify_manifest(rel)
    marker("fetch", seconds=round(time.time() - t0))


def _verify_manifest(rel: Path):
    import hashlib
    man = json.load(open(rel / "manifest.json"))
    for f in ("wam.pt", "dataset_stats.json", "task_skill_prompt.json"):
        want = man["files"][f]
        size = os.path.getsize(rel / f)
        if size != want["size_bytes"]:
            raise RuntimeError(f"{f}: size {size} != manifest {want['size_bytes']}")
        h = hashlib.sha256()
        with open(rel / f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != want["sha256"]:
            raise RuntimeError(f"{f}: sha256 mismatch vs manifest")
    print("[fetch] release files match manifest.json sha256", flush=True)


# ---------------------------------------------------------------------------------------------------------------
def _episodes(a):
    """--spec task:n,... (per-task episode counts) wins over --tasks x --episodes."""
    from research.brainloop.lab.demos import open_episode
    data = str(Path(a.assets) / "robotwin")
    if a.spec:
        pairs = [(x.split(":")[0], int(x.split(":")[1])) for x in a.spec.split(",")]
        return [open_episode(data, t, i) for t, n in pairs for i in range(n)]
    return [open_episode(data, t, i) for t in a.tasks for i in range(a.episodes)]


def _real_adapter(a, task, eps):
    """Precompute every needed instruction embedding (text encoder alone on the GPU), then build the policy."""
    import torch
    from research.brainloop.lab import policy_io
    paths = paths_for(a.assets)
    cache = str(Path(a.assets) / "text_cache")
    prompts = [policy_io.wrapped_instruction(ep.instructions[0]) for ep in eps]
    n = policy_io.precompute_text_embeds(prompts, paths["text_encoder"], cache)
    print(f"[text] {n} new embeddings ({len(set(prompts))} prompts)", flush=True)
    torch.cuda.empty_cache()
    policy, report = policy_io.build_release_policy(task=task, release_dir=str(Path(a.assets) / "release"),
                                                    text_cache_dir=cache, paths=paths, dtype=a.dtype)
    print("[policy] validated:", json.dumps(report["checks"]), flush=True)
    return policy_io.Adapter(policy, task), report


def _adapter(a, task, eps):
    if a.tiny:
        from research.brainloop.lab import policy_io
        return policy_io.build_tiny_adapter(task, str(Path(a.assets) / "release")), {"tiny": True}
    return _real_adapter(a, task, eps)


def cmd_proof(a):
    """Environment proof + the settings that decide validity, on one task, one episode."""
    import numpy as np
    import torch
    from research.brainloop.lab import joint
    from research.brainloop.lab.demos import check_channel_order, open_episode
    from research.brainloop.lab.experiments import device_report
    from research.brainloop.lab.metrics import action_err
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {"device": device_report(), "imports": {}}
    for mod in ("agenticwam.models.cosmos_mot.build", "transformers", "hydra", "h5py", "cv2", "einops", "safetensors",
                "accelerate", "av", "boto3", "matplotlib"):
        __import__(mod)
        rep["imports"][mod] = "ok"
    joint.assert_default_decode_env()
    task = a.tasks[0]
    data = str(Path(a.assets) / "robotwin")
    rep["channel_order"] = check_channel_order(data, task, 0)
    ep = open_episode(data, task, 0)
    adapter, rep["load"] = _adapter(a, task, [ep])
    model = adapter.model
    instr = ep.instructions[0]
    inp = adapter.inputs(ep.obs_dict(0), instr)
    gl = adapter.gen_len
    # 1. parity: the lab decode == the policy's own full path (deploy_policy._infer_action_chunk), same global seed
    if not a.tiny:
        torch.manual_seed(11)
        ref = adapter.policy._infer_action_chunk(ep.obs_dict(0), instr)
        torch.manual_seed(11)
        r = joint.joint_decode(model, inp, num_steps=20, gen_len=gl)
        mine = adapter.denorm(r.action)
        rep["parity_maxabs_rad"] = float(np.abs(mine - ref).max())
        if rep["parity_maxabs_rad"] > 1e-3:
            raise RuntimeError(f"lab decode != policy decode: max |diff| {rep['parity_maxabs_rad']:.2e} rad")
    # 2. determinism and the seed floor
    g = lambda s: (torch.Generator(device=model.device).manual_seed(s))  # noqa: E731
    r1 = joint.joint_decode(model, inp, num_steps=20, gen_len=gl, generator=g(1), capture="all")
    r1b = joint.joint_decode(model, inp, num_steps=20, gen_len=gl, generator=g(1))
    r2 = joint.joint_decode(model, inp, num_steps=20, gen_len=gl, generator=g(2))
    rep["determinism_maxabs"] = float((r1.action - r1b.action).abs().max())
    rep["seed_floor_h8"] = action_err(adapter.denorm(r2.action), adapter.denorm(r1.action), 8)
    # 3. motor-tier exactness at k=0 with per-step caches
    g3 = g(1)
    st = r1.static
    torch.randn(1, st.obs_latents.shape[1], model.n_cameras * st.Tlpv, *st.obs_latents.shape[3:],
                device=model.device, dtype=next(model.action_expert.parameters()).dtype, generator=g3)
    a_fast = joint.fast_action(model, None, inp.prompt_embs, inp.proprio, num_steps=20, generator=g3, caches=r1.caches)
    rep["motor_exact_maxabs_rad"] = float(np.abs(adapter.denorm(a_fast) - adapter.denorm(r1.action)).max())
    rep["motor_exact_vs_floor"] = rep["motor_exact_maxabs_rad"] / max(1e-9, rep["seed_floor_h8"]["arm"])
    del r1
    # 4. is the model doing anything? h8 error vs GT against holding the current pose, over a few anchors
    errs, holds = [], []
    for t in range(0, max(1, ep.T - 40), max(4, (ep.T - 40) // 4 or 4))[:4]:
        ri = joint.joint_decode(model, adapter.inputs(ep.obs_dict(t), instr), num_steps=20, gen_len=gl, generator=g(t))
        act = adapter.denorm(ri.action)
        best = min(action_err(act, ep.qpos[t + o:t + o + 8], 8)["arm"] for o in (0, 1))
        errs.append(best)
        holds.append(action_err(np.repeat(ep.qpos[t:t + 1], 8, 0), ep.qpos[t:t + 8], 8)["arm"])
    rep["gt_h8_model"], rep["gt_h8_hold"] = float(np.median(errs)), float(np.median(holds))
    rep["model_over_hold"] = rep["gt_h8_model"] / max(1e-9, rep["gt_h8_hold"])
    if not a.tiny and rep["model_over_hold"] > 0.8:
        raise RuntimeError(f"model is not tracking the demonstrations: error/hold = {rep['model_over_hold']:.2f}")
    # 5. timing of one 20-step eager decode -> cost projection for the full lab
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    joint.joint_decode(model, inp, num_steps=20, gen_len=gl, generator=g(5))
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    rep["decode20_s"] = time.time() - t1
    rep["peak_gpu_gb"] = (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else 0.0
    rep["projection"] = project(rep["decode20_s"], a)
    rep["code_sample"] = r2.code_text[:400]
    rep["wall_s"] = time.time() - t0
    json.dump(rep, open(out / "proof.json", "w"), indent=1, default=float)
    print(json.dumps({k: rep[k] for k in rep if k not in ("imports",)}, indent=1, default=float), flush=True)
    marker("proof", decode20_s=round(rep["decode20_s"], 2), projection=rep["projection"])


def project(decode20_s, a, eps_len=None):
    """GPU-hours for sweep + e4 + e5 from one measured 20-step decode, using each task's real episode lengths."""
    from research.brainloop.lab.demos import open_episode
    from research.brainloop.lab.experiments import SweepConfig, e4_perturbations
    cfg = SweepConfig()
    per_step = decode20_s / 20.0
    data = str(Path(a.assets) / "robotwin")
    steps = 0
    n_anchor = 0
    for t in a.tasks:
        for i in range(a.episodes):
            try:
                T = open_episode(data, t, i).T
            except Exception:
                T = 150
            n = len(range(0, T - 1 - cfg.horizon + 1, cfg.stride))
            n_anchor += n
            steps += n * 2 * cfg.nis                                                    # fresh A + B
            n_e2 = (n + cfg.e2_every - 1) // cfg.e2_every
            steps += n_e2 * len(cfg.warm_ks) * (sum(cfg.warm_steps) + 2 * len(cfg.warm_rho_extra) + sum(cfg.warm_steps))
            steps += n * len(cfg.ks) * sum(cfg.fast_steps) * 0.15                       # action-only ~15% of a step
    overhead = 1.35                                                                     # VAE enc/dec, data, compare
    sweep_h = steps * per_step * overhead / 3600
    e4_h = len(a.tasks) * 3 * (len(e4_perturbations()) + 2) * 20 * per_step * 1.2 / 3600
    e5_h = 0.25
    setup_h = 0.5                                                                       # image pull + downloads + load
    total = sweep_h + e4_h + e5_h + setup_h
    return {"anchors": n_anchor, "sweep_h": round(sweep_h, 2), "e4_h": round(e4_h, 2), "e5_h": e5_h,
            "setup_h": setup_h, "total_h": round(total, 2)}


def cmd_sweep(a):
    from research.brainloop.lab.experiments import SweepConfig, sweep_episode
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eps = _episodes(a)
    cfg = SweepConfig(max_anchors=a.max_anchors, decode_pixels=not a.no_pixels)
    prog = {"job": "sweep", "t_start": time.time(), "episodes_total": len(eps), "episodes_done": 0}
    adapter = None
    for ep in eps:
        if adapter is None:
            adapter, _ = _adapter(a, ep.task, eps)
        adapter.task = ep.task
        res = sweep_episode(adapter, ep, cfg, str(out), progress=prog)
        prog["episodes_done"] += 1
        prog["mot_steps_base"] = prog.get("mot_steps", 0)
        print(f"[sweep] {ep.task} ep{ep.index}: {res}", flush=True)
    from research.brainloop.lab.experiments import _progress
    _progress(str(out), **{**prog, "job": "sweep DONE"})
    marker("sweep", episodes=len(eps))


def cmd_e4(a):
    from research.brainloop.lab.experiments import e4_episode
    eps = [ep for ep in _episodes(a) if ep.index == 0]
    adapter, _ = _adapter(a, eps[0].task, eps)
    for ep in eps:
        adapter.task = ep.task
        T = ep.T - 33
        anchors = sorted({max(0, int(T * f) // 4 * 4) for f in (0.25, 0.5, 0.75)})[: a.e4_anchors]
        e4_episode(adapter, ep, anchors, a.out)
        print(f"[e4] {ep.task}: anchors {anchors}", flush=True)
    marker("e4", episodes=len(eps))


def cmd_e5(a):
    from research.brainloop.lab.experiments import e5_latency
    eps = _episodes(a)[:1]
    adapter, _ = _adapter(a, eps[0].task, eps)
    rep = e5_latency(adapter, eps[0], a.out, reps=a.reps, compile_modes=not a.no_compile)
    print(json.dumps(rep, indent=1, default=float), flush=True)
    marker("e5", device=rep["device"].get("device"))


def cmd_chain(a):
    from research.brainloop.lab.experiments import chain_episode
    eps = _episodes(a)
    adapter, _ = _adapter(a, eps[0].task, eps)
    for ep in eps:
        adapter.task = ep.task
        res = chain_episode(adapter, ep, a.out)
        print(f"[chain] {ep.task} ep{ep.index}: {res}", flush=True)
    marker("chain", episodes=len(eps))


def cmd_e5b(a):
    from research.brainloop.lab.experiments import e5b_latency
    eps = _episodes(a)[:1]
    adapter, _ = _adapter(a, eps[0].task, eps)
    rep = e5b_latency(adapter, eps[0], a.out, only=a.e5b_mode, reps=a.reps,
                      compile_mode=None if a.no_compile else "reduce-overhead")
    print(json.dumps(rep, indent=1, default=float), flush=True)
    marker("e5b", mode=a.e5b_mode)


ALL_TASKS = sorted(p.name for p in (REPO / "configs" / "robotwin" / "tasks").iterdir() if p.is_dir())


def cmd_teacher(a):
    """C2: D1 (warm-1 deliberation loop) on demos [0, d1_demos) of every task; D2 (fresh 20-step) on demos
    [d2_from, d1_demos) of every `d2_every`-th task."""
    from research.brainloop.lab import teacher
    from research.brainloop.lab.demos import open_episode
    from research.brainloop.lab.experiments import _progress
    data = str(Path(a.assets) / "robotwin")
    tasks = a.tasks if a.tasks != TASKS else ALL_TASKS
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plan = [(t, i, "d1") for j, t in enumerate(tasks) if j % a.d1_every == 0 for i in range(a.d1_from, a.d1_demos)]
    plan += [(t, i, "d2") for j, t in enumerate(tasks) if j % a.d2_every == 0 for i in range(a.d2_from, a.d1_demos)]
    if a.max_episodes:
        plan = plan[: a.max_episodes]
    adapter = None
    t0 = time.time()
    for n, (t, i, kind) in enumerate(plan, 1):
        ep = open_episode(data, t, i)
        if adapter is None:            # every instruction the plan will use gets its embedding up front
            adapter, _ = _adapter(a, t, [open_episode(data, tt, ii) for tt, ii, _ in plan])
        adapter.task = t
        fn = teacher.episode_d1 if kind == "d1" else teacher.episode_d2
        res = fn(adapter, ep, str(out))
        _progress(str(out), job="teacher", episodes_done=n, episodes_total=len(plan), ep_task=t, ep_index=i,
                  anchor=n, anchors=len(plan), elapsed_s=round(time.time() - t0), last_decode_s=res and res.get("s"))
        if n % 20 == 0 or n == len(plan):
            print(f"[teacher] {n}/{len(plan)} {t} ep{i} {kind}: {res} ({time.time() - t0:.0f}s)", flush=True)
    marker("teacher", episodes=len(plan), seconds=round(time.time() - t0))


def _out_tasks(a):
    """(task, ep) pairs that have teacher data, for the text-embedding precompute."""
    import glob as _g
    out = []
    for f in sorted(_g.glob(str(Path(a.out) / "d[12]" / "*.pt"))):
        stem = Path(f).stem
        t, e = stem.rsplit("_ep", 1)
        out.append((t, int(e)))
    return sorted(set(out))


def cmd_train_motor(a):
    from research.brainloop.lab import train_c
    from research.brainloop.lab.demos import open_episode
    data = str(Path(a.assets) / "robotwin")
    te = _out_tasks(a)
    adapter, _ = _adapter(a, te[0][0], [open_episode(data, t, i) for t, i in te])
    res = train_c.train_motor(adapter, str(Path(a.out) / "d1"), data, str(Path(a.out) / "t1"), steps=a.steps,
                              batch=a.batch, lr=a.lr, max_minutes=a.max_minutes, heldout_from=a.heldout_from,
                              params=a.t1_params)
    print(json.dumps(res), flush=True)
    marker("train_motor", **res)


def cmd_train_t23(a):
    import numpy as np
    import torch
    from research.brainloop.lab import train_c
    from research.brainloop.lab.demos import open_episode
    data = str(Path(a.assets) / "robotwin")
    pairs = train_c.t23_pairs(str(Path(a.out) / "d1"), str(Path(a.out) / "d2"), ks=(8, 16))
    r2 = train_c.train_shift(pairs, str(Path(a.out) / "t2"), epochs=a.t2_epochs, eval_from=a.t23_eval_from)
    te = _out_tasks(a)
    adapter, _ = _adapter(a, te[0][0], [open_episode(data, t, i) for t, i in te])
    r3 = train_c.train_surprise(pairs, str(Path(a.out) / "t3"), denorm=lambda x: adapter.denorm(x),
                                eval_from=a.t23_eval_from)
    print(json.dumps({"t2": {k: v for k, v in r2.items() if k != "history"}, "t3": r3}), flush=True)
    marker("train_t23", t2_improvement=r2["improvement"], t3_auroc_eval=r3["auroc_eval"])


def cmd_eval_c(a):
    from research.brainloop.lab import eval_c
    from research.brainloop.lab.demos import open_episode
    data = str(Path(a.assets) / "robotwin")
    o = Path(a.out)
    te = _out_tasks(a)
    eps = [open_episode(data, t, i) for t, i in te]
    adapter, _ = _adapter(a, te[0][0], eps)
    m = eval_c.eval_motor(adapter, str(o / "d1"), str(o / "d2"), data, str(o / "t1" / "motor_t1.pt"), str(o),
                          heldout_from=a.heldout_from)
    print(json.dumps(m), flush=True)
    adapter, _ = _adapter(a, te[0][0], eps)     # fresh base weights
    c = eval_c.eval_chain(adapter, str(o / "d1"), str(o / "d2"), data, str(o / "t2" / "shift_t2.pt"), str(o),
                          eval_from=a.t23_eval_from)
    print(json.dumps(c), flush=True)
    lat = eval_c.latency_c(adapter, eps[0], str(o / "t1" / "motor_t1.pt"),
                           str(o / "t2" / "shift_t2.pt"), str(o))
    print(json.dumps(lat), flush=True)
    marker("eval_c", t1=m.get("verdict_T1"), t2=c.get("verdict_T2"))


def cmd_analyze(a):
    from research.brainloop.lab.analysis import analyze
    S = analyze(a.out)
    print(open(Path(a.out) / "report.md").read(), flush=True)
    marker("analyze", e1=S.get("E1", {}).get("verdict"), e2=S.get("E2", {}).get("verdict"),
           e3=S.get("E3", {}).get("verdict"))


def cmd_all(a):
    for stage in ("proof", "sweep", "e4", "e5", "analyze"):
        rc = subprocess.call([sys.executable, "-u", __file__, stage] + sys.argv[2:])
        if rc != 0:
            marker("all", ok=False, failed_stage=stage)
            sys.exit(rc)
    marker("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "proof", "sweep", "e4", "e5", "chain", "e5b", "analyze", "all", "tiny",
                                    "teacher", "train_motor", "train_t23", "eval_c"])
    ap.add_argument("--d1-demos", type=int, default=20)
    ap.add_argument("--d1-from", type=int, default=0)
    ap.add_argument("--d1-every", type=int, default=1)
    ap.add_argument("--d2-from", type=int, default=16)
    ap.add_argument("--d2-every", type=int, default=2)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3, help="T1 rate for the new zero-init staleness parameters")
    ap.add_argument("--t1-params", default="stale", choices=["stale", "light", "full"])
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--t2-epochs", type=int, default=30)
    ap.add_argument("--heldout-from", type=int, default=16, help="demos >= this are never trained on by T1")
    ap.add_argument("--t23-eval-from", type=int, default=18, help="D2 demos >= this evaluate T2/T3")
    ap.add_argument("--spec", default="", help="per-task episode counts, e.g. click_alarmclock:2,hanging_mug:1")
    ap.add_argument("--e5b-mode", default="core", choices=["core", "fastjoint", "reflex"])
    ap.add_argument("--assets", default="/workspace/assets")
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--max-anchors", type=int, default=None)
    ap.add_argument("--e4-anchors", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-pixels", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.cmd == "tiny":                                  # CPU plumbing run of every stage on the tiny model
        a.tiny = True
        a.max_anchors = a.max_anchors or 6
        a.episodes = 1
        a.e4_anchors = 1
        a.reps = 2
        a.no_compile = True
        for stage in (cmd_proof, cmd_sweep, cmd_e4, cmd_e5, cmd_chain, cmd_e5b, cmd_analyze):
            stage(a)
        marker("tiny")
        return
    try:
        globals()[f"cmd_{a.cmd}"](a)
    except Exception:
        traceback.print_exc()
        marker(a.cmd, ok=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
