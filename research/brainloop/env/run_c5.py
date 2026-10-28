#!/usr/bin/env python3
"""C5: RoboTwin success for the brainloop configs, synchronous and real-time, through the repo's own eval launcher
(robotwin/hybrid_eval/launch.py --mode wam). Same seed for every config, so episodes are paired.

    python -u research/brainloop/env/run_c5.py --tasks click_alarmclock --episodes 2 --configs released --modes sync
Writes /workspace/out/c5/results.jsonl (one row per task x config x mode) and progress.json; resumable.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONFIGS = {
    "released": {"config": "released"},
    "warm1": {"config": "warm1", "replan": 8},
    # C1 sim: pure warm chains stall in closed loop (400-step timeouts); a cold refresh every 3rd plan fixes it
    "warm1c3": {"config": "warm1", "replan": 8, "cold_every": 3},
    "coldasync": {"config": "warm1", "replan": 8, "cold_every": 1},   # control: compiled cold-20 only, async
    # compiled cold-4 decodes only, async (Stage A: 181 ms p95 on the 5090)
    "cold4async": {"config": "warm1", "replan": 8, "cold_every": 1, "cold_steps": 4, "latency_ms": {"cold": 181.0}},
    # cold-8: latency interpolated from the measured cold-4 (181 ms) and cold-20 (799 ms) at 38.6 ms per step
    "cold8async": {"config": "warm1", "replan": 8, "cold_every": 1, "cold_steps": 8, "latency_ms": {"cold": 335.0}},
    # Stage D: slow cold-8 every 24 steps + the fresh-vision reflex every 4 (N action steps from noise)
    "reflex": {"config": "reflex", "cold_steps": 8, "slow_every": 24, "reflex_every": 4, "reflex_steps": 4,
               "latency_ms": {"cold": 335.0}},
    # D3 diagnostics for the sync stall: adopt fresh chunks from index 3 (as realtime does), or re-plan every 8
    "reflex_s3": {"config": "reflex", "cold_steps": 8, "slow_every": 24, "reflex_every": 4, "reflex_steps": 4,
                  "sync_offset": 3, "latency_ms": {"cold": 335.0}},
    "reflex8": {"config": "reflex", "cold_steps": 8, "slow_every": 24, "reflex_every": 8, "reflex_steps": 4,
                "latency_ms": {"cold": 335.0}},
    # D5: a faster slow tier, so the code (sub-task) and video prediction lag less on long multi-stage tasks
    "reflex_c4s12": {"config": "reflex", "cold_steps": 4, "slow_every": 12, "reflex_every": 4, "reflex_steps": 4,
                     "latency_ms": {"cold": 190.0}},
    "reflex_c8s16": {"config": "reflex", "cold_steps": 8, "slow_every": 16, "reflex_every": 4, "reflex_steps": 4,
                     "latency_ms": {"cold": 348.0}},
    "reflex2": {"config": "reflex", "cold_steps": 8, "slow_every": 24, "reflex_every": 4, "reflex_steps": 2,
                "latency_ms": {"cold": 335.0}},
    "brain": {"config": "brain", "replan": 8},
    # warm1 diagnostics: released cadence, a cold refresh every 3rd plan, and a fully re-masked code canvas
    "warm1r24": {"config": "warm1", "replan": 24},
    "warm1m1": {"config": "warm1", "replan": 8, "remask": 1.0},
    "cold8": {"config": "released", "released_replan": 8},     # cadence control: cold-20 every 8 steps
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--modes", nargs="+", default=["sync", "realtime"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latency-json", default="", help='{"cold8": ms, "cold4": ms, "reflex4": ms, ...} from D2; '
                    'overrides the configs\' latency model')
    ap.add_argument("--out", default="/workspace/out/c5")
    ap.add_argument("--ckpt", default="/workspace/assets/release/wam.pt")
    ap.add_argument("--motor-ckpt", default="/workspace/out/t1/motor_t1.pt")
    ap.add_argument("--shift-ckpt", default="/workspace/out/t2/shift_t2.pt")
    ap.add_argument("--surprise-ckpt", default="/workspace/out/t3/surprise_t3.pt")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task"], r["cfg"], r["mode"]))
    jobs = [(t, c, m) for t in a.tasks for c in a.configs for m in a.modes]
    t0 = time.time()
    for n, (task, cfg_name, mode) in enumerate(jobs, 1):
        if (task, cfg_name, mode) in done:
            continue
        cfg = dict(CONFIGS[cfg_name], realtime=(mode == "realtime"))
        if a.latency_json:                                 # measured D2 latencies replace the interpolated ones
            lat = json.loads(Path(a.latency_json).read_text())
            base = cfg.get("config")
            ms = dict(cfg.get("latency_ms", {}))
            if base in ("warm1", "reflex") and f"cold{cfg.get('cold_steps', 20)}" in lat:
                ms["cold"] = lat[f"cold{cfg.get('cold_steps', 20)}"]
            if base == "reflex" and f"reflex{cfg.get('reflex_steps', 4)}" in lat:
                ms["reflex"] = lat[f"reflex{cfg.get('reflex_steps', 4)}"]
            cfg["latency_ms"] = ms
        if CONFIGS[cfg_name]["config"] in ("warm1", "brain") and os.path.exists(a.shift_ckpt):
            cfg["shift_ckpt"] = a.shift_ckpt
        if cfg_name == "brain":
            cfg["motor_ckpt"] = a.motor_ckpt
            if os.path.exists(a.surprise_ckpt):
                cfg["surprise_ckpt"] = a.surprise_ckpt
        tag = f"bl_{cfg_name}_{mode}"
        for stale in (REPO / "runs" / "eval_logs").glob(f"{tag}_{task}_s{a.seed}.log"):
            stale.unlink()
        for stale in (REPO / "evaluate_results" / "robotwin").glob(f"*/{tag}_{task}_g0_s{a.seed}"):
            shutil.rmtree(stale)
        env = dict(os.environ, AGENTICWAM_BRAINLOOP=json.dumps(cfg), AGENTICWAM_DEPLOY_DTYPE="bf16",
                   AGENTICWAM_TEXT_DEVICE="cpu", PYTHONUNBUFFERED="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, "-u", str(REPO / "robotwin" / "hybrid_eval" / "launch.py"), "--task", task,
               "--model", "robotwin_wam", "--ckpt", a.ckpt, "--mode", "wam", "--proto", "demo_clean",
               "--seed", str(a.seed), "--episodes", str(a.episodes), "--shard-gpu", "0", "--no-render", "--tag", tag]
        t1 = time.time()
        rc = subprocess.call(cmd, cwd=REPO, env=env)
        meta_p = next(iter((REPO / "evaluate_results" / "robotwin").glob(f"*/{tag}_{task}_g0_s{a.seed}/{task}/run_meta.json")), None)
        meta = json.loads(meta_p.read_text()) if meta_p else {}
        log = REPO / "runs" / "eval_logs" / f"{tag}_{task}_s{a.seed}.log"
        lines = log.read_text(errors="replace").splitlines() if log.exists() else []
        eps = [json.loads(l.split("BRAINLOOP_EPISODE ", 1)[1]) for l in lines if "BRAINLOOP_EPISODE " in l]
        if not any("BRAINLOOP_INSTALLED" in l for l in lines):     # a run without the wrapper measures the wrong policy
            print(f"BRAINLOOP c5 FAILED: wrapper not installed for {tag} ({log})", flush=True)
            sys.exit(2)
        row = {"task": task, "cfg": cfg_name, "mode": mode, "rc": rc, "result": meta.get("result"),
               "wall_s": round(time.time() - t1), "episodes": a.episodes, "seed": a.seed,
               "steps_mean": float(np.mean([e["steps"] for e in eps])) if eps else None,
               "held_mean": float(np.mean([e["held"] for e in eps])) if eps else None,
               "delib_mean": float(np.mean([e["delib"] for e in eps])) if eps else None, "ep_stats": eps}
        with open(res_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[c5] {n}/{len(jobs)} {row}", flush=True)
        json.dump({"job": "c5", "episodes_done": n, "episodes_total": len(jobs), "ep_task": f"{task}/{cfg_name}/{mode}",
                   "anchor": n, "anchors": len(jobs), "elapsed_s": round(time.time() - t0), "ts": time.time(),
                   "last_decode_s": row["wall_s"]}, open(out.parent / "progress.json", "w"))
    print("BRAINLOOP c5 DONE", flush=True)


if __name__ == "__main__":
    main()
