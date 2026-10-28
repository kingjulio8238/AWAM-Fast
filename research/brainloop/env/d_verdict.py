#!/usr/bin/env python3
"""Stage D verdicts against the pre-registered bars (STAGE_D_PLAN.md). Pools any number of run_c5 results.jsonl files
(Stage C controls + Stage D runs; rows for the same task x config x mode from different seeds are concatenated).

    python research/brainloop/env/d_verdict.py --latency results/stageD/d2/latency_d.json \
        results/stageC/c1/c5first/results.jsonl results/stageD/c5/results.jsonl
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def load(paths):
    eps = defaultdict(list)                     # (task, cfg, mode) -> [((file, (seed, ep_index)), t, success)]
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            for i, e in enumerate(r["ep_stats"]):
                eps[(r["task"], r["cfg"], r["mode"])].append(((p, (r.get("seed", 0), i)), e["t"], e["t"] < e["step_lim"]))
    return eps


def paired(eps, a, b, tasks):
    """Median step ratio a/b over episodes (same task, same seed index) where both succeeded."""
    rs = []
    for t in tasks:
        xa = {i[1]: (s, ok) for i, s, ok in eps.get((t,) + a, [])}
        xb = {i[1]: (s, ok) for i, s, ok in eps.get((t,) + b, [])}
        rs += [xa[k][0] / xb[k][0] for k in xa.keys() & xb.keys() if xa[k][1] and xb[k][1]]
    return (float(np.median(rs)) if rs else None), len(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--latency", required=True)
    ap.add_argument("--tasks", nargs="+", default=["click_alarmclock", "beat_block_hammer", "open_microwave"])
    ap.add_argument("--configs", nargs="+", default=["reflex", "reflex2", "cold8async", "cold4async"])
    a = ap.parse_args()
    lat = json.load(open(a.latency))
    eps = load(a.results)
    ref = ("released", "sync")
    succ = lambda k: (sum(ok for t in a.tasks for _, _, ok in eps.get((t,) + k, [])),          # noqa: E731
                      sum(1 for t in a.tasks for _ in eps.get((t,) + k, [])))
    s_ref = succ(ref)
    print(f"reference released/sync: {s_ref[0]}/{s_ref[1]}")
    print("| config | mode | latency p95 (+tick) | success | per-task (vs ref) | steps vs ref (n) | verdict |")
    print("|---|---|---|---|---|---|---|")
    for cfg in a.configs:
        for mode in ("realtime", "sync"):
            k = (cfg, mode)
            if not any((t,) + k in eps for t in a.tasks):
                continue
            s = succ(k)
            per = []
            for t in a.tasks:
                x, r = eps.get((t,) + k, []), eps.get((t,) + ref, [])
                per.append(f"{sum(o for _, _, o in x)}/{len(x)} ({sum(o for _, _, o in r)}/{len(r)})")
            ratio, n = paired(eps, k, ref, a.tasks)
            key = {"reflex": "reflex4", "reflex2": "reflex2", "cold8async": "cold8", "cold4async": "cold4"}.get(cfg)
            ms = lat.get(key)
            e2e = ms + 1000 / 30 if ms is not None else None
            if cfg.startswith("reflex") and mode == "realtime":
                ok = (e2e is not None and e2e <= 150 and s[0] >= s_ref[0] - 1 and ratio is not None and ratio <= 1.15)
                verdict = "PASS" if ok else "FAIL"
            elif cfg.startswith("reflex") and mode == "sync":
                # kill: >= 2 episodes below the reference on some task
                kill = any(sum(o for _, _, o in eps.get((t,) + k, [])) <= sum(o for _, _, o in eps.get((t,) + ref, [])) - 2
                           for t in a.tasks)
                verdict = "KILL" if kill else "ok"
            else:
                verdict = "-"
            print(f"| {cfg} | {mode} | {('%.0f (%.0f)' % (ms, e2e)) if ms else '-'} | {s[0]}/{s[1]} | {'; '.join(per)} | "
                  f"{('%.2f' % ratio) if ratio else '-'} ({n}) | {verdict} |")


if __name__ == "__main__":
    main()
