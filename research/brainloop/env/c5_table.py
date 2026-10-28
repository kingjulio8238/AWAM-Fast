#!/usr/bin/env python3
"""Markdown table from one or more run_c5.py results.jsonl files: success and mean steps (successful episodes and
all episodes) per task x config x mode. Steps are control steps at 30 Hz, including steps held while the model thinks.

    python research/brainloop/env/c5_table.py results/stageC/c1/c5first/results.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np


def main(paths):
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[(r["cfg"], r["mode"])].append(r)
    print("| task | config | mode | success | steps (succ.) | steps (all) | held/ep |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["task"], r["mode"] != "sync", r["cfg"])):
        res = r["result"] or {}
        eps = r["ep_stats"]
        ok = [e["t"] for e, s in zip(eps, _success(r)) if s]
        print(f"| {r['task']} | {r['cfg']} | {r['mode']} | {res.get('success')}/{res.get('total')} | "
              f"{np.mean(ok):.0f} | {np.mean([e['t'] for e in eps]):.0f} | {np.mean([e['held'] for e in eps]):.0f} |"
              if ok else
              f"| {r['task']} | {r['cfg']} | {r['mode']} | {res.get('success')}/{res.get('total')} | - | "
              f"{np.mean([e['t'] for e in eps]):.0f} | {np.mean([e['held'] for e in eps]):.0f} |")
    print("\n| config | mode | success (pooled) | steps (succ., pooled) |")
    print("|---|---|---|---|")
    for (cfg, mode), rr in sorted(by.items(), key=lambda kv: (kv[0][1] != "sync", kv[0][0])):
        s = sum((r["result"] or {}).get("success", 0) for r in rr)
        n = sum((r["result"] or {}).get("total", 0) for r in rr)
        ok = [e["t"] for r in rr for e, ss in zip(r["ep_stats"], _success(r)) if ss]
        print(f"| {cfg} | {mode} | {s}/{n} | {np.mean(ok):.0f} |" if ok else f"| {cfg} | {mode} | {s}/{n} | - |")


def _success(r):
    """Per-episode success. The wrapper's own flag reads a reset env for all but the last episode, so use step counts:
    an episode that stopped before the step limit ended in success (RoboTwin stops early only on success)."""
    return [e["t"] < e["step_lim"] for e in r["ep_stats"]]


if __name__ == "__main__":
    main(sys.argv[1:])
