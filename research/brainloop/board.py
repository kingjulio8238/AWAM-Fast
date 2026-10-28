#!/usr/bin/env python3
"""One status board for every brainloop job, regenerated from the jobs' own artifacts.

    python research/brainloop/board.py [--jobs research/brainloop/results/jobs.json]

jobs.json: [{"name": ..., "kind": "docker"|"vast"|"progress", "log": path | "instance": id | "progress": path,
             "eta_utc": "...", "budget_usd": float}]
Row: job | phase | progress | result so far | ETA (UTC) | $ spent
"""
import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone


def _age(ts):
    return f"{(time.time() - ts) / 60:.0f}m ago"


def row_docker(j):
    if not os.path.exists(j["log"]):
        return ["pending", "…", "…"]
    txt = open(j["log"], errors="replace").read()
    steps = re.findall(r"#(\d+) \[\s*(\d+)/(\d+)\]", txt)
    phase = "done" if "exit=0" in txt else ("FAILED" if re.search(r"exit=[1-9]", txt) else "building")
    prog = f"step {steps[-1][1]}/{steps[-1][2]}" if steps else "starting"
    res = "imports ok" if "BRAINLOOP check_env DONE" in txt else ("error: " + txt.strip().splitlines()[-1][:60]
                                                                    if phase == "FAILED" else "…")
    return [phase, f"{prog} (log {_age(os.path.getmtime(j['log']))})", res]


def row_vast(j):
    try:
        d = json.loads(subprocess.run(["vastai", "show", "instance", str(j["instance"]), "--raw"], capture_output=True,
                                      text=True, timeout=30).stdout)
        up = (time.time() - float(d.get("start_date") or time.time())) / 3600
        return [d.get("actual_status", "?"), f"up {up:.2f}h", f"${d['dph_total']:.3f}/h"], up * d["dph_total"]
    except Exception as e:  # report, do not hide
        return ["UNKNOWN", f"query failed: {e}", "…"], None


def row_progress(j):
    p = j["progress"]
    if not os.path.exists(p):
        return ["pending", "…", "…"]
    d = json.load(open(p))
    stale = time.time() - d["ts"]
    done = "DONE" in d.get("job", "")
    phase = d.get("job", "?") + (" STALE" if not done and stale > 2 * max(60, 3 * d.get("last_decode_s", 20)) else "")
    prog = f"ep {d.get('episodes_done', 0)}/{d.get('episodes_total', '?')}, anchor {d.get('anchor', 0)}/{d.get('anchors', '?')}"
    return [phase, prog, f"{d.get('ep_task', '')} decode {d.get('last_decode_s', '…')}s, updated {_age(d['ts'])}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=os.path.join(os.path.dirname(__file__), "results", "jobs.json"))
    a = ap.parse_args()
    jobs = json.load(open(a.jobs)) if os.path.exists(a.jobs) else []
    rows = []
    for j in jobs:
        spent = j.get("spent_usd")
        if j["kind"] == "docker":
            r = row_docker(j)
        elif j["kind"] == "vast":
            r, spent = row_vast(j)
        else:
            r = row_progress(j)
        rows.append([j["name"]] + r + [j.get("eta_utc", "…"), "…" if spent is None else f"${spent:.2f}"])
    hdr = ["job", "phase", "progress", "result so far", "ETA (UTC)", "$ spent"]
    w = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)] if rows else [len(h) for h in hdr]
    line = lambda r: "| " + " | ".join(str(x).ljust(n) for x, n in zip(r, w)) + " |"  # noqa: E731
    print(f"board @ {datetime.now(timezone.utc):%H:%M} UTC")
    print(line(hdr))
    print("|" + "|".join("-" * (n + 2) for n in w) + "|")
    for r in rows:
        print(line(r))


if __name__ == "__main__":
    main()
